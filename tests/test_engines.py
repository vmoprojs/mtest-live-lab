from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
from mtest import mtest

from lab.data import BootstrapConfig, DataError, simulate, transport_csv
from lab.engines import EngineError, compare_results, r_probe, rscript_path, run_python, run_r

PREDICTORS = ["X1", "X2"]


@pytest.fixture
def config():
    return BootstrapConfig(n_boot=200, nsam=300, seed=2026)


def test_real_packages_available():
    status = r_probe()
    assert status["available"], status
    assert status["MTest"] == "1.0.4"
    from importlib.metadata import version
    assert version("mtest-py") == "0.1.4"


@pytest.mark.parametrize("rho", [.1, .95, .999])
def test_dual_scenarios_same_data_observed_metrics(rho, config):
    d = simulate(rho=rho)
    r, py = run_r(d, "y", PREDICTORS, config), run_python(d, "y", PREDICTORS, config)
    comparable, reasons, _ = compare_results(r, py)
    assert comparable, reasons
    assert r.identical_input
    theoretical = 1/(1-d.X1.corr(d.X2)**2)
    np.testing.assert_allclose(r.vif, theoretical, rtol=1e-8)
    np.testing.assert_allclose(py.vif, theoretical, rtol=1e-8)
    if rho == .1:
        assert max(r.vif) < 1.2
        assert max(r.asl_vif) < .05
    elif rho == .95:
        assert min(r.vif) > 5
    else:
        assert min(r.vif) > 100
        assert min(r.asl_vif) > .95
    # Independent direct OLS check for global R².
    X = np.column_stack([np.ones(len(d)), d[PREDICTORS]])
    residual = d.y - X@np.linalg.lstsq(X, d.y, rcond=None)[0]
    expected = 1-np.sum(residual**2)/np.sum((d.y-d.y.mean())**2)
    np.testing.assert_allclose(r.r2_global, expected, atol=1e-12)


@pytest.mark.parametrize("engine", [run_r, run_python])
def test_seed_reproducibility(engine, config):
    d = simulate(rho=.95)
    a, b = engine(d, "y", PREDICTORS, config), engine(d, "y", PREDICTORS, config)
    np.testing.assert_array_equal(a.boot_aux, b.boot_aux)
    np.testing.assert_array_equal(a.boot_global, b.boot_global)
    other = engine(d, "y", PREDICTORS, replace(config, seed=2027))
    assert not np.array_equal(a.boot_aux, other.boot_aux)


def test_python_matches_direct_call(config):
    d = simulate(rho=.95)
    actual = run_python(d, "y", PREDICTORS, config)
    direct = mtest(d[PREDICTORS], d.y, n_boot=config.n_boot, nsam=config.nsam,
                   seed=config.seed, r2_threshold=config.r2_threshold, add_intercept=True,
                   return_distributions=True)
    for field, key in [("r2_aux", "R2_aux"), ("vif", "VIF"), ("boot_aux", "B_R2_aux"),
                       ("boot_global", "B_R2_global")]:
        np.testing.assert_array_equal(getattr(actual, field), direct[key])
    assert actual.r2_global == direct["R2_global"]
    assert dict(zip(PREDICTORS, actual.asl_vif)) == direct["p_vif"]
    assert dict(zip(PREDICTORS, actual.asl_klein)) == direct["p_klein"]


def test_r_matches_independent_direct_call(tmp_path, config):
    d = simulate(rho=.95)
    csv = tmp_path/"input.csv"
    csv.write_bytes(transport_csv(d))
    # Independent invocation of MTest::MTest, outside our bridge.
    script = tmp_path/"direct.R"
    script.write_text(f'''
    d <- read.csv(commandArgs(TRUE)[1], colClasses="character")
    d[] <- lapply(d, as.numeric)
    r <- MTest::MTest(lm(y~X1+X2, data=d), nboot={config.n_boot}, nsam={config.nsam},
                     seed={config.seed}, valor_vif={config.r2_threshold})
    cat(jsonlite::toJSON(list(B=unname(r$Bvals), VIF=unname(r$vif.tot), R2=unname(r$R.tot),
                             pv=unname(r$pval_vif), pk=unname(r$pval_klein)), digits=NA))
    ''')
    proc = subprocess.run([rscript_path(), "--vanilla", str(script), str(csv)], capture_output=True, text=True, check=True)
    direct = json.loads(proc.stdout)
    actual = run_r(d, "y", PREDICTORS, config)
    np.testing.assert_array_equal(actual.boot_global, np.array(direct["B"])[:,0])
    np.testing.assert_array_equal(actual.boot_aux, np.array(direct["B"])[:,1:])
    np.testing.assert_array_equal(actual.vif, direct["VIF"])
    np.testing.assert_array_equal(actual.r2_aux, direct["R2"][1:])
    np.testing.assert_array_equal(actual.asl_vif, direct["pv"])
    np.testing.assert_array_equal(actual.asl_klein, direct["pk"])


def test_column_order_and_arbitrary_names(config):
    d = simulate(rho=.5).rename(columns={"X1": "peso ` + raro", "X2": "ingreso (USD)", "y": "respuesta ñ"})
    preds = ["ingreso (USD)", "peso ` + raro"]
    r = run_r(d, "respuesta ñ", preds, config)
    py = run_python(d, "respuesta ñ", preds, config)
    assert compare_results(r, py)[0]
    assert r.predictors == preds


def test_nan_bootstrap_does_not_display_misleading_asl():
    d = simulate(n=100)
    d["X1"] = 0.
    d.loc[0, "X1"] = 1.
    cfg = BootstrapConfig(n_boot=100, nsam=20)
    r, py = run_r(d, "y", PREDICTORS, cfg), run_python(d, "y", PREDICTORS, cfg)
    assert not r.clean and not py.clean
    assert not compare_results(r, py)[0]
    assert np.isnan(r.table().loc[0, "ASL VIF"])
    assert np.isnan(py.table().loc[0, "ASL Klein"])
    assert "ASL VIF original del paquete" in py.export_table()


def test_r_missing_and_timeout(config, monkeypatch):
    d = simulate()
    with pytest.raises(EngineError, match="Rscript"):
        run_r(d, "y", PREDICTORS, config, executable="/nonexistent/Rscript")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(EngineError, match="superó"):
        run_r(d, "y", PREDICTORS, config)


def test_r_actual_script_error(tmp_path):
    # Real failing R invocation, not a mock.
    from lab.engines import r_command
    proc = subprocess.run(r_command(tmp_path/"missing.csv", "y", PREDICTORS, BootstrapConfig()),
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "MTEST_R_ERROR" in proc.stderr


def test_mismatched_parameters_rejected(config):
    d = simulate()
    a = run_r(d, "y", PREDICTORS, config)
    b = run_python(d, "y", PREDICTORS, replace(config, nsam=100))
    assert not compare_results(a, b)[0]


@pytest.mark.parametrize("engine", [run_r, run_python])
@pytest.mark.parametrize("bad", ["constant", "singular", "nan", "infinite"])
def test_invalid_data_blocked_before_engine(engine, bad, config):
    d = simulate()
    if bad == "constant": d["X1"] = 1.
    elif bad == "singular": d["X2"] = 2*d.X1 + 5
    elif bad == "nan": d.loc[0, "X1"] = np.nan
    else: d.loc[0, "y"] = np.inf
    with pytest.raises(DataError):
        engine(d, "y", PREDICTORS, config)
