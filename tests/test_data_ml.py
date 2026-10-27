import io
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from lab.data import BootstrapConfig, DataError, MAX_BYTES, csv_bytes, read_csv, simulate, validate_data
from lab.ml import compare_models
from lab.reproduce import reproduction_zip


def test_simulator_reproducible():
    pd.testing.assert_frame_equal(simulate(seed=27), simulate(seed=27))
    assert not simulate(seed=27).equals(simulate(seed=28))


def test_csv_limits_and_validation():
    d = simulate()
    pd.testing.assert_frame_equal(d, read_csv(csv_bytes(d)))
    for invalid in [b"x"*(MAX_BYTES+1), b"y,x,x\n1,2,3", b"", b"y,x,z\n\xff,2,3"]:
        with pytest.raises(DataError): read_csv(invalid)
    with pytest.raises(DataError): validate_data(d, "y", ["X1", "y"])
    with pytest.raises(DataError): validate_data(d.iloc[:10], "y", ["X1", "X2"])
    d["X1"] = "category"
    with pytest.raises(DataError): validate_data(d, "y", ["X1", "X2"])


@pytest.mark.parametrize("cfg", [BootstrapConfig(n_boot=0), BootstrapConfig(nsam=3),
                                 BootstrapConfig(seed=-1), BootstrapConfig(r2_threshold=1),
                                 BootstrapConfig(n_boot=1.5), BootstrapConfig(n_boot=2000, nsam=5000)])
def test_invalid_bootstrap_parameters(cfg):
    with pytest.raises(DataError): cfg.validate(2)


def test_ml_no_leakage_and_stability_units():
    d = simulate(rho=.95)
    result = compare_models(d, "y", ["X1", "X2"], repeats=5)
    train, test = result["train"], result["test"]
    assert not set(train) & set(test)
    assert all(set(rows) <= set(train) for rows in result["samples"])
    for model in result["models"].values():
        np.testing.assert_allclose(model[0].mean_, d[["X1", "X2"]].iloc[train].mean())
    # Changing held-out X and y must leave all fitted models and stability unchanged.
    changed = d.copy()
    changed.loc[test, ["X1", "X2"]] += 100
    changed.loc[test, "y"] -= 1000
    other = compare_models(changed, "y", ["X1", "X2"], repeats=5)
    pd.testing.assert_frame_equal(result["coefficients"], other["coefficients"])
    pd.testing.assert_frame_equal(result["stability"], other["stability"])
    assert not np.allclose(result["metrics"]["MSE test"], other["metrics"]["MSE test"])


def test_reproducible_download_executes_both_languages(tmp_path):
    data = simulate(n=50)
    archive = reproduction_zip(data, "y", ["X1", "X2"], BootstrapConfig(n_boot=20, nsam=50))
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        z.extractall(tmp_path)
    for command in [[sys.executable, "run_python.py"], ["Rscript", "--vanilla", "run_R.R"]]:
        p = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
