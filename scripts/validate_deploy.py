"""Deployment rehearsal, exclusively synthetic inputs. JSON on stdout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import platform
from importlib.metadata import version
from lab.runtime import prepare_runtime
from lab.data import BootstrapConfig, simulate
from lab.engines import run_r, run_python, compare_results

status = prepare_runtime()
assert status["available"], status
report = {"R": status, "Python": platform.python_version(), "mtest-py": version("mtest-py"),
          "platform": platform.platform(), "scenarios": []}
for rho in [.1, .95, .999]:
    data = simulate(n=300, rho=rho, noise=1, seed=27)
    r = run_r(data, "y", ["X1", "X2"], BootstrapConfig())
    py = run_python(data, "y", ["X1", "X2"], BootstrapConfig())
    ok, reasons, _ = compare_results(r, py)
    assert ok, reasons
    report["scenarios"].append({"rho": rho, "identical_input": r.identical_input,
        "observed_metrics_pass": ok, "R2_global": r.r2_global, "R2_aux": r.r2_aux.tolist(),
        "VIF": r.vif.tolist(), "ASL_VIF_R": r.asl_vif.tolist(), "ASL_VIF_Python": py.asl_vif.tolist(),
        "ASL_Klein_R": r.asl_klein.tolist(), "ASL_Klein_Python": py.asl_klein.tolist(),
        "seconds_R": r.seconds, "seconds_Python": py.seconds,
        "max_VIF_difference": abs(r.vif-py.vif).max()})
print(json.dumps(report, indent=2))
