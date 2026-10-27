"""Official Python call in a bounded child process. Input and output use pipes."""
import io
import json
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
from mtest import mtest


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unsupported package output")


def main():
    if version("mtest-py") != "0.1.4":
        raise RuntimeError("Expected mtest-py 0.1.4")
    payload = json.load(sys.stdin)
    frame = pd.read_csv(io.StringIO(payload["csv"]), dtype=str).map(float.fromhex)
    result = mtest(frame[payload["predictors"]], frame[payload["response"]].to_numpy(),
                   **payload["config"], add_intercept=True, return_distributions=True)
    json.dump(result, sys.stdout, default=encode)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never print user values, column names, paths or tracebacks.
        sys.stderr.write("MTEST_PYTHON_ERROR\n")
        sys.exit(1)
