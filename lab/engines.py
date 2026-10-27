from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version

import numpy as np
import pandas as pd
from mtest import mtest as official_mtest

from lab.data import BootstrapConfig, data_id, transport_csv, validate_data

ROOT = Path(__file__).resolve().parents[1]


class EngineError(RuntimeError):
    pass


def rscript_path():
    path = os.environ.get("MTEST_RSCRIPT") or shutil.which("Rscript")
    if not path:
        raise EngineError("Rscript no está disponible en el entorno.")
    return path


def r_probe():
    try:
        proc = subprocess.run([rscript_path(), "--vanilla", str(ROOT/"r/run_mtest.R"), "--probe"],
                              capture_output=True, text=True, timeout=20, cwd=ROOT)
        if proc.returncode:
            return {"available": False, "error": "No se verificó R >= 4.1 con MTest 1.0.4 y jsonlite."}
        return {"available": True, **json.loads(proc.stdout)}
    except (EngineError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {"available": False, "error": "No se pudo comprobar el entorno R dentro del tiempo permitido."}


@dataclass
class Result:
    engine: str
    version: str
    predictors: list[str]
    fingerprint: str
    config: BootstrapConfig
    r2_global: float
    r2_aux: np.ndarray
    vif: np.ndarray
    asl_vif: np.ndarray
    asl_klein: np.ndarray
    boot_global: np.ndarray
    boot_aux: np.ndarray
    seconds: float
    raw: dict
    identical_input: bool = True

    @property
    def valid_vif(self):
        return np.isfinite(self.boot_aux) & (self.boot_aux < 1) & (self.boot_aux >= -1e-10)

    @property
    def valid_klein(self):
        return self.valid_vif & np.isfinite(self.boot_global[:, None])

    @property
    def clean(self):
        return bool(self.valid_klein.all() and np.isfinite(self.vif).all()
                    and np.isfinite(self.r2_aux).all() and np.isfinite(self.r2_global)
                    and np.isfinite(self.asl_vif).all() and np.isfinite(self.asl_klein).all())

    def table(self):
        valid_vif = self.valid_vif.sum(axis=0)
        valid_klein = self.valid_klein.sum(axis=0)
        return pd.DataFrame({"Predictor": self.predictors, "R² auxiliar": self.r2_aux,
            "VIF observado": self.vif,
            "ASL VIF": np.where(valid_vif == self.config.n_boot, self.asl_vif, np.nan),
            "ASL Klein": np.where(valid_klein == self.config.n_boot, self.asl_klein, np.nan),
            "Réplicas válidas VIF": valid_vif, "Réplicas válidas Klein": valid_klein,
            "Lectura": ["No interpretable: revisar remuestreos" if valid_klein[j] < self.config.n_boot
                        else ("VIF observado supera el umbral" if self.vif[j] > 1/(1-self.config.r2_threshold)
                              else "VIF observado bajo el umbral") for j in range(len(self.predictors))]})

    def export_table(self):
        table = self.table()
        table["ASL VIF original del paquete"] = self.asl_vif
        table["ASL Klein original del paquete"] = self.asl_klein
        table["Motor"] = self.engine
        table["Versión"] = self.version
        table["R² global"] = self.r2_global
        for k, v in asdict(self.config).items():
            table[k] = v
        table["dataset_sha256"] = self.fingerprint
        return table


def _check_shapes(result):
    p, B = len(result.predictors), result.config.n_boot
    if result.boot_aux.shape != (B, p) or result.boot_global.shape != (B,):
        raise EngineError("La API devolvió distribuciones con dimensiones incompatibles.")
    if any(a.shape != (p,) for a in (result.vif, result.r2_aux, result.asl_vif, result.asl_klein)):
        raise EngineError("La API no conserva el orden o el número de predictores.")
    return result


def run_python(frame, response, predictors, config):
    frame, _ = validate_data(frame, response, predictors)
    config.validate(len(predictors))
    started = time.perf_counter()
    try:
        if version("mtest-py") != "0.1.4":
            raise EngineError("Se requiere mtest-py 0.1.4.")
        payload = {"csv": transport_csv(frame).decode(), "response": response,
                   "predictors": list(predictors), "config": asdict(config)}
        proc = subprocess.run([sys.executable, "-m", "lab.python_worker"],
                              input=json.dumps(payload), capture_output=True, text=True,
                              timeout=90, cwd=ROOT)
        if proc.returncode:
            raise EngineError("mtest-py no completó el diagnóstico. Revise el diseño y los parámetros.")
        raw = json.loads(proc.stdout)
        result = Result("Python", version("mtest-py"), list(predictors), data_id(frame), config,
            float(raw["R2_global"]), np.asarray(raw["R2_aux"], float), np.asarray(raw["VIF"], float),
            np.array([raw["p_vif"][p] for p in predictors]),
            np.array([raw["p_klein"][p] for p in predictors]),
            np.asarray(raw["B_R2_global"], float), np.asarray(raw["B_R2_aux"], float),
            time.perf_counter()-started, raw)
        return _check_shapes(result)
    except subprocess.TimeoutExpired as exc:
        raise EngineError("Python superó 90 s; reduzca nboot, nsam o predictores.") from exc
    except (OSError, KeyError, ValueError, np.linalg.LinAlgError, IndexError) as exc:
        raise EngineError("mtest-py no pudo completar el diagnóstico; revise el diseño.") from exc


def r_command(csv_path, response, predictors, config, executable=None):
    return [executable or rscript_path(), "--vanilla", str(ROOT/"r/run_mtest.R"),
            "--csv", str(csv_path), "--response", response, "--predictors", json.dumps(list(predictors)),
            "--nboot", str(config.n_boot), "--nsam", str(config.nsam),
            "--threshold", str(config.r2_threshold), "--seed", str(config.seed)]


def run_r(frame, response, predictors, config, executable=None, timeout=90):
    frame, _ = validate_data(frame, response, predictors)
    config.validate(len(predictors))
    started = time.perf_counter()
    try:
        # A single temporary CSV is removed on success, error and timeout.
        with tempfile.TemporaryDirectory(prefix="mtest-live-") as tmp:
            path = Path(tmp)/"design.csv"
            path.write_bytes(transport_csv(frame))
            proc = subprocess.run(r_command(path, response, predictors, config, executable),
                                  capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        if proc.returncode:
            raise EngineError("R no completó MTest. Revise el diseño y los parámetros.")
        raw = json.loads(proc.stdout)
        expected = ["(Intercept)", *predictors]
        if raw["design_columns"] != expected or not raw["intercept"]:
            raise EngineError("R devolvió una matriz de diseño distinta.")
        # Hex strings preserve IEEE-754 values through JSON without rounding.
        echoed = np.array([[float.fromhex(v) for v in row] for row in raw.pop("input_hex")])
        identical = np.array_equal(echoed, frame.to_numpy())
        if not identical:
            raise EngineError("La lectura CSV en R alteró los datos: comparación bloqueada.")
        observed = np.asarray(raw["R.tot"], dtype=float)
        boot = np.asarray(raw["Bvals"], dtype=float)
        result = Result("R", raw["version"], list(predictors), data_id(frame), config,
            float(observed[0]), observed[1:], np.asarray(raw["vif.tot"], float),
            np.asarray(raw["pval_vif"], float), np.asarray(raw["pval_klein"], float),
            boot[:, 0], boot[:, 1:], time.perf_counter()-started, raw, identical)
        return _check_shapes(result)
    except subprocess.TimeoutExpired as exc:
        raise EngineError("R superó 90 s; reduzca nboot, nsam o predictores.") from exc
    except OSError as exc:
        raise EngineError("No se pudo iniciar Rscript. Compruebe la instalación o MTEST_RSCRIPT.") from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise EngineError("Respuesta R incompatible o no válida; no se muestran resultados.") from exc


def compare_results(r, py):
    reasons = []
    if (r.fingerprint, r.predictors, r.config) != (py.fingerprint, py.predictors, py.config):
        reasons.append("Los datos, columnas o parámetros difieren.")
    if not r.identical_input or not py.identical_input:
        reasons.append("No se verificó identidad de datos.")
    if not r.clean or not py.clean:
        reasons.append("Existen réplicas degeneradas o valores no finitos; las ASL no son comparables.")
    if r.predictors != py.predictors:
        return False, reasons, pd.DataFrame()
    for a, b in [(r.r2_global, py.r2_global), (r.r2_aux, py.r2_aux), (r.vif, py.vif)]:
        if not np.allclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=False):
            reasons.append("Las métricas observadas exceden la tolerancia numérica (rtol=1e-8, atol=1e-10).")
            break
    table = pd.DataFrame({"Predictor": r.predictors})
    for label, a, b in [("R² aux", r.r2_aux, py.r2_aux), ("VIF", r.vif, py.vif),
                        ("ASL VIF", r.table()["ASL VIF"], py.table()["ASL VIF"]),
                        ("ASL Klein", r.table()["ASL Klein"], py.table()["ASL Klein"])]:
        table[label+" · R"] = np.asarray(a)
        table[label+" · Py"] = np.asarray(b)
        table["Δ absoluto · "+label] = np.abs(np.asarray(a)-np.asarray(b))
    return not reasons, reasons, table
