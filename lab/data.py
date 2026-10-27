from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 5000
MAX_PREDICTORS = 8
SCENARIOS = {"Baja": 0.10, "Alta": 0.95, "Casi perfecta": 0.999}


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class BootstrapConfig:
    n_boot: int = 200
    nsam: int = 300
    seed: int = 2026
    r2_threshold: float = 0.90

    def validate(self, p: int):
        for name in ("n_boot", "nsam", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise DataError(f"{name} debe ser entero.")
        if not 20 <= self.n_boot <= 500:
            raise DataError("Use entre 20 y 500 réplicas.")
        if not max(20, p + 3) <= self.nsam <= 2000:
            raise DataError("nsam debe estar entre 20 (y p + 3) y 2000.")
        if not 0 <= self.seed <= 2**31 - 1:
            raise DataError("La semilla debe estar entre 0 y 2147483647.")
        if not np.isfinite(self.r2_threshold) or not 0 < self.r2_threshold < 1:
            raise DataError("El umbral R² debe ser estrictamente entre 0 y 1.")
        if self.n_boot * self.nsam * p > 2_000_000:
            raise DataError("Carga de cálculo excesiva para la demo: reduzca réplicas, nsam o predictores.")


def simulate(n=300, rho=0.10, noise=1.0, seed=27):
    if not 20 <= n <= MAX_ROWS or not -1 < rho < 1 or not 0 <= noise <= 20:
        raise DataError("Parámetros de simulación fuera de rango.")
    rng = np.random.default_rng(seed)
    x1, z, eps = rng.normal(size=(3, n))
    x2 = rho * x1 + np.sqrt(1 - rho**2) * z
    # Opposite effects make cancellation and coefficient instability visible.
    return pd.DataFrame({"y": 1 + 2*x1 - 2*x2 + noise*eps, "X1": x1, "X2": x2})


def csv_bytes(frame):
    return frame.to_csv(index=False, float_format="%.17g").encode("utf-8")


def transport_csv(frame):
    """CSV hexadecimal numeric text avoids language-specific decimal parsing."""
    return frame.map(lambda v: float(v).hex()).to_csv(index=False).encode("utf-8")


def read_csv(payload: bytes):
    if len(payload) > MAX_BYTES:
        raise DataError("El archivo supera el límite de 2 MB.")
    try:
        text = payload.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)))
        if len(header) != len(set(header)) or any(not c.strip() for c in header):
            raise DataError("Los encabezados deben ser únicos y no vacíos.")
        if not 3 <= len(header) <= 30:
            raise DataError("El CSV debe contener entre 3 y 30 columnas.")
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS+1, float_precision="round_trip")
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise DataError("CSV inválido: use UTF-8, coma como separador y punto decimal.") from exc
    if len(frame) > MAX_ROWS:
        raise DataError("El CSV supera las 5000 filas.")
    return frame


def validate_data(frame, response, predictors):
    predictors = list(predictors)
    if frame.columns.duplicated().any() or not 2 <= len(predictors) <= MAX_PREDICTORS:
        raise DataError("Seleccione de 2 a 8 predictores con nombres únicos.")
    if response in predictors or len(set(predictors)) != len(predictors):
        raise DataError("La respuesta no puede ser un predictor; evite nombres repetidos.")
    if any(c not in frame for c in [response, *predictors]):
        raise DataError("Hay columnas seleccionadas que no existen.")
    if not max(20, len(predictors)+3) <= len(frame) <= MAX_ROWS:
        raise DataError("Se requieren entre 20 y 5000 filas, y más observaciones que parámetros.")
    selected = frame[[response, *predictors]].copy()
    if any(not pd.api.types.is_numeric_dtype(selected[c]) or pd.api.types.is_bool_dtype(selected[c])
           for c in selected):
        raise DataError("Solo se admiten columnas numéricas, sin categorías ni booleanos.")
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataError("Existen faltantes, NaN o infinitos. Corríjalos antes de ejecutar; no se eliminan filas automáticamente.")
    spread = values.std(axis=0)
    if np.any(spread <= 1e-12 * np.maximum(1, np.abs(values).max(axis=0))):
        raise DataError("Hay variables constantes o sin variación numérica suficiente.")
    if np.abs(values).max() > 1e10 or np.any(spread < 1e-8):
        raise DataError("Rango numérico extremo: cambie las unidades antes de analizar.")
    X = values[:, 1:]
    design = np.column_stack([np.ones(len(X)), X])
    scaled = np.column_stack([np.ones(len(X)), (X-X.mean(axis=0))/X.std(axis=0)])
    if np.linalg.matrix_rank(scaled) < design.shape[1] or np.linalg.cond(design) > 1e10:
        raise DataError("Matriz singular o numéricamente no identificable: quite predictores redundantes.")
    warnings = []
    for c in selected:
        if selected[c].nunique() <= 10:
            warnings.append(f"{c}: baja cardinalidad ({selected[c].nunique()} valores); algún remuestreo podría ser degenerado.")
    if np.linalg.cond(scaled) > 1000:
        warnings.append("Diseño muy mal condicionado: interprete los coeficientes con cautela.")
    return selected.astype(float), warnings


def data_id(frame):
    return hashlib.sha256(csv_bytes(frame)).hexdigest()


def descriptive_ols(frame, response, predictors):
    """Ordinary OLS diagnostics, never labelled as an MTest bootstrap result."""
    frame, _ = validate_data(frame, response, predictors)
    y, X = frame[response].to_numpy(), frame[predictors].to_numpy()
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design@coef
    r2 = 1 - residuals@residuals / np.sum((y-y.mean())**2)
    aux = []
    for j in range(X.shape[1]):
        Z = np.delete(design, j+1, axis=1)
        res = X[:, j] - Z@np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
        aux.append(1-res@res/np.sum((X[:, j]-X[:, j].mean())**2))
    df = len(y)-design.shape[1]
    # SVD covariance is more stable than inverting X'X.
    _, singular, vh = np.linalg.svd(design, full_matrices=False)
    cov = (vh.T / singular**2) @ vh * (residuals@residuals/df)
    half = t.ppf(.975, df)*np.sqrt(np.maximum(0, np.diag(cov)))
    table = pd.DataFrame({"Predictor": predictors, "Coeficiente": coef[1:],
                          "IC inferior": coef[1:]-half[1:], "IC superior": coef[1:]+half[1:],
                          "R² auxiliar": aux, "VIF": 1/(1-np.asarray(aux))})
    return float(r2), table
