"""Predictive experiment. Test rows never enter scaling, fitting or resampling."""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lab.data import validate_data


def compare_models(frame, response, predictors, seed=27, repeats=80, ridge_alpha=10., lasso_alpha=.1):
    frame, _ = validate_data(frame, response, predictors)
    X, y = frame[predictors].to_numpy(), frame[response].to_numpy()
    train, test = train_test_split(np.arange(len(frame)), test_size=.30, random_state=seed)
    models = {"OLS": LinearRegression(), "Ridge": Ridge(alpha=ridge_alpha),
              "Lasso": Lasso(alpha=lasso_alpha, max_iter=50000, tol=1e-7)}
    rng = np.random.default_rng(seed)
    samples = [rng.choice(train, size=len(train), replace=True) for _ in range(repeats)]
    metrics, coefficients, stability, fitted = [], [], [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, model in models.items():
            pipe = make_pipeline(StandardScaler(), model)
            pipe.fit(X[train], y[train])
            fitted[name] = pipe
            metrics.append({"Modelo": name, "MSE test": mean_squared_error(y[test], pipe.predict(X[test]))})
            coef = pipe[-1].coef_ / pipe[0].scale_
            for j, predictor in enumerate(predictors):
                coefficients.append({"Modelo": name, "Predictor": predictor, "Coeficiente": coef[j]})
            for b, rows in enumerate(samples):
                fitted_boot = clone(pipe).fit(X[rows], y[rows])
                beta = fitted_boot[-1].coef_ / fitted_boot[0].scale_
                for j, predictor in enumerate(predictors):
                    stability.append({"Modelo": name, "Predictor": predictor, "Coeficiente": beta[j], "Réplica": b})
    return {"metrics": pd.DataFrame(metrics), "coefficients": pd.DataFrame(coefficients),
            "stability": pd.DataFrame(stability), "models": fitted, "train": train, "test": test,
            "samples": samples, "seed": seed}
