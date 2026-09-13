"""GARCH-family benchmark models and historical-simulation baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats


def fit_garch(
    returns: np.ndarray,
    model_type: str = "GARCH",
    p: int = 1,
    q: int = 1,
):
    """Fit Gaussian GARCH(1,1) or asymmetric EGARCH(1,1,1)."""
    ret_pct = np.asarray(returns, dtype=float) * 100.0

    if model_type == "GARCH":
        model = arch_model(
            ret_pct, vol="GARCH", p=p, o=0, q=q, mean="Constant", dist="normal"
        )
    elif model_type == "EGARCH":
        # o=1 is essential: without it the fitted EGARCH has no leverage term,
        # contradicting the asymmetric equation used in the paper.
        model = arch_model(
            ret_pct, vol="EGARCH", p=p, o=1, q=q, mean="Constant", dist="normal"
        )
    else:
        raise ValueError("model_type must be 'GARCH' or 'EGARCH'")

    try:
        result = model.fit(disp="off", show_warning=False)
        if model_type == "GARCH":
            alpha = float(result.params.get("alpha[1]", 0.0))
            beta = float(result.params.get("beta[1]", 0.0))
            if alpha + beta >= 1.0:
                return None
        return result
    except Exception:
        return None


def forecast_var_es(result, alpha: float, horizon: int = 1) -> tuple[float, float]:
    try:
        forecasts = result.forecast(horizon=horizon, reindex=False)
        sigma = np.sqrt(forecasts.variance.values[-1, 0]) / 100.0
        mu = float(result.params.get("Const", 0.0)) / 100.0
        z = stats.norm.ppf(alpha)
        var = -(mu + sigma * z)
        es = -(mu - sigma * stats.norm.pdf(z) / alpha)
        return float(var), float(es)
    except Exception:
        return np.nan, np.nan


def rolling_var_es(
    returns: pd.Series,
    model_type: str = "GARCH",
    window: int = 1000,
    alphas: list | None = None,
) -> pd.DataFrame:
    if alphas is None:
        alphas = [0.01, 0.05]
    if len(returns) <= window:
        raise ValueError("Series length must exceed rolling window")

    values = returns.to_numpy(dtype=float)
    dates = returns.index
    rows = []
    for i in range(len(values) - window):
        train = values[i : i + window]
        row = {
            "date": dates[i + window],
            "actual_return": values[i + window],
            "estimation_failed": False,
        }
        result = fit_garch(train, model_type=model_type)
        if result is None:
            row["estimation_failed"] = True
            for alpha in alphas:
                row[f"var_{alpha}"] = np.nan
                row[f"es_{alpha}"] = np.nan
        else:
            for alpha in alphas:
                var, es = forecast_var_es(result, alpha)
                row[f"var_{alpha}"] = var
                row[f"es_{alpha}"] = es
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def historical_simulation_var_es(
    returns: pd.Series,
    window: int = 1000,
    alphas: list | None = None,
) -> pd.DataFrame:
    if alphas is None:
        alphas = [0.01, 0.05]
    values = returns.to_numpy(dtype=float)
    dates = returns.index
    rows = []
    for i in range(len(values) - window):
        train = values[i : i + window]
        row = {
            "date": dates[i + window],
            "actual_return": values[i + window],
            "estimation_failed": False,
        }
        for alpha in alphas:
            q = float(np.quantile(train, alpha))
            tail = train[train <= q]
            row[f"var_{alpha}"] = -q
            row[f"es_{alpha}"] = -float(tail.mean()) if len(tail) else -q
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")
