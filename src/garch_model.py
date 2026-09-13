"""GARCH-family benchmark models and historical-simulation baseline.

The benchmark family deliberately crosses two volatility dynamics (GARCH and
asymmetric EGARCH) with two innovation laws (Gaussian and standardized
Student-t). This makes the comparison with SV-t scientifically fairer: a gain
from SV-t should not be attributable merely to giving only the SV family heavy
tails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

SUPPORTED_MODEL_TYPES = {"GARCH", "EGARCH"}
SUPPORTED_DISTRIBUTIONS = {"normal", "t"}


def _parameter_values(result, prefix: str) -> list[float]:
    """Extract ordered ARCH parameter-family values such as alpha[i]/beta[i]."""
    return [
        float(value)
        for name, value in result.params.items()
        if str(name).startswith(f"{prefix}[")
    ]


def fit_garch(
    returns: np.ndarray,
    model_type: str = "GARCH",
    p: int = 1,
    q: int = 1,
    distribution: str = "normal",
):
    """Fit GARCH(p,q) or asymmetric EGARCH(p,1,q).

    ``arch`` is fit on percentage returns for numerical stability. Student-t
    innovations use ``arch``'s standardized (unit-variance) t distribution.
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {sorted(SUPPORTED_MODEL_TYPES)}")
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {sorted(SUPPORTED_DISTRIBUTIONS)}"
        )
    if p < 1 or q < 1:
        raise ValueError("p and q must both be >= 1")

    ret_pct = np.asarray(returns, dtype=float) * 100.0
    vol = "GARCH" if model_type == "GARCH" else "EGARCH"
    asym_order = 0 if model_type == "GARCH" else 1
    dist = "normal" if distribution == "normal" else "t"

    model = arch_model(
        ret_pct,
        vol=vol,
        p=p,
        o=asym_order,
        q=q,
        mean="Constant",
        dist=dist,
        rescale=False,
    )

    try:
        result = model.fit(disp="off", show_warning=False)

        # Enforce stationarity for any p/q order used in robustness checks.
        beta_values = _parameter_values(result, "beta")
        if not beta_values or not np.all(np.isfinite(beta_values)):
            return None

        if model_type == "GARCH":
            alpha_values = _parameter_values(result, "alpha")
            if not alpha_values or not np.all(np.isfinite(alpha_values)):
                return None
            if any(x < 0 for x in alpha_values + beta_values):
                return None
            if sum(alpha_values) + sum(beta_values) >= 1.0:
                return None
        else:
            # Covariance stationarity of log variance requires the AR roots to
            # be stable. For q<=2 (the planned sensitivity orders), the simple
            # sum condition is a conservative screen; the primary model q=1 is
            # exactly the familiar |beta|<1 condition.
            if q == 1:
                if abs(beta_values[0]) >= 1.0:
                    return None
            elif sum(abs(x) for x in beta_values) >= 1.0:
                return None

        if distribution == "t":
            nu = float(result.params.get("nu", np.nan))
            if not np.isfinite(nu) or nu <= 2.0:
                return None

        return result
    except Exception:
        return None


def _student_t_var_es_multiplier(alpha: float, nu: float) -> tuple[float, float]:
    """Return unit-variance Student-t quantile and positive ES multiplier."""
    if nu <= 2.0:
        raise ValueError("Student-t degrees of freedom must exceed 2")
    q_raw = float(stats.t.ppf(alpha, df=nu))
    pdf_raw = float(stats.t.pdf(q_raw, df=nu))
    scale = float(np.sqrt((nu - 2.0) / nu))
    q_std = scale * q_raw
    es_std = scale * ((nu + q_raw**2) / (nu - 1.0)) * pdf_raw / alpha
    return q_std, es_std


def forecast_var_es(
    result,
    alpha: float,
    horizon: int = 1,
    distribution: str = "normal",
) -> tuple[float, float]:
    """Produce one-step VaR and ES in decimal-return units."""
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {sorted(SUPPORTED_DISTRIBUTIONS)}"
        )
    try:
        forecasts = result.forecast(horizon=horizon, reindex=False)
        sigma = float(np.sqrt(forecasts.variance.values[-1, 0]) / 100.0)
        mu = float(result.params.get("Const", 0.0)) / 100.0

        if distribution == "normal":
            q = float(stats.norm.ppf(alpha))
            es_multiplier = float(stats.norm.pdf(q) / alpha)
        else:
            nu = float(result.params["nu"])
            q, es_multiplier = _student_t_var_es_multiplier(alpha, nu)

        var = -(mu + sigma * q)
        es = -mu + sigma * es_multiplier
        return float(var), float(es)
    except Exception:
        return np.nan, np.nan


def rolling_var_es(
    returns: pd.Series,
    model_type: str = "GARCH",
    window: int = 1000,
    alphas: list | None = None,
    distribution: str = "normal",
    p: int = 1,
    q: int = 1,
) -> pd.DataFrame:
    """Daily rolling one-step VaR/ES for one GARCH-family specification."""
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
        result = fit_garch(
            train,
            model_type=model_type,
            p=p,
            q=q,
            distribution=distribution,
        )
        if result is None:
            row["estimation_failed"] = True
            for alpha in alphas:
                row[f"var_{alpha}"] = np.nan
                row[f"es_{alpha}"] = np.nan
        else:
            for alpha in alphas:
                var, es = forecast_var_es(
                    result,
                    alpha,
                    distribution=distribution,
                )
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
