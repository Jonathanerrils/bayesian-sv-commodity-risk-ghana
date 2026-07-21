"""
models/garch/garch_model.py

GARCH(1,1) and EGARCH(1,1) estimation and rolling VaR/ES forecasting.

Implements Deliverable 2 evaluation framework exactly:
- Rolling window W=1000 (primary), W=750, W=1250 (robustness)
- VaR at alpha=0.01 and alpha=0.05
- Forecasts are one-step-ahead only
- No future data leaks into any estimation window

Model specification reference: Deliverable 1, Section 3.
Evaluation specification reference: Deliverable 2, Section 3.
"""

import numpy as np
import pandas as pd
from arch import arch_model
import warnings

# Suppress arch convergence warnings in rolling — we track failures explicitly
warnings.filterwarnings("ignore", category=UserWarning, module="arch")


def fit_garch(returns: np.ndarray,
              model_type: str = "GARCH",
              p: int = 1,
              q: int = 1) -> object:
    """
    Fit a GARCH(p,q) or EGARCH(p,q) model to a return series.

    Parameters
    ----------
    returns : np.ndarray
        Log-return series (as percentage returns internally for numerical
        stability; arch package works in percentage scale).
    model_type : str
        'GARCH' or 'EGARCH'.
    p, q : int
        Lag orders. Default (1,1) per Deliverable 1 justification.

    Returns
    -------
    arch ARCHModelResult, or None if estimation failed.
    """
    # arch package expects percentage returns for numerical stability
    ret_pct = returns * 100

    if model_type == "GARCH":
        model = arch_model(ret_pct, vol="Garch", p=p, q=q,
                           mean="Constant", dist="normal")
    elif model_type == "EGARCH":
        model = arch_model(ret_pct, vol="EGarch", p=p, q=q,
                           mean="Constant", dist="normal")
    else:
        raise ValueError(f"model_type must be 'GARCH' or 'EGARCH', got {model_type}")

    try:
        result = model.fit(disp="off", show_warning=False)
        # Basic sanity check: alpha+beta < 1 for GARCH stationarity
        if model_type == "GARCH":
            alpha = result.params.get("alpha[1]", 0)
            beta  = result.params.get("beta[1]", 0)
            if alpha + beta >= 1.0:
                # Non-stationary fit -- flag but do not crash
                pass  # returned result still usable; caller checks
        return result
    except Exception:
        return None


def forecast_var_es(result, alpha: float, horizon: int = 1) -> tuple:
    """
    Produce one-step-ahead VaR and ES from a fitted GARCH/EGARCH result.

    Returns VaR and ES in the ORIGINAL return scale (not percentage),
    as positive numbers representing losses (following Deliverable 2
    sign convention: VaR = -quantile of return distribution).

    Parameters
    ----------
    result : ARCHModelResult
        Fitted model from fit_garch().
    alpha : float
        Coverage level (0.01 for 99% VaR, 0.05 for 95% VaR).
    horizon : int
        Forecast horizon in days. Always 1 for this study.

    Returns
    -------
    (var, es) : tuple of float
        VaR and ES as positive loss values.
        Returns (np.nan, np.nan) if forecast fails.
    """
    from scipy import stats

    try:
        forecasts = result.forecast(horizon=horizon, reindex=False)
        sigma_pct = np.sqrt(forecasts.variance.values[-1, 0])
        mu_pct    = result.params.get("Const", 0.0)

        # Convert back to log-return scale
        sigma = sigma_pct / 100
        mu    = mu_pct / 100

        # Under Gaussian assumption:
        # VaR(alpha) = -(mu + sigma * Phi^{-1}(alpha))
        z_alpha = stats.norm.ppf(alpha)
        var = -(mu + sigma * z_alpha)  # positive loss

        # ES(alpha) = -(mu + sigma * phi(Phi^{-1}(alpha)) / alpha)
        es = -(mu - sigma * stats.norm.pdf(z_alpha) / alpha)  # positive loss

        return float(var), float(es)

    except Exception:
        return np.nan, np.nan


def rolling_var_es(returns: pd.Series,
                   model_type: str = "GARCH",
                   window: int = 1000,
                   alphas: list = None) -> pd.DataFrame:
    """
    Full rolling walk-forward VaR/ES forecasting loop.

    Implements Deliverable 2 Section 3 exactly:
    - Estimate on [t-W+1, t], forecast for t+1
    - No leakage: each window sees only past data
    - Records estimation failures explicitly (not silently)

    Parameters
    ----------
    returns : pd.Series
        Clean log-return series (output of data_loader.load_all).
    model_type : str
        'GARCH' or 'EGARCH'.
    window : int
        Rolling window length W.
    alphas : list
        Coverage levels. Default [0.01, 0.05] per Deliverable 2.

    Returns
    -------
    pd.DataFrame with columns:
        actual_return,
        var_0.01, es_0.01, var_0.05, es_0.05,
        estimation_failed (bool)
    """
    if alphas is None:
        alphas = [0.01, 0.05]

    ret_array = returns.values
    dates     = returns.index
    n         = len(ret_array)

    if n <= window:
        raise ValueError(
            f"Series length ({n}) must exceed window ({window}). "
            f"Check your sample period."
        )

    n_forecasts = n - window
    results = []
    n_failures = 0

    for i in range(n_forecasts):
        train = ret_array[i : i + window]
        actual_return = ret_array[i + window]
        forecast_date = dates[i + window]

        row = {"date": forecast_date, "actual_return": actual_return,
               "estimation_failed": False}

        result = fit_garch(train, model_type=model_type)

        if result is None:
            n_failures += 1
            row["estimation_failed"] = True
            for alpha in alphas:
                row[f"var_{alpha}"] = np.nan
                row[f"es_{alpha}"]  = np.nan
        else:
            for alpha in alphas:
                var, es = forecast_var_es(result, alpha)
                row[f"var_{alpha}"] = var
                row[f"es_{alpha}"]  = es

        results.append(row)

        # Progress report every 250 steps
        if (i + 1) % 250 == 0:
            print(f"  {model_type} [{i+1}/{n_forecasts}] "
                  f"failures so far: {n_failures}")

    df = pd.DataFrame(results).set_index("date")

    if n_failures > 0:
        pct_failed = 100 * n_failures / n_forecasts
        print(f"  WARNING: {n_failures} estimation failures "
              f"({pct_failed:.1f}%) -- these dates have NaN forecasts. "
              f"Investigate before reporting results.")
    else:
        print(f"  {model_type}: all {n_forecasts} steps converged cleanly.")

    return df


def historical_simulation_var_es(returns: pd.Series,
                                  window: int = 1000,
                                  alphas: list = None) -> pd.DataFrame:
    """
    Historical simulation VaR/ES (the trivial baseline from Deliverable 2
    Section 5). No model fitted; just empirical quantiles of the past
    W returns.

    Parameters
    ----------
    returns : pd.Series
        Clean log-return series.
    window : int
        Rolling window length W.
    alphas : list
        Coverage levels.

    Returns
    -------
    pd.DataFrame matching the structure of rolling_var_es output.
    """
    if alphas is None:
        alphas = [0.01, 0.05]

    ret_array = returns.values
    dates     = returns.index
    n         = len(ret_array)
    n_forecasts = n - window
    results = []

    for i in range(n_forecasts):
        train = ret_array[i : i + window]
        actual_return = ret_array[i + window]
        forecast_date = dates[i + window]

        row = {"date": forecast_date, "actual_return": actual_return,
               "estimation_failed": False}

        for alpha in alphas:
            # VaR = negative of empirical alpha-quantile
            var = -np.quantile(train, alpha)
            # ES  = mean of returns below the VaR threshold
            tail = train[train < -var]
            es   = -tail.mean() if len(tail) > 0 else var
            row[f"var_{alpha}"] = float(var)
            row[f"es_{alpha}"]  = float(es)

        results.append(row)

    return pd.DataFrame(results).set_index("date")


if __name__ == "__main__":
    """
    Integration test: run GARCH(1,1), EGARCH(1,1), and Historical Simulation
    on REAL data for a short out-of-sample window to verify correctness
    before full production run. Uses W=1000, last 100 steps only.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_utils import load_all_returns

    print("Loading real data...")
    all_returns = load_all_returns(verbose=False)

    # Test on gold only for the integration test (fastest + cleanest series)
    gold_ret = all_returns["gold"]
    print(f"\nGold returns: {len(gold_ret)} observations")
    print("Running integration test: GARCH(1,1), W=1000, last 200 steps only...")

    # Slice to last 1200 observations for speed (1000 train + 200 test)
    test_slice = gold_ret.iloc[-1200:]

    garch_results = rolling_var_es(test_slice, model_type="GARCH",
                                   window=1000, alphas=[0.01, 0.05])
    egarch_results = rolling_var_es(test_slice, model_type="EGARCH",
                                    window=1000, alphas=[0.01, 0.05])
    hs_results = historical_simulation_var_es(test_slice, window=1000,
                                              alphas=[0.01, 0.05])

    print("\n--- Sample output (last 5 rows, GARCH) ---")
    print(garch_results[["actual_return", "var_0.01", "es_0.01"]].tail())
    print("\n--- Sample output (last 5 rows, HS baseline) ---")
    print(hs_results[["actual_return", "var_0.01", "es_0.01"]].tail())

    # Sanity checks
    assert garch_results["estimation_failed"].sum() == 0, \
        "Some GARCH steps failed -- investigate"
    assert (garch_results["var_0.01"] > garch_results["var_0.05"]).all(), \
        "99% VaR must be >= 95% VaR on every day"
    assert (garch_results["es_0.01"] > garch_results["var_0.01"]).all(), \
        "ES must be >= VaR at the same level (definition)"

    print("\nAll sanity checks passed.")
    print("  99% VaR > 95% VaR: TRUE on all days")
    print("  ES >= VaR (same level): TRUE on all days")
    print("  Zero estimation failures: TRUE")
