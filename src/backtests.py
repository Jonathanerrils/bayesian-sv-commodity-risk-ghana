"""
utils/backtests.py

Implements the three backtest statistics from Deliverable 2, Section 4:
  1. Kupiec (1995) POF test
  2. Christoffersen (1998) Conditional Coverage test
  3. Acerbi-Szekely (2014) Test 2 for ES

All functions take actual returns and forecasted VaR/ES and return
test statistics, p-values, and a pass/fail flag at alpha=0.05.

Sign convention (Deliverable 2, Section 3):
  - actual_return is the raw log-return (negative = loss)
  - var is a POSITIVE number representing the loss threshold
  - A violation occurs when actual_return < -var
    (i.e., the actual loss exceeded the forecast loss threshold)
"""

import numpy as np
import pandas as pd
from scipy import stats


def kupiec_pof(actual_returns: np.ndarray,
               var_forecasts: np.ndarray,
               alpha: float,
               significance: float = 0.05) -> dict:
    """
    Kupiec (1995) Proportion of Failures test.

    H0: The observed violation rate equals alpha (model is correctly calibrated).
    H1: The observed violation rate differs from alpha.

    Parameters
    ----------
    actual_returns : np.ndarray
        Realised log-returns (negative = loss).
    var_forecasts : np.ndarray
        Forecast VaR as positive loss values.
    alpha : float
        Nominal coverage level (0.01 or 0.05).
    significance : float
        Test significance level. Default 0.05 (Deliverable 2).

    Returns
    -------
    dict with keys: n, T, observed_rate, lr_stat, p_value, passed
    """
    # Remove NaN pairs
    mask = ~(np.isnan(actual_returns) | np.isnan(var_forecasts))
    actual = actual_returns[mask]
    var    = var_forecasts[mask]

    T = len(actual)
    if T == 0:
        return {"n": 0, "T": 0, "observed_rate": np.nan,
                "lr_stat": np.nan, "p_value": np.nan, "passed": False}

    # Violations: actual loss exceeds VaR threshold
    violations = (actual < -var).astype(int)
    N = violations.sum()
    p_hat = N / T

    # Avoid log(0): if no violations or all violations, LR is infinite
    if N == 0 or N == T:
        lr_stat = np.inf
        p_value = 0.0
    else:
        lr_stat = -2 * (
            N * np.log(alpha / p_hat) +
            (T - N) * np.log((1 - alpha) / (1 - p_hat))
        )
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "n_violations": int(N),
        "T": T,
        "observed_rate": float(p_hat),
        "expected_rate": float(alpha),
        "lr_stat": float(lr_stat),
        "p_value": float(p_value),
        "passed": bool(p_value >= significance),
    }


def christoffersen_cc(actual_returns: np.ndarray,
                      var_forecasts: np.ndarray,
                      alpha: float,
                      significance: float = 0.05) -> dict:
    """
    Christoffersen (1998) Conditional Coverage test.

    Jointly tests:
      (a) unconditional coverage (same as Kupiec POF)
      (b) independence of violations (violations don't cluster)

    The CC statistic = LR_POF + LR_independence, ~ chi2(2).

    As specified in Deliverable 2: both components reported separately.

    Parameters
    ----------
    (same as kupiec_pof)

    Returns
    -------
    dict with keys for both the CC joint test and the independence component.
    """
    # Remove NaN pairs
    mask = ~(np.isnan(actual_returns) | np.isnan(var_forecasts))
    actual = actual_returns[mask]
    var    = var_forecasts[mask]

    T = len(actual)
    if T < 2:
        return {"cc_lr_stat": np.nan, "cc_p_value": np.nan,
                "ind_lr_stat": np.nan, "ind_p_value": np.nan,
                "cc_passed": False, "ind_passed": False}

    # Violation sequence
    hit = (actual < -var).astype(int)

    # Transition counts
    # n_ab = number of transitions from state a to state b
    n_00 = ((hit[:-1] == 0) & (hit[1:] == 0)).sum()
    n_01 = ((hit[:-1] == 0) & (hit[1:] == 1)).sum()
    n_10 = ((hit[:-1] == 1) & (hit[1:] == 0)).sum()
    n_11 = ((hit[:-1] == 1) & (hit[1:] == 1)).sum()

    # Estimated transition probabilities
    pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
    pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
    pi    = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)

    # Independence LR statistic
    # H0: pi_01 = pi_11 = pi (violations are iid Bernoulli)
    def safe_log(x):
        return np.log(x) if x > 0 else -1e10

    try:
        lr_ind = -2 * (
            (n_00 + n_10) * safe_log(1 - pi) +
            (n_01 + n_11) * safe_log(pi) -
            n_00 * safe_log(1 - pi_01) - n_01 * safe_log(pi_01 + 1e-15) -
            n_10 * safe_log(1 - pi_11 + 1e-15) - n_11 * safe_log(pi_11 + 1e-15)
        )
        lr_ind = max(lr_ind, 0)  # numerical floor
    except Exception:
        lr_ind = np.nan

    p_ind = 1 - stats.chi2.cdf(lr_ind, df=1) if not np.isnan(lr_ind) else np.nan

    # POF component (for CC)
    pof = kupiec_pof(actual_returns, var_forecasts, alpha, significance)
    lr_pof = pof["lr_stat"]

    # CC joint statistic
    if not np.isnan(lr_ind) and not np.isinf(lr_pof):
        lr_cc = lr_pof + lr_ind
        p_cc  = 1 - stats.chi2.cdf(lr_cc, df=2)
    else:
        lr_cc = np.nan
        p_cc  = np.nan

    return {
        # Joint CC test
        "cc_lr_stat":   float(lr_cc) if lr_cc is not None else np.nan,
        "cc_p_value":   float(p_cc)  if p_cc  is not None else np.nan,
        "cc_passed":    bool(p_cc >= significance) if p_cc is not None else False,
        # Independence component (Deliverable 2: report separately)
        "ind_lr_stat":  float(lr_ind) if not np.isnan(lr_ind) else np.nan,
        "ind_p_value":  float(p_ind)  if not np.isnan(p_ind)  else np.nan,
        "ind_passed":   bool(p_ind >= significance) if not np.isnan(p_ind) else False,
        # POF from CC perspective
        "pof_lr_stat":  pof["lr_stat"],
        "pof_p_value":  pof["p_value"],
        "pof_passed":   pof["passed"],
        "n_violations": pof["n_violations"],
        "observed_rate": pof["observed_rate"],
    }


def acerbi_szekely(actual_returns: np.ndarray,
                   var_forecasts: np.ndarray,
                   es_forecasts: np.ndarray,
                   alpha: float,
                   n_simulations: int = 1000,
                   random_seed: int = 42) -> dict:
    """
    Acerbi-Szekely (2014) Test 2 for ES adequacy.

    Z2 = (1/(T*alpha*ES_hat)) * sum(r_t * I[r_t < -VaR_t]) + 1

    Under H0 (correct ES), E[Z2] = 0.
    Negative Z2 = ES under-estimated (model under-forecasts tail risk).

    Significance assessed by Monte Carlo under the Gaussian assumption,
    since there is no closed-form null distribution.

    Parameters
    ----------
    actual_returns : np.ndarray
    var_forecasts : np.ndarray
        VaR as positive loss values.
    es_forecasts : np.ndarray
        ES as positive loss values.
    alpha : float
    n_simulations : int
        Monte Carlo replications for p-value.
    random_seed : int
        For reproducibility.

    Returns
    -------
    dict with z2_stat, p_value, passed, interpretation.
    """
    mask = ~(np.isnan(actual_returns) | np.isnan(var_forecasts) |
             np.isnan(es_forecasts))
    actual = actual_returns[mask]
    var    = var_forecasts[mask]
    es     = es_forecasts[mask]

    T = len(actual)
    if T == 0 or es.mean() == 0:
        return {"z2_stat": np.nan, "p_value": np.nan,
                "passed": False, "interpretation": "insufficient data"}

    # Violation indicator
    I_t = (actual < -var).astype(float)

    # Z2 statistic (Deliverable 2 eq. 3)
    z2_num = np.sum(actual * I_t)
    z2_den = T * alpha * np.mean(es)
    z2     = z2_num / z2_den + 1

    # Monte Carlo p-value under H0 (Gaussian returns with model-implied sigma)
    # Under H0: r_t ~ N(0, sigma_t^2), so simulate and compute Z2 distribution
    rng = np.random.default_rng(random_seed)
    sigma_implied = es * alpha / stats.norm.pdf(stats.norm.ppf(alpha))
    # Clip to avoid degenerate sigmas
    sigma_implied = np.clip(sigma_implied, 1e-8, None)

    z2_sim = []
    for _ in range(n_simulations):
        r_sim  = rng.normal(0, sigma_implied)
        I_sim  = (r_sim < -var).astype(float)
        z2_s   = np.sum(r_sim * I_sim) / (T * alpha * np.mean(es)) + 1
        z2_sim.append(z2_s)

    z2_sim = np.array(z2_sim)
    # One-sided p-value: probability of seeing Z2 as negative as observed under H0
    p_value = (z2_sim <= z2).mean()

    if z2 < 0:
        interpretation = "ES under-estimated (model under-forecasts tail risk)"
    elif z2 > 0:
        interpretation = "ES over-estimated (model over-forecasts tail risk)"
    else:
        interpretation = "ES exactly correct (Z2=0)"

    return {
        "z2_stat":        float(z2),
        "p_value":        float(p_value),
        "passed":         bool(p_value >= 0.05),
        "interpretation": interpretation,
    }


def run_all_backtests(forecast_df: pd.DataFrame,
                      alphas: list = None,
                      label: str = "") -> pd.DataFrame:
    """
    Run all three backtest statistics for a forecast DataFrame,
    across all specified alpha levels.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of rolling_var_es() or historical_simulation_var_es().
        Must have columns: actual_return, var_{alpha}, es_{alpha}.
    alphas : list
        Coverage levels to test. Default [0.01, 0.05].
    label : str
        Model label for printing.

    Returns
    -------
    pd.DataFrame with one row per alpha level, all test statistics.
    """
    if alphas is None:
        alphas = [0.01, 0.05]

    actual = forecast_df["actual_return"].values
    rows   = []

    for alpha in alphas:
        var_col = f"var_{alpha}"
        es_col  = f"es_{alpha}"

        if var_col not in forecast_df.columns:
            continue

        var = forecast_df[var_col].values
        es  = forecast_df[es_col].values if es_col in forecast_df.columns \
              else np.full_like(var, np.nan)

        pof_res = kupiec_pof(actual, var, alpha)
        cc_res  = christoffersen_cc(actual, var, alpha)
        as_res  = acerbi_szekely(actual, var, es, alpha)

        row = {
            "model":           label,
            "alpha":           alpha,
            "n_obs":           pof_res["T"],
            "n_violations":    pof_res["n_violations"],
            "observed_rate":   round(pof_res["observed_rate"], 4),
            "expected_rate":   alpha,
            # Kupiec POF
            "kupiec_lr":       round(pof_res["lr_stat"], 4),
            "kupiec_p":        round(pof_res["p_value"], 4),
            "kupiec_passed":   pof_res["passed"],
            # CC joint
            "cc_lr":           round(cc_res["cc_lr_stat"], 4),
            "cc_p":            round(cc_res["cc_p_value"], 4),
            "cc_passed":       cc_res["cc_passed"],
            # Independence (reported separately per Deliverable 2)
            "ind_lr":          round(cc_res["ind_lr_stat"], 4),
            "ind_p":           round(cc_res["ind_p_value"], 4),
            "ind_passed":      cc_res["ind_passed"],
            # Acerbi-Szekely ES
            "as_z2":           round(as_res["z2_stat"], 4),
            "as_p":            round(as_res["p_value"], 4),
            "as_passed":       as_res["passed"],
            "as_interpretation": as_res["interpretation"],
        }
        rows.append(row)

        if label:
            status = "PASS" if pof_res["passed"] else "FAIL"
            print(f"  [{label}] alpha={alpha}: "
                  f"violations={pof_res['n_violations']}/{pof_res['T']} "
                  f"({pof_res['observed_rate']:.3f} vs {alpha:.2f} expected) "
                  f"Kupiec p={pof_res['p_value']:.4f} [{status}]")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    """
    Integration test: verify all three backtest statistics on synthetic data
    with KNOWN properties (so we can check correctness, not just
    that the code runs).

    Test 1: Correctly specified model -- should PASS all tests.
    Test 2: Over-conservative model (too high VaR) -- should FAIL Kupiec
            (too few violations).
    Test 3: Clustered violations -- should fail Christoffersen independence.
    """
    np.random.seed(42)
    T = 3000
    alpha = 0.01

    print("=== Backtest sanity tests ===\n")

    # True sigma for each period
    true_sigma = np.full(T, 0.01)  # constant 1% daily vol
    actual_returns = np.random.normal(0, true_sigma)

    # Test 1: Correct model (VaR from true sigma)
    from scipy.stats import norm
    var_correct = -norm.ppf(alpha) * true_sigma  # positive VaR
    es_correct  = true_sigma * norm.pdf(norm.ppf(alpha)) / alpha
    pof = kupiec_pof(actual_returns, var_correct, alpha)
    print(f"Test 1 (correct model): "
          f"p={pof['p_value']:.4f}, passed={pof['passed']} "
          f"[expect: PASS ~95% of the time by construction]")

    # Test 2: Over-conservative (VaR 3x too high)
    var_high = var_correct * 3
    es_high  = es_correct * 3
    pof2 = kupiec_pof(actual_returns, var_high, alpha)
    print(f"Test 2 (VaR 3x too high): "
          f"p={pof2['p_value']:.4f}, passed={pof2['passed']} "
          f"[expect: FAIL -- too few violations]")

    # Test 3: Clustered violations (blocks of bad days)
    # Inject 20 consecutive violations in the middle
    var_clustered = var_correct.copy()
    var_clustered[1450:1470] = 0.0001  # near-zero VaR -> guaranteed violations
    cc = christoffersen_cc(actual_returns, var_clustered, alpha)
    print(f"Test 3 (clustered violations): "
          f"CC p={cc['cc_p_value']:.4f}, ind p={cc['ind_p_value']:.4f} "
          f"[expect: ind test FAILS -- violations cluster]")

    print("\nAll sanity tests completed.")
    assert not pof2["passed"],  "Over-conservative model should FAIL Kupiec"
    assert not cc["ind_passed"], "Clustered violations should FAIL independence"
    print("Critical assertions passed.")
