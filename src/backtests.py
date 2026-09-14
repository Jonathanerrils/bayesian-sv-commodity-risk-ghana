"""VaR and Expected Shortfall backtests.

Sign convention throughout:
* realised returns are negative for losses;
* VaR and ES forecasts are positive loss magnitudes;
* a VaR violation occurs when ``actual_return < -VaR``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def bonferroni_adjust_pvalue(p_value: float, family_size: int) -> float:
    """Return a Bonferroni-adjusted p-value without altering the raw value."""
    if family_size < 1:
        raise ValueError("family_size must be at least 1")
    if not np.isfinite(p_value):
        return np.nan
    return float(min(max(float(p_value), 0.0) * family_size, 1.0))


def apply_bonferroni_reporting(
    results: pd.DataFrame,
    family_size: int,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Add adjusted p-values and corrected non-rejection decisions.

    The paper defines the multiplicity family in terms of model x commodity x
    confidence-level comparison cells.  Each primary test keeps its raw p-value
    and receives a Bonferroni-adjusted p-value using that cell-family size.

    Because these are adequacy tests with calibration as the null, ``passed``
    means *failure to reject inadequacy*. Bonferroni therefore controls false
    rejection and makes rejection harder; it must not be described as a more
    stringent proof that a model is adequate.
    """
    if family_size < 1:
        raise ValueError("family_size must be at least 1")
    out = results.copy()
    out["bonferroni_family_size"] = int(family_size)
    out["bonferroni_alpha"] = float(significance / family_size)

    for prefix in ("kupiec", "cc", "as"):
        p_col = f"{prefix}_p"
        if p_col not in out:
            continue
        raw = pd.to_numeric(out[p_col], errors="coerce")
        adjusted = np.minimum(raw * family_size, 1.0)
        out[f"{prefix}_p_bonferroni"] = adjusted
        out[f"{prefix}_passed_bonferroni"] = adjusted >= significance
    return out


def kupiec_pof(
    actual_returns: np.ndarray,
    var_forecasts: np.ndarray,
    alpha: float,
    significance: float = 0.05,
) -> dict:
    """Kupiec (1995) unconditional coverage test."""
    actual_returns = np.asarray(actual_returns, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)
    mask = np.isfinite(actual_returns) & np.isfinite(var_forecasts)
    actual = actual_returns[mask]
    var = var_forecasts[mask]
    T = len(actual)
    if T == 0:
        return {
            "n_violations": 0,
            "T": 0,
            "observed_rate": np.nan,
            "expected_rate": alpha,
            "lr_stat": np.nan,
            "p_value": np.nan,
            "passed": False,
        }

    hit = actual < -var
    N = int(hit.sum())
    p_hat = N / T
    log_null = xlogy(N, alpha) + xlogy(T - N, 1 - alpha)
    log_alt = xlogy(N, p_hat) + xlogy(T - N, 1 - p_hat)
    lr_stat = float(max(-2.0 * (log_null - log_alt), 0.0))
    p_value = float(stats.chi2.sf(lr_stat, df=1))
    return {
        "n_violations": N,
        "T": T,
        "observed_rate": float(p_hat),
        "expected_rate": float(alpha),
        "lr_stat": lr_stat,
        "p_value": p_value,
        "passed": bool(p_value >= significance),
    }


def christoffersen_cc(
    actual_returns: np.ndarray,
    var_forecasts: np.ndarray,
    alpha: float,
    significance: float = 0.05,
) -> dict:
    """Christoffersen (1998) independence and conditional-coverage tests."""
    actual_returns = np.asarray(actual_returns, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)
    mask = np.isfinite(actual_returns) & np.isfinite(var_forecasts)
    actual = actual_returns[mask]
    var = var_forecasts[mask]
    if len(actual) < 2:
        return {
            "cc_lr_stat": np.nan,
            "cc_p_value": np.nan,
            "cc_passed": False,
            "ind_lr_stat": np.nan,
            "ind_p_value": np.nan,
            "ind_passed": False,
            "pof_lr_stat": np.nan,
            "pof_p_value": np.nan,
            "pof_passed": False,
            "n_violations": 0,
            "observed_rate": np.nan,
        }

    hit = (actual < -var).astype(int)
    n00 = int(((hit[:-1] == 0) & (hit[1:] == 0)).sum())
    n01 = int(((hit[:-1] == 0) & (hit[1:] == 1)).sum())
    n10 = int(((hit[:-1] == 1) & (hit[1:] == 0)).sum())
    n11 = int(((hit[:-1] == 1) & (hit[1:] == 1)).sum())
    denom0 = n00 + n01
    denom1 = n10 + n11
    total = denom0 + denom1
    pi01 = n01 / denom0 if denom0 else 0.0
    pi11 = n11 / denom1 if denom1 else 0.0
    pi = (n01 + n11) / total if total else 0.0

    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = (
        xlogy(n00, 1 - pi01)
        + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11)
        + xlogy(n11, pi11)
    )
    lr_ind = float(max(-2.0 * (log_l0 - log_l1), 0.0))
    p_ind = float(stats.chi2.sf(lr_ind, df=1))
    pof = kupiec_pof(actual, var, alpha, significance)
    lr_cc = float(pof["lr_stat"] + lr_ind)
    p_cc = float(stats.chi2.sf(lr_cc, df=2))
    return {
        "cc_lr_stat": lr_cc,
        "cc_p_value": p_cc,
        "cc_passed": bool(p_cc >= significance),
        "ind_lr_stat": lr_ind,
        "ind_p_value": p_ind,
        "ind_passed": bool(p_ind >= significance),
        "pof_lr_stat": pof["lr_stat"],
        "pof_p_value": pof["p_value"],
        "pof_passed": pof["passed"],
        "n_violations": pof["n_violations"],
        "observed_rate": pof["observed_rate"],
    }


@lru_cache(maxsize=64)
def _z2_normal_reference(
    T: int,
    alpha_rounded: float,
    n_simulations: int,
    random_seed: int,
) -> np.ndarray:
    alpha = float(alpha_rounded)
    rng = np.random.default_rng(random_seed)
    counts = rng.binomial(T, alpha, size=n_simulations)
    total_tail = int(counts.sum())
    sums = np.zeros(n_simulations, dtype=float)
    if total_tail:
        x_tail = stats.norm.ppf(rng.uniform(1e-12, alpha, size=total_tail))
        groups = np.repeat(np.arange(n_simulations), counts)
        sums = np.bincount(groups, weights=x_tail, minlength=n_simulations)
    q = stats.norm.ppf(alpha)
    es = stats.norm.pdf(q) / alpha
    return np.sort(sums / (T * alpha * es) + 1.0)


def acerbi_szekely(
    actual_returns: np.ndarray,
    var_forecasts: np.ndarray,
    es_forecasts: np.ndarray,
    alpha: float,
    significance: float = 0.05,
    n_simulations: int = 20_000,
    random_seed: int = 42,
) -> dict:
    """Acerbi-Szekely (2014) Test 2 / unconditional ES reference test."""
    actual_returns = np.asarray(actual_returns, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)
    es_forecasts = np.asarray(es_forecasts, dtype=float)
    mask = (
        np.isfinite(actual_returns)
        & np.isfinite(var_forecasts)
        & np.isfinite(es_forecasts)
        & (es_forecasts > 0)
    )
    actual = actual_returns[mask]
    var = var_forecasts[mask]
    es = es_forecasts[mask]
    T = len(actual)
    if T == 0:
        return {
            "z2_stat": np.nan,
            "p_value": np.nan,
            "critical_value": np.nan,
            "passed": False,
            "interpretation": "insufficient data",
            "reference": "standardized normal",
        }

    hit = actual < -var
    z2 = float(np.sum((actual * hit) / es) / (T * alpha) + 1.0)
    ref = _z2_normal_reference(T, round(float(alpha), 10), int(n_simulations), int(random_seed))
    rank = int(np.searchsorted(ref, z2, side="right"))
    p_value = float(rank / len(ref))
    critical_value = float(np.quantile(ref, significance))
    if z2 < 0:
        interpretation = "ES under-estimated (tail risk under-forecast)"
    elif z2 > 0:
        interpretation = "ES over-estimated (tail risk over-forecast)"
    else:
        interpretation = "ES exactly calibrated (Z2=0)"
    return {
        "z2_stat": z2,
        "p_value": p_value,
        "critical_value": critical_value,
        "passed": bool(z2 >= critical_value),
        "interpretation": interpretation,
        "reference": "standardized normal",
    }


def run_all_backtests(
    forecast_df: pd.DataFrame,
    alphas: list | None = None,
    label: str = "",
) -> pd.DataFrame:
    if alphas is None:
        alphas = [0.01, 0.05]
    actual = forecast_df["actual_return"].to_numpy(dtype=float)
    rows = []
    for alpha in alphas:
        var_col = f"var_{alpha}"
        es_col = f"es_{alpha}"
        if var_col not in forecast_df:
            continue
        var = forecast_df[var_col].to_numpy(dtype=float)
        es = forecast_df[es_col].to_numpy(dtype=float) if es_col in forecast_df else np.full_like(var, np.nan)
        pof = kupiec_pof(actual, var, alpha)
        cc = christoffersen_cc(actual, var, alpha)
        as_test = acerbi_szekely(actual, var, es, alpha)
        rows.append(
            {
                "model": label,
                "alpha": alpha,
                "n_obs": pof["T"],
                "n_violations": pof["n_violations"],
                "observed_rate": round(pof["observed_rate"], 6),
                "expected_rate": alpha,
                "kupiec_lr": round(pof["lr_stat"], 6),
                "kupiec_p": round(pof["p_value"], 6),
                "kupiec_passed": pof["passed"],
                "cc_lr": round(cc["cc_lr_stat"], 6),
                "cc_p": round(cc["cc_p_value"], 6),
                "cc_passed": cc["cc_passed"],
                "ind_lr": round(cc["ind_lr_stat"], 6),
                "ind_p": round(cc["ind_p_value"], 6),
                "ind_passed": cc["ind_passed"],
                "as_z2": round(as_test["z2_stat"], 6),
                "as_p": round(as_test["p_value"], 6),
                "as_critical": round(as_test["critical_value"], 6),
                "as_passed": as_test["passed"],
                "as_reference": as_test["reference"],
                "as_interpretation": as_test["interpretation"],
            }
        )
    return pd.DataFrame(rows)
