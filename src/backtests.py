"""VaR and Expected Shortfall backtests.

Sign convention:
* realised returns are negative for losses;
* VaR and ES forecasts are positive loss magnitudes;
* a VaR violation occurs when ``actual_return < -VaR``.

Hypothesis-test p-values are retained at full machine precision.  Formatting is
strictly a presentation concern and must never affect multiplicity decisions.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def bonferroni_adjust_pvalue(p_value: float, family_size: int) -> float:
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
    """Add adjusted p-values without overwriting raw inference.

    For the simulation-based Acerbi-Szekely test we also propagate the exact
    binomial confidence interval for the Monte Carlo p-value.  A corrected ES
    decision is labelled ``mc-uncertain`` whenever that interval straddles the
    raw Bonferroni cutoff, rather than pretending the simulation point estimate
    is exact.
    """
    if family_size < 1:
        raise ValueError("family_size must be at least 1")
    out = results.copy()
    threshold = float(significance / family_size)
    out["bonferroni_family_size"] = int(family_size)
    out["bonferroni_alpha"] = threshold

    for prefix in ("kupiec", "cc", "as"):
        p_col = f"{prefix}_p"
        if p_col not in out:
            continue
        raw = pd.to_numeric(out[p_col], errors="coerce")
        adjusted = np.minimum(raw * family_size, 1.0)
        out[f"{prefix}_p_bonferroni"] = adjusted
        out[f"{prefix}_passed_bonferroni"] = adjusted >= significance

    if {"as_mc_ci_low", "as_mc_ci_high"}.issubset(out.columns):
        low = pd.to_numeric(out["as_mc_ci_low"], errors="coerce")
        high = pd.to_numeric(out["as_mc_ci_high"], errors="coerce")
        stable_nonreject = low >= threshold
        stable_reject = high < threshold
        decision = np.full(len(out), "mc-uncertain", dtype=object)
        decision[stable_nonreject.fillna(False).to_numpy()] = "non-reject"
        decision[stable_reject.fillna(False).to_numpy()] = "reject"
        invalid = ~(np.isfinite(low.to_numpy()) & np.isfinite(high.to_numpy()))
        decision[invalid] = "unavailable"
        out["as_bonferroni_decision"] = decision
        out["as_bonferroni_decision_stable"] = stable_nonreject | stable_reject
    return out


def kupiec_pof(actual_returns, var_forecasts, alpha, significance=0.05) -> dict:
    """Kupiec (1995) unconditional coverage test."""
    actual_returns = np.asarray(actual_returns, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)
    mask = np.isfinite(actual_returns) & np.isfinite(var_forecasts)
    actual = actual_returns[mask]
    var = var_forecasts[mask]
    T = len(actual)
    if T == 0:
        return {
            "n_violations": 0, "T": 0, "observed_rate": np.nan,
            "expected_rate": alpha, "lr_stat": np.nan,
            "p_value": np.nan, "passed": False,
        }
    hit = actual < -var
    N = int(hit.sum())
    p_hat = N / T
    log_null = xlogy(N, alpha) + xlogy(T - N, 1 - alpha)
    log_alt = xlogy(N, p_hat) + xlogy(T - N, 1 - p_hat)
    lr = float(max(-2.0 * (log_null - log_alt), 0.0))
    p = float(stats.chi2.sf(lr, 1))
    return {
        "n_violations": N, "T": T, "observed_rate": float(p_hat),
        "expected_rate": float(alpha), "lr_stat": lr,
        "p_value": p, "passed": bool(p >= significance),
    }


def christoffersen_cc(actual_returns, var_forecasts, alpha, significance=0.05) -> dict:
    """Christoffersen (1998) independence and conditional-coverage tests."""
    actual = np.asarray(actual_returns, dtype=float)
    var = np.asarray(var_forecasts, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(var)
    actual, var = actual[mask], var[mask]
    if len(actual) < 2:
        return {
            "cc_lr_stat": np.nan, "cc_p_value": np.nan, "cc_passed": False,
            "ind_lr_stat": np.nan, "ind_p_value": np.nan, "ind_passed": False,
            "pof_lr_stat": np.nan, "pof_p_value": np.nan, "pof_passed": False,
            "n_violations": 0, "observed_rate": np.nan,
        }
    hit = (actual < -var).astype(int)
    n00 = int(((hit[:-1] == 0) & (hit[1:] == 0)).sum())
    n01 = int(((hit[:-1] == 0) & (hit[1:] == 1)).sum())
    n10 = int(((hit[:-1] == 1) & (hit[1:] == 0)).sum())
    n11 = int(((hit[:-1] == 1) & (hit[1:] == 1)).sum())
    d0, d1 = n00 + n01, n10 + n11
    total = d0 + d1
    pi01 = n01 / d0 if d0 else 0.0
    pi11 = n11 / d1 if d1 else 0.0
    pi = (n01 + n11) / total if total else 0.0
    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = (
        xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    )
    lr_ind = float(max(-2.0 * (log_l0 - log_l1), 0.0))
    p_ind = float(stats.chi2.sf(lr_ind, 1))
    pof = kupiec_pof(actual, var, alpha, significance)
    lr_cc = float(pof["lr_stat"] + lr_ind)
    p_cc = float(stats.chi2.sf(lr_cc, 2))
    return {
        "cc_lr_stat": lr_cc, "cc_p_value": p_cc,
        "cc_passed": bool(p_cc >= significance),
        "ind_lr_stat": lr_ind, "ind_p_value": p_ind,
        "ind_passed": bool(p_ind >= significance),
        "pof_lr_stat": pof["lr_stat"], "pof_p_value": pof["p_value"],
        "pof_passed": pof["passed"], "n_violations": pof["n_violations"],
        "observed_rate": pof["observed_rate"],
    }


@lru_cache(maxsize=64)
def _z2_normal_reference(T, alpha_rounded, n_simulations, random_seed) -> np.ndarray:
    """Chunked standardized-normal Test-2 reference distribution."""
    alpha = float(alpha_rounded)
    n_simulations = int(n_simulations)
    rng = np.random.default_rng(int(random_seed))
    q = float(stats.norm.ppf(alpha))
    es = float(stats.norm.pdf(q) / alpha)
    out = np.empty(n_simulations, dtype=float)
    chunk_size = 5_000
    for start in range(0, n_simulations, chunk_size):
        stop = min(start + chunk_size, n_simulations)
        m = stop - start
        counts = rng.binomial(int(T), alpha, size=m)
        total_tail = int(counts.sum())
        sums = np.zeros(m, dtype=float)
        if total_tail:
            tail = stats.norm.ppf(rng.uniform(1e-12, alpha, size=total_tail))
            groups = np.repeat(np.arange(m), counts)
            sums = np.bincount(groups, weights=tail, minlength=m)
        out[start:stop] = sums / (T * alpha * es) + 1.0
    return np.sort(out)


def _clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for the Monte Carlo tail probability."""
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if k == 0 else float(stats.beta.ppf(tail, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1.0 - tail, k + 1, n - k))
    return low, high


def acerbi_szekely(
    actual_returns,
    var_forecasts,
    es_forecasts,
    alpha,
    significance=0.05,
    n_simulations=100_000,
    random_seed=42,
) -> dict:
    """Acerbi-Szekely (2014) Test 2 with simulated normal critical values."""
    actual = np.asarray(actual_returns, dtype=float)
    var = np.asarray(var_forecasts, dtype=float)
    es = np.asarray(es_forecasts, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(var) & np.isfinite(es) & (es > 0)
    actual, var, es = actual[mask], var[mask], es[mask]
    T = len(actual)
    if T == 0:
        return {
            "z2_stat": np.nan, "p_value": np.nan, "critical_value": np.nan,
            "passed": False, "interpretation": "insufficient data",
            "reference": "standardized normal (Monte Carlo)",
            "mc_n": int(n_simulations), "mc_extreme_count": 0,
            "mc_ci_low": np.nan, "mc_ci_high": np.nan,
        }

    hit = actual < -var
    z2 = float(np.sum((actual * hit) / es) / (T * alpha) + 1.0)
    ref = _z2_normal_reference(T, round(float(alpha), 12), int(n_simulations), int(random_seed))
    k = int(np.searchsorted(ref, z2, side="right"))
    # Add-one Monte Carlo p-value prevents a finite simulation from reporting 0.
    p = float((k + 1) / (len(ref) + 1))
    ci_low, ci_high = _clopper_pearson(k, len(ref))
    critical = float(np.quantile(ref, significance))
    interpretation = (
        "ES under-estimated (tail risk under-forecast)" if z2 < 0
        else "ES over-estimated (tail risk over-forecast)" if z2 > 0
        else "ES exactly calibrated (Z2=0)"
    )
    return {
        "z2_stat": z2, "p_value": p, "critical_value": critical,
        "passed": bool(z2 >= critical), "interpretation": interpretation,
        "reference": "standardized normal (Monte Carlo)",
        "mc_n": int(len(ref)), "mc_extreme_count": k,
        "mc_ci_low": ci_low, "mc_ci_high": ci_high,
    }


def run_all_backtests(forecast_df: pd.DataFrame, alphas=None, label="") -> pd.DataFrame:
    if alphas is None:
        alphas = [0.01, 0.05]
    actual = forecast_df["actual_return"].to_numpy(dtype=float)
    rows = []
    for alpha in alphas:
        var_col, es_col = f"var_{alpha}", f"es_{alpha}"
        if var_col not in forecast_df:
            continue
        var = forecast_df[var_col].to_numpy(dtype=float)
        es = (
            forecast_df[es_col].to_numpy(dtype=float)
            if es_col in forecast_df else np.full_like(var, np.nan)
        )
        pof = kupiec_pof(actual, var, alpha)
        cc = christoffersen_cc(actual, var, alpha)
        aset = acerbi_szekely(actual, var, es, alpha)
        rows.append({
            "model": label,
            "alpha": float(alpha),
            "n_obs": int(pof["T"]),
            "n_violations": int(pof["n_violations"]),
            "observed_rate": pof["observed_rate"],
            "expected_rate": float(alpha),
            "kupiec_lr": pof["lr_stat"],
            "kupiec_p": pof["p_value"],
            "kupiec_passed": pof["passed"],
            "cc_lr": cc["cc_lr_stat"],
            "cc_p": cc["cc_p_value"],
            "cc_passed": cc["cc_passed"],
            "ind_lr": cc["ind_lr_stat"],
            "ind_p": cc["ind_p_value"],
            "ind_passed": cc["ind_passed"],
            "as_z2": aset["z2_stat"],
            "as_p": aset["p_value"],
            "as_critical": aset["critical_value"],
            "as_passed": aset["passed"],
            "as_reference": aset["reference"],
            "as_interpretation": aset["interpretation"],
            "as_mc_n": aset["mc_n"],
            "as_mc_extreme_count": aset["mc_extreme_count"],
            "as_mc_ci_low": aset["mc_ci_low"],
            "as_mc_ci_high": aset["mc_ci_high"],
        })
    return pd.DataFrame(rows)
