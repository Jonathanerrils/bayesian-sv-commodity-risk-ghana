"""
src/plot_utils.py

Publication-quality figures for the paper.
All figures saved to outputs/figures/ as high-resolution PDF and PNG.

Figures produced:
    1. Rolling log-returns for all three commodities (stylised facts)
    2. Rolling VaR comparison: SV-t vs GARCH vs HistSim per commodity
    3. VaR violation scatter plots (timing of exceptions)
    4. Backtest summary heatmap (pass/fail across models × commodities)
    5. Posterior distributions for key SV parameters (phi, sigma_eta, nu)

Author: Jonathan
Project: Stochastic Volatility Models for Commodity Price Risk in West Africa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------
# STYLE SETTINGS — consistent across all figures
# -----------------------------------------------------------------------
COLORS = {
    "cocoa":       "#8B4513",   # brown
    "gold":        "#DAA520",   # goldenrod
    "oil":         "#2F4F4F",   # dark slate
    "GARCH":       "#2196F3",   # blue
    "EGARCH":      "#03A9F4",   # light blue
    "SV-t":        "#E53935",   # red
    "SV-Gaussian": "#FF7043",   # deep orange
    "SV-Leverage": "#D81B60",   # pink
    "SV-t-Leverage": "#880E4F", # dark pink
    "OU":          "#4CAF50",   # green
    "HistSim":     "#9E9E9E",   # grey
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


def _save(fig, name: str):
    """Save figure as both PDF (for paper) and PNG (for quick review)."""
    for ext in ["pdf", "png"]:
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


# -----------------------------------------------------------------------
# FIGURE 1: Log-returns overview (stylised facts)
# -----------------------------------------------------------------------
def plot_returns_overview(returns: dict):
    """
    Three-panel figure showing daily log-returns for cocoa, gold, oil.
    Highlights the key crisis periods identified in Phase 2.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Daily Log-Returns: Cocoa, Gold, and Brent Crude Oil\n"
                 "Common sample: 2003–2026", fontsize=12, fontweight="bold")

    crisis_bands = {
        "GFC 2008":        ("2008-09-01", "2009-03-31"),
        "COVID 2020":      ("2020-02-01", "2020-06-30"),
        "Cocoa crisis 2024": ("2024-01-01", "2024-12-31"),
        "Oil shock 2026":  ("2026-02-01", "2026-06-22"),
    }

    commodity_labels = {
        "cocoa": "Cocoa (CC=F, Yahoo Finance)",
        "gold":  "Gold (GC=F, Yahoo Finance)",
        "oil":   "Brent Crude (DCOILBRENTEU, FRED/EIA)",
    }

    for ax, (commodity, ret) in zip(axes, returns.items()):
        color = COLORS[commodity]
        ax.plot(ret.index, ret.values * 100, color=color,
                linewidth=0.4, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Log-return (%)", fontsize=9)
        ax.set_title(commodity_labels[commodity], fontsize=10, loc="left")

        # Shade crisis periods
        for label, (start, end) in crisis_bands.items():
            try:
                s = pd.Timestamp(start)
                e = min(pd.Timestamp(end), ret.index.max())
                ax.axvspan(s, e, alpha=0.08, color="red")
            except Exception:
                pass

        # Summary stats annotation
        kurt = ret.kurt()
        skew = ret.skew()
        ax.text(0.02, 0.95,
                f"Skew: {skew:.2f}  |  Excess kurtosis: {kurt:.1f}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", color="dimgray")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    _save(fig, "fig1_returns_overview")


# -----------------------------------------------------------------------
# FIGURE 2: Rolling VaR comparison per commodity
# -----------------------------------------------------------------------
def plot_rolling_var(returns: dict,
                     forecasts: dict,
                     alpha: float = 0.01,
                     models_to_plot: list = None):
    """
    For each commodity: plot actual returns vs rolling 99% VaR forecasts
    from selected models. Violation dots shown in red.

    Parameters
    ----------
    returns : dict
        Output of load_all_returns().
    forecasts : dict
        {(commodity, model): forecast_df} from production run.
    alpha : float
        VaR confidence level to plot (default 0.01 = 99%).
    models_to_plot : list
        Models to overlay. Default: GARCH, SV-t, HistSim.
    """
    if models_to_plot is None:
        models_to_plot = ["GARCH", "SV-t", "HistSim"]

    var_col = f"var_{alpha}"
    commodities = ["cocoa", "gold", "oil"]
    n = len(commodities)

    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n))
    fig.suptitle(f"Rolling {int((1-alpha)*100)}% VaR Forecasts vs Actual Returns",
                 fontsize=12, fontweight="bold")

    for ax, commodity in zip(axes, commodities):
        ret = returns[commodity]
        ax.plot(ret.index, ret.values*100, color="lightgray",
                linewidth=0.4, alpha=0.9, label="Actual return", zorder=1)

        for model in models_to_plot:
            key = (commodity, model)
            if key not in forecasts:
                continue
            fc = forecasts[key]
            if var_col not in fc.columns:
                continue
            color = COLORS.get(model, "black")
            ax.plot(fc.index, -fc[var_col]*100, color=color,
                    linewidth=0.8, label=f"{model} –VaR", zorder=2)

            # Mark violations
            viol = fc[fc["actual_return"] < -fc[var_col]]
            if len(viol) > 0:
                ax.scatter(viol.index, viol["actual_return"]*100,
                           color=color, s=8, zorder=3, alpha=0.7)

        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_title(f"{commodity.capitalize()}", loc="left", fontsize=10)
        ax.set_ylabel("Log-return / –VaR (%)")
        ax.legend(loc="lower left", ncol=len(models_to_plot)+1,
                  fontsize=8, framealpha=0.7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    _save(fig, f"fig2_rolling_var_{int((1-alpha)*100)}pct")


# -----------------------------------------------------------------------
# FIGURE 3: Backtest summary heatmap
# -----------------------------------------------------------------------
def plot_backtest_heatmap(results_df: pd.DataFrame):
    """
    Heatmap of pass/fail across all model × commodity × test combinations.
    Green = pass, Red = fail. Rows = models, columns = commodity + test.
    """
    models      = results_df["model"].unique()
    commodities = results_df["commodity"].unique()
    tests       = ["kupiec_passed", "cc_passed", "ind_passed", "as_passed"]
    test_labels = ["Kupiec", "CC", "Ind.", "AS-ES"]
    alpha_vals  = results_df["alpha"].unique()

    # Build matrix: rows=models, cols=commodity×alpha×test
    col_labels = []
    col_keys   = []
    for c in commodities:
        for a in sorted(alpha_vals):
            for t, tl in zip(tests, test_labels):
                col_labels.append(f"{c[:3].upper()}\nα={a}\n{tl}")
                col_keys.append((c, a, t))

    matrix = np.full((len(models), len(col_keys)), np.nan)
    for i, model in enumerate(models):
        for j, (c, a, t) in enumerate(col_keys):
            row = results_df[
                (results_df["model"] == model) &
                (results_df["commodity"] == c) &
                (results_df["alpha"] == a)
            ]
            if len(row) > 0:
                matrix[i, j] = float(row.iloc[0][t])

    fig, ax = plt.subplots(figsize=(max(14, len(col_keys)*0.8),
                                     max(6, len(models)*0.5)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title("Backtest Results: Pass (green) / Fail (red)\n"
                 "Kupiec POF, Christoffersen CC, Independence, Acerbi-Szekely ES",
                 fontsize=11, fontweight="bold")

    for i in range(len(models)):
        for j in range(len(col_keys)):
            val = matrix[i, j]
            if not np.isnan(val):
                text = "P" if val == 1 else "F"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=9, color="white" if val == 0 else "black",
                        fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.4, label="Pass (1) / Fail (0)")
    plt.tight_layout()
    _save(fig, "fig3_backtest_heatmap")


# -----------------------------------------------------------------------
# FIGURE 4: OU failure rate — mean-reversion diagnostic
# -----------------------------------------------------------------------
def plot_ou_kappa_over_time(prices: dict):
    """
    Rolling kappa (mean-reversion speed) for OU model over time.
    Shows visually where and why OU fails — which periods are trending.
    """
    from ou_model import fit_ou

    window = 1000
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
    fig.suptitle("OU Mean-Reversion Speed (κ) Over Time\n"
                 "κ < 0.01 (grey zone) indicates no detectable mean-reversion",
                 fontsize=11, fontweight="bold")

    for ax, (commodity, price_series) in zip(axes, prices.items()):
        log_p = np.log(price_series.dropna().values)
        dates = price_series.dropna().index
        n     = len(log_p)
        kappas, kdates = [], []

        for i in range(0, n - window, 21):  # every 21 days
            fitted = fit_ou(log_p[i:i+window])
            kappas.append(fitted["kappa"] if fitted["converged"] else np.nan)
            kdates.append(dates[i + window])

        kappas = np.array(kappas)
        ax.plot(kdates, kappas, color=COLORS[commodity], linewidth=1.0)
        ax.axhline(0.01, color="red", linewidth=0.8, linestyle="--",
                   label="κ = 0.01 threshold")
        ax.fill_between(kdates, 0, 0.01, alpha=0.1, color="red",
                        label="Non-converging zone")
        ax.set_title(f"{commodity.capitalize()}", loc="left", fontsize=10)
        ax.set_ylabel("κ (mean-reversion speed)")
        ax.legend(fontsize=8, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    _save(fig, "fig4_ou_kappa_diagnostic")


# -----------------------------------------------------------------------
# FIGURE 5: SV posterior distributions (if traces available)
# -----------------------------------------------------------------------
def plot_sv_posteriors(sv_traces: dict):
    """
    Posterior density plots for key SV parameters (phi, sigma_eta, nu)
    across commodities. Only works if SV traces have been saved.

    Parameters
    ----------
    sv_traces : dict
        {commodity: fit_result_dict} from fit_sv().
    """
    params_to_plot = {
        "phi":       (r"$\phi$ (persistence)", None),
        "sigma_eta": (r"$\sigma_\eta$ (vol-of-vol)", None),
        "nu":        (r"$\nu$ (degrees of freedom)", None),
    }

    commodities = list(sv_traces.keys())
    n_params    = len(params_to_plot)

    fig, axes = plt.subplots(n_params, len(commodities),
                              figsize=(4*len(commodities), 3*n_params))
    fig.suptitle("SV-t Posterior Distributions by Commodity",
                 fontsize=12, fontweight="bold")

    for j, commodity in enumerate(commodities):
        trace = sv_traces[commodity].get("trace")
        if trace is None:
            continue
        for i, (param, (label, _)) in enumerate(params_to_plot.items()):
            ax = axes[i][j] if len(commodities) > 1 else axes[i]
            if param not in trace.posterior:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center")
                continue
            samples = trace.posterior[param].values.flatten()
            ax.hist(samples, bins=40, density=True,
                    color=COLORS[commodity], alpha=0.7, edgecolor="none")
            ax.axvline(np.median(samples), color="black",
                       linewidth=1.2, linestyle="--",
                       label=f"Median: {np.median(samples):.3f}")
            ax.set_xlabel(label, fontsize=9)
            if j == 0:
                ax.set_ylabel("Density")
            if i == 0:
                ax.set_title(commodity.capitalize(), fontsize=10,
                             fontweight="bold")
            ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "fig5_sv_posteriors")


if __name__ == "__main__":
    print("plot_utils.py loaded. Call individual functions with real data.")
    print(f"Figures will be saved to: {FIGURES_DIR}")
