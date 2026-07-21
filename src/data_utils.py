"""
src/data_utils.py

Data loading, cleaning, and validation utilities.
All decisions here trace back to the Phase 2 data cleaning policy document.

File format handling (tested against actual files):
  - cocoa_raw.csv / gold_raw.csv: yfinance multi-header format
      Row 0: Price, Close, High, Low, Open, Volume
      Row 1: Ticker, CC=F, CC=F, ...
      Row 2: Date, (blank), ...
      Row 3+: dates and prices
    -> Skip first 3 rows, assign column names manually

  - brent_oil_primary.csv: plain two-column CSV, no header
      Row 0: 1987-05-20, 18.63
      Row 1: 1987-05-21, 18.45
    -> Read with header=None, assign names ['Date','Close']

Author: Jonathan
Project: Stochastic Volatility Models for Commodity Price Risk in West Africa
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------
# DATA FILE PATHS — update these to your actual locations
# -----------------------------------------------------------------------
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

RAW_FILES = {
    "cocoa": RAW_DATA_DIR / "cocoa_raw.csv",
    "gold":  RAW_DATA_DIR / "gold_raw.csv",
    "oil":   RAW_DATA_DIR / "brent_oil_primary.csv",
}

VALIDATION_FILES = {
    "icco":       RAW_DATA_DIR / "icco_daily_prices_1994-2025.csv",
    "wti_spot":   RAW_DATA_DIR / "fred_oil.csv",
    "stooq_gold": RAW_DATA_DIR / "xauusd_d.csv",
}

# -----------------------------------------------------------------------
# SAMPLE PERIOD (Deliverable 2, Section 2)
# -----------------------------------------------------------------------
SAMPLE_START = "2003-01-01"
SAMPLE_END   = "2026-06-22"

# -----------------------------------------------------------------------
# PHASE 2 POLICY: Flagged dates for robustness checks
# -----------------------------------------------------------------------
COCOA_FLAGGED_DATES = ["2024-05-13", "2024-05-16", "2024-09-16"]


def _read_commodity_file(path: Path) -> pd.DataFrame:
    """
    Read a commodity CSV file robustly, handling all formats present in this project.
    
    Strategy: read everything as strings, filter rows to only those where the 
    index matches a YYYY-MM-DD date pattern, then convert types.
    This works regardless of how many header rows the file has.
    """
    # Peek to determine number of data columns
    peek = pd.read_csv(path, nrows=5, header=None, dtype=str)
    n_cols = peek.shape[1]
    
    if n_cols >= 5:
        # yfinance format: Date, Close, High, Low, Open, Volume
        col_names = ["Date", "Close", "High", "Low", "Open", "Volume"][:n_cols]
    else:
        # Plain two-column: Date, Close (Brent oil)
        col_names = ["Date", "Close"]
    
    # Read everything as strings with no header
    df = pd.read_csv(path, header=None, names=col_names,
                     index_col=0, dtype=str)
    
    # Keep only rows where index is a valid YYYY-MM-DD date
    date_mask = df.index.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
    df = df[date_mask]
    
    # Convert index to datetime with explicit format
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    df.index.name = "Date"
    
    # Convert Close to numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    
    # Convert Volume to numeric if present
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    
    return df.sort_index()


def load_prices(commodity: str,
                exclude_zero_volume: bool = True,
                verbose: bool = True) -> pd.Series:
    """
    Load daily closing prices for one commodity.

    Applies Phase 2 cleaning policy:
    - Restricts to common sample period (2003-01-01 to 2026-06-22)
    - For cocoa: treats zero-volume days as missing (rollover-illiquidity)
    - No interpolation of any missing values

    Parameters
    ----------
    commodity : str
        One of 'cocoa', 'gold', 'oil'.
    exclude_zero_volume : bool
        Apply zero-volume exclusion to cocoa. True by default.
    verbose : bool
        Print cleaning summary.

    Returns
    -------
    pd.Series of daily closing prices, indexed by date.
    """
    path = RAW_FILES.get(commodity)
    if path is None:
        raise ValueError(f"Unknown commodity '{commodity}'. "
                         f"Choose from {list(RAW_FILES.keys())}")
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Make sure your data files are in data/raw/"
        )

    df = _read_commodity_file(path)

    # Restrict to common sample period
    df = df.loc[SAMPLE_START:SAMPLE_END]

    prices = df["Close"].copy()

    # Cocoa: zero-volume days → NaN (Phase 2 policy Section 1)
    if commodity == "cocoa" and exclude_zero_volume and "Volume" in df.columns:
        zero_vol = df["Volume"] == 0
        n_zeroed = int(zero_vol.sum())
        if n_zeroed > 0:
            prices[zero_vol] = np.nan
            if verbose:
                print(f"  [cocoa] {n_zeroed} zero-volume days set to NaN "
                      f"(Phase 2 cleaning policy)")

    prices.name = f"{commodity}_price"
    return prices


def compute_log_returns(prices: pd.Series,
                        flag_dates: list = None,
                        verbose: bool = True) -> pd.Series:
    """
    Compute daily log-returns from a price series.

    Handles non-positive prices (undefined log) by coercing to NaN
    with an explicit warning. Does NOT interpolate.
    """
    non_positive = prices[prices <= 0].dropna()
    if len(non_positive) > 0 and verbose:
        print(f"  WARNING: {len(non_positive)} non-positive price(s) — "
              f"log-return undefined, set to NaN:")
        for d, v in non_positive.items():
            print(f"    {d.date()}: {v}")

    with np.errstate(invalid="ignore", divide="ignore"):
        log_ret = np.log(prices / prices.shift(1))

    log_ret.name = prices.name.replace("_price", "_logret")

    if flag_dates and verbose:
        flagged = [pd.Timestamp(d) for d in flag_dates
                   if pd.Timestamp(d) in log_ret.index]
        if flagged:
            print(f"  [flag] {len(flagged)} pre-committed robustness-check "
                  f"dates flagged: {[d.date() for d in flagged]}")

    return log_ret


def load_all_returns(exclude_zero_volume_cocoa: bool = True,
                     verbose: bool = True) -> dict:
    """
    Load clean log-return series for all three commodities.

    Returns
    -------
    dict with keys 'cocoa', 'gold', 'oil', each a pd.Series of log-returns
    with NaN rows dropped (not interpolated).
    """
    result = {}
    summaries = []

    for commodity in ["cocoa", "gold", "oil"]:
        if verbose:
            print(f"\n{'='*40}")
            print(f"Loading {commodity.upper()}")

        prices = load_prices(
            commodity,
            exclude_zero_volume=(commodity == "cocoa" and exclude_zero_volume_cocoa),
            verbose=verbose,
        )

        flag_dates = COCOA_FLAGGED_DATES if commodity == "cocoa" else None
        log_ret = compute_log_returns(prices, flag_dates=flag_dates,
                                      verbose=verbose)

        n_before = len(log_ret)
        log_ret = log_ret.dropna()
        n_dropped = n_before - len(log_ret)
        result[commodity] = log_ret

        if verbose:
            print(f"  Observations: {len(log_ret)} ({n_dropped} dropped)")
            print(f"  Period: {log_ret.index.min().date()} "
                  f"to {log_ret.index.max().date()}")
            print(f"  Ann. vol: {log_ret.std()*np.sqrt(252):.4f}  "
                  f"Skew: {log_ret.skew():.4f}  "
                  f"Kurt: {log_ret.kurt():.4f}")

        summaries.append({
            "commodity": commodity,
            "n_obs": len(log_ret),
            "start": str(log_ret.index.min().date()),
            "end":   str(log_ret.index.max().date()),
            "mean":     round(log_ret.mean(), 6),
            "ann_vol":  round(log_ret.std() * np.sqrt(252), 4),
            "skewness": round(log_ret.skew(), 4),
            "excess_kurtosis": round(log_ret.kurt(), 4),
        })

    if verbose:
        print(f"\n{'='*40}")
        print("SUMMARY")
        print(pd.DataFrame(summaries).set_index("commodity").to_string())

    return result


def load_all_prices() -> dict:
    """
    Load raw price level series for all three commodities.
    Used by the OU model (which fits on price levels, not returns).
    """
    return {c: load_prices(c, verbose=False) for c in ["cocoa", "gold", "oil"]}


def save_processed_series(returns: dict, prices: dict):
    """
    Save cleaned series to data/processed/ for reproducibility.
    """
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)

    for commodity in ["cocoa", "gold", "oil"]:
        returns[commodity].to_csv(processed_dir / f"{commodity}_log_returns.csv",
                                  header=True)
        prices[commodity].dropna().to_csv(processed_dir / f"{commodity}_prices.csv",
                                          header=True)

    print(f"Processed series saved to {processed_dir}")


if __name__ == "__main__":
    print("Testing data_utils.py...")
    returns = load_all_returns(verbose=True)
    prices  = load_all_prices()

    for c in ["cocoa", "gold", "oil"]:
        assert len(returns[c]) > 4000, f"{c}: too few observations"
        assert returns[c].isna().sum() == 0, f"{c}: NaN in returns"
    print("\nAll assertions passed.")

    save_processed_series(returns, prices)
