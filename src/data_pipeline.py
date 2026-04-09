"""
data_pipeline.py
================
P1 — Data Pipeline for SET50 Portfolio Optimization

Responsibilities:
  1. Fetch OHLCV data for all SET50 tickers via yfinance
  2. Align all tickers to a common trading calendar
  3. Compute daily log returns
  4. Generate rolling windows:
       - Correlation matrix C_t  [N × N]  → GAN "real" samples
       - Node feature matrix X_t [N × d]  → GAT input
       - Adjacency matrix A_t    [N × N]  → GAT graph structure
  5. Save processed artefacts to disk (parquet + npy)

Usage:
    python src/data_pipeline.py --config configs/config.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Step 1 — Fetch raw OHLCV
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    raw_dir: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted-close prices for all tickers.
    Caches each ticker as a CSV so re-runs skip network calls.

    Returns
    -------
    prices : pd.DataFrame  [dates × tickers]  — adjusted close prices
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    frames = {}
    failed = []

    for ticker in tqdm(tickers, desc="Fetching tickers"):
        # Cache stores only the close price as a clean single-column CSV
        csv_file = raw_path / f"{ticker.replace('.', '_')}.csv"

        if csv_file.exists() and not force_refresh:
            # Cached CSV: single column (date → close price)
            series = pd.read_csv(
                csv_file, index_col=0, parse_dates=True
            ).squeeze("columns")
            series.name = ticker
            frames[ticker] = series
            continue

        # ── Fresh download ────────────────────────────────────────────────
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                logger.warning(f"No data returned for {ticker} — skipping.")
                failed.append(ticker)
                continue
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")
            failed.append(ticker)
            continue

        # ── Flatten MultiIndex columns (new yfinance ≥ 0.2) ──────────────
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ── Extract close price ───────────────────────────────────────────
        close_col = "Close" if "Close" in df.columns else \
                    "Adj Close" if "Adj Close" in df.columns else None
        if close_col is None:
            logger.warning(f"No close column found for {ticker} — skipping.")
            failed.append(ticker)
            continue

        series = df[close_col]
        # If MultiIndex flatten produced duplicate col names, df[col] is a
        # DataFrame — take the first column in that case
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        series = series.dropna()
        series.name = ticker

        # Cache as clean single-column CSV (no MultiIndex header)
        series.to_frame(name="Close").to_csv(csv_file)
        frames[ticker] = series

    if failed:
        logger.warning(f"Skipped {len(failed)} tickers: {failed}")

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    logger.info(f"Fetched {prices.shape[1]} tickers × {prices.shape[0]} dates")
    return prices


# ---------------------------------------------------------------------------
# Step 2 — Clean & align
# ---------------------------------------------------------------------------

def clean_prices(
    prices: pd.DataFrame,
    min_valid_fraction: float = 0.90,
) -> pd.DataFrame:
    """
    1. Drop tickers with too many missing values.
    2. Forward-fill remaining gaps (max 3 consecutive days).
    3. Drop any remaining NaN rows.

    Returns
    -------
    prices : pd.DataFrame  cleaned, all-numeric
    """
    n_rows = len(prices)

    # Drop tickers below threshold
    valid_frac = prices.notna().mean()
    keep = valid_frac[valid_frac >= min_valid_fraction].index.tolist()
    dropped = set(prices.columns) - set(keep)
    if dropped:
        logger.warning(f"Dropping tickers with sparse data: {dropped}")
    prices = prices[keep]

    # Forward-fill short gaps (holiday closures, suspension)
    prices = prices.ffill(limit=3)

    # Drop rows that still have NaNs (e.g., very beginning of series)
    prices = prices.dropna()

    logger.info(
        f"Cleaned prices: {prices.shape[1]} tickers × {prices.shape[0]} trading days"
    )
    return prices


# ---------------------------------------------------------------------------
# Step 3 — Log returns
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Daily log returns:  r_t = log(P_t / P_{t-1})

    Returns
    -------
    returns : pd.DataFrame  same shape as prices, first row dropped
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    logger.info(f"Log returns: {returns.shape}")
    return returns


# ---------------------------------------------------------------------------
# Step 4 — Node features
# ---------------------------------------------------------------------------

SECTOR_LIST = [
    "Banking", "Energy", "Telecom", "Retail",
    "Food", "Industrial", "Property", "Transport",
    "Healthcare", "Finance", "Media",
]


def build_node_features(
    returns: pd.DataFrame,
    t: int,
    window: int,
    sector_map: dict[str, str],
) -> np.ndarray:
    """
    Build node feature matrix X_t for window ending at index t.

    Features per stock (d = 4 numeric + len(SECTOR_LIST) sector one-hot):
        [0]  5-day return           (short momentum)
        [1]  21-day return          (medium momentum)
        [2]  realized volatility    (std of returns in window)
        [3]  volume z-score         (set to 0 if volume not available)
        [4:] sector one-hot

    Parameters
    ----------
    returns : pd.DataFrame  [dates × tickers]
    t       : int           end index of current window (exclusive)
    window  : int           look-back window length
    sector_map : dict       ticker → sector string

    Returns
    -------
    X : np.ndarray  [N × d]
    """
    tickers = returns.columns.tolist()
    N = len(tickers)
    window_returns = returns.iloc[t - window : t]  # [window × N]

    n_numeric = 4
    d = n_numeric + len(SECTOR_LIST)
    X = np.zeros((N, d), dtype=np.float32)

    for i, ticker in enumerate(tickers):
        r = window_returns[ticker].values  # [window]

        # --- numeric features ---
        # 5-day return
        X[i, 0] = r[-5:].sum() if len(r) >= 5 else r.sum()
        # 21-day return (full window)
        X[i, 1] = r.sum()
        # Realized volatility
        X[i, 2] = r.std()
        # Volume z-score placeholder (0 until volume data added)
        X[i, 3] = 0.0

        # --- sector one-hot ---
        sector = sector_map.get(ticker, "")
        if sector in SECTOR_LIST:
            sector_idx = SECTOR_LIST.index(sector)
            X[i, n_numeric + sector_idx] = 1.0

    return X  # [N × d]


# ---------------------------------------------------------------------------
# Step 5 — Correlation matrix (GAN "real" samples)
# ---------------------------------------------------------------------------

def build_correlation_matrix(
    returns: pd.DataFrame,
    t: int,
    window: int,
) -> np.ndarray:
    """
    Empirical Pearson correlation matrix over the rolling window.

    Returns
    -------
    C : np.ndarray  [N × N]  symmetric, diagonal = 1
    """
    window_returns = returns.iloc[t - window : t]
    C = window_returns.corr().values.astype(np.float32)

    # Safety: replace NaN (zero-variance stocks) with 0 off-diagonal
    np.fill_diagonal(C, 1.0)
    C = np.nan_to_num(C, nan=0.0)

    # Clip to [-1, 1] for numerical safety
    C = np.clip(C, -1.0, 1.0)
    return C  # [N × N]


# ---------------------------------------------------------------------------
# Step 6 — Adjacency matrix (GAT graph)
# ---------------------------------------------------------------------------

def build_adjacency_matrix(
    C: np.ndarray,
    threshold: float = 0.3,
    abs_correlation: bool = True,
) -> np.ndarray:
    """
    Build adjacency matrix from correlation matrix.

    Edge rule: A[i,j] = |C[i,j]|  if |C[i,j]| >= threshold, else 0
    Self-loops: A[i,i] = 1  (allows each node to keep its own features)
    Row-normalize: A_hat = D^{-1} A  (so attention operates on unit scale)

    Returns
    -------
    A : np.ndarray  [N × N]  row-normalized adjacency
    """
    W = np.abs(C) if abs_correlation else C.copy()
    W[W < threshold] = 0.0
    np.fill_diagonal(W, 1.0)  # self-loops

    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div-by-zero
    A = (W / row_sums).astype(np.float32)

    return A  # [N × N]


# ---------------------------------------------------------------------------
# Step 7 — Run full pipeline and save
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str, force_refresh: bool = False) -> dict:
    """
    End-to-end pipeline. Returns a dict of processed arrays and metadata.

    Saves to disk:
        data/processed/returns.parquet
        data/processed/prices.parquet
        data/processed/windows.npz        ← X, A, C arrays for all windows
        data/processed/metadata.yaml      ← tickers, dates, window indices
    """
    cfg = load_config(config_path)

    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch ──────────────────────────────────────────────────────────
    prices = fetch_prices(
        tickers=cfg["data"]["tickers"],
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        raw_dir=cfg["data"]["raw_dir"],
        force_refresh=force_refresh,
    )

    # ── 2. Clean ──────────────────────────────────────────────────────────
    prices = clean_prices(prices, cfg["data"]["min_valid_fraction"])
    tickers = prices.columns.tolist()
    N = len(tickers)

    # ── 3. Returns ────────────────────────────────────────────────────────
    returns = compute_log_returns(prices)

    # Save intermediate
    prices.to_parquet(processed_dir / "prices.parquet")
    returns.to_parquet(processed_dir / "returns.parquet")
    logger.info("Saved prices and returns to parquet.")

    # ── 4. Rolling windows ────────────────────────────────────────────────
    window = cfg["data"]["window"]
    sector_map = cfg["data"]["sector_map"]
    threshold = cfg["graph"]["threshold"]
    abs_corr = cfg["graph"]["abs_correlation"]

    T = len(returns)
    n_windows = T - window  # first usable t = window

    # Compute feature dim dynamically: 4 numeric + len(SECTOR_LIST) one-hot
    feature_dim = 4 + len(SECTOR_LIST)

    logger.info(f"Generating {n_windows} rolling windows (T={window}, d={feature_dim})...")

    all_X = np.zeros((n_windows, N, feature_dim), dtype=np.float32)  # node features
    all_A = np.zeros((n_windows, N, N), dtype=np.float32)            # adjacency
    all_C = np.zeros((n_windows, N, N), dtype=np.float32)            # correlation

    for idx, t in enumerate(tqdm(range(window, T), desc="Rolling windows")):
        C = build_correlation_matrix(returns, t, window)
        X = build_node_features(returns, t, window, sector_map)
        A = build_adjacency_matrix(C, threshold, abs_corr)

        all_C[idx] = C
        all_X[idx] = X
        all_A[idx] = A

    # ── 5. Save processed arrays ──────────────────────────────────────────
    np.savez_compressed(
        processed_dir / "windows.npz",
        X=all_X,   # [n_windows, N, d]
        A=all_A,   # [n_windows, N, N]
        C=all_C,   # [n_windows, N, N]
    )

    # Save metadata
    window_dates = returns.index[window:].strftime("%Y-%m-%d").tolist()
    metadata = {
        "tickers": tickers,
        "n_stocks": N,
        "n_windows": n_windows,
        "window_size": window,
        "feature_dim": all_X.shape[-1],
        "window_dates": window_dates,
        "start_date": cfg["data"]["start_date"],
        "end_date": cfg["data"]["end_date"],
    }
    with open(processed_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    logger.info(
        f"Pipeline complete — "
        f"{n_windows} windows | "
        f"X: {all_X.shape} | A: {all_A.shape} | C: {all_C.shape}"
    )

    return {
        "prices": prices,
        "returns": returns,
        "X": all_X,
        "A": all_A,
        "C": all_C,
        "tickers": tickers,
        "window_dates": window_dates,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Quick diagnostic
# ---------------------------------------------------------------------------

def print_diagnostics(data: dict) -> None:
    """Print sanity-check stats after pipeline runs."""
    C = data["C"]
    A = data["A"]
    X = data["X"]

    print("\n" + "=" * 55)
    print("  PIPELINE DIAGNOSTICS")
    print("=" * 55)
    print(f"  Tickers kept       : {len(data['tickers'])}")
    print(f"  Trading days       : {len(data['returns'])}")
    print(f"  Windows generated  : {C.shape[0]}")
    print(f"  Node feature dim   : {X.shape[-1]}")
    print()
    print("  Correlation matrices (C):")
    N = C.shape[1]
    off_diag_mask = ~np.eye(N, dtype=bool)                  # [N, N]
    off_diag_vals = C[:, off_diag_mask]                     # [n_windows, N*(N-1)]
    diag_ok = np.allclose(C[:, np.arange(N), np.arange(N)], 1.0)
    print(f"    min={C.min():.3f}  max={C.max():.3f}  "
          f"mean(off-diag)={off_diag_vals.mean():.3f}")
    print(f"    diagonal == 1.0  : {diag_ok}")

    print()
    print("  Adjacency matrices (A):")
    avg_degree = (A > 0).sum(axis=(1, 2)).mean() / A.shape[1]
    print(f"    avg degree per node : {avg_degree:.1f}")
    row_sum_ok = np.allclose(A.sum(axis=2), 1.0, atol=1e-5)
    print(f"    row-normalized      : {row_sum_ok}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SET50 Data Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-download raw data even if cached CSVs exist"
    )
    args = parser.parse_args()

    data = run_pipeline(args.config, force_refresh=args.force_refresh)
    print_diagnostics(data)
