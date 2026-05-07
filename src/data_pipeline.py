"""
data_pipeline.py
================
P1 — Data Pipeline for SET50 Portfolio Optimization

Responsibilities:
  1. Fetch OHLCV data for all SET50 tickers via yfinance
  2. Align all tickers to a common trading calendar
  3. Compute daily log returns
  4. Generate rolling windows using two look-back horizons (short=21d, long=63d):
       - Node feature matrix X_t [N × d]  → GAT input
         Features per stock (7 numeric per window × 2 windows + 11 sector one-hot):
           [0]  5-day return           (short momentum)
           [1]  w-day return           (window momentum)
           [2]  realized volatility    (std of returns in window)
           [3]  volume z-score         (last-day vol relative to window)
           [4]  RSI-14                 (momentum oscillator, normalised [0,1])
           [5]  Bollinger band width   (4·σ / |SMA| from price series)
           [6]  return skewness        (distribution asymmetry)
           ... repeated for long window (63d) ...
           [14:] sector one-hot        (11 sectors)
       - Correlation matrix C_t  [N × N]  → GAN "real" samples (short window)
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
from scipy.stats import skew as scipy_skew
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
# Step 1 — Fetch raw OHLCV (close + volume)
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    raw_dir: str,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted-close prices and volume for all tickers.
    Caches each ticker as a two-column CSV (Close, Volume) so re-runs skip
    network calls.

    Returns
    -------
    prices  : pd.DataFrame  [dates × tickers]  — adjusted close prices
    volumes : pd.DataFrame  [dates × tickers]  — daily traded volume
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    price_frames  = {}
    volume_frames = {}
    failed        = []

    for ticker in tqdm(tickers, desc="Fetching tickers"):
        csv_file = raw_path / f"{ticker.replace('.', '_')}.csv"

        if csv_file.exists() and not force_refresh:
            df_cached = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            # Support both old (1-col Close-only) and new (Close+Volume) formats
            if "Close" in df_cached.columns:
                price_frames[ticker]  = df_cached["Close"].rename(ticker)
                if "Volume" in df_cached.columns:
                    volume_frames[ticker] = df_cached["Volume"].rename(ticker)
            else:
                # Legacy: single unnamed column is Close
                price_frames[ticker] = df_cached.squeeze("columns").rename(ticker)
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

        series_close = df[close_col]
        if isinstance(series_close, pd.DataFrame):
            series_close = series_close.iloc[:, 0]
        series_close = series_close.dropna()
        series_close.name = ticker

        # ── Extract volume ────────────────────────────────────────────────
        series_vol = None
        if "Volume" in df.columns:
            series_vol = df["Volume"]
            if isinstance(series_vol, pd.DataFrame):
                series_vol = series_vol.iloc[:, 0]
            series_vol = series_vol.reindex(series_close.index).fillna(0)
            series_vol.name = ticker

        # ── Cache as two-column CSV ───────────────────────────────────────
        cache_df = series_close.to_frame(name="Close")
        if series_vol is not None:
            cache_df["Volume"] = series_vol.values
        cache_df.to_csv(csv_file)

        price_frames[ticker] = series_close
        if series_vol is not None:
            volume_frames[ticker] = series_vol

    if failed:
        logger.warning(f"Skipped {len(failed)} tickers: {failed}")

    prices = pd.DataFrame(price_frames)
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    # Build volumes aligned to prices; fill missing with 0
    if volume_frames:
        volumes = pd.DataFrame(volume_frames).reindex(prices.index).fillna(0)
    else:
        volumes = pd.DataFrame(
            np.zeros_like(prices.values), index=prices.index, columns=prices.columns
        )

    logger.info(f"Fetched {prices.shape[1]} tickers × {prices.shape[0]} dates")
    return prices, volumes


# ---------------------------------------------------------------------------
# Step 2 — Clean & align
# ---------------------------------------------------------------------------

def clean_prices(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    min_valid_fraction: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Drop tickers with too many missing values.
    2. Forward-fill remaining gaps (max 3 consecutive days).
    3. Drop any remaining NaN rows.

    Returns
    -------
    prices  : pd.DataFrame  cleaned, all-numeric
    volumes : pd.DataFrame  aligned to cleaned prices
    """
    # Drop tickers below threshold
    valid_frac = prices.notna().mean()
    keep = valid_frac[valid_frac >= min_valid_fraction].index.tolist()
    dropped = set(prices.columns) - set(keep)
    if dropped:
        logger.warning(f"Dropping tickers with sparse data: {dropped}")
    prices  = prices[keep]
    volumes = volumes.reindex(columns=keep)

    # Forward-fill short gaps (holiday closures, suspension)
    prices  = prices.ffill(limit=3)
    volumes = volumes.ffill(limit=3).fillna(0)

    # Drop rows that still have NaNs (e.g., very beginning of series)
    prices  = prices.dropna()
    volumes = volumes.reindex(prices.index).fillna(0)

    logger.info(
        f"Cleaned prices: {prices.shape[1]} tickers × {prices.shape[0]} trading days"
    )
    return prices, volumes


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
# Step 4 — Feature helper functions
# ---------------------------------------------------------------------------

def _compute_rsi(r: np.ndarray, period: int = 14) -> float:
    """
    RSI normalised to [0, 1].  0 = oversold, 1 = overbought.
    Uses a simple arithmetic average of gains and losses over the last
    `period` bars (computationally cheap; captures the same signal as
    Wilder's EMA for the rolling-window use-case here).
    """
    if len(r) < period:
        return 0.5
    tail = r[-period:]
    gains  = np.where(tail > 0, tail, 0.0).mean()
    losses = np.where(tail < 0, -tail, 0.0).mean()
    if losses < 1e-10:
        return 1.0 if gains > 0 else 0.5
    rs = gains / losses
    rsi_100 = 100.0 - 100.0 / (1.0 + rs)
    return rsi_100 / 100.0


def _compute_bb_width(p: np.ndarray) -> float:
    """
    Bollinger Band width = (upper − lower) / middle
                         = 4·σ / |SMA|

    Applied to price series `p` over the window.
    """
    if len(p) < 2:
        return 0.0
    sma = np.abs(p.mean())
    if sma < 1e-8:
        return 0.0
    return 4.0 * p.std() / sma


def _compute_skewness(r: np.ndarray) -> float:
    """Return the skewness of `r`, clipped to [-3, 3] for numerical safety."""
    if len(r) < 3:
        return 0.0
    try:
        s = float(scipy_skew(r, bias=False))
        return float(np.clip(s, -3.0, 3.0))
    except Exception:
        return 0.0


def _window_features(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    tickers: list[str],
    t: int,
    w: int,
) -> np.ndarray:
    """
    Compute 7 numeric features per stock over look-back window of length `w`.

    Feature layout (per stock):
        [0] 5-day return  (short momentum, clipped at 5 or w)
        [1] w-day return  (full window momentum)
        [2] realized volatility (std of returns)
        [3] volume z-score (last-day vol vs window mean/std)
        [4] RSI-14
        [5] Bollinger Band width (price-based)
        [6] return skewness

    Returns
    -------
    F : np.ndarray  [N, 7]
    """
    N  = len(tickers)
    F  = np.zeros((N, 7), dtype=np.float32)

    r_block = returns.iloc[t - w : t]   # [w, N]
    p_block = prices.iloc[t - w : t]    # [w, N]
    v_block = volumes.iloc[t - w : t]   # [w, N]

    for i, ticker in enumerate(tickers):
        r = r_block[ticker].values  # [w]
        p = p_block[ticker].values  # [w]
        v = v_block[ticker].values  # [w]

        # 5-day return
        F[i, 0] = r[-5:].sum() if len(r) >= 5 else r.sum()
        # w-day return
        F[i, 1] = r.sum()
        # Realized volatility
        F[i, 2] = float(r.std())
        # Volume z-score: clip to [-3, 3] to suppress extreme outliers
        # (same clipping convention used for the vol_zscore in the long window)
        v_std = float(v.std())
        if v_std > 1e-8:
            F[i, 3] = float(np.clip((v[-1] - v.mean()) / v_std, -3.0, 3.0))
        # RSI-14
        F[i, 4] = _compute_rsi(r, 14)
        # Bollinger Band width (from prices)
        F[i, 5] = _compute_bb_width(p)
        # Skewness
        F[i, 6] = _compute_skewness(r)

    return F   # [N, 7]


SECTOR_LIST = [
    "Banking", "Energy", "Telecom", "Retail",
    "Food", "Industrial", "Property", "Transport",
    "Healthcare", "Finance", "Media",
]


def build_node_features(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: int,
    window: int,
    long_window: int,
    sector_map: dict[str, str],
) -> np.ndarray:
    """
    Build node feature matrix X_t for window ending at index t.

    Features per stock  (d = 7×2 + 11 = 25):
        [0:7]  7 numeric features over the short window (21d)
        [7:14] 7 numeric features over the long  window (63d)
        [14:]  11 sector one-hot

    Parameters
    ----------
    returns     : pd.DataFrame  [dates × tickers]
    prices      : pd.DataFrame  [dates × tickers]
    volumes     : pd.DataFrame  [dates × tickers]
    t           : int           end index of current window (exclusive)
    window      : int           short look-back window length
    long_window : int           long  look-back window length
    sector_map  : dict          ticker → sector string

    Returns
    -------
    X : np.ndarray  [N × d]
    """
    tickers = returns.columns.tolist()
    N       = len(tickers)
    n_sector = len(SECTOR_LIST)
    d = 7 + 7 + n_sector   # = 25

    X = np.zeros((N, d), dtype=np.float32)

    # ── Numeric features ──────────────────────────────────────────────────
    F_short = _window_features(returns, prices, volumes, tickers, t, window)      # [N,7]
    F_long  = _window_features(returns, prices, volumes, tickers, t, long_window)  # [N,7]

    X[:, 0:7]  = F_short
    X[:, 7:14] = F_long

    # ── Sector one-hot ────────────────────────────────────────────────────
    for i, ticker in enumerate(tickers):
        sector = sector_map.get(ticker, "")
        if sector in SECTOR_LIST:
            X[i, 14 + SECTOR_LIST.index(sector)] = 1.0

    return X   # [N × 25]


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
    prices, volumes = fetch_prices(
        tickers=cfg["data"]["tickers"],
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        raw_dir=cfg["data"]["raw_dir"],
        force_refresh=force_refresh,
    )

    # ── 2. Clean ──────────────────────────────────────────────────────────
    prices, volumes = clean_prices(prices, volumes, cfg["data"]["min_valid_fraction"])
    tickers = prices.columns.tolist()
    N = len(tickers)

    # ── 3. Returns ────────────────────────────────────────────────────────
    returns = compute_log_returns(prices)
    # Align volumes to the returns index (volumes keeps full price-aligned index)
    volumes = volumes.reindex(returns.index).fillna(0)

    # Save intermediate
    prices.to_parquet(processed_dir / "prices.parquet")
    returns.to_parquet(processed_dir / "returns.parquet")
    logger.info("Saved prices and returns to parquet.")

    # ── 4. Rolling windows ────────────────────────────────────────────────
    window      = cfg["data"]["window"]
    long_window = cfg["data"].get("long_window", 63)
    sector_map  = cfg["data"]["sector_map"]
    threshold   = cfg["graph"]["threshold"]
    abs_corr    = cfg["graph"]["abs_correlation"]

    if window <= 0 or long_window <= 0:
        raise ValueError(
            f"Configured rolling windows must be > 0, got window={window}, "
            f"long_window={long_window}"
        )

    T            = len(returns)
    start_t      = max(window, long_window)  # need enough history for all windows
    n_windows    = T - start_t

    # Feature dim: 7 short + 7 long + 11 sector = 25
    feature_dim = 7 + 7 + len(SECTOR_LIST)

    logger.info(
        f"Generating {n_windows} rolling windows "
        f"(short={window}d, long={long_window}d, d={feature_dim})..."
    )

    all_X = np.zeros((n_windows, N, feature_dim), dtype=np.float32)
    all_A = np.zeros((n_windows, N, N), dtype=np.float32)
    all_C = np.zeros((n_windows, N, N), dtype=np.float32)

    for idx, t in enumerate(tqdm(range(start_t, T), desc="Rolling windows")):
        C = build_correlation_matrix(returns, t, window)
        X = build_node_features(returns, prices, volumes, t, window, long_window, sector_map)
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
    window_dates = returns.index[start_t:].strftime("%Y-%m-%d").tolist()
    metadata = {
        "tickers":      tickers,
        "n_stocks":     N,
        "n_windows":    n_windows,
        "window_size":  window,
        "long_window":  long_window,
        "feature_dim":  all_X.shape[-1],
        "window_dates": window_dates,
        "start_date":   cfg["data"]["start_date"],
        "end_date":     cfg["data"]["end_date"],
    }
    with open(processed_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    logger.info(
        f"Pipeline complete — "
        f"{n_windows} windows | "
        f"X: {all_X.shape} | A: {all_A.shape} | C: {all_C.shape}"
    )

    return {
        "prices":        prices,
        "returns":       returns,
        "X":             all_X,
        "A":             all_A,
        "C":             all_C,
        "tickers":       tickers,
        "window_dates":  window_dates,
        "metadata":      metadata,
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

    print()
    print("  Node feature ranges (X):")
    for j, name in enumerate(
        ["5d_ret", "21d_ret", "vol", "vol_z", "RSI", "BB_w", "skew",
         "5d_ret_L", "63d_ret", "vol_L", "vol_z_L", "RSI_L", "BB_w_L", "skew_L"]
    ):
        col = X[:, :, j].flatten()
        print(f"    [{j:2d}] {name:<10}: mean={col.mean():+.4f}  std={col.std():.4f}")

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

