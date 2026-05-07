"""
backtest.py
===========
P4 — Walk-Forward Backtest Engine

Walk-forward scheme (no look-ahead bias):
    For each rebalance date t:
        1. Train window  : all data up to t
        2. Generate Σ̂_t  : use trained GAT + Generator
        3. Optimise w*_t : GMV solver
        4. Hold portfolio : until next rebalance date t+1
        5. Realise return : subtract transaction costs

Metrics computed:
    Annualised Return, Annualised Volatility, Sharpe Ratio,
    Maximum Drawdown, Calmar Ratio, Portfolio Turnover

Baselines included:
    1. Equal Weight (1/N)
    2. Sample Covariance GMV
    3. DCC-GARCH GMV
    3. Buy-and-Hold SET50 Index (approximated as equal weight, static)

Usage:
    # point to run folder 
    python src/backtest.py --run-dir results/v1_lr3e4

    # custom checkpoint name
    python src/backtest.py --run-dir results/v1_lr3e4 --checkpoint epoch_0240.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline   import load_config
from dataset         import load_datasets
from gat             import GAT
from generator       import Generator
from optimizer       import gmv_optimize, correlation_to_covariance
from baselines       import dcc_garch_covariance

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(returns: np.ndarray, label: str = "") -> dict:
    """
    Compute standard portfolio performance metrics.

    Parameters
    ----------
    returns : np.ndarray [T]   daily portfolio returns (decimal, e.g. 0.01)
    label   : str              strategy name

    Returns
    -------
    metrics : dict
    """
    if len(returns) == 0:
        return {}

    ann_factor = 252
    r          = np.array(returns)
    cum_ret    = np.cumprod(1 + r)

    ann_return = float(cum_ret[-1] ** (ann_factor / len(r)) - 1)
    ann_vol    = float(r.std() * np.sqrt(ann_factor))
    sharpe     = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(cum_ret)
    drawdowns   = (cum_ret - running_max) / running_max
    max_dd      = float(drawdowns.min())

    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "label":       label,
        "ann_return":  ann_return,
        "ann_vol":     ann_vol,
        "sharpe":      sharpe,
        "max_drawdown": max_dd,
        "calmar":      calmar,
        "total_return": float(cum_ret[-1] - 1),
        "n_days":      len(r),
    }


def turnover(w_prev: np.ndarray, w_next: np.ndarray) -> float:
    """One-way portfolio turnover between two weight vectors."""
    return float(np.abs(w_next - w_prev).sum()) / 2


# ---------------------------------------------------------------------------
# Generate covariance matrix from trained models
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_covariance(
    gat: GAT,
    generator: Generator,
    X: torch.Tensor,
    A: torch.Tensor,
    sigma_diag: np.ndarray,
    noise_dim: int,
    device: torch.device,
    n_samples: int = 10,
) -> np.ndarray:
    """
    Generate Σ̂ for one window using the trained GAT + Generator.

    Averages n_samples generated correlation matrices to reduce variance.

    Parameters
    ----------
    X          : FloatTensor [1, N, d]   node features
    A          : FloatTensor [1, N, N]   adjacency matrix
    sigma_diag : np.ndarray [N]          per-stock rolling volatility
    n_samples  : int                     number of matrices to average

    Returns
    -------
    Sigma : np.ndarray [N, N]   covariance matrix
    """
    gat.eval()
    generator.eval()

    _, g, _ = gat(X, A)                          # g: [1, emb_dim]
    R_hat   = generator.sample(g, n_samples)     # [n_samples, N, N]
    R_mean  = R_hat.mean(dim=0).cpu().numpy()    # [N, N] — average for stability

    return correlation_to_covariance(R_mean, sigma_diag)


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(
    config_path: str,
    checkpoint_path: str,
    n_gen_samples: int = 10,
) -> dict:
    """
    Run the full walk-forward backtest.

    Returns a dict with:
        "our_model"  : dict of metrics
        "equal_weight": dict of metrics
        "sample_cov" : dict of metrics
        "returns"    : dict of daily return arrays per strategy
        "weights"    : dict of weight histories per strategy
        "dates"      : list of holding-period dates
    """
    cfg    = load_config(config_path)
    device = get_device()
    logger.info(f"Backtest on device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    processed_dir = cfg["data"]["processed_dir"]
    _, _, test_ds, _ = load_datasets(processed_dir, cfg)

    # Load raw returns for holding period realisation
    returns_df = pd.read_parquet(Path(processed_dir) / "returns.parquet")
    tickers    = test_ds.X.shape  # we'll get tickers from metadata
    N          = test_ds.n_stocks
    window     = cfg["data"]["window"]

    logger.info(f"Test set: {len(test_ds)} windows  |  N={N}")

    # ── Load models ───────────────────────────────────────────────────────
    cfg["model"]["gat"]["node_feature_dim"] = test_ds.feature_dim
    gat       = GAT.from_config(cfg).to(device)
    generator = Generator.from_config(cfg, n_stocks=N).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    gat.load_state_dict(ckpt["gat"])
    generator.load_state_dict(ckpt["generator"])
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # ── Walk-forward loop ─────────────────────────────────────────────────
    noise_dim       = cfg["model"]["gan"]["noise_dim"]
    txn_cost        = cfg["backtest"]["transaction_cost"]
    max_w           = cfg["backtest"]["max_weight"]
    rebalance_freq  = 21    # monthly ≈ 21 trading days

    # Longer history window for sample covariance (must be > N to be non-singular)
    sample_cov_window = max(252, N + 10)

    # Strategy return containers
    all_returns = {
        "our_model":    [],
        "equal_weight": [],
        "sample_cov":   [],
        "dcc_garch":    [],
    }
    all_weights = {
        "our_model":    [],
        "equal_weight": [],
        "sample_cov":   [],
        "dcc_garch":    [],
    }
    all_turnovers = {k: [] for k in all_returns}
    holding_dates = []

    # Previous weights (for turnover computation)
    prev_w = {k: np.ones(N) / N for k in all_returns}

    returns_arr = returns_df.values   # [T, N]
    dates_arr   = returns_df.index

    with open(Path(processed_dir) / "metadata.yaml") as f:
        import yaml as _yaml
        meta = _yaml.safe_load(f)

    test_indices = test_ds.indices   # indices into the full window array

    # DCC-GARCH needs a longer window: at least 252 days for stable GARCH fit
    dcc_window = max(252, N + 10)

    # Current portfolio weights (held between rebalances)
    current_w = {k: np.ones(N) / N for k in all_returns}
    steps_since_rebalance = 0

    for step, idx in enumerate(test_indices):
        hold_start = idx + window
        hold_end   = hold_start + 1
        if hold_end > len(returns_arr):
            break

        # ── Rebalance only on schedule ────────────────────────────────────
        if steps_since_rebalance == 0:
            X_t = torch.from_numpy(test_ds.X[idx]).unsqueeze(0).to(device)
            A_t = torch.from_numpy(test_ds.A[idx]).unsqueeze(0).to(device)

            # Rolling vol for GAN covariance (short window = 21 days)
            window_ret = returns_arr[idx : idx + window]
            sigma_diag = window_ret.std(axis=0) * np.sqrt(252)

            # Our model
            Sigma_gan = generate_covariance(
                gat, generator, X_t, A_t, sigma_diag,
                noise_dim, device, n_samples=n_gen_samples,
            )
            current_w["our_model"] = gmv_optimize(Sigma_gan, max_weight=max_w)

            # Equal weight (never changes but compute for turnover tracking)
            current_w["equal_weight"] = np.ones(N) / N

            # Sample cov GMV — use long lookback window so N_obs >> N_stocks
            long_start = max(0, hold_start - sample_cov_window)
            long_ret   = returns_arr[long_start : hold_start]
            if long_ret.shape[0] > N:
                Sigma_sample = np.cov(long_ret.T) + np.eye(N) * 1e-6
            else:
                Sigma_sample = np.eye(N)
            current_w["sample_cov"] = gmv_optimize(Sigma_sample, max_weight=max_w)

            # DCC-GARCH GMV — fit on rolling 252-day window
            dcc_start  = max(0, hold_start - dcc_window)
            dcc_ret    = returns_arr[dcc_start : hold_start]
            try:
                Sigma_dcc, _ = dcc_garch_covariance(dcc_ret)
                current_w["dcc_garch"] = gmv_optimize(Sigma_dcc, max_weight=max_w)
            except Exception as e:
                logger.warning(f"DCC-GARCH failed at step {step}: {e} — using equal weight")
                current_w["dcc_garch"] = np.ones(N) / N

        # ── Daily holding period return ───────────────────────────────────
        day_returns = returns_arr[hold_start]
        holding_dates.append(dates_arr[hold_start])

        for name in all_returns:
            w  = current_w[name]
            to = turnover(prev_w[name], w) if steps_since_rebalance == 0 else 0.0
            gross = float(w @ day_returns)
            net   = gross - txn_cost * to

            all_returns[name].append(net)
            all_weights[name].append(w.copy())
            all_turnovers[name].append(to)
            prev_w[name] = w.copy()

        steps_since_rebalance = (steps_since_rebalance + 1) % rebalance_freq

        if step % 50 == 0:
            logger.info(
                f"  Step {step+1:4d}/{len(test_indices)} | "
                f"date={dates_arr[hold_start].date()} | "
                f"our_model_ret={all_returns['our_model'][-1]:.4f}"
            )

    # ── Compute metrics ───────────────────────────────────────────────────
    results = {}
    for name in all_returns:
        r = np.array(all_returns[name])
        m = compute_metrics(r, label=name)
        m["avg_turnover"] = float(np.mean(all_turnovers[name]))
        results[name]     = m

    results["returns"]      = {k: np.array(v) for k, v in all_returns.items()}
    results["weights"]      = {k: np.array(v) for k, v in all_weights.items()}
    results["holding_dates"]= holding_dates

    return results


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------

def print_results_table(results: dict) -> None:
    strategies = ["our_model", "equal_weight", "sample_cov"]
    labels     = ["Our Model (GAT+GAN)", "Equal Weight (1/N)", "Sample Cov GMV"]

    print("\n" + "=" * 72)
    print(f"  {'Strategy':<24} {'Ann.Ret':>8} {'Ann.Vol':>8} "
          f"{'Sharpe':>8} {'MaxDD':>8} {'Turnover':>9}")
    print("=" * 72)

    for strat, label in zip(strategies, labels):
        m = results[strat]
        print(
            f"  {label:<24} "
            f"{m['ann_return']:>7.2%}  "
            f"{m['ann_vol']:>7.2%}  "
            f"{m['sharpe']:>7.3f}  "
            f"{m['max_drawdown']:>7.2%}  "
            f"{m.get('avg_turnover', 0):>8.4f}"
        )

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--run-dir",    required=True,
                        help="Path of run folder e.g. results/v1_lr3e4")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="name of checkpoint in <run-dir>/checkpoints/ (default: best.pt)")
    parser.add_argument("--n-samples",  type=int, default=10,
                        help="GAN samples to average per window (higher = more stable)")
    args = parser.parse_args()

    run_dir     = Path(args.run_dir)
    config_path = str(run_dir / "config.yaml")
    ckpt_path   = str(run_dir / "checkpoints" / args.checkpoint)

    logger.info(f"Run dir    : {run_dir}")
    logger.info(f"Config     : {config_path}")
    logger.info(f"Checkpoint : {ckpt_path}")

    results = run_backtest(
        config_path     = config_path,
        checkpoint_path = ckpt_path,
        n_gen_samples   = args.n_samples,
    )

    print_results_table(results)

    # Save results
    np.save(str(run_dir / "backtest_results.npy"), results)
    logger.info(f"Results saved → {run_dir}/backtest_results.npy")
