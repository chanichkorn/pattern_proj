"""
evaluation.py
=============
P6 — Evaluation, Visualisation, and Error Analysis

Generates all plots needed for the project presentation:

  1. Cumulative return curves       (all strategies vs market)
  2. Rolling volatility             (30-day window)
  3. Portfolio weight evolution     (heatmap over time)
  4. Correlation matrix comparison  (real vs GAN-generated)
  5. Training loss curves           (W, L_D, L_G over epochs)
  6. Attention weight heatmap       (which stock pairs GAT attends to)
  7. Results summary table          (printed + saved as CSV)

Usage:
    # point to run folder — read config, checkpoint, training_history
    python src/evaluation.py --run-dir results/v1_lr3e4

    # custom checkpoint name 
    python src/evaluation.py --run-dir results/v1_lr3e4 --checkpoint epoch_0240.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from backtest   import run_backtest, compute_metrics
from data_pipeline import load_config
from dataset    import load_datasets
from gat        import GAT, attention_to_matrix
from generator  import Generator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── Plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
})

COLORS = {
    "our_model":    "#2563EB",   # blue
    "sample_cov":   "#16A34A",   # green
    "equal_weight": "#DC2626",   # red
    "dcc_garch":    "#9333EA",   # purple
}
LABELS = {
    "our_model":    "Our Model (GAT+GAN)",
    "sample_cov":   "Sample Cov GMV",
    "equal_weight": "Equal Weight (1/N)",
    "dcc_garch":    "DCC-GARCH GMV",
}


# ---------------------------------------------------------------------------
# 1. Cumulative Return Curves
# ---------------------------------------------------------------------------

def plot_cumulative_returns(results: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = ["our_model", "dcc_garch", "sample_cov", "equal_weight"]
    dates      = pd.to_datetime(results["holding_dates"])

    for name in strategies:
        r    = results["returns"][name]
        cum  = np.cumprod(1 + r) - 1          # cumulative return
        ax.plot(dates, cum * 100,
                label=LABELS[name],
                color=COLORS[name],
                linewidth=2.0 if name == "our_model" else 1.5,
                linestyle="-" if name == "our_model" else "--")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Cumulative Returns — Test Period (Apr–Dec 2019)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 2. Rolling Volatility
# ---------------------------------------------------------------------------

def plot_rolling_volatility(results: dict, out_path: str, window: int = 21) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    strategies = ["our_model", "dcc_garch", "sample_cov", "equal_weight"]
    dates      = pd.to_datetime(results["holding_dates"])

    for name in strategies:
        r        = pd.Series(results["returns"][name], index=dates)
        roll_vol = r.rolling(window).std() * np.sqrt(252) * 100
        ax.plot(dates, roll_vol,
                label=LABELS[name],
                color=COLORS[name],
                linewidth=2.0 if name == "our_model" else 1.5,
                linestyle="-" if name == "our_model" else "--")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(f"Rolling {window}-Day Annualised Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 3. Portfolio Weight Evolution (heatmap)
# ---------------------------------------------------------------------------

def plot_weight_heatmap(results: dict, tickers: list, out_path: str) -> None:
    weights = results["weights"]["our_model"]    # [T, N]
    W       = np.array(weights).T                # [N, T]
    dates   = pd.to_datetime(results["holding_dates"])

    # Only show rebalance points (non-zero turnover)
    to_arr = results.get("returns", {})          # use full time axis

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(W * 100, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=10, interpolation="nearest")

    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=7)

    # X-axis: show every ~20 dates
    step = max(1, len(dates) // 8)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels(
        [d.strftime("%b %y") for d in dates[::step]], rotation=45, fontsize=8
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Weight (%)")

    ax.set_title("Portfolio Weight Evolution — Our Model (GAT+GAN)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 4. Correlation Matrix: Real vs Generated
# ---------------------------------------------------------------------------

def plot_correlation_comparison(
    C_real: np.ndarray,
    C_fake: np.ndarray,
    tickers: list,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _heatmap(ax, mat, title, vmin=-1, vmax=1):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title)
        step = max(1, len(tickers) // 8)
        ax.set_xticks(range(0, len(tickers), step))
        ax.set_xticklabels(tickers[::step], rotation=90, fontsize=6)
        ax.set_yticks(range(0, len(tickers), step))
        ax.set_yticklabels(tickers[::step], fontsize=6)
        return im

    _heatmap(axes[0], C_real, "Empirical Correlation (Real)")
    _heatmap(axes[1], C_fake, "GAN-Generated Correlation")
    diff = C_fake - C_real
    im3  = _heatmap(axes[2], diff, "Difference (GAN − Real)",
                    vmin=-0.5, vmax=0.5)

    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Correlation Matrix: Real vs GAN-Generated", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 5. Training Loss Curves
# ---------------------------------------------------------------------------

def plot_training_curves(history_path: str, out_path: str) -> None:
    if not Path(history_path).exists():
        logger.warning(f"Training history not found: {history_path}")
        return

    history = np.load(history_path, allow_pickle=True).item()
    train_h = history.get("train", [])
    val_h   = history.get("val",   [])

    if not train_h:
        logger.warning("Empty training history.")
        return

    epochs   = list(range(1, len(train_h) + 1))
    loss_D   = [h.get("loss_D",      float("nan")) for h in train_h]
    loss_G   = [h.get("loss_G",      float("nan")) for h in train_h]
    wass     = [h.get("wasserstein", float("nan")) for h in train_h]
    val_wass = [h.get("val_wasserstein", float("nan")) for h in val_h]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, wass,     color="#2563EB", linewidth=1.5, label="Train W")
    axes[0].plot(epochs, val_wass, color="#F59E0B", linewidth=1.5,
                 linestyle="--", label="Val W")
    axes[0].set_title("Wasserstein Distance")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, loss_D, color="#DC2626", linewidth=1.5)
    axes[1].set_title("Critic Loss (L_D)")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, loss_G, color="#16A34A", linewidth=1.5)
    axes[2].set_title("Generator Loss (L_G)")
    axes[2].set_xlabel("Epoch")

    fig.suptitle("Training Curves — GAT + WGAN-GP", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 6. GAT Attention Heatmap
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_attention_heatmap(
    gat: GAT,
    X: torch.Tensor,
    A: torch.Tensor,
    tickers: list,
    out_path: str,
    head: int = 0,
) -> None:
    gat.eval()
    _, _, alpha = gat(X, A)              # alpha: [1, N, N, H]
    attn = attention_to_matrix(alpha, head=head, sample=0).numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=attn.max(),
                   interpolation="nearest")

    step = max(1, len(tickers) // 8)
    ax.set_xticks(range(0, len(tickers), step))
    ax.set_xticklabels(tickers[::step], rotation=90, fontsize=7)
    ax.set_yticks(range(0, len(tickers), step))
    ax.set_yticklabels(tickers[::step], fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"GAT Attention Weights — Layer 2, Head {head+1}\n"
                 f"(Brighter = model paid more attention to this pair)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 7. Results Summary Table
# ---------------------------------------------------------------------------

def print_and_save_table(results: dict, out_path: str) -> None:
    strategies = ["our_model", "dcc_garch", "sample_cov", "equal_weight"]

    rows = []
    for name in strategies:
        m = results[name]
        rows.append({
            "Strategy":         LABELS[name],
            "Ann. Return (%)":  round(m["ann_return"]   * 100, 2),
            "Ann. Vol (%)":     round(m["ann_vol"]       * 100, 2),
            "Sharpe Ratio":     round(m["sharpe"],              3),
            "Max Drawdown (%)": round(m["max_drawdown"]  * 100, 2),
            "Avg Turnover":     round(m.get("avg_turnover", 0), 4),
            "Total Return (%)": round(m["total_return"]  * 100, 2),
        })

    df = pd.DataFrame(rows).set_index("Strategy")

    print("\n" + "=" * 80)
    print("  BACKTEST RESULTS SUMMARY — SET50 (Test: Apr–Dec 2019)")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80 + "\n")

    df.to_csv(out_path)
    logger.info(f"Saved: {out_path}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    run_dir: Path | None = None,
) -> None:
    """
    Parameters
    ----------
    config_path     : path of config.yaml
    checkpoint_path : path of .pt checkpoint
    run_dir         : run folder
    """
    cfg    = load_config(config_path)
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )

    out_dir = (Path(run_dir) / "figures") if run_dir else Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run backtest ──────────────────────────────────────────────────────
    logger.info("Running backtest...")
    results = run_backtest(config_path, checkpoint_path, n_gen_samples=10)

    # ── Load metadata ─────────────────────────────────────────────────────
    processed_dir = cfg["data"]["processed_dir"]
    with open(Path(processed_dir) / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    tickers = meta["tickers"]

    # ── Load models for attention + correlation plots ─────────────────────
    _, _, test_ds, _ = load_datasets(processed_dir, cfg)
    N = test_ds.n_stocks
    cfg["model"]["gat"]["node_feature_dim"] = test_ds.feature_dim

    gat       = GAT.from_config(cfg).to(device)
    generator = Generator.from_config(cfg, n_stocks=N).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    gat.load_state_dict(ckpt["gat"])
    generator.load_state_dict(ckpt["generator"])

    # Pick a representative test window (middle of test set)
    mid_idx = test_ds.indices[len(test_ds) // 2]
    X_t  = torch.from_numpy(test_ds.X[mid_idx]).unsqueeze(0).to(device)
    A_t  = torch.from_numpy(test_ds.A[mid_idx]).unsqueeze(0).to(device)
    C_t  = test_ds.C[mid_idx]

    # ── Generate a sample correlation matrix ─────────────────────────────
    with torch.no_grad():
        _, g, _ = gat(X_t, A_t)
        R_fake  = generator.sample(g, n_samples=1)[0].cpu().numpy()

    # ── Plot 1: Cumulative returns ─────────────────────────────────────────
    plot_cumulative_returns(results, str(out_dir / "1_cumulative_returns.png"))

    # ── Plot 2: Rolling volatility ────────────────────────────────────────
    plot_rolling_volatility(results, str(out_dir / "2_rolling_volatility.png"))

    # ── Plot 3: Weight heatmap ────────────────────────────────────────────
    plot_weight_heatmap(results, tickers, str(out_dir / "3_weight_heatmap.png"))

    # ── Plot 4: Correlation comparison ────────────────────────────────────
    plot_correlation_comparison(
        C_t, R_fake, tickers,
        str(out_dir / "4_correlation_comparison.png")
    )

    # ── Plot 5: Training curves ───────────────────────────────────────────
    history_path = (
        str(Path(run_dir) / "training_history.npy")
        if run_dir
        else "results/training_history.npy"
    )
    plot_training_curves(
        history_path,
        str(out_dir / "5_training_curves.png")
    )

    # ── Plot 6: Attention heatmap ─────────────────────────────────────────
    plot_attention_heatmap(
        gat, X_t, A_t, tickers,
        str(out_dir / "6_attention_heatmap.png"),
        head=0,
    )

    # ── Table 7: Results summary ──────────────────────────────────────────
    print_and_save_table(results, str(out_dir / "7_results_table.csv"))

    logger.info(f"\n✅  All figures saved to {out_dir}/")
    logger.info("    1_cumulative_returns.png")
    logger.info("    2_rolling_volatility.png")
    logger.info("    3_weight_heatmap.png")
    logger.info("    4_correlation_comparison.png")
    logger.info("    5_training_curves.png")
    logger.info("    6_attention_heatmap.png")
    logger.info("    7_results_table.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation & Visualisation")
    parser.add_argument("--run-dir",    required=True,
                        help="Path of run folder e.g. results/v1_lr3e4")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="Checkpoint name in <run-dir>/checkpoints/ (default: best.pt)")
    args = parser.parse_args()

    run_dir     = Path(args.run_dir)
    config_path = str(run_dir / "config.yaml")
    ckpt_path   = str(run_dir / "checkpoints" / args.checkpoint)

    logger.info(f"Run dir    : {run_dir}")
    logger.info(f"Config     : {config_path}")
    logger.info(f"Checkpoint : {ckpt_path}")

    run_evaluation(config_path, ckpt_path, run_dir=run_dir)
