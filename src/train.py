"""
train.py
========
P3 — Full Training Loop: GAT + WGAN-GP

Pipeline per training step:
    1. Sample batch (X, A, C_real) from DataLoader
    2. GAT forward: X, A → Z, g, alpha
    3. [Critic steps × n_critic]
         a. Generate C_fake = G(noise, g)
         b. L_D = D(C_fake) - D(C_real) + λ·GP
         c. Backprop, clip gradients, update Critic
    4. [Generator step]
         a. Generate C_fake = G(noise, g)
         b. L_G = -D(C_fake)
         c. Backprop, clip gradients, update GAT + Generator jointly
    5. Update EMA shadow weights for Generator

Both GAT and Generator are updated together in the generator step —
the gradient flows: D → G → GAT. This means the GAT learns to produce
graph embeddings g that help the Generator fool the Critic.

Training improvements over baseline:
  • CosineAnnealingLR schedulers for both optimisers (lr → eta_min=1e-6)
  • Gradient clipping (max_norm from config) for stable training
  • Exponential Moving Average (EMA) of Generator weights for stable inference

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import copy
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import run_pipeline, load_config
from dataset import load_datasets, make_dataloaders
from gat import GAT
from generator import Generator
from discriminator import Critic, gradient_penalty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA) for Generator
# ---------------------------------------------------------------------------

class EMA:
    """
    Maintains an exponential moving average of a model's parameters.

    Usage:
        ema = EMA(generator, decay=0.999)
        # after each generator update:
        ema.update()
        # for validation / inference:
        with ema.average_parameters():
            R = generator.sample(g)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model  = model
        self.decay  = decay
        self.shadow: dict = {}
        self._register()

    def _register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self) -> None:
        """Call after each generator optimiser step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self) -> None:
        """Swap live weights with EMA weights (for inference)."""
        self._backup: dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore live weights after inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    class _ShadowContext:
        def __init__(self, ema: "EMA"):
            self._ema = ema

        def __enter__(self):
            self._ema.apply_shadow()
            return self

        def __exit__(self, *args):
            self._ema.restore()

    def average_parameters(self) -> "_ShadowContext":
        """Context manager: temporarily swap in EMA weights."""
        return self._ShadowContext(self)


# ---------------------------------------------------------------------------
# Loss tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Lightweight running average tracker for training metrics."""

    def __init__(self):
        self._sums   = {}
        self._counts = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            self._sums[k]   = self._sums.get(k, 0.0)   + val
            self._counts[k] = self._counts.get(k, 0)   + 1

    def mean(self, key: str) -> float:
        if self._counts.get(key, 0) == 0:
            return float("nan")
        return self._sums[key] / self._counts[key]

    def reset(self):
        self._sums.clear()
        self._counts.clear()

    def summary(self) -> dict:
        return {k: self.mean(k) for k in self._sums}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    epoch: int,
    gat: GAT,
    generator: Generator,
    critic: Critic,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    metrics: dict,
    path: str,
    ema: EMA | None = None,
) -> None:
    payload = {
        "epoch":     epoch,
        "gat":       gat.state_dict(),
        "generator": generator.state_dict(),
        "critic":    critic.state_dict(),
        "opt_G":     opt_G.state_dict(),
        "opt_D":     opt_D.state_dict(),
        "metrics":   metrics,
    }
    if ema is not None:
        payload["ema_shadow"] = ema.shadow
    torch.save(payload, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    path: str,
    gat: GAT,
    generator: Generator,
    critic: Critic,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    device: torch.device,
    ema: EMA | None = None,
) -> int:
    ckpt = torch.load(path, map_location=device)
    gat.load_state_dict(ckpt["gat"])
    generator.load_state_dict(ckpt["generator"])
    critic.load_state_dict(ckpt["critic"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    if ema is not None and "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]
    logger.info(f"Resumed from checkpoint {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    gat: GAT,
    generator: Generator,
    critic: Critic,
    loader: torch.utils.data.DataLoader,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    cfg: dict,
    device: torch.device,
    tracker: MetricTracker,
    ema: EMA | None = None,
) -> None:
    gat.train()
    generator.train()
    critic.train()

    noise_dim  = cfg["model"]["gan"]["noise_dim"]
    n_critic   = cfg["model"]["gan"]["n_critic_steps"]
    gp_lambda  = cfg["model"]["gan"]["gp_lambda"]
    grad_clip  = cfg["model"]["gan"].get("grad_clip", 0.0)   # 0 = disabled

    for batch in loader:
        X      = batch["X"].to(device)   # [B, N, d]
        A      = batch["A"].to(device)   # [B, N, N]
        C_real = batch["C"].to(device)   # [B, N, N]
        B      = X.shape[0]

        # ── GAT: get graph embedding ──────────────────────────────────────
        # Detach for critic steps to avoid unnecessary GAT backprop
        with torch.no_grad():
            _, g_detached, _ = gat(X, A)   # [B, emb_dim]

        # ═══════════════════════════════════════════════════════════════════
        # CRITIC STEPS  (n_critic times per generator step)
        # ═══════════════════════════════════════════════════════════════════
        for _ in range(n_critic):
            noise  = torch.randn(B, noise_dim, device=device)
            C_fake, _ = generator(noise, g_detached)

            score_real = critic(C_real, g_detached)
            score_fake = critic(C_fake.detach(), g_detached)

            gp     = gradient_penalty(critic, C_real, C_fake.detach(),
                                      g_detached, device)
            loss_D = score_fake.mean() - score_real.mean() + gp_lambda * gp

            opt_D.zero_grad()
            loss_D.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip)
            opt_D.step()

            tracker.update(
                loss_D       = loss_D,
                wasserstein  = score_real.mean() - score_fake.mean(),
                grad_penalty = gp,
            )

        # ═══════════════════════════════════════════════════════════════════
        # GENERATOR + GAT STEP
        # ═══════════════════════════════════════════════════════════════════
        # Full forward through GAT (with grad) so GAT weights update too
        _, g, _ = gat(X, A)
        noise   = torch.randn(B, noise_dim, device=device)
        C_fake, _ = generator(noise, g)

        loss_G  = -critic(C_fake, g).mean()

        opt_G.zero_grad()
        loss_G.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(gat.parameters()) + list(generator.parameters()),
                max_norm=grad_clip,
            )
        opt_G.step()

        # Update EMA after generator step
        if ema is not None:
            ema.update()

        tracker.update(loss_G=loss_G)


# ---------------------------------------------------------------------------
# Validation — Wasserstein distance proxy on val set
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    gat: GAT,
    generator: Generator,
    critic: Critic,
    loader: torch.utils.data.DataLoader,
    cfg: dict,
    device: torch.device,
    ema: EMA | None = None,
) -> dict:
    gat.eval()
    generator.eval()
    critic.eval()

    noise_dim = cfg["model"]["gan"]["noise_dim"]
    tracker   = MetricTracker()

    # Use EMA weights for validation if available
    ctx = ema.average_parameters() if ema is not None else nullcontext()
    with ctx:
        for batch in loader:
            X      = batch["X"].to(device)
            A      = batch["A"].to(device)
            C_real = batch["C"].to(device)
            B      = X.shape[0]

            _, g, _ = gat(X, A)
            noise   = torch.randn(B, noise_dim, device=device)
            C_fake, _ = generator(noise, g)

            score_real = critic(C_real, g)
            score_fake = critic(C_fake, g)

            tracker.update(
                val_wasserstein = score_real.mean() - score_fake.mean(),
                val_score_real  = score_real.mean(),
                val_score_fake  = score_fake.mean(),
            )

    return tracker.summary()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config_path: str, resume: str | None = None) -> None:
    cfg    = load_config(config_path)
    device = get_device()
    logger.info(f"Training on device: {device}")

    # ── Reproducibility ───────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────
    processed_dir = cfg["data"]["processed_dir"]
    if not (Path(processed_dir) / "windows.npz").exists():
        logger.info("Processed data not found — running pipeline first.")
        run_pipeline(config_path)

    train_ds, val_ds, _, _ = load_datasets(processed_dir, cfg)
    gcfg = cfg["model"]["gan"]

    train_loader, val_loader, _ = make_dataloaders(
        train_ds, val_ds, val_ds,   # test_ds reuses val here; test reserved for backtest
        batch_size  = gcfg["batch_size"],
        num_workers = 0,            # 0 = safe on MPS
    )

    N = train_ds.n_stocks
    logger.info(f"Dataset ready — N={N}, train={len(train_ds)}, val={len(val_ds)}")

    # ── Models ────────────────────────────────────────────────────────────
    # Fix feature dim to match actual pipeline output
    cfg["model"]["gat"]["node_feature_dim"] = train_ds.feature_dim

    gat       = GAT.from_config(cfg).to(device)
    generator = Generator.from_config(cfg, n_stocks=N).to(device)
    critic    = Critic.from_config(cfg, n_stocks=N).to(device)

    logger.info(f"GAT params       : {gat.count_parameters():,}")
    logger.info(f"Generator params : {generator.count_parameters():,}")
    logger.info(f"Critic params    : {critic.count_parameters():,}")

    # ── Optimisers ────────────────────────────────────────────────────────
    # Generator and GAT share one optimiser — they update jointly
    opt_G = torch.optim.Adam(
        list(gat.parameters()) + list(generator.parameters()),
        lr=gcfg["lr"], betas=(gcfg["beta1"], gcfg["beta2"]),
    )
    opt_D = torch.optim.Adam(
        critic.parameters(),
        lr=gcfg["lr"], betas=(gcfg["beta1"], gcfg["beta2"]),
    )

    # ── LR Schedulers ─────────────────────────────────────────────────────
    epochs       = gcfg["epochs"]
    lr_scheduler = gcfg.get("lr_scheduler", "none")
    sched_G = sched_D = None
    if lr_scheduler == "cosine":
        sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_G, T_max=epochs, eta_min=1e-6
        )
        sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_D, T_max=epochs, eta_min=1e-6
        )
        logger.info("CosineAnnealingLR schedulers enabled.")

    # ── EMA ───────────────────────────────────────────────────────────────
    use_ema = gcfg.get("use_ema", False)
    ema_decay = gcfg.get("ema_decay", 0.999)
    ema = EMA(generator, decay=ema_decay) if use_ema else None
    if use_ema:
        logger.info(f"EMA enabled (decay={ema_decay}).")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    ckpt_dir    = Path("results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        start_epoch = load_checkpoint(
            resume, gat, generator, critic, opt_G, opt_D, device, ema
        )

    # ── Training ──────────────────────────────────────────────────────────
    log_every  = max(1, epochs // 20)    # log ~20 times
    save_every = max(1, epochs // 5)     # save 5 checkpoints

    best_val_w = float("-inf")
    history    = {"train": [], "val": []}

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        t0      = time.time()
        tracker = MetricTracker()

        train_epoch(
            gat, generator, critic,
            train_loader, opt_G, opt_D,
            cfg, device, tracker, ema,
        )

        train_metrics = tracker.summary()
        val_metrics   = validate(gat, generator, critic, val_loader, cfg, device, ema)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Step LR schedulers
        if sched_G is not None:
            sched_G.step()
            sched_D.step()

        # Log
        if (epoch + 1) % log_every == 0 or epoch == 0:
            elapsed = time.time() - t0
            lr_g = opt_G.param_groups[0]["lr"]
            lr_d = opt_D.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"L_D={train_metrics.get('loss_D', float('nan')):.4f} | "
                f"L_G={train_metrics.get('loss_G', float('nan')):.4f} | "
                f"W={train_metrics.get('wasserstein', float('nan')):.4f} | "
                f"val_W={val_metrics.get('val_wasserstein', float('nan')):.4f} | "
                f"lr_G={lr_g:.2e} lr_D={lr_d:.2e} | "
                f"{elapsed:.1f}s"
            )

        # Save best model
        val_w = val_metrics.get("val_wasserstein", float("-inf"))
        if val_w > best_val_w:
            best_val_w = val_w
            save_checkpoint(
                epoch+1, gat, generator, critic, opt_G, opt_D,
                {**train_metrics, **val_metrics},
                str(ckpt_dir / "best.pt"),
                ema=ema,
            )

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                epoch+1, gat, generator, critic, opt_G, opt_D,
                {**train_metrics, **val_metrics},
                str(ckpt_dir / f"epoch_{epoch+1:04d}.pt"),
                ema=ema,
            )

    # Final save
    save_checkpoint(
        epochs, gat, generator, critic, opt_G, opt_D,
        history["val"][-1] if history["val"] else {},
        str(ckpt_dir / "final.pt"),
        ema=ema,
    )

    # Save training history
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    np.save(str(results_dir / "training_history.npy"), history)
    logger.info("Training complete.")
    logger.info(f"Best val Wasserstein distance: {best_val_w:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT + WGAN-GP")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, resume=args.resume)
