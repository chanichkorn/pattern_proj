"""
dataset.py
==========
P1 — PyTorch Dataset for SET50 Rolling Windows

Wraps the processed .npz arrays into a Dataset usable by DataLoader
for both the GAT and the WGAN-GP training loops.

Each sample represents one rolling window t:
    X_t : node feature matrix   [N × d]   → GAT input
    A_t : adjacency matrix      [N × N]   → GAT graph structure
    C_t : correlation matrix    [N × N]   → GAN "real" sample
    g_t : graph conditioning    [4]       → hand-crafted market state
                                            (fallback if GAT not used)

Usage:
    from src.dataset import SET50Dataset, load_datasets

    train_ds, val_ds, test_ds = load_datasets("data/processed", cfg)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    for batch in train_loader:
        X, A, C, g = batch["X"], batch["A"], batch["C"], batch["g"]
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph-level conditioning vector
# ---------------------------------------------------------------------------

def compute_graph_conditioning(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute a 4-dimensional market-state conditioning vector from one window.

    This is the hand-crafted fallback conditioning for the GAN when GAT
    embeddings are not yet available (e.g., during standalone GAN tests).
    When the full GAT is plugged in, the graph-level pooling of GAT
    embeddings replaces this.

    Features:
        [0]  avg_pairwise_corr   — mean off-diagonal |C|, proxy for crowding
        [1]  avg_realized_vol    — mean of per-stock realized vols (X[:, 2])
        [2]  avg_momentum_21d    — mean of per-stock 21d returns (X[:, 1])
        [3]  corr_dispersion     — std of off-diagonal |C|, proxy for rotation

    Parameters
    ----------
    C : np.ndarray  [N × N]
    X : np.ndarray  [N × d]

    Returns
    -------
    g : np.ndarray  [4]
    """
    N = C.shape[0]
    mask = ~np.eye(N, dtype=bool)
    off_diag = np.abs(C[mask])

    g = np.array([
        off_diag.mean(),          # avg pairwise |corr|
        X[:, 2].mean(),           # avg realized vol
        X[:, 1].mean(),           # avg 21d momentum
        off_diag.std(),           # dispersion of correlations
    ], dtype=np.float32)

    return g


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SET50Dataset(Dataset):
    """
    PyTorch Dataset wrapping pre-processed rolling windows.

    Each item is a dict with tensors:
        "X" : FloatTensor  [N, d]   node features
        "A" : FloatTensor  [N, N]   adjacency matrix
        "C" : FloatTensor  [N, N]   empirical correlation  (GAN real sample)
        "g" : FloatTensor  [4]      graph conditioning vector
        "t" : int                   window index
        "date" : str                date of window end
    """

    def __init__(
        self,
        X: np.ndarray,
        A: np.ndarray,
        C: np.ndarray,
        dates: list[str],
        indices: list[int] | None = None,
    ):
        """
        Parameters
        ----------
        X       : [n_windows, N, d]
        A       : [n_windows, N, N]
        C       : [n_windows, N, N]
        dates   : list of date strings, length n_windows
        indices : optional subset of window indices (for train/val/test split)
        """
        self.X = X
        self.A = A
        self.C = C
        self.dates = dates
        self.indices = indices if indices is not None else list(range(len(X)))

        # Pre-compute conditioning vectors for all windows
        logger.info("Pre-computing graph conditioning vectors...")
        self.G = np.stack([
            compute_graph_conditioning(C[i], X[i])
            for i in range(len(X))
        ])  # [n_windows, 4]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        return {
            "X": torch.from_numpy(self.X[i]),       # [N, d]
            "A": torch.from_numpy(self.A[i]),       # [N, N]
            "C": torch.from_numpy(self.C[i]),       # [N, N]
            "g": torch.from_numpy(self.G[i]),       # [4]
            "t": i,
            "date": self.dates[i],
        }

    @property
    def n_stocks(self) -> int:
        return self.X.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.X.shape[2]


# ---------------------------------------------------------------------------
# Train / Val / Test split (chronological — NO shuffle across splits)
# ---------------------------------------------------------------------------

def chronological_split(
    n_windows: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split window indices chronologically.
    Default: 70% train | 15% val | 15% test

    No random shuffling across splits — time-series integrity preserved.
    DataLoader can shuffle within the training split only.
    """
    n_train = int(n_windows * train_ratio)
    n_val   = int(n_windows * val_ratio)
    n_test  = n_windows - n_train - n_val

    train_idx = list(range(0, n_train))
    val_idx   = list(range(n_train, n_train + n_val))
    test_idx  = list(range(n_train + n_val, n_windows))

    logger.info(
        f"Chronological split — "
        f"train: {n_train} | val: {n_val} | test: {n_test} windows"
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Normalisation stats (fit on train, apply to all)
# ---------------------------------------------------------------------------

class FeatureNormalizer:
    """
    Z-score normalizer fitted on the training split only.
    Applied identically to val and test splits to prevent leakage.

    Normalizes node features X and graph conditioning vectors g.
    Correlation matrices C and adjacency matrices A are NOT normalized
    (they already live in [-1, 1] and [0, 1] respectively).
    """

    def __init__(self):
        self.X_mean: np.ndarray | None = None
        self.X_std:  np.ndarray | None = None
        self.g_mean: np.ndarray | None = None
        self.g_std:  np.ndarray | None = None

    def fit(self, X: np.ndarray, G: np.ndarray) -> "FeatureNormalizer":
        """
        Parameters
        ----------
        X : [n_train, N, d]
        G : [n_train, 4]
        """
        # Collapse windows and stocks → fit across all training observations
        X_flat = X.reshape(-1, X.shape[-1])           # [n_train*N, d]
        self.X_mean = X_flat.mean(axis=0)
        self.X_std  = X_flat.std(axis=0) + 1e-8       # avoid div-by-zero

        self.g_mean = G.mean(axis=0)
        self.g_std  = G.std(axis=0) + 1e-8

        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_std

    def transform_g(self, G: np.ndarray) -> np.ndarray:
        return (G - self.g_mean) / self.g_std

    def save(self, path: str) -> None:
        np.savez(
            path,
            X_mean=self.X_mean, X_std=self.X_std,
            g_mean=self.g_mean, g_std=self.g_std,
        )

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        data = np.load(path)
        norm = cls()
        norm.X_mean = data["X_mean"]
        norm.X_std  = data["X_std"]
        norm.g_mean = data["g_mean"]
        norm.g_std  = data["g_std"]
        return norm


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_datasets(
    processed_dir: str,
    config: dict,
    normalize: bool = True,
) -> tuple[SET50Dataset, SET50Dataset, SET50Dataset, FeatureNormalizer]:
    """
    Load processed windows and return train / val / test datasets.

    Parameters
    ----------
    processed_dir : path to data/processed/
    config        : full config dict
    normalize     : whether to z-score node features and conditioning vectors

    Returns
    -------
    train_ds, val_ds, test_ds : SET50Dataset
    normalizer                : FeatureNormalizer  (fitted on train)
    """
    pdir = Path(processed_dir)

    # Load arrays
    npz = np.load(pdir / "windows.npz")
    X, A, C = npz["X"], npz["A"], npz["C"]

    with open(pdir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    dates = meta["window_dates"]

    n_windows = len(X)
    logger.info(f"Loaded {n_windows} windows — X{X.shape} A{A.shape} C{C.shape}")

    # Chronological split
    train_idx, val_idx, test_idx = chronological_split(n_windows)

    # Build datasets (pre-computes G internally)
    train_ds = SET50Dataset(X, A, C, dates, train_idx)
    val_ds   = SET50Dataset(X, A, C, dates, val_idx)
    test_ds  = SET50Dataset(X, A, C, dates, test_idx)

    # Normalise using training stats only
    normalizer = FeatureNormalizer()
    if normalize:
        normalizer.fit(
            X[train_idx],
            train_ds.G[train_idx],
        )
        # Apply in-place to shared arrays
        X[:] = normalizer.transform_X(X)
        train_ds.G[:] = normalizer.transform_g(train_ds.G)

        # Save normalizer for inference
        normalizer.save(str(pdir / "normalizer.npz"))
        logger.info("Feature normalizer fitted and saved.")

    return train_ds, val_ds, test_ds, normalizer


# ---------------------------------------------------------------------------
# Convenience DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    train_ds: SET50Dataset,
    val_ds:   SET50Dataset,
    test_ds:  SET50Dataset,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns DataLoaders.
    Training loader shuffles within the split (valid for i.i.d. GAN training).
    Val/Test loaders preserve chronological order.
    """
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from data_pipeline import run_pipeline, load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Run pipeline if not already done
    processed_dir = cfg["data"]["processed_dir"]
    if not (Path(processed_dir) / "windows.npz").exists():
        run_pipeline(args.config)

    train_ds, val_ds, test_ds, norm = load_datasets(processed_dir, cfg)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=cfg["model"]["gan"]["batch_size"],
    )

    # Inspect one batch
    batch = next(iter(train_loader))
    print("\n── Batch shapes ──────────────────────────")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:6s} : {tuple(v.shape)}  dtype={v.dtype}")
    print(f"\n  Train windows : {len(train_ds)}")
    print(f"  Val   windows : {len(val_ds)}")
    print(f"  Test  windows : {len(test_ds)}")
    print(f"  N stocks      : {train_ds.n_stocks}")
    print(f"  Feature dim   : {train_ds.feature_dim}")
    print("──────────────────────────────────────────\n")
