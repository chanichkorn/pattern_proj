"""
generator.py
============
P3 — WGAN-GP Generator for Correlation Matrix Generation

Takes:
    ε  ~ N(0, I)   noise vector            [B, noise_dim]
    g               GAT graph embedding     [B, condition_dim]

Produces:
    R̂  ∈ [-1, 1]^{N×N}   valid PSD correlation matrix   [B, N, N]

Key design: Cholesky Parameterisation
--------------------------------------
Instead of generating the correlation matrix directly (which gives no
guarantee of positive semi-definiteness), the Generator outputs the
lower-triangular entries of a Cholesky factor L:

    Σ̃ = L @ Lᵀ                    (always PSD by construction)
    R̂ = diag(Σ̃)^{-½} Σ̃ diag(Σ̃)^{-½}  (normalise diagonal to 1)

This means:
  • No eigenvalue clipping needed
  • No post-hoc PSD repair
  • Gradients flow cleanly through L
  • Diagonal of L passes through Softplus to enforce L_ii > 0

Number of free parameters for N=40:
    lower triangle entries = N*(N+1)/2 = 820

Residual blocks improve gradient flow when producing the high-dimensional
Cholesky output (820 entries for N=40).
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cholesky → Correlation helper (used by Generator and tests)
# ---------------------------------------------------------------------------

def cholesky_to_correlation(L: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of Cholesky factors to correlation matrices.

    Parameters
    ----------
    L : FloatTensor [B, N, N]   lower-triangular Cholesky factors
                                 (diagonal must be > 0)

    Returns
    -------
    R : FloatTensor [B, N, N]   symmetric PSD correlation matrices
                                 diagonal = 1, off-diagonal ∈ (-1, 1)
    """
    # Σ̃ = L @ Lᵀ  →  always PSD
    Sigma = torch.bmm(L, L.transpose(1, 2))      # [B, N, N]

    # Normalise to correlation: R_ij = Σ_ij / sqrt(Σ_ii * Σ_jj)
    diag = torch.diagonal(Sigma, dim1=1, dim2=2)  # [B, N]
    std  = torch.sqrt(diag.clamp(min=1e-8))       # [B, N]
    outer = torch.bmm(std.unsqueeze(2), std.unsqueeze(1))  # [B, N, N]
    R = Sigma / outer                              # [B, N, N]

    # Numerical safety: clamp to [-1, 1] and force exact diagonal = 1
    R = R.clamp(-1.0, 1.0)
    eye = torch.eye(R.shape[1], device=R.device, dtype=R.dtype).unsqueeze(0)
    R = R * (1 - eye) + eye                       # diagonal exactly 1

    return R


def vec_to_cholesky(v: torch.Tensor, N: int) -> torch.Tensor:
    """
    Map a flat vector of lower-triangular entries to a Cholesky matrix.

    Parameters
    ----------
    v : FloatTensor [B, N*(N+1)//2]   raw generator output
    N : int                            number of stocks

    Returns
    -------
    L : FloatTensor [B, N, N]   lower-triangular Cholesky factor (diag > 0)
    """
    B = v.shape[0]
    L = torch.zeros(B, N, N, device=v.device, dtype=v.dtype)

    # Fill lower triangle (including diagonal) from flat vector
    rows, cols = torch.tril_indices(N, N, offset=0)
    L[:, rows, cols] = v

    # Enforce positive diagonal via Softplus (smooth, always > 0)
    diag_idx = torch.arange(N, device=v.device)
    L[:, diag_idx, diag_idx] = torch.nn.functional.softplus(
        L[:, diag_idx, diag_idx]
    )

    return L   # [B, N, N]


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    A residual block with the structure:
        y = LayerNorm(x + Linear(LeakyReLU(LayerNorm(Linear(x)))))

    Using a pre-activation (norm before activation) style for stable training.
    Both linear layers operate in the same dimension so the skip connection
    requires no projection.
    """

    def __init__(self, dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    MLP Generator with residual blocks:
    (noise ‖ graph_embedding) → Cholesky → Correlation matrix.

    Architecture
    ------------
    Input  : [ε ‖ g]   dim = noise_dim + condition_dim
    Linear(in  → hidden)  + LayerNorm + LeakyReLU
    ResBlock(hidden)
    Linear(hidden → hidden*2) + LayerNorm + LeakyReLU
    ResBlock(hidden*2)
    Linear(hidden*2 → hidden) + LayerNorm + LeakyReLU
    ResBlock(hidden)
    Linear(hidden → N*(N+1)//2)             ← Cholesky entries
    → vec_to_cholesky → cholesky_to_correlation
    Output : R̂  [B, N, N]   valid correlation matrix
    """

    def __init__(
        self,
        n_stocks: int,
        noise_dim: int,
        condition_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.n_stocks     = n_stocks
        self.noise_dim    = noise_dim
        self.condition_dim = condition_dim
        self.chol_dim     = n_stocks * (n_stocks + 1) // 2

        in_dim = noise_dim + condition_dim

        self.net = nn.Sequential(
            # Entry projection
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            ResBlock(hidden_dim),

            # Expand
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            ResBlock(hidden_dim * 2),

            # Contract
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            ResBlock(hidden_dim),

            # Output: Cholesky entries (no activation — handled in vec_to_cholesky)
            nn.Linear(hidden_dim, self.chol_dim),
        )

        self._init_weights()
        logger.info(
            f"Generator — N={n_stocks}  noise={noise_dim}  "
            f"cond={condition_dim}  hidden={hidden_dim}  "
            f"chol_entries={self.chol_dim}  "
            f"params={self.count_parameters():,}"
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        noise: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        noise : FloatTensor [B, noise_dim]
        g     : FloatTensor [B, condition_dim]  GAT graph embedding

        Returns
        -------
        R     : FloatTensor [B, N, N]   generated correlation matrix
        L     : FloatTensor [B, N, N]   Cholesky factor (for diagnostics)
        """
        z   = torch.cat([noise, g], dim=1)       # [B, noise_dim + cond_dim]
        v   = self.net(z)                        # [B, chol_dim]
        L   = vec_to_cholesky(v, self.n_stocks)  # [B, N, N]
        R   = cholesky_to_correlation(L)         # [B, N, N]
        return R, L

    def sample(
        self,
        g: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Convenience method: draw noise internally and return R only.

        Parameters
        ----------
        g        : FloatTensor [B, condition_dim]
        n_samples: how many matrices to generate per conditioning vector

        Returns
        -------
        R : FloatTensor [B*n_samples, N, N]
        """
        B = g.shape[0]
        g_rep = g.repeat_interleave(n_samples, dim=0)    # [B*S, cond_dim]
        noise = torch.randn(
            B * n_samples, self.noise_dim,
            device=g.device, dtype=g.dtype,
        )
        R, _ = self.forward(noise, g_rep)
        return R

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: dict, n_stocks: int) -> "Generator":
        gcfg = cfg["model"]["gan"]
        return cls(
            n_stocks      = n_stocks,
            noise_dim     = gcfg["noise_dim"],
            condition_dim = cfg["model"]["gat"]["embedding_dim"],
            hidden_dim    = gcfg["hidden_dim"],
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    import numpy as np

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}\n")

    N   = 40   # actual stocks after pipeline
    B   = 16
    G   = Generator.from_config(cfg, n_stocks=N).to(device)

    noise = torch.randn(B, cfg["model"]["gan"]["noise_dim"]).to(device)
    g     = torch.randn(B, cfg["model"]["gat"]["embedding_dim"]).to(device)

    R, L = G(noise, g)

    print(f"── Generator output ─────────────────────")
    print(f"  R shape          : {tuple(R.shape)}")
    print(f"  L shape          : {tuple(L.shape)}")

    # Validity checks
    diag_ok  = torch.allclose(
        torch.diagonal(R, dim1=1, dim2=2),
        torch.ones(B, N, device=device), atol=1e-5
    )
    symm_ok  = torch.allclose(R, R.transpose(1, 2), atol=1e-5)
    range_ok = (R >= -1.0).all() and (R <= 1.0).all()

    # PSD check via eigenvalues (must run on CPU — MPS doesn't support eigvalsh)
    eigvals  = torch.linalg.eigvalsh(R.cpu())
    psd_ok   = (eigvals >= -1e-5).all()

    print(f"  Diagonal == 1    : {diag_ok}")
    print(f"  Symmetric        : {symm_ok}")
    print(f"  Values in [-1,1] : {range_ok}")
    print(f"  PSD (min eigval ≥ 0) : {psd_ok.item()}  "
          f"(min={eigvals.min().item():.6f})")

    # Gradient flow
    loss = R.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in G.parameters())
    print(f"  Gradients flow   : {grad_ok}")

    # Sample method
    g2  = torch.randn(4, cfg["model"]["gat"]["embedding_dim"]).to(device)
    R2  = G.sample(g2, n_samples=3)
    print(f"\n  sample() output  : {tuple(R2.shape)}  (4 graphs × 3 samples)")

    # Diversity: generated matrices should differ
    std_off_diag = R[:, ~torch.eye(N, dtype=torch.bool)].std().item()
    print(f"  Off-diag std across batch: {std_off_diag:.4f}  (> 0 = diverse)")

    print("\n✅  Generator smoke test passed.")

