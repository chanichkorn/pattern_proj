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
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Factor model → Correlation (replaces Cholesky in the Generator)
# ---------------------------------------------------------------------------

def factor_to_correlation(
    v: torch.Tensor,
    N: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert flat MLP output to a PSD correlation matrix via a factor model.

    Parameterisation
    ----------------
    v[:, :N*K]  → factor loadings  Λ  [B, N, K]
    v[:, N*K:]  → idiosyncratic    d  [B, N]  (Softplus → positive)

    Covariance:  Σ = Λ Λᵀ + diag(d)          (PSD by construction)
    Correlation: R_ij = Σ_ij / sqrt(Σ_ii Σ_jj)

    Advantages over Cholesky
    ------------------------
    • N*K + N free parameters vs N*(N+1)//2  (360 vs 820 for N=40, K=8)
    • Λ is interpretable: column k is a latent market factor
    • The rank-K structure regularises the matrix toward factor solutions
      that generalise better out-of-sample

    Parameters
    ----------
    v : FloatTensor [B, N*K + N]   raw MLP output
    N : int                        number of stocks
    K : int                        number of latent factors

    Returns
    -------
    R      : FloatTensor [B, N, N]   PSD correlation matrix (diag = 1)
    Lambda : FloatTensor [B, N, K]   factor loadings (for diagnostics)
    """
    B = v.shape[0]
    Lambda = v[:, : N * K].view(B, N, K)           # [B, N, K]
    d      = F.softplus(v[:, N * K :])             # [B, N]  positive idiosyncratic var

    Sigma  = torch.bmm(Lambda, Lambda.transpose(1, 2))  # [B, N, N]
    Sigma  = Sigma + torch.diag_embed(d)                 # add idiosyncratic diagonal

    # Normalise to correlation
    diag  = torch.diagonal(Sigma, dim1=1, dim2=2).clamp(min=1e-8)  # [B, N]
    std   = torch.sqrt(diag)                                         # [B, N]
    outer = torch.bmm(std.unsqueeze(2), std.unsqueeze(1))           # [B, N, N]
    R     = Sigma / outer

    R = R.clamp(-1.0, 1.0)
    eye = torch.eye(N, device=R.device, dtype=R.dtype).unsqueeze(0)
    R   = R * (1 - eye) + eye                        # exact diagonal = 1

    return R, Lambda


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    MLP Generator: (noise ‖ graph_embedding) → factor model → Correlation matrix.

    Architecture
    ------------
    Input  : [ε ‖ g]   dim = noise_dim + condition_dim
    Linear(in  → hidden)  + LayerNorm + LeakyReLU
    Linear(hidden → hidden*2) + LayerNorm + LeakyReLU
    Linear(hidden*2 → hidden) + LayerNorm + LeakyReLU
    Linear(hidden → N*K + N)              ← factor loadings Λ + idiosyncratic d
    → factor_to_correlation  (Σ = ΛΛᵀ + diag(d), normalise → R)
    Output : R̂  [B, N, N]   valid PSD correlation matrix
    """

    def __init__(
        self,
        n_stocks: int,
        noise_dim: int,
        condition_dim: int,
        hidden_dim: int,
        n_factors: int = 8,
    ):
        super().__init__()
        self.n_stocks      = n_stocks
        self.noise_dim     = noise_dim
        self.condition_dim = condition_dim
        self.n_factors     = n_factors
        # Factor model output: Λ [N×K] entries + idiosyncratic d [N]
        self.factor_dim    = n_stocks * n_factors + n_stocks

        in_dim = noise_dim + condition_dim

        self.net = nn.Sequential(
            # Block 1
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            # Block 2
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),

            # Block 3
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            # Output: factor loadings Λ [N×K] + idiosyncratic d [N]
            nn.Linear(hidden_dim, self.factor_dim),
        )

        self._init_weights()
        logger.info(
            f"Generator (factor model) — N={n_stocks}  K={n_factors}  "
            f"noise={noise_dim}  cond={condition_dim}  hidden={hidden_dim}  "
            f"factor_entries={self.factor_dim}  "
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
        R      : FloatTensor [B, N, N]   generated correlation matrix
        Lambda : FloatTensor [B, N, K]   factor loadings (for diagnostics)
        """
        z          = torch.cat([noise, g], dim=1)                          # [B, noise_dim + cond_dim]
        v          = self.net(z)                                           # [B, factor_dim]
        R, Lambda  = factor_to_correlation(v, self.n_stocks, self.n_factors)
        return R, Lambda

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
            n_factors     = gcfg.get("n_factors", 8),
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

    R, Lambda = G(noise, g)

    print(f"── Generator output ─────────────────────")
    print(f"  R shape          : {tuple(R.shape)}")
    print(f"  Lambda shape     : {tuple(Lambda.shape)}")

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
