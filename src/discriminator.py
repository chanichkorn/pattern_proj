"""
discriminator.py
================
P3 — WGAN-GP Critic for Correlation Matrix Discrimination

Takes:
    C  : correlation matrix (real or generated)   [B, N, N]
    g  : GAT graph embedding (conditioning)        [B, condition_dim]

Produces:
    score : Wasserstein distance scalar (no sigmoid)  [B, 1]

Why no sigmoid?
---------------
In WGAN the critic outputs an *unbounded* real number approximating the
Wasserstein distance between real and fake distributions. Adding a sigmoid
would constrain the output to (0,1) and lose this property. A higher score
means "more real", a lower score means "more fake" — but the absolute
values don't matter, only the difference between real and fake.

Architecture
------------
Input: [flatten(upper_triangle(C)) ‖ g]
       upper triangle has N*(N-1)//2 entries (excludes diagonal which is always 1)

SN(Linear(in → hidden))  + LayerNorm + LeakyReLU
SN(Linear(hidden → hidden)) + LayerNorm + LeakyReLU
SN(Linear(hidden → hidden//2)) + LeakyReLU
SN(Linear(hidden//2 → 1))                            ← raw score

Note: LayerNorm (not BatchNorm) is used because BatchNorm interferes with
the gradient penalty computation in WGAN-GP.

Spectral Normalization (SN) on each linear layer enforces the 1-Lipschitz
constraint directly, complementing the gradient penalty for extra stability.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: flatten upper triangle (excludes diagonal)
# ---------------------------------------------------------------------------

def flatten_upper_triangle(C: torch.Tensor) -> torch.Tensor:
    """
    Extract upper-triangular entries (excluding diagonal) from a batch
    of symmetric matrices.

    The diagonal is always 1 in a correlation matrix — no information
    content — so we drop it to reduce input dimensionality.

    Parameters
    ----------
    C : FloatTensor [B, N, N]

    Returns
    -------
    v : FloatTensor [B, N*(N-1)//2]
    """
    N = C.shape[1]
    rows, cols = torch.triu_indices(N, N, offset=1)   # exclude diagonal
    return C[:, rows, cols]                            # [B, N*(N-1)//2]


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    WGAN-GP Critic (Discriminator) with Spectral Normalization.

    Scores a correlation matrix as real or fake, conditioned on the
    GAT graph embedding g so it understands the market context.

    Spectral normalization on each Linear layer enforces the 1-Lipschitz
    constraint directly, complementing the gradient penalty for extra
    training stability.

    Parameters
    ----------
    n_stocks      : number of stocks N (determines input dim)
    condition_dim : dimension of g from GAT
    hidden_dim    : width of hidden layers
    """

    def __init__(
        self,
        n_stocks: int,
        condition_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.n_stocks      = n_stocks
        self.condition_dim = condition_dim

        # Upper triangle entries (no diagonal): N*(N-1)//2
        tri_dim  = n_stocks * (n_stocks - 1) // 2
        in_dim   = tri_dim + condition_dim

        sn = nn.utils.spectral_norm   # shorthand

        self.net = nn.Sequential(
            # Block 1
            sn(nn.Linear(in_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            # Block 2
            sn(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            # Block 3 — bottleneck
            sn(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),

            # Output — unbounded scalar (no activation)
            sn(nn.Linear(hidden_dim // 2, 1)),
        )

        self._init_weights()
        logger.info(
            f"Critic — N={n_stocks}  tri_dim={tri_dim}  "
            f"cond={condition_dim}  hidden={hidden_dim}  "
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
        C: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        C : FloatTensor [B, N, N]   correlation matrix (real or fake)
        g : FloatTensor [B, condition_dim]

        Returns
        -------
        score : FloatTensor [B, 1]   Wasserstein score (unbounded)
        """
        v     = flatten_upper_triangle(C)      # [B, N*(N-1)//2]
        x     = torch.cat([v, g], dim=1)       # [B, tri_dim + cond_dim]
        score = self.net(x)                    # [B, 1]
        return score

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: dict, n_stocks: int) -> "Critic":
        gcfg = cfg["model"]["gan"]
        return cls(
            n_stocks      = n_stocks,
            condition_dim = cfg["model"]["gat"]["embedding_dim"],
            hidden_dim    = gcfg["hidden_dim"],
        )


# ---------------------------------------------------------------------------
# Gradient Penalty (WGAN-GP)
# ---------------------------------------------------------------------------

def gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    g: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the WGAN-GP gradient penalty.

    Interpolates between real and fake matrices, computes the critic score
    on the interpolation, and penalises deviations of the gradient norm
    from 1 (enforcing the 1-Lipschitz constraint).

    GP = E[ (‖∇_x̂ D(x̂)‖₂ - 1)² ]
    where x̂ = ε·real + (1-ε)·fake,  ε ~ Uniform(0,1)

    Parameters
    ----------
    critic : Critic module
    real   : FloatTensor [B, N, N]   real correlation matrices
    fake   : FloatTensor [B, N, N]   generated correlation matrices
    g      : FloatTensor [B, cond]   conditioning (same for both)
    device : torch.device

    Returns
    -------
    gp : FloatTensor scalar   gradient penalty term
    """
    B, N, _ = real.shape

    # Random interpolation coefficient ε per sample
    eps = torch.rand(B, 1, 1, device=device)        # [B, 1, 1]
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)

    # Critic score on interpolated samples
    score = critic(interpolated, g)                  # [B, 1]

    # Gradients w.r.t. the interpolated input
    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]                                             # [B, N, N]

    # Flatten and compute L2 norm per sample
    gradients = gradients.reshape(B, -1)             # [B, N*N]
    grad_norm = gradients.norm(2, dim=1)             # [B]

    gp = ((grad_norm - 1) ** 2).mean()
    return gp


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from generator import Generator, cholesky_to_correlation

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}\n")

    N, B = 40, 16
    G = Generator.from_config(cfg, n_stocks=N).to(device)
    D = Critic.from_config(cfg, n_stocks=N).to(device)

    # Mock inputs
    noise  = torch.randn(B, cfg["model"]["gan"]["noise_dim"]).to(device)
    g      = torch.randn(B, cfg["model"]["gat"]["embedding_dim"]).to(device)
    C_real = torch.randn(B, N, N).to(device)
    # Make C_real a valid-ish correlation matrix
    C_real = (C_real + C_real.transpose(1, 2)) / 2
    eye    = torch.eye(N, device=device).unsqueeze(0)
    C_real = C_real * (1 - eye) * 0.3 + eye

    # Generated
    C_fake, _ = G(noise, g)

    print("── Critic scores ─────────────────────────")
    score_real = D(C_real, g)
    score_fake = D(C_fake.detach(), g)
    print(f"  Score real  : mean={score_real.mean().item():.4f}")
    print(f"  Score fake  : mean={score_fake.mean().item():.4f}")
    print(f"  Shape       : {tuple(score_real.shape)}")

    # WGAN-GP losses
    loss_D = score_fake.mean() - score_real.mean()
    gp     = gradient_penalty(D, C_real, C_fake.detach(), g, device)
    total  = loss_D + cfg["model"]["gan"]["gp_lambda"] * gp

    print(f"\n── WGAN-GP losses ────────────────────────")
    print(f"  Wasserstein loss : {loss_D.item():.4f}")
    print(f"  Gradient penalty : {gp.item():.4f}")
    print(f"  Total D loss     : {total.item():.4f}")

    # Generator loss
    C_fake2, _ = G(noise, g)
    loss_G = -D(C_fake2, g).mean()
    print(f"  Generator loss   : {loss_G.item():.4f}")

    # Gradient flow check
    total.backward()
    D_grad_ok = all(p.grad is not None for p in D.parameters())
    print(f"\n  Critic grads flow    : {D_grad_ok}")

    loss_G.backward()
    G_grad_ok = all(p.grad is not None for p in G.parameters())
    print(f"  Generator grads flow : {G_grad_ok}")

    print("\n✅  Critic + gradient penalty smoke test passed.")

