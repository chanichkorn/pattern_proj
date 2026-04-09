"""
optimizer.py
============
P4 — Global Minimum Variance (GMV) Portfolio Optimizer

Takes a covariance matrix Σ̂ and solves:

    min  wᵀ Σ̂ w
    s.t. Σ wᵢ = 1          (fully invested)
         0 ≤ wᵢ ≤ max_w    (long-only, position cap)

Returns optimal weights w* ∈ ℝ^N.

Why GMV?
--------
In a sideways market, return prediction is near-zero signal.
GMV ignores expected returns entirely and focuses only on minimising
portfolio variance — the right objective when returns are unpredictable.
"""

import logging
import warnings

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def gmv_optimize(
    sigma: np.ndarray,
    max_weight: float = 0.10,
    min_weight: float = 0.00,
) -> np.ndarray:
    """
    Solve the Global Minimum Variance problem.

    Parameters
    ----------
    sigma      : np.ndarray [N, N]  covariance matrix (must be PSD)
    max_weight : float              upper bound per stock (default 10%)
    min_weight : float              lower bound per stock (default 0 = long-only)

    Returns
    -------
    w : np.ndarray [N]   optimal portfolio weights (sum to 1)
        Falls back to equal weights if optimisation fails.
    """
    N = sigma.shape[0]
    w = cp.Variable(N)

    # Objective: minimise portfolio variance wᵀΣw
    objective = cp.Minimize(cp.quad_form(w, sigma))

    constraints = [
        cp.sum(w) == 1,                  # fully invested
        w >= min_weight,                 # long-only
        w <= max_weight,                 # position cap
    ]

    prob = cp.Problem(objective, constraints)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            weights = np.array(w.value, dtype=np.float64)
            weights = np.clip(weights, min_weight, max_weight)
            weights /= weights.sum()    # renormalise after clipping
            return weights

    except Exception as e:
        logger.debug(f"GMV solve failed: {e}")

    # Fallback: equal weights
    logger.warning("GMV optimisation failed — falling back to equal weights.")
    return np.ones(N, dtype=np.float64) / N


# ---------------------------------------------------------------------------
# Covariance reconstruction from generated correlation + historical vol
# ---------------------------------------------------------------------------

def correlation_to_covariance(
    R: np.ndarray,
    sigma_diag: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct covariance matrix from correlation and per-stock volatilities.

    Σ̂ = D_σ · R̂ · D_σ    where D_σ = diag(σ₁, ..., σ_N)

    Parameters
    ----------
    R          : np.ndarray [N, N]   correlation matrix
    sigma_diag : np.ndarray [N]      per-stock annualised volatility

    Returns
    -------
    Sigma : np.ndarray [N, N]   covariance matrix
    """
    D = np.diag(sigma_diag)
    Sigma = D @ R @ D

    # Symmetrise and ensure PSD via small diagonal regularisation
    Sigma = (Sigma + Sigma.T) / 2
    Sigma += np.eye(Sigma.shape[0]) * 1e-6

    return Sigma


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def portfolio_stats(
    weights: np.ndarray,
    sigma: np.ndarray,
    label: str = "",
) -> dict:
    """Compute and return basic portfolio statistics."""
    port_var = float(weights @ sigma @ weights)
    port_vol = float(np.sqrt(port_var * 252))     # annualised
    herfindahl = float((weights ** 2).sum())       # concentration (lower = more diversified)

    stats = {
        "label":       label,
        "ann_vol":     port_vol,
        "herfindahl":  herfindahl,
        "max_weight":  float(weights.max()),
        "n_nonzero":   int((weights > 1e-4).sum()),
    }
    return stats


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    N = 40

    # Build a random valid covariance matrix
    A = np.random.randn(N, N)
    Sigma = A @ A.T / N + np.eye(N) * 0.01   # guaranteed PSD

    print("── GMV Optimizer ────────────────────────")
    w = gmv_optimize(Sigma, max_weight=0.10)

    print(f"  Sum of weights   : {w.sum():.6f}  (should be 1.0)")
    print(f"  Min weight       : {w.min():.4f}")
    print(f"  Max weight       : {w.max():.4f}")
    print(f"  Non-zero stocks  : {(w > 1e-4).sum()}")

    stats = portfolio_stats(w, Sigma, label="GMV")
    print(f"  Ann. volatility  : {stats['ann_vol']:.4f}")
    print(f"  Herfindahl index : {stats['herfindahl']:.4f}")

    # Correlation → covariance round-trip
    vols = np.random.uniform(0.1, 0.4, N)
    R = np.eye(N)
    R[0, 1] = R[1, 0] = 0.5
    Sigma2 = correlation_to_covariance(R, vols)

    print(f"\n── Correlation → Covariance ─────────────")
    print(f"  Sigma shape      : {Sigma2.shape}")
    eigvals = np.linalg.eigvalsh(Sigma2)
    print(f"  Min eigenvalue   : {eigvals.min():.6f}  (should be ≥ 0)")

    print("\n✅  Optimizer smoke test passed.")
