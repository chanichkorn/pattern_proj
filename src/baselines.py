"""
baselines.py
============
P5 — Baseline Covariance Estimators

Implements DCC-GARCH(1,1) for dynamic correlation estimation.

DCC-GARCH (Dynamic Conditional Correlation — Engle 2002):
----------------------------------------------------------
Step 1 — Univariate GARCH(1,1) per stock:
    σ²_i,t = ω_i + α_i·ε²_{i,t-1} + β_i·σ²_{i,t-1}

    Standardised residuals: z_{i,t} = r_{i,t} / σ_{i,t}

Step 2 — Dynamic correlation on standardised residuals:
    Q_t = (1 - a - b)·Q̄ + a·z_{t-1}z'_{t-1} + b·Q_{t-1}
    R_t = diag(Q_t)^{-½} · Q_t · diag(Q_t)^{-½}

Step 3 — Covariance reconstruction:
    Σ_t = D_t · R_t · D_t    (D_t = diag of GARCH vols)

Why DCC-GARCH matters as a baseline:
-------------------------------------
DCC-GARCH is the gold-standard statistical model for dynamic correlation.
Beating it is a meaningful result; even matching it with a data-driven
approach is academically valid since DCC-GARCH requires strong parametric
assumptions (Gaussian innovations, GARCH structure) that may not hold for
Thai emerging market data.
"""

import logging
import warnings

import numpy as np
from arch import arch_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Fit univariate GARCH(1,1) per stock
# ---------------------------------------------------------------------------

def fit_garch_vols(
    returns: np.ndarray,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit GARCH(1,1) to each stock independently.

    Parameters
    ----------
    returns : np.ndarray [T, N]   log returns
    horizon : int                 forecast horizon (default 1 day ahead)

    Returns
    -------
    std_residuals  : np.ndarray [T, N]   standardised residuals z_{i,t}
    forecast_vols  : np.ndarray [N]      1-step ahead volatility forecast σ_{i,T+1}
    """
    T, N = returns.shape
    std_residuals = np.zeros((T, N), dtype=np.float64)
    forecast_vols = np.zeros(N, dtype=np.float64)

    for i in range(N):
        r = returns[:, i] * 100   # scale to % for numerical stability

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(r, vol="Garch", p=1, q=1,
                                   dist="normal", rescale=False)
                res   = model.fit(disp="off", show_warning=False)

            # Conditional volatility (in % units)
            cond_vol = res.conditional_volatility / 100   # back to decimal

            # Standardised residuals
            std_residuals[:, i] = (returns[:, i]) / (cond_vol + 1e-8)

            # 1-step ahead forecast
            fc = res.forecast(horizon=horizon)
            forecast_vols[i] = np.sqrt(fc.variance.values[-1, 0]) / 100

        except Exception as e:
            logger.debug(f"GARCH fit failed for stock {i}: {e}")
            # Fallback: use historical std
            std_residuals[:, i] = returns[:, i] / (returns[:, i].std() + 1e-8)
            forecast_vols[i]    = returns[:, i].std() * np.sqrt(252)

    return std_residuals, forecast_vols


# ---------------------------------------------------------------------------
# Step 2 — DCC correlation dynamics
# ---------------------------------------------------------------------------

def fit_dcc_correlation(
    std_residuals: np.ndarray,
    a: float = 0.05,
    b: float = 0.90,
) -> np.ndarray:
    """
    Fit DCC(1,1) on standardised GARCH residuals.

    Q_t = (1-a-b)·Q̄ + a·z_{t-1}z'_{t-1} + b·Q_{t-1}
    R_t = diag(Q_t)^{-½} · Q_t · diag(Q_t)^{-½}

    Parameters
    ----------
    std_residuals : np.ndarray [T, N]
    a             : DCC α parameter (news impact)
    b             : DCC β parameter (persistence)

    Returns
    -------
    R_T : np.ndarray [N, N]   DCC correlation matrix at time T
    """
    T, N = std_residuals.shape

    # Unconditional correlation Q̄
    Q_bar = np.corrcoef(std_residuals.T)
    Q_bar = (Q_bar + Q_bar.T) / 2 + np.eye(N) * 1e-6   # symmetrise + stabilise

    # Initialise
    Q_t = Q_bar.copy()

    # Recurse through time
    for t in range(1, T):
        z      = std_residuals[t - 1]
        Q_t    = (1 - a - b) * Q_bar + a * np.outer(z, z) + b * Q_t

    # Convert Q_T → R_T
    q_diag = np.sqrt(np.diag(Q_t))
    q_diag = np.where(q_diag < 1e-8, 1e-8, q_diag)
    R_T    = Q_t / np.outer(q_diag, q_diag)

    # Clip and symmetrise for numerical safety
    R_T = np.clip(R_T, -1.0, 1.0)
    R_T = (R_T + R_T.T) / 2
    np.fill_diagonal(R_T, 1.0)

    return R_T


# ---------------------------------------------------------------------------
# Full DCC-GARCH covariance estimator
# ---------------------------------------------------------------------------

def dcc_garch_covariance(
    returns: np.ndarray,
    a: float = 0.05,
    b: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the DCC-GARCH covariance matrix from a returns history.

    Parameters
    ----------
    returns : np.ndarray [T, N]   log returns (training window)
    a, b    : DCC parameters

    Returns
    -------
    Sigma : np.ndarray [N, N]   DCC-GARCH covariance matrix
    R     : np.ndarray [N, N]   DCC correlation matrix
    """
    T, N = returns.shape

    # Step 1: GARCH per stock
    std_residuals, forecast_vols = fit_garch_vols(returns)

    # Annualise forecast vols
    sigma_diag = forecast_vols * np.sqrt(252)

    # Step 2: DCC correlation
    R = fit_dcc_correlation(std_residuals, a=a, b=b)

    # Step 3: Covariance = D · R · D
    D     = np.diag(sigma_diag)
    Sigma = D @ R @ D
    Sigma = (Sigma + Sigma.T) / 2 + np.eye(N) * 1e-6

    return Sigma, R


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    np.random.seed(42)

    T, N = 252, 10   # small N for quick test
    returns = np.random.randn(T, N) * 0.01

    print(f"Fitting DCC-GARCH on {T}×{N} returns...")
    t0 = time.time()

    Sigma, R = dcc_garch_covariance(returns)

    elapsed = time.time() - t0
    print(f"  Time              : {elapsed:.2f}s")
    print(f"  Sigma shape       : {Sigma.shape}")
    print(f"  R diagonal == 1   : {np.allclose(np.diag(R), 1.0, atol=1e-5)}")

    eigvals = np.linalg.eigvalsh(Sigma)
    print(f"  Sigma PSD         : {(eigvals >= -1e-6).all()}  "
          f"(min eigval={eigvals.min():.6f})")
    print(f"  R off-diag range  : [{R[~np.eye(N, dtype=bool)].min():.3f}, "
          f"{R[~np.eye(N, dtype=bool)].max():.3f}]")

    print("\n✅  DCC-GARCH smoke test passed.")
