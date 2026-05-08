# SET50 Portfolio Optimization — Change Log

## Baseline (v1) — Original Code

**Architecture:** GATv1 · Cholesky Generator · val_W model selection

**Known bug:** `best.pt` saved only at epoch 1. val_W (Wasserstein from a simultaneously-training Critic) flipped negative immediately — the Generator overpowered an untrained Critic and the model selection metric never improved again. All 300 training epochs were wasted.

**Backtest result (Apr–Dec 2019, 10 samples):**
```
Our Model (GAT+GAN)      -14.62%   10.75%   Sharpe -1.359   MaxDD -14.62%
Equal Weight (1/N)        -8.40%   12.55%   Sharpe -0.670   MaxDD -12.18%
Sample Cov GMV           -10.27%    9.64%   Sharpe -1.065   MaxDD -11.85%
```

---

## v2 — Architecture + Training Fixes

### P1 · GATv2 Attention [`src/gat.py`]

**What changed:** `GATLayer` upgraded from GATv1 to GATv2.

| | GATv1 | GATv2 |
|---|---|---|
| Attention score | `e = LeakyReLU(aᵀ [Wh_i ‖ Wh_j ‖ A_ij])` | `e = aᵀ LeakyReLU(W_pair [h_i ‖ h_j ‖ A_ij])` |
| Expressiveness | Decomposable — score for i→j separates into `f(i) + g(j)` | Non-decomposable — nonlinearity mixes i and j before dot product |
| Failure mode | Rank collapse in deep graphs | Fixed |

**Files changed:** `src/gat.py`
- Added `self.W_pair = nn.Parameter(torch.empty(n_heads, attn_dim, out_features))`
- Shrank `self.attn` from `[H, attn_dim]` → `[H, out_features]`
- Forward: `transformed = einsum('bnmhd,hdo->bnmho', pair, W_pair)` then `LeakyReLU` then dot with `attn`
- GAT params: 78,912 → 79,424 (+512 from W_pair)

---

### P3 · Factor Model Generator [`src/generator.py`]

**What changed:** Generator output parameterisation replaced.

| | Cholesky (v1) | Factor Model (v2) |
|---|---|---|
| Output entries | `N*(N+1)//2 = 820` Cholesky entries | `N*K + N = 360` (K=8 factors + idiosyncratic) |
| PSD guarantee | Softplus diagonal of L, then `LLᵀ` | `ΛΛᵀ + diag(d)`, always PSD |
| Interpretability | None | Column k of Λ = latent market factor |
| Regularisation | None | Rank-K structure regularises toward factor solutions |

**Files changed:** `src/generator.py`
- Added `factor_to_correlation(v, N, K)` function
- `Generator.__init__`: `chol_dim` → `factor_dim = N*K + N`
- `Generator.forward`: returns `(R, Lambda)` instead of `(R, L)`
- `Generator.from_config`: reads `n_factors` from config

**Config changed:** `configs/config.yaml`
```yaml
model:
  gan:
    n_factors: 8    # added
```

---

### Training Fix 1 · Critic Warm-up [`src/train.py`]

**Problem:** Generator + GAT updated from epoch 1 against a random Critic. First gradients were garbage — those bad updates baked into GAT weights before Critic learned anything.

**Fix:** Critic trains alone for 30 epochs before Generator and GAT touch their weights.

**Implementation:**
- Added `generator_step: bool = True` parameter to `train_epoch()`
- Generator+GAT block wrapped in `if generator_step:`
- `train()` loops `train_epoch(..., generator_step=False)` for `warmup_epochs` before the main loop

**Config added:**
```yaml
model:
  gan:
    warmup_epochs: 30
```

---

### Training Fix 2 · Frobenius Auxiliary Loss [`src/train.py`]

**Problem:** Generator loss `L_G = -D(C_fake)` depends entirely on the Critic, which is simultaneously unstable. No stable learning signal.

**Fix:** Add per-element MSE between generated and real correlation matrices:
```
loss_G = -D(C_fake) + λ * MSE(C_fake, C_real)
```

MSE is Critic-independent and stable throughout training. Acts as a soft constraint pulling generated matrices toward the real distribution even when the adversarial game is imbalanced.

**Config added:**
```yaml
model:
  gan:
    lambda_frob: 0.1
```

---

### Training Fix 3 · Model Selection [`src/train.py`]

**Problem:** `best.pt` saved when `val_wasserstein` improves — circular, because the Critic measuring it is simultaneously training. At epoch 1 the score was 0.41; after that the Generator won the game and val_W flipped negative. Model selection froze at epoch 1.

**Fix:** Switch to `val_frobenius` — MSE between generated and real correlation matrices on the val set. No Critic involved. Monotonically tracks actual generator quality.

```python
# OLD — maximise (circular)
if val_w > best_val_w: save best.pt

# NEW — minimise (stable)
if val_frob < best_val_frob: save best.pt
```

**v2 training outcome:** `best.pt` saved many times throughout training. Best val_frob = **0.0763** at epoch ~90.

**Backtest result (Apr–Dec 2019, 50 samples — honest estimate):**
```
Our Model (GAT+GAN)      -10.53%    9.92%   Sharpe -1.062   MaxDD -13.26%
Equal Weight (1/N)        -8.40%   12.55%   Sharpe -0.670   MaxDD -12.18%
Sample Cov GMV           -10.27%    9.64%   Sharpe -1.065   MaxDD -11.85%
```

*Note: A single 10-sample run produced -7.47% / -0.751 Sharpe. This was a lucky random draw from the GAN. At 50 samples the estimate stabilises to -1.062.*

**What improved vs v1:** Sharpe −1.359 → −1.062. Model now beats Sample Cov GMV. training stable.
**Still trailing:** Equal Weight by ~0.4 Sharpe.

---

## v3 — P4 Multi-Scale Features

### 63-Day Regime Features [`src/data_pipeline.py`]

**What changed:** Two new numeric node features added to give GAT a slow-regime signal.

| Index | Feature | Window |
|---|---|---|
| 0 | Cumulative return | 5d |
| 1 | Cumulative return | 21d |
| 2 | Realized volatility | 21d |
| 3 | Volume z-score | — (placeholder 0) |
| **4** | **Cumulative return** | **63d (new)** |
| **5** | **Realized volatility** | **63d (new)** |
| 6–16 | Sector one-hot | 11 categories |

Feature dimension: 15 → **17**

**Rationale:** In a declining market, 63d correlations tighten before 21d windows detect the shift. A richer graph embedding `g` means the Generator gets better regime conditioning and produces more consistent, less noise-sensitive outputs.

**Files changed:** `src/data_pipeline.py`
- `build_node_features`: added `long_window=63` parameter, `n_numeric 4→6`, new X[:,4] and X[:,5]
- `run_pipeline`: reads `long_window` from config, updates `feature_dim = 6 + len(SECTOR_LIST)`

**Config changed:** `configs/config.yaml`
```yaml
data:
  long_window: 63       # added

model:
  gat:
    node_feature_dim: 17  # was 15
```

**v3 training outcome:** 300 epochs completed. Best val_frob = **0.0695** (better than v2's 0.0763).

**Backtest result (Apr–Dec 2019, 50 samples):**
```
Our Model (GAT+GAN)      -13.48%   10.85%   Sharpe -1.243   MaxDD -14.27%
Equal Weight (1/N)        -8.40%   12.55%   Sharpe -0.670   MaxDD -12.18%
Sample Cov GMV           -10.27%    9.64%   Sharpe -1.065   MaxDD -11.85%
```

**Outcome:** val_frob improved but backtest regressed vs v2. The 63d features helped the generator reconstruct training-period (2015–2018) correlation matrices more accurately, but the test period (Apr–Dec 2019 bear market) has different dynamics. The model learned bear-market correlation patterns from training that do not match 2019's specific decline.

---

## Summary of Results

| Version | Changes | val_frob | Sharpe | vs Equal Weight |
|---|---|---|---|---|
| v1 | Original | broken (epoch 1 bug) | −1.359 | −0.689 |
| v2 | GATv2 + Factor model + training fixes | 0.0763 | −1.062 | −0.392 |
| v3 | + 63d regime features | 0.0695 | −1.243 | −0.573 |
| Equal Weight | 1/N | — | −0.670 | 0 |
| Sample Cov GMV | 252d historical cov | — | −1.065 | −0.395 |

**Key finding:** The largest single gain came from fixing the training loop (v1→v2, +0.3 Sharpe). Architectural improvements (GATv2, factor model) and feature additions (63d) produced mixed results on this specific test window. The 8-month bear market test period is too short and regime-specific to draw definitive conclusions about model quality.

---

## Pending Improvements

| Priority | Change | Status |
|---|---|---|
| P2 | Temporal GRU wrapper over sequential `g_t` | Not started |
| — | Extend test period / use longer data range | Not started |
| — | Volume z-score (currently placeholder 0) | Not started |
