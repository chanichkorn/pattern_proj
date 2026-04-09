"""
gat.py
======
P2 — Graph Attention Network (GAT) for SET50 Portfolio Optimization

Architecture:
    Input  : X [B, N, d]   node features  (d=15)
             A [B, N, N]   adjacency matrix (edge weight = |correlation|)

    Layer 1: GATLayer(d → hidden_dim, n_heads=4, concat=True)
             → [B, N, hidden_dim * 4]   LayerNorm + ELU + Dropout

    Layer 2: GATLayer(hidden_dim*4 → embedding_dim, n_heads=4, concat=False)
             → [B, N, embedding_dim]    LayerNorm

    Pooling: mean over N  →  graph embedding g [B, embedding_dim]
             g conditions the WGAN-GP Generator

Output:
    Z : [B, N, embedding_dim]   per-stock node embeddings
    g : [B, embedding_dim]      graph-level market state (→ GAN)
    α : [B, N, N, H]            attention weights (for interpretability)

Usage:
    from src.gat import GAT
    model = GAT.from_config(cfg)
    Z, g, alpha = model(X, A)
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single GAT Layer
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    One Graph Attention layer with optional edge feature incorporation.

    Attention score between node i and node j:

        e_ij = LeakyReLU( a^T [ W·h_i ‖ W·h_j ‖ A_ij ] )

    where A_ij is the edge weight (absolute correlation) from the adjacency
    matrix.  Including A_ij lets the model up-weight highly correlated pairs
    directly without having to infer it solely from node features.

    Parameters
    ----------
    in_features  : input node feature dimension
    out_features : output dimension *per head*
    n_heads      : number of parallel attention heads
    concat       : if True  → output = concat of heads  [N, out_features * n_heads]
                   if False → output = mean  of heads   [N, out_features]
    dropout      : dropout applied to attention weights during training
    use_edge_feat: whether to include edge weight A_ij in attention
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        concat: bool = True,
        dropout: float = 0.1,
        use_edge_feat: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.n_heads      = n_heads
        self.out_features = out_features
        self.concat       = concat
        self.use_edge_feat = use_edge_feat

        # Shared linear transform  (no bias — LayerNorm handles offset)
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)

        # Attention vector per head:  [H, 2F]  or  [H, 2F+1]
        attn_dim = 2 * out_features + (1 if use_edge_feat else 0)
        self.attn = nn.Parameter(torch.empty(n_heads, attn_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout    = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))   # treat as [1, H, attn_dim]

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x   : FloatTensor [B, N, in_features]
        adj : FloatTensor [B, N, N]  edge weights ∈ [0, 1]

        Returns
        -------
        out   : FloatTensor [B, N, out_features*n_heads]  if concat
                             [B, N, out_features]          if mean
        alpha : FloatTensor [B, N, N, H]  attention weights (for logging)
        """
        B, N, _ = x.shape
        H  = self.n_heads
        Fh = self.out_features          # renamed: Fh avoids shadowing nn.functional F

        # ── Linear transform ──────────────────────────────────────────────
        h = self.W(x)               # [B, N, H*Fh]
        h = h.view(B, N, H, Fh)    # [B, N, H, Fh]

        # ── Pairwise feature construction ─────────────────────────────────
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B, N, N, H, Fh]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [B, N, N, H, Fh]

        if self.use_edge_feat:
            e_ij = adj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, 1)
            pair = torch.cat([h_i, h_j, e_ij], dim=-1)  # [B, N, N, H, 2Fh+1]
        else:
            pair = torch.cat([h_i, h_j], dim=-1)         # [B, N, N, H, 2Fh]

        # ── Attention scores ──────────────────────────────────────────────
        attn_vec = self.attn.view(1, 1, 1, H, -1)
        e = (pair * attn_vec).sum(dim=-1)    # [B, N, N, H]
        e = self.leaky_relu(e)

        # Mask zero-weight edges (no connection)
        no_edge = (adj == 0).unsqueeze(-1).expand(-1, -1, -1, H)
        e = e.masked_fill(no_edge, float("-inf"))

        # ── Softmax + dropout ─────────────────────────────────────────────
        alpha = torch.softmax(e, dim=2)              # [B, N, N, H]
        alpha = torch.nan_to_num(alpha, nan=0.0)     # isolated nodes → 0
        alpha_drop = self.dropout(alpha)

        # ── Aggregate neighbour features ──────────────────────────────────
        out = (alpha_drop.unsqueeze(-1) * h_j).sum(dim=2)   # [B, N, H, Fh]

        # ── Concat or mean over heads ─────────────────────────────────────
        if self.concat:
            out = out.reshape(B, N, H * Fh)   # [B, N, H*Fh]
        else:
            out = out.mean(dim=2)              # [B, N, Fh]

        return out, alpha   # alpha kept for interpretability


# ---------------------------------------------------------------------------
# Full 2-Layer GAT
# ---------------------------------------------------------------------------

class GAT(nn.Module):
    """
    2-layer Graph Attention Network.

    Layer 1 (concat heads):   [B, N, d]      →  [B, N, hidden_dim * n_heads_l1]
    Layer 2 (mean  heads):    [B, N, ...]     →  [B, N, embedding_dim]
    Graph pooling (mean):     [B, N, emb_dim] →  [B, embedding_dim]

    The graph embedding g is passed to the WGAN-GP Generator as the
    conditioning vector, replacing hand-crafted market features.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        n_heads_l1: int = 4,
        n_heads_l2: int = 4,
        dropout: float = 0.1,
        use_edge_feat: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Layer 1: concat heads → output dim = hidden_dim * n_heads_l1
        self.layer1 = GATLayer(
            in_features=node_feature_dim,
            out_features=hidden_dim,
            n_heads=n_heads_l1,
            concat=True,
            dropout=dropout,
            use_edge_feat=use_edge_feat,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * n_heads_l1)

        # Layer 2: mean heads → output dim = embedding_dim
        self.layer2 = GATLayer(
            in_features=hidden_dim * n_heads_l1,
            out_features=embedding_dim,
            n_heads=n_heads_l2,
            concat=False,          # ← mean, not concat
            dropout=dropout,
            use_edge_feat=use_edge_feat,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.activation = nn.ELU()
        self.dropout    = nn.Dropout(dropout)

        logger.info(
            f"GAT initialised — "
            f"d={node_feature_dim} → {hidden_dim*n_heads_l1} → {embedding_dim} "
            f"(heads: {n_heads_l1}/{n_heads_l2})"
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x   : FloatTensor [B, N, node_feature_dim]
        adj : FloatTensor [B, N, N]

        Returns
        -------
        Z     : FloatTensor [B, N, embedding_dim]   node embeddings
        g     : FloatTensor [B, embedding_dim]       graph embedding → GAN
        alpha : FloatTensor [B, N, N, n_heads_l2]   layer-2 attention weights
        """
        # ── Layer 1 ───────────────────────────────────────────────────────
        h, _      = self.layer1(x, adj)   # [B, N, hidden_dim * H1]
        h         = self.norm1(h)
        h         = self.activation(h)
        h         = self.dropout(h)

        # ── Layer 2 ───────────────────────────────────────────────────────
        Z, alpha  = self.layer2(h, adj)   # [B, N, embedding_dim]
        Z         = self.norm2(Z)

        # ── Graph-level pooling ───────────────────────────────────────────
        g = Z.mean(dim=1)                 # [B, embedding_dim]

        return Z, g, alpha

    # ── Convenience constructors ──────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "GAT":
        """Build a GAT from the central config.yaml dict."""
        gcfg = cfg["model"]["gat"]
        return cls(
            node_feature_dim = gcfg["node_feature_dim"],
            hidden_dim       = gcfg["hidden_dim"],
            embedding_dim    = gcfg["embedding_dim"],
            n_heads_l1       = gcfg["num_heads_l1"],
            n_heads_l2       = gcfg["num_heads_l2"],
            dropout          = gcfg["dropout"],
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Attention visualisation helper
# ---------------------------------------------------------------------------

def attention_to_matrix(
    alpha: torch.Tensor,
    head: int = 0,
    sample: int = 0,
) -> torch.Tensor:
    """
    Extract a single attention matrix for inspection or plotting.

    Parameters
    ----------
    alpha  : [B, N, N, H]
    head   : which attention head to extract
    sample : which batch sample to extract

    Returns
    -------
    A_attn : [N, N]  attention weight matrix
    """
    return alpha[sample, :, :, head].detach().cpu()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Override feature dim to match actual pipeline output (15, not config 14)
    cfg["model"]["gat"]["node_feature_dim"] = 15

    device = (
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cuda")  if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    model = GAT.from_config(cfg).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Mock batch matching real pipeline output
    B, N, d = 16, 40, 15
    X   = torch.randn(B, N, d).to(device)
    adj = torch.rand(B, N, N).to(device)
    adj = (adj + adj.transpose(1, 2)) / 2   # symmetric
    adj = (adj > 0.6).float()               # sparse, like real graph

    Z, g, alpha = model(X, adj)

    print(f"\n── Forward pass shapes ──────────────────")
    print(f"  Input  X     : {tuple(X.shape)}")
    print(f"  Input  adj   : {tuple(adj.shape)}")
    print(f"  Output Z     : {tuple(Z.shape)}")
    print(f"  Output g     : {tuple(g.shape)}")
    print(f"  Attention α  : {tuple(alpha.shape)}")

    # Sanity checks
    assert Z.shape   == (B, N, cfg["model"]["gat"]["embedding_dim"])
    assert g.shape   == (B, cfg["model"]["gat"]["embedding_dim"])
    assert alpha.shape == (B, N, N, cfg["model"]["gat"]["num_heads_l2"])
    assert torch.isfinite(Z).all(),   "Z contains NaN/Inf"
    assert torch.isfinite(g).all(),   "g contains NaN/Inf"

    # Attention weights should sum to 1 over neighbors (for non-isolated nodes)
    attn_sum = alpha.sum(dim=2)   # [B, N, H]
    # Nodes with no edges will have sum=0 — filter them out
    connected = (adj.sum(dim=2) > 0).unsqueeze(-1).expand_as(attn_sum)
    attn_ok = torch.allclose(attn_sum[connected], torch.ones_like(attn_sum[connected]), atol=1e-5)
    print(f"  Attention sums to 1 (connected nodes): {attn_ok}")

    # Test gradient flow
    loss = Z.sum() + g.sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters())
    print(f"  Gradients flow to all params         : {grad_ok}")

    print("\n✅  GAT smoke test passed.")
