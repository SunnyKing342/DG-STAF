import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalAdaptiveFusion(nn.Module):
    """Spatio-Temporal Adaptive Fusion (STAF), Eqs. st_emb - unified_matrix.

    e_{S,T} = E(S) (+) E(T); Q = W_Q e_{S,T}; k_m = W_K (e_m (+) omega_m);
    w = softmax(Q K^T / sqrt(d_k)); M_unified = sum_m w_m * M_hat_m.
    No value projection (the attention distribution itself is the fusion weight).
    """

    def __init__(self, n_spatial, n_temporal, n_modal=4, ctx_dim=32, d_k=32, d_e=16):
        super().__init__()
        self.spatial_emb = nn.Embedding(max(n_spatial, 1), ctx_dim)
        self.temporal_emb = nn.Embedding(max(n_temporal, 1), ctx_dim)
        self.W_Q = nn.Linear(2 * ctx_dim, d_k, bias=False)
        self.modal_emb = nn.Parameter(torch.randn(n_modal, d_e) * 0.1)  # e_m
        self.W_K = nn.Linear(d_e + 1, d_k, bias=False)
        self.d_k = d_k

    def weights(self, spatial_ids, temporal_ids, omega):
        """spatial_ids, temporal_ids: (B,) long. omega: (M,). Returns w: (B, M)."""
        e_st = torch.cat([self.spatial_emb(spatial_ids), self.temporal_emb(temporal_ids)], -1)
        q = self.W_Q(e_st)                                              # (B, d_k)
        keys = self.W_K(torch.cat([self.modal_emb, omega.unsqueeze(-1)], -1))  # (M, d_k)
        return F.softmax(q @ keys.t() / math.sqrt(self.d_k), dim=-1)

    def forward(self, gated_mats, spatial_ids, temporal_ids, omega):
        """gated_mats: list of M matrices (B, N). Returns (M_unified (B, N), w (B, M))."""
        w = self.weights(spatial_ids, temporal_ids, omega)
        stacked = torch.stack(gated_mats, dim=0)                        # (M, B, N)
        unified = (w.t().unsqueeze(-1) * stacked).sum(dim=0)
        return unified, w
