import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Multi-head graph attention layer (Eqs. gat_layer, attention_coeff), dense masked form.

    h_i' = sigma( sum_{j in N_i} alpha_ij W h_j ), heads are concatenated.
    """

    def __init__(self, in_dim, out_dim, heads=8, dropout=0.2, negative_slope=0.2):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads, self.head_dim = heads, out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)
        self.slope = negative_slope

    def forward(self, h, adj):
        """h: (N, in_dim); adj: (N, N) bool, adj[i, j] = j is a neighbour of i (self-loops incl.)."""
        n = h.size(0)
        wh = self.W(h).view(n, self.heads, self.head_dim)              # (N, H, d)
        e_i = (wh * self.a_src).sum(-1)                                # (N, H)
        e_j = (wh * self.a_dst).sum(-1)                                # (N, H)
        e = F.leaky_relu(e_i.unsqueeze(1) + e_j.unsqueeze(0), self.slope)  # (N, N, H)
        e = e.masked_fill(~adj.unsqueeze(-1), float("-inf"))
        alpha = self.drop(torch.softmax(e, dim=1))
        out = torch.einsum("ijh,jhd->ihd", alpha, wh).reshape(n, -1)
        return F.leaky_relu(out, self.slope)


class StructuralEncoder(nn.Module):
    """L-layer GAT; output h^S = h^0 || h^1 || ... || h^L (Eq. structural_embedding)."""

    def __init__(self, in_dim, hidden=256, layers=2, heads=8, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([GATLayer(hidden, hidden, heads, dropout) for _ in range(layers)])
        self.out_dim = hidden * (layers + 1)

    def forward(self, x, adj):
        h = self.inp(x)
        outs = [h]
        for layer in self.layers:
            h = layer(h, adj)
            outs.append(h)
        return torch.cat(outs, dim=-1)


class TextEncoder(nn.Module):
    """h^W = W_w * BERT_CLS + b_w (Eq. text_emb). BERT [CLS] vectors are pre-extracted."""

    def __init__(self, in_dim=768, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


class VisualEncoder(nn.Module):
    """Two fully connected layers 2048 -> 1024 -> 512 on pre-extracted ResNet-152 features."""

    def __init__(self, in_dim=2048, hidden=1024, out_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)
