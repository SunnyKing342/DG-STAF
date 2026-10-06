"""Distance measures: Traffic-Robust Bray-Curtis (TR-BC) and Normalized Levenshtein."""
import numpy as np
import torch


def trbc_paired(x, y, lam=0.1, eps=1e-6, sigma2_max=1.0):
    """TR-BC dissimilarity D_b(x, y) for aligned rows (Eq. tr_bc).

    D_b = mean_k |x_k - y_k| / (x_k + y_k + eps) + lam * Var(x - y) / sigma2_max

    x, y: (..., n) non-negative features (the model bounds them to (0, 1), so the maximum
    possible variance of x - y is 1 and sigma2_max = 1). Returns (...,).
    """
    diff = x - y
    bc = (diff.abs() / (x + y + eps)).mean(dim=-1)
    var = diff.pow(2).mean(dim=-1) - diff.mean(dim=-1).pow(2)
    return bc + lam * var.clamp_min(0.0) / sigma2_max


def trbc_matrix(xs, ys, lam=0.1, eps=1e-6, sigma2_max=1.0, chunk=32):
    """Pairwise TR-BC matrix (N_x, N_y), computed in row chunks to bound memory."""
    out = []
    y = ys.unsqueeze(0)
    for i in range(0, xs.size(0), chunk):
        x = xs[i:i + chunk].unsqueeze(1)
        out.append(trbc_paired(x, y, lam, eps, sigma2_max))
    return torch.cat(out, dim=0)


def normalized_levenshtein_matrix(src_names, tgt_names):
    """Normalized Levenshtein distance of Yujian & Bo (2007): 2d / (|a| + |b| + d).

    Returns a float32 numpy array of shape (len(src_names), len(tgt_names)) in [0, 1].
    """
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist
    src = [str(s) for s in src_names]
    tgt = [str(t) for t in tgt_names]
    d = cdist(src, tgt, scorer=Levenshtein.distance, dtype=np.float64, workers=-1)
    ls = np.array([len(s) for s in src], dtype=np.float64)[:, None]
    lt = np.array([len(t) for t in tgt], dtype=np.float64)[None, :]
    denom = ls + lt + d
    out = np.zeros_like(d)
    np.divide(2.0 * d, denom, out=out, where=denom > 0)
    return out.astype(np.float32)
