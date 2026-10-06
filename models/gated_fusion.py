import torch
import torch.nn as nn


class DynamicConfidenceGating(nn.Module):
    """Dynamic Confidence Gating (DCG), Eqs. global_var, gate_weight, gate_recalibrate.

    Parameter-free. For every modality distance matrix M_m it computes the global variance
    sigma_m^2 over all entries, then omega = softmax(sigma^2 / tau) and M_hat_m = omega_m M_m.
    """

    def __init__(self, tau=0.5, var_scale=1.0):
        super().__init__()
        self.tau = tau
        # var_scale = 1.0 is the paper formula. The raw variances of distances in [0, ~1] are
        # tiny, so with tau = 0.5 the softmax is close to uniform; see README ("DCG scale").
        self.var_scale = var_scale

    def forward(self, mats, stats_mats=None):
        """mats: list of M x (rows, cols) matrices (order s, w, v, l).
        stats_mats: optional matrices used only to compute sigma^2 (e.g. the full matrix at
        inference). Returns (list of gated matrices, omega of shape (M,), sigma2 of shape (M,)).
        """
        ref = mats if stats_mats is None else stats_mats
        sigma2 = torch.stack([m.var(unbiased=False) for m in ref])
        omega = torch.softmax(self.var_scale * sigma2 / self.tau, dim=0)
        gated = [omega[i] * m for i, m in enumerate(mats)]
        return gated, omega, sigma2
