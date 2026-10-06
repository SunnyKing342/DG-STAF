"""Traffic Entity Alignment module: similarity, top-K smoothing, Sinkhorn OT, reciprocal check."""
import torch


def similarity_matrix(m_unified):
    """S_match = exp(-M_unified)  (Eq. sim_matrix)."""
    return torch.exp(-m_unified)


def topk_smoothing(sim, K=3, beta=0.8):
    """Top-K neighbourhood smoothing (Eq. topk_smooth), implemented exactly as written.

    N_K(i) is the set of the K most similar targets of source i. NOTE: the second term does
    not depend on j, so it shifts every entry of row i by the same constant.
    """
    K = min(K, sim.size(1))
    topk = sim.topk(K, dim=1).values.mean(dim=1, keepdim=True)
    return beta * sim + (1.0 - beta) * topk


@torch.no_grad()
def sinkhorn_log(sim, eps=0.05, n_iters=100):
    """Entropy-regularised OT (Eq. sinkhorn_ot) with cost 1 - sim, uniform marginals,
    solved in the log domain. Returns log P of shape (N_S, N_T)."""
    cost = 1.0 - sim
    n_s, n_t = cost.shape
    log_k = -cost / eps
    log_a = torch.full((n_s,), -torch.log(torch.tensor(float(n_s))), device=sim.device)
    log_b = torch.full((n_t,), -torch.log(torch.tensor(float(n_t))), device=sim.device)
    f = torch.zeros(n_s, device=sim.device)
    g = torch.zeros(n_t, device=sim.device)
    for _ in range(n_iters):
        f = log_a - torch.logsumexp(log_k + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(log_k + f.unsqueeze(1), dim=0)
    return log_k + f.unsqueeze(1) + g.unsqueeze(0)


@torch.no_grad()
def reciprocal_matches(plan):
    """Mutual arg-max verification (Eq. bidirectional_match).
    Returns a long tensor (n, 2) of accepted (source, target) index pairs."""
    row_best = plan.argmax(dim=1)
    col_best = plan.argmax(dim=0)
    src = torch.arange(plan.size(0), device=plan.device)
    ok = col_best[row_best] == src
    return torch.stack([src[ok], row_best[ok]], dim=1)
