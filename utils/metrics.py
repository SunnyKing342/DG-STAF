import torch


def ranking_metrics(scores, true_cols, ks=(1, 10)):
    """Hits@k and MRR for score rows where HIGHER is better.

    scores: (n, N_T) over the full target set (no candidate filtering).
    true_cols: (n,) index of the gold target for every row.
    Ties are counted pessimistically: rank = 1 + #{other candidates with score >= gold score}.
    """
    true = scores.gather(1, true_cols.unsqueeze(1))
    rank = (scores >= true).sum(dim=1).float()          # includes the gold itself -> >= 1
    out = {f"hits@{k}": (rank <= k).float().mean().item() for k in ks}
    out["mrr"] = (1.0 / rank).mean().item()
    return out


def reciprocal_metrics(matches, gold_pairs):
    """Precision / recall of reciprocally verified matches against the gold pairs.

    matches: (m, 2) accepted (src, tgt); gold_pairs: (n, 2). Recall is over the gold pairs only;
    precision is over accepted matches whose source is in the gold set.
    """
    gold = {(int(s), int(t)) for s, t in gold_pairs.tolist()}
    gold_src = {s for s, _ in gold}
    acc = [(int(s), int(t)) for s, t in matches.tolist() if int(s) in gold_src]
    correct = sum(1 for p in acc if p in gold)
    precision = correct / len(acc) if acc else 0.0
    recall = correct / len(gold) if gold else 0.0
    return {"recip_precision": precision, "recip_recall": recall}
