import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distance import trbc_matrix
from utils.metrics import ranking_metrics, reciprocal_metrics
from .alignment import similarity_matrix, topk_smoothing, sinkhorn_log, reciprocal_matches
from .encoders import StructuralEncoder, TextEncoder, VisualEncoder
from .gated_fusion import DynamicConfidenceGating
from .spatio_temporal import SpatioTemporalAdaptiveFusion

MODALITIES = ("s", "w", "v", "l")  # structural, semantic (BERT), visual, literal (Levenshtein)


class DGSTAF(nn.Module):
    """DG-STAF: TR-BC distance matrices -> DCG -> STAF -> (smoothing, Sinkhorn OT, reciprocal check).

    Encoders are shared between the source and target graphs. Embeddings are mapped into
    (0, 1) with a sigmoid so that Bray-Curtis is well defined and the maximum variance of
    x - y in TR-BC is 1.
    """

    def __init__(self, cfg, n_spatial, n_temporal):
        super().__init__()
        m = cfg["model"]
        self.m = m
        self.struct_enc = StructuralEncoder(m["struct_in"], m["gat_hidden"], m["gat_layers"],
                                            m["gat_heads"], m["dropout"])
        self.text_enc = TextEncoder(m["text_in"], m["text_dim"])
        self.vis_enc = VisualEncoder(m["vis_in"], m["vis_hidden"], m["vis_dim"], m["dropout"])
        self.active = [MODALITIES.index(x) for x in m["modalities"]]   # ablation: modality subset
        self.dcg = DynamicConfidenceGating(m["dcg_tau"], m["dcg_var_scale"])
        self.staf = SpatioTemporalAdaptiveFusion(n_spatial, n_temporal, len(self.active),
                                                 m["staf_ctx_dim"], m["staf_d_k"], m["staf_d_e"])

    # ------------------------------------------------------------------ embeddings
    def encode(self, d):
        emb = {}
        for side in ("s", "t"):
            emb[side] = {
                "s": torch.sigmoid(self.struct_enc(d[f"{side}_struct"], d[f"{side}_adj"])),
                "w": torch.sigmoid(self.text_enc(d[f"{side}_text"])),
                "v": torch.sigmoid(self.vis_enc(d[f"{side}_vis"])),
            }
        return emb

    # ------------------------------------------------------------------ distance matrices
    def modal_matrices(self, emb, src_idx, tgt_idx, m_lit):
        """Four distance matrices (len(src_idx), len(tgt_idx)) in the order s, w, v, l."""
        m = self.m
        mats = []
        for i in self.active:
            k = MODALITIES[i]
            if k == "l":
                mats.append(m_lit[src_idx.unsqueeze(1), tgt_idx.unsqueeze(0)])
            else:
                mats.append(trbc_matrix(emb["s"][k][src_idx], emb["t"][k][tgt_idx], m["trbc_lambda"],
                                        m["trbc_eps"], m["trbc_sigma2_max"], m["dist_chunk"]))
        return mats

    def fuse(self, mats, spatial_ids, temporal_ids):
        """DCG then STAF. `use_dcg` / `use_staf` = false give the uniform-weight ablations."""
        m = self.m
        n = len(mats)
        if m["use_dcg"]:
            gated, omega, sigma2 = self.dcg(mats)
        else:
            sigma2 = torch.stack([x.var(unbiased=False) for x in mats])
            omega = torch.full((n,), 1.0 / n, device=mats[0].device)
            gated = [omega[i] * x for i, x in enumerate(mats)]
        if m["use_staf"]:
            unified, w = self.staf(gated, spatial_ids, temporal_ids, omega)
        else:
            w = torch.full((mats[0].size(0), n), 1.0 / n, device=mats[0].device)
            unified = sum(w[:, i:i + 1] * g for i, g in enumerate(gated))
        return unified, w, omega, sigma2

    # ------------------------------------------------------------------ training
    def training_loss(self, d, pair_src, pair_tgt, gamma, mu, num_neg=1):
        emb = self.encode(d)
        mats = self.modal_matrices(emb, pair_src, pair_tgt, d["m_lit"])          # B x B each
        ctx = d["s_ctx"][pair_src]
        unified, w, omega, _ = self.fuse(mats, ctx[:, 0], ctx[:, 1])
        b = unified.size(0)
        pos = unified.diagonal()
        # random in-batch negatives j != i (a negative equal to the gold target is masked out)
        r = torch.randint(0, b - 1, (b, num_neg), device=unified.device)
        j = r + (r >= torch.arange(b, device=unified.device).unsqueeze(1)).long()
        neg = unified.gather(1, j)
        valid = (pair_tgt[j] != pair_tgt.unsqueeze(1)).float()
        hinge = F.relu(gamma + pos.unsqueeze(1) - neg) * valid
        l_align = hinge.sum() / valid.sum().clamp_min(1.0)
        l_reg = -(w * torch.log(w + 1e-12)).sum(dim=-1).mean()                  # entropy of w
        return l_align + mu * l_reg, {"align": l_align.item(), "reg": l_reg.item(),
                                      "omega": omega.detach().cpu().tolist()}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def unified_matrix(self, d):
        """Full M_unified over all source x all target entities; DCG statistics are global."""
        emb = self.encode(d)
        all_s = torch.arange(d["n_s"], device=d["s_ctx"].device)
        all_t = torch.arange(d["n_t"], device=d["s_ctx"].device)
        mats = self.modal_matrices(emb, all_s, all_t, d["m_lit"])
        unified, w, omega, sigma2 = self.fuse(mats, d["s_ctx"][:, 0], d["s_ctx"][:, 1])
        return unified, {"omega": omega.cpu().tolist(), "sigma2": sigma2.cpu().tolist(),
                         "mean_w": w.mean(0).cpu().tolist()}

    @torch.no_grad()
    def evaluate(self, d, pair_idx, use_ot=True):
        """Hits@1/10 and MRR on `pair_idx` ranking over the FULL target set."""
        self.eval()
        m = self.m
        unified, info = self.unified_matrix(d)
        smooth = topk_smoothing(similarity_matrix(unified), m["topk_K"], m["topk_beta"])
        pairs = d["pairs"][pair_idx]
        src, tgt = pairs[:, 0], pairs[:, 1]
        res = ranking_metrics(smooth[src], tgt)
        if use_ot:
            plan = sinkhorn_log(smooth, m["ot_eps"], m["ot_iters"])
            ot = ranking_metrics(plan[src], tgt)
            res.update({f"ot_{k}": v for k, v in ot.items()})
            res.update(reciprocal_metrics(reciprocal_matches(plan), pairs))
        res.update(info)
        return res

