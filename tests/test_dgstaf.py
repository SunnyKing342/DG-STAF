"""Unit and smoke tests.  Run from the repository root:  python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from models import DGSTAF, GATLayer, DynamicConfidenceGating, SpatioTemporalAdaptiveFusion  # noqa: E402
from models.alignment import sinkhorn_log, topk_smoothing, reciprocal_matches  # noqa: E402
from utils.data import apply_overrides, load_benchmark, load_config  # noqa: E402
from utils.distance import trbc_paired, trbc_matrix, normalized_levenshtein_matrix  # noqa: E402
from utils.metrics import ranking_metrics, reciprocal_metrics  # noqa: E402
from make_toy_data import make_toy  # noqa: E402
import synchronize_data as sd  # noqa: E402


class TestDistances(unittest.TestCase):
    def test_trbc_matches_reference(self):
        torch.manual_seed(0)
        x, y, lam, eps = torch.rand(5, 16), torch.rand(7, 16), 0.1, 1e-6
        got = trbc_matrix(x, y, lam, eps, 1.0, chunk=2)
        for i in range(5):
            for j in range(7):
                d = x[i] - y[j]
                ref = (d.abs() / (x[i] + y[j] + eps)).mean() + lam * d.var(unbiased=False)
                self.assertAlmostEqual(got[i, j].item(), ref.item(), places=5)

    def test_trbc_zero_vectors_are_finite(self):
        z = torch.zeros(3, 4)
        self.assertTrue(torch.isfinite(trbc_paired(z, z)).all())
        self.assertEqual(trbc_paired(z, z)[0].item(), 0.0)

    def test_trbc_is_differentiable(self):
        x = torch.rand(4, 8, requires_grad=True)
        trbc_matrix(x, torch.rand(3, 8)).sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_normalized_levenshtein(self):
        m = normalized_levenshtein_matrix(["kitten", "a"], ["sitting", "a"])
        self.assertAlmostEqual(float(m[0, 0]), 2 * 3 / (6 + 7 + 3), places=5)   # Yujian & Bo
        self.assertEqual(float(m[1, 1]), 0.0)
        self.assertTrue(((m >= 0) & (m <= 1)).all())


class TestMetrics(unittest.TestCase):
    def test_ranking_and_ties(self):
        scores = torch.tensor([[0.9, 0.1, 0.5], [0.2, 0.2, 0.2]])
        r = ranking_metrics(scores, torch.tensor([0, 1]))
        self.assertAlmostEqual(r["hits@1"], 0.5)
        self.assertAlmostEqual(r["mrr"], (1 + 1 / 3) / 2, places=5)      # ties are pessimistic

    def test_mrr_not_below_hits1(self):
        torch.manual_seed(1)
        r = ranking_metrics(torch.randn(50, 40), torch.randint(0, 40, (50,)))
        self.assertGreaterEqual(r["mrr"], r["hits@1"])
        self.assertGreaterEqual(r["hits@10"], r["hits@1"])

    def test_reciprocal(self):
        plan = torch.tensor([[.9, .1], [.8, .2], [.0, .7]])
        m = reciprocal_matches(plan)
        self.assertEqual(m.tolist(), [[0, 0], [2, 1]])
        r = reciprocal_metrics(m, torch.tensor([[0, 0], [2, 1], [1, 0]]))
        self.assertAlmostEqual(r["recip_precision"], 1.0)
        self.assertAlmostEqual(r["recip_recall"], 2 / 3, places=5)


class TestAlignment(unittest.TestCase):
    def test_sinkhorn_marginals(self):
        torch.manual_seed(0)
        p = sinkhorn_log(torch.rand(30, 25), 0.05, 300).exp()
        self.assertTrue(torch.allclose(p.sum(1), torch.full((30,), 1 / 30), atol=1e-5))
        self.assertTrue(torch.allclose(p.sum(0), torch.full((25,), 1 / 25), atol=1e-5))

    def test_topk_smoothing_keeps_ranking(self):
        torch.manual_seed(0)
        s = torch.rand(6, 9)
        self.assertTrue((topk_smoothing(s, 3, 0.8).argsort(1) == s.argsort(1)).all())


class TestModules(unittest.TestCase):
    def test_dcg_weights_sum_to_one(self):
        mats = [torch.rand(8, 8) * s for s in (1, .1, .5, 2)]
        gated, omega, sigma2 = DynamicConfidenceGating(0.5)(mats)
        self.assertAlmostEqual(omega.sum().item(), 1.0, places=5)
        self.assertGreater(omega[3], omega[1])             # higher variance -> larger weight
        self.assertTrue(torch.allclose(gated[0], omega[0] * mats[0]))

    def test_staf_weights_and_shape(self):
        st = SpatioTemporalAdaptiveFusion(3, 3)
        mats = [torch.rand(5, 6) for _ in range(4)]
        unified, w = st(mats, torch.tensor([0, 1, 2, 0, 1]), torch.tensor([2, 1, 0, 0, 1]),
                        torch.full((4,), 0.25))
        self.assertEqual(unified.shape, (5, 6))
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(5), atol=1e-5))

    def test_gat_attends_only_to_neighbours(self):
        layer = GATLayer(8, 16, 4, dropout=0.0)
        h = torch.randn(5, 8)
        adj = torch.eye(5, dtype=torch.bool)
        adj[0, 1] = True
        out = layer(h, adj)
        h2 = h.clone()
        h2[3] += 10                                          # node 3 is not a neighbour of node 0
        self.assertTrue(torch.allclose(layer(h2, adj)[0], out[0], atol=1e-6))
        self.assertTrue(torch.isfinite(out).all())


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg_path = make_toy(os.path.join(cls.tmp.name, "toy"), 60, 50, 40, seed=0)
        cls.cfg = load_config(cls.cfg_path)
        cls.d = load_benchmark(cls.cfg["data"]["dir"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_splits_are_disjoint_at_entity_level(self):
        pairs, d = self.d["pairs"], self.d
        parts = [d["split_train"], d["split_val"], d["split_test"]]
        self.assertEqual(sum(len(p) for p in parts), len(pairs))
        seen_s, seen_t = set(), set()
        for p in parts:
            s = set(pairs[p, 0].tolist())
            t = set(pairs[p, 1].tolist())
            self.assertFalse(s & seen_s)
            self.assertFalse(t & seen_t)
            seen_s |= s
            seen_t |= t

    def test_training_step_and_evaluation(self):
        torch.manual_seed(0)
        cfg, d = self.cfg, self.d
        model = DGSTAF(cfg, len(d["spatial_vocab"]), len(d["temporal_vocab"]))
        p = d["pairs"][d["split_train"][:16]]
        loss, parts = model.training_loss(d, p[:, 0], p[:, 1], 0.5, 0.15, 1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None))
        res = model.evaluate(d, d["split_test"], use_ot=True)
        for k in ("hits@1", "hits@10", "mrr", "ot_mrr", "recip_precision", "omega", "sigma2"):
            self.assertIn(k, res)
        self.assertLessEqual(res["hits@1"], res["mrr"] + 1e-9)
        self.assertAlmostEqual(sum(res["omega"]), 1.0, places=4)

    def test_ablation_switches(self):
        for over in (["model.use_dcg=false"], ["model.use_staf=false"], ["model.modalities=[s,w,l]"],
                     ["model.modalities=[v]"]):
            cfg = apply_overrides(load_config(self.cfg_path), over)
            d = self.d
            model = DGSTAF(cfg, len(d["spatial_vocab"]), len(d["temporal_vocab"]))
            res = model.evaluate(d, d["split_val"], use_ot=False)
            self.assertTrue(np.isfinite(res["mrr"]), over)

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(KeyError):
            apply_overrides(load_config(self.cfg_path), ["model.not_a_key=1"])

    def test_exclude_matched_frames_changes_visual_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_toy(os.path.join(tmp, "a"), 60, 50, 40, seed=0, exclude_matched_frames=False)
            make_toy(os.path.join(tmp, "b"), 60, 50, 40, seed=0, exclude_matched_frames=True)
            a = load_benchmark(os.path.join(tmp, "a", "synchronized"))
            b = load_benchmark(os.path.join(tmp, "b", "synchronized"))
            lab = a["pairs"][:, 0]
            self.assertTrue((a["s_vis"][lab] != b["s_vis"][lab]).any())
            unl = torch.ones(a["n_s"], dtype=torch.bool)
            unl[lab] = False
            self.assertTrue(torch.allclose(a["s_vis"][unl], b["s_vis"][unl]))


if __name__ == "__main__":
    unittest.main()
