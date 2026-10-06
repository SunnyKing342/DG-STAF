"""Generate a small synthetic benchmark so the whole pipeline can be run without real data.

    python scripts/make_toy_data.py --out data/toy
    python train.py --config data/toy/config.yaml
    python test.py  --config data/toy/config.yaml

Writes the CSV files described in the README (entities, edges, frames, pairs), dummy image
files, and runs the normal `build` step with synthetic BERT / ResNet features (no model
download). The numbers it produces say nothing about real performance: this is a smoke test.
"""
import argparse
import os
import sys
import zlib

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import synchronize_data as sd  # noqa: E402


def _latent(key, dim):
    return np.random.RandomState(zlib.crc32(str(key).encode()) % (2 ** 31)).randn(dim)


def make_toy(out, n_src=200, n_tgt=180, n_pairs=150, seed=0, exclude_matched_frames=False):
    out = out if os.path.isabs(out) else os.path.join(ROOT, out)
    rng = np.random.RandomState(seed)
    bj, dair, sync = (os.path.join(out, d) for d in ("bjtt", "dair-v2x", "synchronized"))
    for d in (bj, dair, sync, os.path.join(dair, "img")):
        os.makedirs(d, exist_ok=True)

    lat0, lon0 = 39.9, 116.4
    t_lat, t_lon = lat0 + rng.rand(n_tgt) * 0.02, lon0 + rng.rand(n_tgt) * 0.02
    letters = list("abcdefghijklmnopqrstuvwxyz")
    t_name = ["RSU-%04d-%s" % (j, "".join(rng.choice(letters, 4))) for j in range(n_tgt)]
    matched = rng.permutation(n_tgt)[:n_pairs]

    s_lat, s_lon, s_name, pairs = np.zeros(n_src), np.zeros(n_src), [""] * n_src, []
    for i in range(n_src):
        if i < n_pairs:
            j = matched[i]
            s_lat[i], s_lon[i] = t_lat[j] + rng.randn() * 2e-5, t_lon[j] + rng.randn() * 2e-5
            s_name[i] = t_name[j][:-1] + "x"          # near-duplicate tag, like noisy names
            pairs.append((f"s{i}", f"t{j}"))
        else:
            s_lat[i], s_lon[i] = lat0 + rng.rand() * 0.02, lon0 + rng.rand() * 0.02
            s_name[i] = "ROAD-%04d-%s" % (i, "".join(rng.choice(letters, 4)))

    pd.DataFrame({"id": [f"s{i}" for i in range(n_src)], "name": s_name, "lat": s_lat, "lon": s_lon,
                  "spatial_ctx": rng.choice(["intersection", "arterial", "residential"], n_src),
                  "temporal_ctx": rng.choice(["rush", "night", "offpeak"], n_src)}
                 ).to_csv(os.path.join(bj, "entities.csv"), index=False)
    edges = [(f"s{i}", f"s{(i + k) % n_src}") for i in range(n_src) for k in (1, 7)]
    pd.DataFrame(edges, columns=["src", "dst"]).to_csv(os.path.join(bj, "edges.csv"), index=False)
    pd.DataFrame({"id": [f"t{j}" for j in range(n_tgt)], "name": t_name, "lat": t_lat, "lon": t_lon}
                 ).to_csv(os.path.join(dair, "entities.csv"), index=False)
    frames = []
    for j in range(n_tgt):
        for v in range(3):
            rel = f"img/{j}_{v}.jpg"
            open(os.path.join(dair, rel), "w").close()  # empty placeholder, never decoded
            frames.append((f"f{j}_{v}", f"t{j}", t_lat[j] + rng.randn() * 1e-5,
                           t_lon[j] + rng.randn() * 1e-5, rel))
    pd.DataFrame(frames, columns=["frame_id", "entity_id", "lat", "lon", "image_path"]
                 ).to_csv(os.path.join(dair, "frames.csv"), index=False)
    pd.DataFrame(pairs, columns=["src_id", "tgt_id"]).to_csv(os.path.join(sync, "pairs.csv"), index=False)

    # synthetic stand-ins for the BERT / ResNet extractors (no download, deterministic)
    feat_rng = np.random.RandomState(seed + 1)
    proj = np.random.RandomState(seed + 2).randn(16, 2048)

    def fake_images(paths, cfg, device):
        out_ = [_latent(("img", os.path.basename(p).split("_")[0]), 16) @ proj + feat_rng.randn(2048) * 0.3
                for p in paths]
        return torch.tensor(np.array(out_), dtype=torch.float32)

    def fake_text(texts, cfg, device):
        return torch.tensor(np.stack([_latent(t[:-1], 768) + feat_rng.randn(768) * 0.3 for t in texts]),
                            dtype=torch.float32)

    with open(os.path.join(ROOT, "configs", "dgstaf.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["device"] = "cpu"
    cfg["seeds"] = [0, 1]
    cfg["data"].update({"dir": sync, "bjtt_dir": bj, "dair_dir": dair,
                        "exclude_matched_frames": exclude_matched_frames})
    cfg["train"].update({"epochs": 20, "eval_interval": 2, "cycle_epochs": 5, "patience": 5,
                         "save_dir": os.path.join(out, "checkpoints"), "log_dir": os.path.join(out, "logs")})
    real = (sd.extract_image_features, sd.extract_text_features)
    sd.extract_image_features, sd.extract_text_features = fake_images, fake_text
    try:
        sd.cmd_build(cfg)
    finally:
        sd.extract_image_features, sd.extract_text_features = real
    cfg_path = os.path.join(out, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--n_src", type=int, default=200)
    ap.add_argument("--n_tgt", type=int, default=180)
    ap.add_argument("--n_pairs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    path = make_toy(a.out, a.n_src, a.n_tgt, a.n_pairs, a.seed)
    print(f"\nToy benchmark ready. Next:\n  python train.py --config {path}\n  python test.py  --config {path}")
