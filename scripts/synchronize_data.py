"""Benchmark construction for DG-STAF.

Two sub-commands (run from the repository root):

  python scripts/synchronize_data.py candidates   # GPS (5 m) + time-window candidate pairs,
                                                  # to be manually verified
  python scripts/synchronize_data.py build        # graphs, features, distance inputs, splits

Input files (CSV, UTF-8; see README for the schema)
  data/bjtt/entities.csv       id,name,lat,lon[,spatial_ctx,temporal_ctx,timestamp]
  data/bjtt/edges.csv          src,dst                      (road topology of G_S)
  data/dair-v2x/entities.csv   id,name,lat,lon[,timestamp]   (name = RSU / sensor tag string)
  data/dair-v2x/frames.csv     frame_id,entity_id,lat,lon,image_path[,timestamp]
                               (image_path relative to data/dair-v2x/; entity_id may be empty)
  data/synchronized/pairs.csv  src_id,tgt_id                 (manually verified ground truth)
Output: data/synchronized/benchmark.pt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import load_config, apply_overrides, set_seed  # noqa: E402
from utils.distance import normalized_levenshtein_matrix  # noqa: E402

EARTH_RADIUS_M = 6371000.0


def _tree(lat, lon):
    return BallTree(np.radians(np.stack([lat, lon], axis=1)), metric="haversine")


def _rad(meters):
    return meters / EARTH_RADIUS_M


# --------------------------------------------------------------------------- candidates
def cmd_candidates(cfg):
    dc = cfg["data"]
    src = pd.read_csv(os.path.join(dc["bjtt_dir"], "entities.csv"))
    tgt = pd.read_csv(os.path.join(dc["dair_dir"], "entities.csv"))
    tree = _tree(tgt["lat"].values, tgt["lon"].values)
    ind, dist = tree.query_radius(np.radians(src[["lat", "lon"]].values), r=_rad(dc["gt_radius_m"]),
                                  return_distance=True)
    use_time = "timestamp" in src.columns and "timestamp" in tgt.columns
    if use_time:
        ts_s, ts_t = pd.to_datetime(src["timestamp"]), pd.to_datetime(tgt["timestamp"])
    else:
        print("warning: no timestamp column in both tables; time-window filter skipped")
    rows = []
    for i, (js, ds) in enumerate(zip(ind, dist)):
        for j, dd in zip(js, ds):
            if use_time and abs((ts_s[i] - ts_t[j]).total_seconds()) > dc["time_window_s"]:
                continue
            rows.append((src["id"][i], tgt["id"][j], round(float(dd) * EARTH_RADIUS_M, 2)))
    out = os.path.join(dc["dir"], "candidate_pairs.csv")
    os.makedirs(dc["dir"], exist_ok=True)
    pd.DataFrame(rows, columns=["src_id", "tgt_id", "distance_m"]).to_csv(out, index=False)
    print(f"{len(rows)} candidate pairs written to {out}. Verify them manually, then save the "
          f"accepted ones as {os.path.join(dc['dir'], 'pairs.csv')} (columns src_id,tgt_id).")


# --------------------------------------------------------------------------- features
def extract_text_features(texts, cfg, device):
    """BERT [CLS] vectors (768-d), max `bert_max_len` tokens."""
    from transformers import AutoModel, AutoTokenizer
    dc = cfg["data"]
    tok = AutoTokenizer.from_pretrained(dc["bert_name"])
    bert = AutoModel.from_pretrained(dc["bert_name"]).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), dc["extract_batch"]):
            enc = tok(list(texts[i:i + dc["extract_batch"]]), padding="max_length", truncation=True,
                      max_length=dc["bert_max_len"], return_tensors="pt").to(device)
            out.append(bert(**enc).last_hidden_state[:, 0].cpu())
    return torch.cat(out).float()


def extract_image_features(paths, cfg, device):
    """ImageNet-pretrained ResNet-152 global-average-pooled features (2048-d)."""
    import torchvision
    from PIL import Image
    weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2
    net = torchvision.models.resnet152(weights=weights)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    prep = weights.transforms()
    out, bs = [], cfg["data"]["extract_batch"]
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])
            out.append(net(batch.to(device)).cpu())
    return torch.cat(out).float()


# --------------------------------------------------------------------------- graphs
def structural_features(edges, n):
    """Graph-only node descriptors (no GPS): log-degree, mean neighbour degree, clustering
    coefficient, 2-hop neighbourhood size; z-scored within the graph."""
    a = np.zeros((n, n), dtype=np.float32)
    if len(edges):
        a[edges[:, 0], edges[:, 1]] = 1
        a[edges[:, 1], edges[:, 0]] = 1
    np.fill_diagonal(a, 0)
    deg = a.sum(1)
    nbr = (a @ deg) / np.maximum(deg, 1)
    tri = ((a @ a) * a).sum(1)
    clus = tri / np.maximum(deg * (deg - 1), 1)
    two = ((a @ a) > 0).sum(1).astype(np.float32)
    f = np.stack([np.log1p(deg), np.log1p(nbr), clus, np.log1p(two)], axis=1)
    return ((f - f.mean(0)) / (f.std(0) + 1e-6)).astype(np.float32)


def proximity_edges(lat, lon, radius_m):
    ind = _tree(lat, lon).query_radius(np.radians(np.stack([lat, lon], 1)), r=_rad(radius_m))
    return np.array([(i, j) for i, js in enumerate(ind) for j in js if j > i], dtype=np.int64).reshape(-1, 2)


def split_pairs(pairs, ratios, seed):
    """Entity-level split: pairs sharing a source or target entity stay in the same partition."""
    parent = list(range(len(pairs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for i, (s, t) in enumerate(pairs.tolist()):
        for key in (("s", s), ("t", t)):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    groups = {}
    for i in range(len(pairs)):
        groups.setdefault(find(i), []).append(i)
    keys = list(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    n = len(pairs)
    cut = [ratios[0] * n, (ratios[0] + ratios[1]) * n]
    parts, count = ([], [], []), 0
    for k in keys:
        which = 0 if count < cut[0] else (1 if count < cut[1] else 2)
        parts[which].extend(groups[k])
        count += len(groups[k])
    return [torch.tensor(sorted(p), dtype=torch.long) for p in parts]


# --------------------------------------------------------------------------- build
def cmd_build(cfg):
    dc = cfg["data"]
    set_seed(dc["split_seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    src = pd.read_csv(os.path.join(dc["bjtt_dir"], "entities.csv"))
    edges_df = pd.read_csv(os.path.join(dc["bjtt_dir"], "edges.csv"))
    tgt = pd.read_csv(os.path.join(dc["dair_dir"], "entities.csv"))
    frames = pd.read_csv(os.path.join(dc["dair_dir"], "frames.csv"))
    pairs_df = pd.read_csv(os.path.join(dc["dir"], "pairs.csv"))
    for name, df in (("bjtt entities", src), ("dair entities", tgt)):
        assert df["id"].is_unique, f"duplicate ids in {name}"
    s_index = {k: i for i, k in enumerate(src["id"])}
    t_index = {k: i for i, k in enumerate(tgt["id"])}
    n_s, n_t = len(src), len(tgt)

    pairs = np.array([(s_index[s], t_index[t]) for s, t in zip(pairs_df["src_id"], pairs_df["tgt_id"])
                      if s in s_index and t in t_index], dtype=np.int64)
    missing = len(pairs_df) - len(pairs)
    if missing:
        print(f"warning: {missing} pairs reference unknown ids and were dropped")

    # structural graphs
    s_edges = np.array([(s_index[a], s_index[b]) for a, b in zip(edges_df["src"], edges_df["dst"])
                        if a in s_index and b in s_index], dtype=np.int64).reshape(-1, 2)
    t_edges = proximity_edges(tgt["lat"].values, tgt["lon"].values, dc["proxy_radius_m"])
    s_struct, t_struct = structural_features(s_edges, n_s), structural_features(t_edges, n_t)

    # contexts for STAF
    def vocab(col):
        if col not in src.columns:
            return ["<none>"], np.zeros(n_s, dtype=np.int64)
        v = sorted(src[col].astype(str).unique())
        return v, src[col].astype(str).map({k: i for i, k in enumerate(v)}).values

    sp_vocab, sp_ids = vocab("spatial_ctx")
    tm_vocab, tm_ids = vocab("temporal_ctx")

    # visual features
    paths = [os.path.join(dc["dair_dir"], p) for p in frames["image_path"]]
    miss = [p for p in paths if not os.path.isfile(p)]
    if miss:
        raise FileNotFoundError(f"{len(miss)} image files not found, e.g. {miss[0]}")
    print(f"extracting ResNet-152 features for {len(paths)} frames ...")
    f_feat = extract_image_features(paths, cfg, device)
    vis_dim = f_feat.size(1)

    # target: average over the entity's own frames
    t_vis = torch.zeros(n_t, vis_dim)
    t_mask = torch.zeros(n_t, dtype=torch.bool)
    ent = frames["entity_id"].map(t_index)
    for j, idxs in pd.Series(np.arange(len(frames))).groupby(ent.values):
        if not np.isnan(j):
            t_vis[int(j)] = f_feat[torch.as_tensor(idxs.to_numpy().copy())].mean(0)
            t_mask[int(j)] = True

    # source: k nearest frames within r_img (optionally excluding frames of the gold target)
    gold = {}
    for s, t in pairs.tolist():
        gold.setdefault(s, set()).add(t)
    ftree = _tree(frames["lat"].values, frames["lon"].values)
    ind, dist = ftree.query_radius(np.radians(src[["lat", "lon"]].values),
                                   r=_rad(dc["image_radius_m"]), return_distance=True)
    s_vis = torch.zeros(n_s, vis_dim)
    s_mask = torch.zeros(n_s, dtype=torch.bool)
    ent_np = ent.values
    for i, (js, ds) in enumerate(zip(ind, dist)):
        order = np.argsort(ds)
        js = js[order]
        if dc["exclude_matched_frames"] and i in gold:
            js = np.array([j for j in js if ent_np[j] not in gold[i]], dtype=np.int64)
        js = js[:dc["image_k"]]
        if len(js):
            s_vis[i] = f_feat[js].mean(0)
            s_mask[i] = True
    print(f"entities without visual evidence: source {int((~s_mask).sum())}/{n_s}, "
          f"target {int((~t_mask).sum())}/{n_t}")

    # text and literal features
    print("extracting BERT features ...")
    s_text = extract_text_features(src["name"].astype(str).tolist(), cfg, device)
    t_text = extract_text_features(tgt["name"].astype(str).tolist(), cfg, device)
    m_lit = normalized_levenshtein_matrix(src["name"].astype(str), tgt["name"].astype(str))

    tr, va, te = split_pairs(pairs, dc["split"], dc["split_seed"])
    bench = {
        "s_struct": torch.from_numpy(s_struct), "t_struct": torch.from_numpy(t_struct),
        "s_edges": torch.from_numpy(s_edges.T.copy()), "t_edges": torch.from_numpy(t_edges.T.copy()),
        "s_text": s_text, "t_text": t_text, "s_vis": s_vis, "t_vis": t_vis,
        "s_vis_mask": s_mask, "t_vis_mask": t_mask, "m_lit": torch.from_numpy(m_lit),
        "s_ctx": torch.from_numpy(np.stack([sp_ids, tm_ids], 1).astype(np.int64)),
        "pairs": torch.from_numpy(pairs), "split_train": tr, "split_val": va, "split_test": te,
        "spatial_vocab": sp_vocab, "temporal_vocab": tm_vocab,
        "s_ids": [str(x) for x in src["id"]], "t_ids": [str(x) for x in tgt["id"]],
    }
    out = os.path.join(dc["dir"], "benchmark.pt")
    torch.save(bench, out)
    print(f"{n_s} source / {n_t} target entities, {len(pairs)} pairs "
          f"(train {len(tr)}, val {len(va)}, test {len(te)}) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["candidates", "build"])
    ap.add_argument("--config", default="configs/dgstaf.yaml")
    ap.add_argument("--data_dir", default=None, help="override data.dir (e.g. data/synchronized_s1)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="config overrides, e.g. --set model.use_dcg=false train.epochs=50")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    {"candidates": cmd_candidates, "build": cmd_build}[args.command](cfg)


if __name__ == "__main__":
    main()
