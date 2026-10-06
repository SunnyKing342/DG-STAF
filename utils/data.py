import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, items):
    """Apply `section.key=value` overrides (values parsed as YAML), e.g. model.use_dcg=false."""
    for it in items or []:
        key, val = it.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dense_adj(edges, n, device):
    adj = torch.eye(n, dtype=torch.bool)
    if edges.numel() > 0:
        adj[edges[0], edges[1]] = True
        adj[edges[1], edges[0]] = True
    return adj.to(device)


def load_benchmark(path, device="cpu"):
    """Load the benchmark written by scripts/synchronize_data.py and move it to `device`."""
    if os.path.isdir(path):
        path = os.path.join(path, "benchmark.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Download the datasets into data/ and run "
            "`python scripts/synchronize_data.py build` first (see README).")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    d = {}
    for k, v in raw.items():
        d[k] = v.to(device) if torch.is_tensor(v) else v
    d["s_adj"] = _dense_adj(raw["s_edges"], raw["s_struct"].size(0), device)
    d["t_adj"] = _dense_adj(raw["t_edges"], raw["t_struct"].size(0), device)
    d["n_s"], d["n_t"] = raw["s_struct"].size(0), raw["t_struct"].size(0)
    return d


def iterate_batches(pair_idx, batch_size, shuffle, generator=None, min_size=2):
    """Yield index tensors over `pair_idx`; a trailing batch smaller than `min_size` is dropped."""
    idx = pair_idx[torch.randperm(len(pair_idx), generator=generator)] if shuffle else pair_idx
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        if len(b) >= min_size:
            yield b
