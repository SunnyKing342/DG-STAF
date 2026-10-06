import argparse
import glob
import json
import os

import numpy as np
import torch

from models import DGSTAF
from utils.data import load_config, apply_overrides, load_benchmark

KEYS = ["hits@1", "hits@10", "mrr", "ot_hits@1", "ot_hits@10", "ot_mrr",
        "recip_precision", "recip_recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dgstaf.yaml")
    ap.add_argument("--checkpoint", nargs="*", default=None,
                    help="checkpoint file(s); default: every <save_dir>/seed*.pth")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--no_ot", action="store_true", help="skip Sinkhorn / reciprocal evaluation")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="config overrides, e.g. --set model.use_dcg=false train.epochs=50")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    paths = args.checkpoint or sorted(glob.glob(os.path.join(cfg["train"]["save_dir"], "seed*.pth")))
    if not paths:
        raise FileNotFoundError("no checkpoints found; run train.py first")
    d = load_benchmark(cfg["data"]["dir"], device)
    idx = d[f"split_{args.split}"].to(device)

    runs = []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        model = DGSTAF(ck["cfg"], ck["n_spatial"], ck["n_temporal"]).to(device)
        model.load_state_dict(ck["model"])
        res = model.evaluate(d, idx, use_ot=not args.no_ot)
        runs.append(res)
        print(f"{os.path.basename(p)}  " + "  ".join(
            f"{k}={res[k]:.4f}" for k in KEYS if k in res))
        print(f"   DCG omega (s,w,v,l)={[round(x, 3) for x in res['omega']]}  "
              f"sigma2={[round(x, 5) for x in res['sigma2']]}  "
              f"mean STAF w={[round(x, 3) for x in res['mean_w']]}")

    summary = {}
    for k in KEYS:
        if k in runs[0]:
            v = np.array([r[k] for r in runs])
            summary[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    print(f"\n{args.split} split, {len(runs)} run(s), {len(idx)} pairs, ranking over "
          f"{d['n_t']} target entities")
    for k, v in summary.items():
        print(f"  {k:16s} {v['mean']:.4f} +- {v['std']:.4f}")
    os.makedirs(cfg["train"]["log_dir"], exist_ok=True)
    with open(os.path.join(cfg["train"]["log_dir"], f"test_{args.split}.json"), "w") as f:
        json.dump({"summary": summary, "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
