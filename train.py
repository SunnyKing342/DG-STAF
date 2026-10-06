import argparse
import json
import os

import torch

from models import DGSTAF
from utils.data import load_config, apply_overrides, set_seed, load_benchmark, iterate_batches


def train_one(cfg, seed, device, verbose=True):
    set_seed(seed)
    t = cfg["train"]
    d = load_benchmark(cfg["data"]["dir"], device)
    model = DGSTAF(cfg, len(d["spatial_vocab"]), len(d["temporal_vocab"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr_max"], betas=(0.9, 0.999),
                           weight_decay=t["weight_decay"])
    train_idx, val_idx = d["split_train"], d["split_val"]
    steps = max(1, (len(train_idx) + t["batch_size"] - 1) // t["batch_size"])
    sched = torch.optim.lr_scheduler.CyclicLR(
        opt, base_lr=t["lr_base"], max_lr=t["lr_max"], step_size_up=steps * t["cycle_epochs"],
        mode="triangular", cycle_momentum=False)
    gen = torch.Generator().manual_seed(seed)
    os.makedirs(t["save_dir"], exist_ok=True)
    os.makedirs(t["log_dir"], exist_ok=True)
    ckpt_path = os.path.join(t["save_dir"], f"seed{seed}.pth")
    best, bad, history = -1.0, 0, []

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        tot, nb = 0.0, 0
        for b in iterate_batches(train_idx, t["batch_size"], True, gen):
            pairs = d["pairs"][b.to(d["pairs"].device)]
            loss, parts = model.training_loss(d, pairs[:, 0], pairs[:, 1], t["gamma"], t["mu"],
                                              t["num_neg"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        rec = {"epoch": epoch, "loss": tot / max(nb, 1), "omega": parts["omega"]}
        if epoch % t["eval_interval"] == 0:
            res = model.evaluate(d, val_idx.to(d["pairs"].device), use_ot=False)
            rec.update({"val_mrr": res["mrr"], "val_hits@1": res["hits@1"],
                        "val_hits@10": res["hits@10"]})
            if res["mrr"] > best:
                best, bad = res["mrr"], 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "seed": seed,
                            "n_spatial": len(d["spatial_vocab"]),
                            "n_temporal": len(d["temporal_vocab"]), "epoch": epoch,
                            "val_mrr": best}, ckpt_path)
            else:
                bad += 1
        history.append(rec)
        if verbose:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in rec.items() if k != "omega"}))
        if bad >= t["patience"]:
            if verbose:
                print(f"early stopping at epoch {epoch}")
            break
    with open(os.path.join(t["log_dir"], f"train_seed{seed}.json"), "w") as f:
        json.dump(history, f)
    print(f"seed {seed}: best val MRR {best:.4f} -> {ckpt_path}")
    return ckpt_path, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dgstaf.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override the seeds listed in the config (one run per seed)")
    ap.add_argument("--data_dir", default=None, help="override data.dir (e.g. a BjTT-S1 build)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="config overrides, e.g. --set model.use_dcg=false train.epochs=50")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    for seed in (args.seeds if args.seeds else cfg["seeds"]):
        train_one(cfg, seed, device)


if __name__ == "__main__":
    main()
