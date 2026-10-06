# DG-STAF
PyTorch implementation of **DG-STAF: A Spatio-Temporal Adaptive Fusion for Cross-Source Multimodal
Entity Alignment in ITS Data Management** (BjTT -> DAIR-V2X).

The repository contains the model (TR-BC, DCG, STAF, Sinkhorn alignment), the benchmark construction
script and the training / evaluation scripts. **No raw data, annotations or checkpoints are
redistributed**: `data/` is empty and must be filled from the official sources.

## Layout
```
configs/dgstaf.yaml          all hyper-parameters ([paper] vs [choice] marked inside)
data/{bjtt,dair-v2x,synchronized}/   empty; filled by you (see below)
models/   encoders.py (GAT, BERT head, ResNet head) | gated_fusion.py (DCG) |
          spatio_temporal.py (STAF) | alignment.py (smoothing, Sinkhorn, reciprocal check) |
          dgstaf.py (full model, loss, evaluation)
utils/    distance.py (TR-BC, Normalized Levenshtein) | metrics.py | data.py
scripts/synchronize_data.py  candidate generation + benchmark build
train.py, test.py
```

## Install
```bash
pip install -r requirements.txt
```

## Data
Download BjTT (https://github.com/ChyaZhang/BjTT) and DAIR-V2X (https://github.com/AIR-THU/DAIR-V2X)
and convert them to the following CSVs (UTF-8). The converters are dataset specific and are not
included; the schema below is what the code reads.

| file | columns |
|---|---|
| `data/bjtt/entities.csv` | `id,name,lat,lon` + optional `spatial_ctx,temporal_ctx,timestamp` |
| `data/bjtt/edges.csv` | `src,dst` (road topology of G_S) |
| `data/dair-v2x/entities.csv` | `id,name,lat,lon` + optional `timestamp` (`name` = RSU / sensor tag string) |
| `data/dair-v2x/frames.csv` | `frame_id,entity_id,lat,lon,image_path` (path relative to `data/dair-v2x/`; `entity_id` may be empty) |
| `data/synchronized/pairs.csv` | `src_id,tgt_id` - the manually verified ground-truth pairs |

`spatial_ctx` / `temporal_ctx` are the discrete S and T of STAF (e.g. intersection / arterial /
residential, rush / night / off-peak). If absent, all entities share one context.

```bash
python scripts/synchronize_data.py candidates   # optional: 5 m (+60 s) candidate pairs to verify by hand
python scripts/synchronize_data.py build        # graphs, ResNet-152 / BERT features, M_L, splits -> data/synchronized/benchmark.pt
```
For the three subsets build each one into its own folder, e.g. `--data_dir data/s1` (put that
subset's `pairs.csv` there) and pass the same `--data_dir` to `train.py` / `test.py`.

## Train and evaluate
```bash
python train.py                         # one run per seed in configs/dgstaf.yaml (5 seeds)
python test.py                          # test split of every checkpoint, mean +- std, + DCG / STAF weights
python test.py --split val --no_ot
```
Any config value can be overridden, which is how the ablations are run:
```bash
python train.py --set model.use_dcg=false                 # w/o DCG   (uniform gating weights)
python train.py --set model.use_staf=false                # w/o STAF  (uniform fusion weights)
python train.py --set "model.modalities=[s,w,l]"          # w/o visual
python train.py --set data.exclude_matched_frames=true    # circularity control (rebuild first)
```
Use a different `train.save_dir` per setting so checkpoints are not overwritten.

## Evaluation protocol implemented
Entity-level 70/15/15 split (pairs sharing an entity stay together); ranking over the **full**
target entity set; Hits@1, Hits@10, MRR from (i) the smoothed similarity and (ii) the Sinkhorn
plan (`ot_*`); ties are counted pessimistically; reciprocal-verification precision/recall are
also reported. Model selection uses validation MRR with early stopping.

## Where this differs from, or is not specified by, the paper
Please read this before reporting numbers.
1. **Frozen backbones.** BERT and ResNet-152 features are extracted once; only the heads
   (text projection, visual MLP) are trained. The paper fine-tunes the last two layers of both
   backbones - not implemented.
2. **Bounded embeddings.** Embeddings go through a sigmoid so that Bray-Curtis is defined and
   the maximum variance of x - y in TR-BC is 1 (`sigma2_max = 1`).
3. **Structural input.** The GAT input features are graph-only descriptors (degree, mean
   neighbour degree, clustering, 2-hop size), not GPS, so coordinates cannot leak into the
   structural modality. The paper does not specify this input.
4. **Training statistics.** DCG variances are computed on the B x B batch matrix during
   training and on the full N_S x N_T matrix at inference. Negatives are random in-batch
   targets; the hinge loss is averaged (not summed) over pairs. STAF uses the *source*
   entity's (S, T) context for each row.
5. **Unspecified values** (BERT checkpoint, cyclic-LR bounds, Sinkhorn iterations, STAF
   dimensions, early-stopping patience, seeds, ...) are marked `[choice]` in the config.
6. **The DCG neighbour count k' = 10** listed in the paper's hyper-parameter table is not used:
   the DCG as written uses the *global* matrix variance.

## Two properties of the paper's formulation to be aware of
* **DCG scale.** `omega = softmax(sigma^2 / tau)` with `tau = 0.5` and distances in [0, ~1] gives
  variances of order 1e-3..1e-1, so the weights stay close to uniform (in our synthetic smoke
  tests omega stayed within 0.23-0.27). Large weight differences (e.g. 0.82 vs 0.08) require much
  larger variances or a smaller tau. `test.py` prints `sigma2` and `omega` for every run; check
  them against the numbers you report. `model.dcg_var_scale` rescales sigma^2 if you want to study this.
* **Top-K smoothing.** The second term of `S~ = beta S + (1-beta)/K * sum_{k in N_K(i)} S(i,k)` does
  not depend on j, so it only shifts each row by a constant: rankings are unchanged and the
  effect on Sinkhorn is limited to the `beta` rescaling. It is implemented exactly as written.

## Tests performed
All components were verified on synthetic data only (TR-BC against a naive reference, Sinkhorn
marginals, metric definitions, GAT masking, end-to-end build -> train -> test, ablation switches).
The real BjTT / DAIR-V2X pipeline has not been run, and this repository claims no result of the paper.

## License
MIT (see `LICENSE`).
