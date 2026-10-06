# DG-STAF
Official PyTorch implementation of **DG-STAF: A Spatio-Temporal Adaptive Fusion for Cross-Source Multimodal Entity Alignment in ITS Data Management**

This repository contains the DG-STAF model (TR-BC, DCG, STAF and Sinkhorn alignment), the benchmark construction pipeline, and the training and evaluation scripts. Raw datasets are not redistributed.


## 📋 Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```
## 🌱 Repository Structure
```
DG-STAF/
├── data/                     # Raw & synchronized benchmark (empty, filled after download)
│   ├── bjtt/                 # Raw BjTT dataset
│   ├── dair-v2x/             # Raw DAIR-V2X dataset
│   └── synchronized/         # Final aligned benchmark
├── scripts/                  # Data synchronization
│   └── synchronize_data.py
├── configs/                  # Model configurations
│   └── dgstaf.yaml
├── models/                   # DG-STAF core architecture
│   ├── encoders.py           # GAT / BERT / ResNet heads
│   ├── gated_fusion.py       # Dynamic Confidence Gating (DCG)
│   ├── spatio_temporal.py    # Spatio-Temporal Adaptive Fusion (STAF)
│   ├── alignment.py          # Top-K smoothing, Sinkhorn OT, reciprocal verification
│   └── dgstaf.py             # Full model, loss and evaluation
├── utils/                    # TR-BC distance, metrics & helper functions
├── train.py                  # Training script
├── test.py                   # Evaluation script
├── requirements.txt          # Dependencies
└── README.md
```

## 📂 Dataset & Synchronization
This work uses the BjTT+DAIR-V2X benchmark. We provide a pipeline to build the aligned benchmark from the two datasets.
Download the raw datasets:

BjTT:  https://github.com/ChyaZhang/BjTT

DAIR-V2X: https://github.com/AIR-THU/DAIR-V2X

Place them (converted to the CSV files below) in `data/`:

| File | Columns |
|---|---|
| `data/bjtt/entities.csv` | `id, name, lat, lon` (optional: `spatial_ctx, temporal_ctx, timestamp`) |
| `data/bjtt/edges.csv` | `src, dst` |
| `data/dair-v2x/entities.csv` | `id, name, lat, lon` (optional: `timestamp`) |
| `data/dair-v2x/frames.csv` | `frame_id, entity_id, lat, lon, image_path` |
| `data/synchronized/pairs.csv` | `src_id, tgt_id` (manually verified ground truth) |

Generate candidate pairs (5 m / 60 s) for manual verification, then build the benchmark:
```bash
python scripts/synchronize_data.py candidates
python scripts/synchronize_data.py build
```
The aligned benchmark will be saved to data/synchronized/.

## 🚀 Training & Evaluation
Train the full DG-STAF model (one run per seed in the config):
```bash
python train.py --config configs/dgstaf.yaml
```
Test and reproduce the results:

```bash
python test.py --config configs/dgstaf.yaml
```
Ablations and other settings can be run by overriding the config:
```bash
python train.py --set model.use_dcg=false                 # w/o DCG
python train.py --set model.use_staf=false                # w/o STAF
python train.py --set "model.modalities=[s,w,l]"          # w/o visual modality
```

## 📊 Main Results
Performance on the BjTT+DAIR-V2X benchmark:

| Model          | Hits@1 | Hits@10 | MRR   |
|----------------|--------|---------|-------|
| DG-STAF (Ours) | 0.948    | 0.975     | 0.961   |


## 📄 Citation
If you find this work useful, please cite our paper:
```
@article{ghaffar2026dgstaf,
  title={DG-STAF: A Spatio-Temporal Adaptive Fusion for Cross-Source Multimodal Entity Alignment in ITS Data Management},
  author={Ghaffar, Muhammad Arslan and Zhang, Kangshuai and Pan, Nuo and Peng, Lei},
  year={2026}
}
```
