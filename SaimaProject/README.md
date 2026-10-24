# Multimodal Breast Cancer Classification

A multimodal deep learning system for breast cancer classification using the **CBIS-DDSM** dataset. Combines mammography images (DenseNet201) with clinical tabular features via bidirectional cross-attention fusion.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **71.4%** |
| Val Accuracy | 80.8% |
| Dataset | CBIS-DDSM (mass cases) |
| Classes | MALIGNANT / BENIGN / BENIGN_WITHOUT_CALLBACK |

> Literature baseline for 3-class CBIS-DDSM: 55–72%. This model sits at the top of that range with no label leakage.

## Architecture

```
Mammogram Image (320×320)          Clinical Features (45-dim)
        ↓                                    ↓
  DenseNet201 backbone              FC → BN → ReLU → Dropout
  (ImageNet pretrained)             (Clinical MLP)
  → 512-dim features                → 256-dim features
        ↓                                    ↓
        └──────── Cross-Attention Fusion ────┘
                          ↓
                    Fusion MLP (256-dim)
                          ↓
               3-class Softmax Output
```

**Key design choices:**
- Layer-wise LR: backbone at `1e-5`, head at `1e-4` (prevents destroying pretrained features)
- 30-epoch image-only warmup before multimodal training
- `WarmupAwareEarlyStopping` — ignores early stop during warmup
- Weighted CrossEntropyLoss (balanced class weights)
- Clinical dropout (30%) during training to prevent over-reliance on tabular features

## Project Structure

```
RoomaProject/
├── src/
│   ├── train_with_images.py        # Main training script
│   ├── evaluate_checkpoint.py      # Checkpoint evaluation + per-class metrics
│   ├── multimodal_framework.py     # Model architecture
│   ├── clinical_preprocessor.py   # Clinical feature encoding
│   ├── xai/
│   │   └── run_xai.py             # Grad-CAM + attention + clinical importance
│   └── preprocessing/
│       ├── dicom_to_png.py        # DICOM → PNG conversion
│       └── download_data.py       # Dataset download utilities
├── data/                           # NOT in repo (168GB — download separately)
│   └── processed/csv/
│       ├── mass_train_mapped.csv   # Train set with image_path column
│       ├── mass_test_mapped.csv    # Test set with image_path column
│       ├── mass_train_encoded.csv  # 45-dim clinical features (train)
│       └── mass_test_encoded.csv  # 45-dim clinical features (test)
├── checkpoints/                    # NOT in repo (large .ckpt files)
├── results/
│   ├── diagrams/methodology.drawio # Architecture diagram
│   └── logs/                       # Training CSV logs
└── requirements.txt
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/RoomaProject.git
cd RoomaProject
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download CBIS-DDSM dataset

Download from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).

Place images under `data/` and run preprocessing:

```bash
python3 src/preprocessing/dicom_to_png.py
python3 src/clinical_preprocessor.py
```

### 3. Train

```bash
python3 src/train_with_images.py
```

Training config (in `train_with_images.py`):
- `max_epochs`: 200
- `batch_size`: 12 (320×320 images)
- `lr`: 1e-4 (head), 1e-5 (backbone)
- `warmup_image_only_epochs`: 30
- `early_stop_patience`: 30

### 4. Evaluate

```bash
# Single checkpoint
python3 src/evaluate_checkpoint.py \
    --checkpoint checkpoints/multimodal_real_images/best.ckpt

# All checkpoints ranked
python3 src/evaluate_checkpoint.py \
    --checkpoint_dir checkpoints/multimodal_real_images

# All checkpoints + pairwise McNemar significance tests + auto markdown report
python3 src/evaluate_checkpoint.py \
    --checkpoint_dir checkpoints/multimodal_real_images \
    --significance_out results/mcnemar_significance.json
```

When evaluating multiple checkpoints, the script automatically generates:
- **Compact markdown report** (`results/significance_report.md`) comparing best vs. baseline model with McNemar test result and interpretation.
- **JSON file** with pairwise McNemar results if `--significance_out` is provided.

Bootstrap confidence intervals (95%) for accuracy, precision, recall, and F1 are computed by default.
You can control resampling settings:

```bash
python3 src/evaluate_checkpoint.py \
    --checkpoint checkpoints/multimodal_real_images/best.ckpt \
    --n_bootstrap 3000 \
    --ci_alpha 0.05 \
    --bootstrap_seed 42
```

### 4.1 Objective O6 Validation (Sensitivity, Specificity, AUC)

The evaluator now reports:
- Macro sensitivity (`recall_macro`)
- Macro specificity (`specificity_macro`)
- AUC (`auc_macro_ovr`, `auc_weighted_ovr`)
- Clinical O6 binary metrics for MALIGNANT vs REST:
  - `malignant_binary_sensitivity`
  - `malignant_binary_specificity`
  - `malignant_binary_auc`

Run O6 validation and save machine-readable outputs:

```bash
python3 src/evaluate_checkpoint.py \
    --checkpoint checkpoints/multimodal_real_images/best.ckpt \
    --dataset_name CBIS-DDSM \
    --benchmark_json results/benchmark_sota.json \
    --metrics_out results/o6_metrics_cbis.json \
    --comparison_out results/o6_benchmark_comparison_cbis.json
```

For TCGA experiments (after preparing TCGA-compatible mapped/encoded CSVs), run the same with:

```bash
python3 src/evaluate_checkpoint.py \
    --checkpoint checkpoints/multimodal_real_images/best.ckpt \
    --dataset_name TCGA \
    --benchmark_json results/benchmark_sota.json \
    --metrics_out results/o6_metrics_tcga.json \
    --comparison_out results/o6_benchmark_comparison_tcga.json
```

Update benchmark references in `results/benchmark_sota.json` with real cited values before final reporting.

### 5. XAI Analysis

```bash
python3 src/xai/run_xai.py \
    --checkpoint checkpoints/multimodal_real_images/best.ckpt \
    --n_samples 20
```

Outputs:
- `results/xai/gradcam/` — Grad-CAM heatmaps overlaid on mammograms
- `results/xai/attention/` — Cross-attention weight charts
- `results/xai/clinical_importance.png` — Clinical feature importance (gradient-based)

## Multi-Seed Training (Mean +- Std)

Run the same training with multiple random seeds and aggregate test metrics as mean +- standard deviation:

```bash
python3 src/train_with_images.py \
    --seeds 42,43,44,45,46 \
    --multi_seed_summary_out results/multi_seed_summary.json
```

Recommended: use 3-5 seeds for robust reporting.

## Requirements

```
torch
torchvision
pytorch-lightning
albumentations
Pillow
numpy
pandas
scikit-learn
matplotlib
```

See `requirements.txt` for pinned versions.

## Data Note

CBIS-DDSM images are **not included** in this repository due to size (168GB) and TCIA licensing. Download directly from TCIA. Clinical CSVs (`mass_train_mapped.csv`, `mass_test_mapped.csv`) are also excluded as they contain absolute local paths — regenerate using the preprocessing scripts after downloading images.

## Label Leakage Note

Earlier model runs achieved ~97.88% accuracy due to `pathology_*` one-hot columns leaking the target label into clinical features. This has been fixed — `CLINICAL_COLS` now explicitly excludes all `pathology_*` columns. The 71.4% result is the honest, leakage-free performance.

## Data Leakage Analysis (Documented Artifacts)

This repository now includes explicit leakage-analysis artifacts for DenseNet-based experiments.

### Leakage run used for analysis

- Checkpoint: `checkpoints/multimodal_real_images/epoch=00-val_acc=0.046-val_loss=1.119.ckpt`
- Test accuracy: `10.85%`
- Macro precision: `22.37%`
- Macro recall: `34.02%`
- Macro F1: `7.37%`

Confusion matrix (rows=true, cols=pred):

```
[[  0   3 144]
 [  0   4 190]
 [  0   0  37]]
```

### Generated figures

- Raw confusion matrix image:
    - `results/diagrams/confusion_matrix_densenet_leakage.png`
- Normalized confusion matrix (% per true class):
    - `results/diagrams/confusion_matrix_densenet_leakage_normalized.png`
- Leakage training progress graph:
    - `results/diagrams/accuracy_graph_leakage_progress.png`

### How to regenerate

```bash
# Confusion matrix (raw + normalized)
python src/evaluate_checkpoint.py \
    --checkpoint checkpoints/multimodal_real_images/epoch=00-val_acc=0.046-val_loss=1.119.ckpt \
    --cm_out results/diagrams/confusion_matrix_densenet_leakage.png \
    --cm_norm_out results/diagrams/confusion_matrix_densenet_leakage_normalized.png
```

```bash
# Leakage progress graph from logged runs (version_0, version_1)
python -c "import pandas as pd, matplotlib.pyplot as plt; from pathlib import Path; base=Path('results/logs/multimodal_real_images/multimodal_real'); versions=['version_0','version_1']; run_idx=[]; train_acc=[]; val_acc=[]; [ (run_idx.append(i), train_acc.append(float(pd.read_csv(base/v/'metrics.csv')['train_acc'].dropna().iloc[-1])), val_acc.append(float(pd.read_csv(base/v/'metrics.csv')['val_acc'].dropna().iloc[-1])) ) for i,v in enumerate(versions) if len(pd.read_csv(base/v/'metrics.csv')['train_acc'].dropna()) and len(pd.read_csv(base/v/'metrics.csv')['val_acc'].dropna()) ]; fig,ax=plt.subplots(figsize=(8,5)); ax.plot(run_idx,train_acc,marker='o',label='Train Accuracy'); ax.plot(run_idx,val_acc,marker='o',label='Validation Accuracy'); ax.set_xticks(run_idx); ax.set_xticklabels([versions[i] for i in run_idx]); ax.set_ylim(0,1); ax.set_xlabel('Leakage Training Run Progress'); ax.set_ylabel('Accuracy'); ax.set_title('Leakage Training Progress (DenseNet)'); ax.grid(alpha=0.25); ax.legend(); out=Path('results/diagrams/accuracy_graph_leakage_progress.png'); out.parent.mkdir(parents=True, exist_ok=True); plt.tight_layout(); plt.savefig(out, dpi=200)"
```

Note: current leakage logs contain only early runs (epoch 0 checkpoints). For a full epoch-wise leakage curve, run leakage training for more epochs so `metrics.csv` contains multiple epoch entries.

## Citation

```
@misc{cbis-ddsm,
  title   = {CBIS-DDSM: Curated Breast Imaging Subset of DDSM},
  author  = {Lee, Rebecca Sawyer and Gimenez, Francisco and Hoogi, Assaf and Rubin, Daniel},
  journal = {Scientific Data},
  year    = {2017}
}
```
