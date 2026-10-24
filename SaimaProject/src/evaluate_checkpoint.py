"""
Evaluate checkpoints/multimodal_real_images/*.ckpt on the held-out test set.

This script mirrors train_with_images.py exactly:
  - LABEL_MAP: MALIGNANT=0, BENIGN=1, BENIGN_WITHOUT_CALLBACK=2
  - Test set: data/processed/csv/mass_test_mapped.csv  (separate file, no split needed)
  - Clinical features: 48-col encoded CSV joined on patient_id
  - Dataset class: MappedDataset (image_path from mapped CSV)
  - Model class: MultimodalLit from train_with_images.py

Usage (from project root):
    python src/evaluate_checkpoint.py \
        --checkpoint checkpoints/multimodal_real_images/epoch=135-val_loss=0.038-val_acc=0.980.ckpt

    # All checkpoints ranked:
    python src/evaluate_checkpoint.py \
        --checkpoint_dir checkpoints/multimodal_real_images
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import binomtest, chi2, spearmanr
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

# Import everything from the actual training script so we get exact classes
from train_with_images import (
    MappedDataset,
    MultimodalLit,
    LABEL_MAP,
    CLINICAL_COLS,
    CFG,
    get_val_aug,
)
from multimodal_framework import MultimodalClassifier, create_model_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Invert label map for display
CLASS_NAMES = [k for k, v in sorted(LABEL_MAP.items(), key=lambda x: x[1])]

TRAIN_CSV   = Path("data/processed/csv/mass_train_mapped.csv")
TEST_CSV    = Path("data/processed/csv/mass_test_mapped.csv")
ENC_TRAIN   = Path("data/processed/csv/mass_train_encoded.csv")
ENC_TEST    = Path("data/processed/csv/mass_test_encoded.csv")
DEFAULT_BENCHMARKS = Path("results/benchmark_sota.json")


def save_confusion_matrix_image(cm, class_names, out_path: Path, title: str):
    """Save confusion matrix heatmap as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def save_confusion_matrix_normalized_image(cm, class_names, out_path: Path, title: str):
    """Save row-normalized confusion matrix heatmap as PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm, dtype=np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.1f',
        cmap='Greens',
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def build_test_loader(batch_size=32):
    test_df  = pd.read_csv(TEST_CSV)
    enc_test = pd.read_csv(ENC_TEST)
    # Combine encodings so patient_id lookup works for both splits
    enc_train = pd.read_csv(ENC_TRAIN)
    enc_all   = pd.concat([enc_train, enc_test], ignore_index=True)

    logger.info(f"Test set: {len(test_df)} samples")
    for cls, cnt in test_df['pathology'].value_counts().items():
        logger.info(f"  {cls:30s}: {cnt} ({100*cnt/len(test_df):.1f}%)")

    dataset = MappedDataset(
        df=test_df,
        encoded_df=enc_all,
        transform=get_val_aug(),
        augment=None,
        is_train=False,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )


def _normalize_birads(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()
    s = s.replace('BI-RADS', '').replace('BIRADS', '').replace('BI RADS', '').replace(':', '').strip()
    if not s:
        return np.nan
    if s.endswith('A'):
        s = s[:-1]
    if s.endswith('B'):
        s = s[:-1]
    if s.endswith('C'):
        s = s[:-1]
    digits = ''.join(ch for ch in s if ch.isdigit())
    if not digits:
        return np.nan
    val = int(digits)
    if val < 0:
        return np.nan
    if val > 6:
        val = 6
    return val


def _pred_to_birads(pred_idx: int, probs_row: np.ndarray) -> int:
    malignant_idx = LABEL_MAP.get('MALIGNANT', 0)
    benign_idx = LABEL_MAP.get('BENIGN', 1)
    benign_wo_idx = LABEL_MAP.get('BENIGN_WITHOUT_CALLBACK', 2)

    p_m = float(probs_row[malignant_idx])
    p_b = float(probs_row[benign_idx])
    p_bw = float(probs_row[benign_wo_idx])

    if pred_idx == malignant_idx:
        if p_m >= 0.85:
            return 5
        if p_m >= 0.65:
            return 4
        return 4

    if pred_idx == benign_wo_idx:
        if p_bw >= 0.70:
            return 2
        return 3

    if p_b >= 0.75:
        return 2
    return 3


def compute_birads_hierarchical_metrics(preds: np.ndarray, probs: np.ndarray):
    test_df = pd.read_csv(TEST_CSV)
    if len(test_df) != len(preds):
        raise ValueError(
            f"BI-RADS metric mismatch: len(test_df)={len(test_df)} vs len(preds)={len(preds)}"
        )
    if 'assessment' not in test_df.columns:
        return None

    true_birads = test_df['assessment'].apply(_normalize_birads).to_numpy()
    pred_birads = np.array([_pred_to_birads(int(preds[i]), probs[i]) for i in range(len(preds))], dtype=np.int32)

    valid_mask = ~np.isnan(true_birads)
    if valid_mask.sum() == 0:
        return None

    y_true = true_birads[valid_mask].astype(np.int32)
    y_pred = pred_birads[valid_mask].astype(np.int32)

    labels = [2, 3, 4, 5, 6]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))

    pathology_map = {
        'BENIGN': 0,
        'BENIGN_WITHOUT_CALLBACK': 0,
        'MALIGNANT': 1,
    }
    pathology_num = test_df.loc[valid_mask, 'pathology'].map(pathology_map).to_numpy(dtype=np.float32)
    corr, p_val = spearmanr(y_pred.astype(np.float32), pathology_num)

    return {
        'n_valid_birads': int(valid_mask.sum()),
        'birads_accuracy': acc,
        'birads_labels': labels,
        'birads_confusion_matrix': cm.tolist(),
        'birads_pred_pathology_spearman_corr': float(corr) if not np.isnan(corr) else np.nan,
        'birads_pred_pathology_spearman_pvalue': float(p_val) if not np.isnan(p_val) else np.nan,
    }


def save_birads_confusion_matrix_image(cm, labels, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Oranges',
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel('Predicted BI-RADS')
    ax.set_ylabel('True BI-RADS')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def load_model(checkpoint_path):
    logger.info(f"Loading: {checkpoint_path}")
    raw    = torch.load(checkpoint_path, map_location='cpu')
    hparams = raw.get('hyper_parameters', {})
    cfg    = hparams.get('cfg', CFG)  # fall back to default CFG if not saved

    # Build MultimodalClassifier with correct backbone key
    arch_cfg = {
        'backbone':          cfg.get('image_backbone', 'densenet201'),
        'clinical_input_dim': cfg.get('clinical_input_dim', len(CLINICAL_COLS)),
        'fusion_dim':        cfg.get('fusion_dim', 256),
        'num_classes':       cfg.get('num_classes', 3),
        'dropout':           cfg.get('dropout', 0.5),
    }
    base_model = MultimodalClassifier(arch_cfg)

    # Dummy weight tensor — only model.* keys matter for inference
    dummy_weights = torch.ones(cfg.get('num_classes', 3))
    lit = MultimodalLit(model=base_model, class_weights=dummy_weights, cfg=cfg)

    missing, unexpected = lit.load_state_dict(raw['state_dict'], strict=False)
    if [k for k in missing if 'criterion' not in k]:
        logger.warning(f"Missing non-criterion keys: {[k for k in missing if 'criterion' not in k][:5]}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}")

    lit.eval()
    return lit


def evaluate(model, dataloader, device):
    model.to(device)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            images   = batch['image'].to(device)
            clinical = batch['clinical_features'].to(device)
            labels   = batch['label']

            logits = model(images, clinical)
            probs = F.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_specificity_by_class(cm: np.ndarray):
    """Compute one-vs-rest specificity for each class from confusion matrix."""
    specificities = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specificities.append(float(tn / denom) if denom > 0 else np.nan)
    return np.array(specificities, dtype=np.float64)


def compute_auc_metrics(labels: np.ndarray, probs: np.ndarray):
    """Compute multiclass AUC metrics and per-class OvR AUC."""
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes, dtype=np.float32)[labels]

    auc_macro_ovr = np.nan
    auc_weighted_ovr = np.nan
    try:
        auc_macro_ovr = roc_auc_score(one_hot, probs, average='macro', multi_class='ovr')
        auc_weighted_ovr = roc_auc_score(one_hot, probs, average='weighted', multi_class='ovr')
    except ValueError:
        # Happens when not all classes appear in labels.
        pass

    per_class_auc = {}
    for i, name in enumerate(CLASS_NAMES):
        y_true_bin = (labels == i).astype(np.int32)
        if np.unique(y_true_bin).size < 2:
            per_class_auc[name] = np.nan
            continue
        per_class_auc[name] = float(roc_auc_score(y_true_bin, probs[:, i]))

    return float(auc_macro_ovr), float(auc_weighted_ovr), per_class_auc


def compute_malignant_binary_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray):
    """Compute malignant-vs-rest sensitivity, specificity, and AUC."""
    malignant_idx = LABEL_MAP.get('MALIGNANT')
    if malignant_idx is None:
        return {
            'malignant_binary_sensitivity': np.nan,
            'malignant_binary_specificity': np.nan,
            'malignant_binary_auc': np.nan,
        }

    y_true = (labels == malignant_idx).astype(np.int32)
    y_pred = (preds == malignant_idx).astype(np.int32)
    y_score = probs[:, malignant_idx]

    cm_bin = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_bin.ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    auc = np.nan
    if np.unique(y_true).size > 1:
        auc = float(roc_auc_score(y_true, y_score))

    return {
        'malignant_binary_sensitivity': sensitivity,
        'malignant_binary_specificity': specificity,
        'malignant_binary_auc': auc,
    }


def bootstrap_metric_cis(
    labels: np.ndarray,
    preds: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_seed: int = 42,
):
    """Compute bootstrap CIs for accuracy/precision/recall/F1 (macro)."""
    n = len(labels)
    if n == 0:
        return {}

    rng = np.random.default_rng(random_seed)
    stats = {
        'accuracy': np.zeros(n_bootstrap, dtype=np.float64),
        'precision_macro': np.zeros(n_bootstrap, dtype=np.float64),
        'recall_macro': np.zeros(n_bootstrap, dtype=np.float64),
        'f1_macro': np.zeros(n_bootstrap, dtype=np.float64),
    }

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true = labels[idx]
        y_pred = preds[idx]
        stats['accuracy'][i] = accuracy_score(y_true, y_pred)
        stats['precision_macro'][i] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        stats['recall_macro'][i] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        stats['f1_macro'][i] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    lo_q = 100 * (alpha / 2.0)
    hi_q = 100 * (1.0 - alpha / 2.0)
    ci_out = {}
    for key, vals in stats.items():
        ci_out[key] = {
            'mean_bootstrap': float(np.mean(vals)),
            'ci_lower': float(np.percentile(vals, lo_q)),
            'ci_upper': float(np.percentile(vals, hi_q)),
            'alpha': float(alpha),
            'n_bootstrap': int(n_bootstrap),
        }
    return ci_out


def generate_significance_markdown_report(
    results: list,
    significance_payload: list,
    out_path: Path | None = None,
) -> str:
    """
    Generate compact markdown report comparing best vs baseline models
    with McNemar significance test result.
    """
    if not results:
        return ""

    lines = [
        "# Statistical Significance Report\n",
        f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
    ]

    # Rank by accuracy
    ranked = sorted(results, key=lambda x: x['metrics']['accuracy'], reverse=True)
    best = ranked[0]
    baseline = ranked[-1]

    lines.append("## Model Rankings\n")
    lines.append("| Rank | Model | Accuracy | Precision (M) | Recall (M) | F1 (M) |\n")
    lines.append("|------|-------|----------|---------------|-----------|--------|\n")
    for rank, r in enumerate(ranked, 1):
        m = r['metrics']
        lines.append(
            f"| {rank} | {m['checkpoint']} | {m['accuracy']*100:.2f}% | "
            f"{m['precision_macro']*100:.2f}% | {m['recall_macro']*100:.2f}% | "
            f"{m['f1_macro']*100:.2f}% |\n"
        )

    lines.append("\n## Best vs Baseline Comparison\n")
    lines.append(f"**Best Model:** {best['metrics']['checkpoint']}\n")
    lines.append(f"- Accuracy: {best['metrics']['accuracy']*100:.2f}%\n")
    lines.append(f"- Precision (Macro): {best['metrics']['precision_macro']*100:.2f}%\n")
    lines.append(f"- Recall (Macro): {best['metrics']['recall_macro']*100:.2f}%\n")
    lines.append(f"- F1 (Macro): {best['metrics']['f1_macro']*100:.2f}%\n\n")

    lines.append(f"**Baseline Model:** {baseline['metrics']['checkpoint']}\n")
    lines.append(f"- Accuracy: {baseline['metrics']['accuracy']*100:.2f}%\n")
    lines.append(f"- Precision (Macro): {baseline['metrics']['precision_macro']*100:.2f}%\n")
    lines.append(f"- Recall (Macro): {baseline['metrics']['recall_macro']*100:.2f}%\n")
    lines.append(f"- F1 (Macro): {baseline['metrics']['f1_macro']*100:.2f}%\n\n")

    lines.append("**Accuracy Delta:** ")
    delta = best['metrics']['accuracy'] - baseline['metrics']['accuracy']
    lines.append(f"{delta:+.2f} ({delta*100:+.2f} percentage points)\n\n")

    # Find McNemar test result between best and baseline
    mcnemar_result = None
    for sig in significance_payload:
        if (
            (sig['model_a'] == best['metrics']['checkpoint']
             and sig['model_b'] == baseline['metrics']['checkpoint'])
            or (sig['model_a'] == baseline['metrics']['checkpoint']
                and sig['model_b'] == best['metrics']['checkpoint'])
        ):
            mcnemar_result = sig
            break

    if mcnemar_result:
        lines.append("## McNemar Significance Test\n")
        lines.append(
            f"**Test:** {mcnemar_result['model_a']} vs {mcnemar_result['model_b']}\n\n"
        )
        lines.append("| Metric | Value |\n")
        lines.append("|--------|-------|\n")
        lines.append(
            f"| Discordant pairs | {mcnemar_result['n_discordant']} |\n"
        )
        lines.append(
            f"| Model A correct, B wrong | {mcnemar_result['b_model_a_correct_model_b_wrong']} |\n"
        )
        lines.append(
            f"| Model A wrong, B correct | {mcnemar_result['c_model_a_wrong_model_b_correct']} |\n"
        )
        lines.append(
            f"| χ² (continuity-corrected) | {mcnemar_result['chi2_stat_cc']:.4f} |\n"
        )
        lines.append(
            f"| p-value (χ²) | {mcnemar_result['chi2_pvalue_cc']:.6f} |\n"
        )
        lines.append(
            f"| p-value (exact binomial) | {mcnemar_result['exact_binom_pvalue']:.6f} |\n\n"
        )
        alpha = 0.05
        is_sig = mcnemar_result['exact_binom_pvalue'] < alpha
        lines.append("### Interpretation (α=0.05)\n")
        if is_sig:
            lines.append(
                f"✓ **Statistically Significant** (p={mcnemar_result['exact_binom_pvalue']:.6f})\n\n"
            )
            lines.append(
                f"The best model ({best['metrics']['checkpoint']}) "
                f"is significantly different from the baseline at the 5% level.\n"
            )
        else:
            lines.append(
                f"✗ **Not Statistically Significant** (p={mcnemar_result['exact_binom_pvalue']:.6f})\n\n"
            )
            lines.append(
                f"The performance difference between best and baseline "
                f"is not statistically significant at the 5% level.\n"
            )
    else:
        lines.append("\n*Note: McNemar test not available for single model.*\n")

    if out_path:
        lines.append("\n---\n")
        lines.append(f"*Report saved to: {out_path}*\n")

    report = "".join(lines)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as f:
            f.write(report)

    return report


def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray):
    """Paired significance test comparing two classifiers on same samples."""
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c

    if n == 0:
        return {
            'b_model_a_correct_model_b_wrong': b,
            'c_model_a_wrong_model_b_correct': c,
            'n_discordant': n,
            'chi2_stat_cc': 0.0,
            'chi2_pvalue_cc': 1.0,
            'exact_binom_pvalue': 1.0,
        }

    chi2_stat = ((abs(b - c) - 1.0) ** 2) / n
    chi2_p = float(1.0 - chi2.cdf(chi2_stat, df=1))
    p_exact = float(binomtest(min(b, c), n=n, p=0.5, alternative='two-sided').pvalue)

    return {
        'b_model_a_correct_model_b_wrong': b,
        'c_model_a_wrong_model_b_correct': c,
        'n_discordant': n,
        'chi2_stat_cc': float(chi2_stat),
        'chi2_pvalue_cc': chi2_p,
        'exact_binom_pvalue': p_exact,
    }


def maybe_load_benchmarks(path: str | None):
    if not path:
        return None
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        logger.warning(f"Benchmark file not found: {benchmark_path}")
        return None
    with benchmark_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def compare_with_benchmark(metrics: dict, benchmark_data: dict | None, dataset_name: str):
    if not benchmark_data:
        return None

    dataset_entry = benchmark_data.get(dataset_name)
    if dataset_entry is None:
        logger.warning(f"No benchmark entry found for dataset '{dataset_name}'")
        return None

    benchmark_metrics = dataset_entry.get('metrics', {})
    comparisons = {}
    for key, benchmark_value in benchmark_metrics.items():
        benchmark_value = float(benchmark_value)
        model_value = metrics.get(key)
        if model_value is None or pd.isna(model_value):
            comparisons[key] = {
                'benchmark': benchmark_value,
                'model': None,
                'delta': None,
            }
            continue
        comparisons[key] = {
            'benchmark': float(benchmark_value),
            'model': float(model_value),
            'delta': float(model_value - benchmark_value),
        }

    return {
        'dataset': dataset_name,
        'source': dataset_entry.get('source', ''),
        'comparisons': comparisons,
    }


def print_report(
    preds,
    labels,
    probs,
    checkpoint_name,
    dataset_name,
    benchmark_data: dict | None = None,
    cm_out: Path | None = None,
    cm_norm_out: Path | None = None,
    birads_cm_out: Path | None = None,
    n_bootstrap: int = 2000,
    ci_alpha: float = 0.05,
    bootstrap_seed: int = 42,
):
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec  = recall_score(labels, preds, average='macro', zero_division=0)
    f1   = f1_score(labels, preds, average='macro', zero_division=0)
    cm   = confusion_matrix(labels, preds)
    specificity_per_class = compute_specificity_by_class(cm)
    specificity_macro = float(np.nanmean(specificity_per_class))
    auc_macro_ovr, auc_weighted_ovr, per_class_auc = compute_auc_metrics(labels, probs)
    malignant_metrics = compute_malignant_binary_metrics(labels, preds, probs)
    bootstrap_cis = bootstrap_metric_cis(
        labels=labels,
        preds=preds,
        n_bootstrap=n_bootstrap,
        alpha=ci_alpha,
        random_seed=bootstrap_seed,
    )

    birads_results = compute_birads_hierarchical_metrics(preds=preds, probs=probs)

    metrics = {
        'checkpoint': checkpoint_name,
        'dataset': dataset_name,
        'accuracy': float(acc),
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'f1_macro': float(f1),
        'specificity_macro': specificity_macro,
        'auc_macro_ovr': auc_macro_ovr,
        'auc_weighted_ovr': auc_weighted_ovr,
        'malignant_binary_sensitivity': malignant_metrics['malignant_binary_sensitivity'],
        'malignant_binary_specificity': malignant_metrics['malignant_binary_specificity'],
        'malignant_binary_auc': malignant_metrics['malignant_binary_auc'],
        'bootstrap_ci_95': bootstrap_cis,
        'birads_hierarchical': birads_results,
    }
    benchmark_comparison = compare_with_benchmark(metrics, benchmark_data, dataset_name)

    print("\n" + "=" * 60)
    print(f"CHECKPOINT : {checkpoint_name}")
    print(f"DATASET    : {dataset_name}")
    print(f"LABEL ORDER: {CLASS_NAMES}")
    print("=" * 60)
    print(f"  Test Accuracy  : {acc*100:.2f}%")
    print(f"  Macro Precision: {prec*100:.2f}%")
    print(f"  Macro Recall   : {rec*100:.2f}% (Sensitivity)")
    print(f"  Macro F1-Score : {f1*100:.2f}%")
    print(f"  Macro Specificity: {specificity_macro*100:.2f}%")
    print(f"  AUC (Macro OvR): {auc_macro_ovr*100:.2f}%")
    print(f"  AUC (Weighted OvR): {auc_weighted_ovr*100:.2f}%")
    print("  95% bootstrap confidence intervals:")
    print(
        "    Accuracy      : "
        f"[{bootstrap_cis['accuracy']['ci_lower']*100:.2f}, {bootstrap_cis['accuracy']['ci_upper']*100:.2f}]%"
    )
    print(
        "    Precision(M)  : "
        f"[{bootstrap_cis['precision_macro']['ci_lower']*100:.2f}, {bootstrap_cis['precision_macro']['ci_upper']*100:.2f}]%"
    )
    print(
        "    Recall(M)     : "
        f"[{bootstrap_cis['recall_macro']['ci_lower']*100:.2f}, {bootstrap_cis['recall_macro']['ci_upper']*100:.2f}]%"
    )
    print(
        "    F1(M)         : "
        f"[{bootstrap_cis['f1_macro']['ci_lower']*100:.2f}, {bootstrap_cis['f1_macro']['ci_upper']*100:.2f}]%"
    )
    print()
    print("Objective O6 clinical metrics (malignant vs rest):")
    print(f"  Sensitivity: {malignant_metrics['malignant_binary_sensitivity']*100:.2f}%")
    print(f"  Specificity: {malignant_metrics['malignant_binary_specificity']*100:.2f}%")
    print(f"  AUC        : {malignant_metrics['malignant_binary_auc']*100:.2f}%")
    print()
    print("Per-class report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))
    print("Per-class specificity:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {name:30s}: {specificity_per_class[idx]*100:.2f}%")
    print("Per-class AUC (OvR):")
    for name in CLASS_NAMES:
        print(f"  {name:30s}: {per_class_auc[name]*100:.2f}%")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    if birads_results:
        print()
        print("Hierarchical BI-RADS evaluation:")
        print(f"  Valid BI-RADS samples: {birads_results['n_valid_birads']}")
        print(f"  BI-RADS accuracy: {birads_results['birads_accuracy']*100:.2f}%")
        print(
            "  Spearman corr(pred BI-RADS, pathology): "
            f"{birads_results['birads_pred_pathology_spearman_corr']:.4f} "
            f"(p={birads_results['birads_pred_pathology_spearman_pvalue']:.4e})"
        )
        print("  BI-RADS confusion matrix (rows=true, cols=pred):")
        print(np.array(birads_results['birads_confusion_matrix']))

    if benchmark_comparison:
        print("\nBenchmark comparison:")
        print(f"  Source: {benchmark_comparison['source']}")
        for metric_name, values in benchmark_comparison['comparisons'].items():
            bm = values['benchmark']
            mv = values['model']
            delta = values['delta']
            if mv is None:
                print(f"  {metric_name:35s}: benchmark={bm:.4f}, model=NA")
            else:
                print(f"  {metric_name:35s}: benchmark={bm:.4f}, model={mv:.4f}, delta={delta:+.4f}")

    print("=" * 60)

    if cm_out is not None:
        save_confusion_matrix_image(
            cm=cm,
            class_names=CLASS_NAMES,
            out_path=cm_out,
            title=f"DenseNet201 Test Confusion Matrix ({checkpoint_name})",
        )
        print(f"Saved confusion matrix image: {cm_out}")

    if cm_norm_out is not None:
        save_confusion_matrix_normalized_image(
            cm=cm,
            class_names=CLASS_NAMES,
            out_path=cm_norm_out,
            title=f"DenseNet201 Test Confusion Matrix % ({checkpoint_name})",
        )
        print(f"Saved normalized confusion matrix image: {cm_norm_out}")

    if birads_cm_out is not None and birads_results:
        save_birads_confusion_matrix_image(
            cm=np.array(birads_results['birads_confusion_matrix']),
            labels=birads_results['birads_labels'],
            out_path=birads_cm_out,
            title=f"BI-RADS Confusion Matrix ({checkpoint_name})",
        )
        print(f"Saved BI-RADS confusion matrix image: {birads_cm_out}")

    if acc > 0.90:
        print("\nWARNING: Test acc >90% - verify no train/test leakage in mapped CSVs.")

    return {
        'metrics': metrics,
        'benchmark_comparison': benchmark_comparison,
        'birads': birads_results,
        'preds': preds.tolist(),
        'labels': labels.tolist(),
    }


def run_eval(
    checkpoint_path,
    dataset_name,
    benchmark_data: dict | None = None,
    batch_size=32,
    cm_out: Path | None = None,
    cm_norm_out: Path | None = None,
    birads_cm_out: Path | None = None,
    n_bootstrap: int = 2000,
    ci_alpha: float = 0.05,
    bootstrap_seed: int = 42,
):
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    test_loader = build_test_loader(batch_size)
    model       = load_model(checkpoint_path)
    preds, labels, probs = evaluate(model, test_loader, device)
    return print_report(
        preds,
        labels,
        probs,
        Path(checkpoint_path).name,
        dataset_name,
        benchmark_data=benchmark_data,
        cm_out=cm_out,
        cm_norm_out=cm_norm_out,
        birads_cm_out=birads_cm_out,
        n_bootstrap=n_bootstrap,
        ci_alpha=ci_alpha,
        bootstrap_seed=bootstrap_seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',     type=str)
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--dataset_name',   type=str, default='CBIS-DDSM',
                        help='Dataset label used for reporting and benchmark lookup.')
    parser.add_argument('--benchmark_json', type=str, default=str(DEFAULT_BENCHMARKS),
                        help='Optional JSON file with state-of-the-art benchmark metrics.')
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--cm_out',         type=str, default=None,
                        help='Optional path to save confusion matrix image (PNG).')
    parser.add_argument('--cm_norm_out',    type=str, default=None,
                        help='Optional path to save normalized confusion matrix image (PNG).')
    parser.add_argument('--birads_cm_out',  type=str, default=None,
                        help='Optional path to save BI-RADS confusion matrix image (PNG).')
    parser.add_argument('--metrics_out',    type=str, default=None,
                        help='Optional path to save computed metrics JSON.')
    parser.add_argument('--comparison_out', type=str, default=None,
                        help='Optional path to save benchmark comparison JSON.')
    parser.add_argument('--birads_out',     type=str, default=None,
                        help='Optional path to save BI-RADS hierarchical metrics JSON.')
    parser.add_argument('--n_bootstrap',    type=int, default=2000,
                        help='Number of bootstrap resamples for confidence intervals.')
    parser.add_argument('--ci_alpha',       type=float, default=0.05,
                        help='Alpha for bootstrap CI (0.05 => 95% CI).')
    parser.add_argument('--bootstrap_seed', type=int, default=42,
                        help='Random seed for bootstrap resampling.')
    parser.add_argument('--significance_out', type=str, default=None,
                        help='Optional path to save McNemar significance tests between checkpoints.')
    args = parser.parse_args()

    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Provide --checkpoint or --checkpoint_dir")

    checkpoints = []
    if args.checkpoint:
        checkpoints.append(args.checkpoint)
    if args.checkpoint_dir:
        checkpoints.extend(sorted(Path(args.checkpoint_dir).glob('*.ckpt')))

    benchmark_data = maybe_load_benchmarks(args.benchmark_json)

    results = []
    for ckpt in checkpoints:
        try:
            cm_out = None
            cm_norm_out = None
            birads_cm_out = None
            if args.cm_out:
                base = Path(args.cm_out)
                if len(checkpoints) > 1:
                    cm_out = base.parent / f"{base.stem}_{Path(ckpt).stem}{base.suffix or '.png'}"
                else:
                    cm_out = base if base.suffix else (base.parent / f"{base.name}.png")

            if args.cm_norm_out:
                base_n = Path(args.cm_norm_out)
                if len(checkpoints) > 1:
                    cm_norm_out = base_n.parent / f"{base_n.stem}_{Path(ckpt).stem}{base_n.suffix or '.png'}"
                else:
                    cm_norm_out = base_n if base_n.suffix else (base_n.parent / f"{base_n.name}.png")

            if args.birads_cm_out:
                base_b = Path(args.birads_cm_out)
                if len(checkpoints) > 1:
                    birads_cm_out = base_b.parent / f"{base_b.stem}_{Path(ckpt).stem}{base_b.suffix or '.png'}"
                else:
                    birads_cm_out = base_b if base_b.suffix else (base_b.parent / f"{base_b.name}.png")

            results.append(
                run_eval(
                    str(ckpt),
                    dataset_name=args.dataset_name,
                    benchmark_data=benchmark_data,
                    batch_size=args.batch_size,
                    cm_out=cm_out,
                    cm_norm_out=cm_norm_out,
                    birads_cm_out=birads_cm_out,
                    n_bootstrap=args.n_bootstrap,
                    ci_alpha=args.ci_alpha,
                    bootstrap_seed=args.bootstrap_seed,
                )
            )
        except Exception as e:
            logger.error(f"Failed on {ckpt}: {e}", exc_info=True)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("RANKING BY TEST ACCURACY")
        print("=" * 60)
        for r in sorted(results, key=lambda x: x['metrics']['accuracy'], reverse=True):
            print(f"  {r['metrics']['accuracy']*100:.2f}%  {r['metrics']['checkpoint']}")

        print("\n" + "=" * 60)
        print("PAIRWISE MCNEMAR SIGNIFICANCE TESTS")
        print("=" * 60)
        significance_payload = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                a = results[i]
                b = results[j]
                y_true_a = np.array(a['labels'], dtype=np.int32)
                y_true_b = np.array(b['labels'], dtype=np.int32)
                if y_true_a.shape != y_true_b.shape or not np.array_equal(y_true_a, y_true_b):
                    logger.warning(
                        "Skipping McNemar for %s vs %s due to label mismatch.",
                        a['metrics']['checkpoint'],
                        b['metrics']['checkpoint'],
                    )
                    continue

                test_res = mcnemar_test(
                    y_true=y_true_a,
                    preds_a=np.array(a['preds'], dtype=np.int32),
                    preds_b=np.array(b['preds'], dtype=np.int32),
                )
                rec = {
                    'model_a': a['metrics']['checkpoint'],
                    'model_b': b['metrics']['checkpoint'],
                    **test_res,
                }
                significance_payload.append(rec)
                print(
                    f"  {rec['model_a']} vs {rec['model_b']}: "
                    f"p_exact={rec['exact_binom_pvalue']:.6f}, "
                    f"p_chi2={rec['chi2_pvalue_cc']:.6f}, "
                    f"discordant={rec['n_discordant']}"
                )

        if args.significance_out and significance_payload:
            sig_path = Path(args.significance_out)
            sig_path.parent.mkdir(parents=True, exist_ok=True)
            with sig_path.open('w', encoding='utf-8') as f:
                json.dump(significance_payload, f, indent=2)
            print(f"Saved significance JSON: {sig_path}")

        # Generate automatic best-vs-baseline markdown report
        md_out = Path("results/significance_report.md")
        md_report = generate_significance_markdown_report(
            results=results,
            significance_payload=significance_payload,
            out_path=md_out,
        )
        print(f"\nSaved significance markdown report: {md_out}")
        print("\n" + md_report)

    if args.metrics_out:
        metrics_payload = [
            {
                'metrics': r.get('metrics'),
                'birads': r.get('birads'),
            }
            for r in results
        ]
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open('w', encoding='utf-8') as f:
            json.dump(metrics_payload, f, indent=2)
        print(f"Saved metrics JSON: {metrics_path}")

    if args.comparison_out:
        comparison_payload = [r['benchmark_comparison'] for r in results if r['benchmark_comparison'] is not None]
        comparison_path = Path(args.comparison_out)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        with comparison_path.open('w', encoding='utf-8') as f:
            json.dump(comparison_payload, f, indent=2)
        print(f"Saved benchmark comparison JSON: {comparison_path}")

    if args.birads_out:
        birads_payload = [r['metrics'].get('birads_hierarchical', {}) for r in results]
        birads_path = Path(args.birads_out)
        birads_path.parent.mkdir(parents=True, exist_ok=True)
        with birads_path.open('w', encoding='utf-8') as f:
            json.dump(birads_payload, f, indent=2)
        print(f"Saved BI-RADS metrics JSON: {birads_path}")


if __name__ == '__main__':
    main()
