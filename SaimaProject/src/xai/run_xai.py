"""
XAI analysis for multimodal breast cancer model.

Produces outputs in results/xai/:
    1. gradcam/              Grad-CAM heatmaps overlaid on mammogram
    2. attention/            Cross-attention weight bar charts
    3. clinical_importance.png
                                                     Mean absolute gradient of clinical features
    4. shap/clinical/        SHAP clinical feature attributions
    5. lime/image/           LIME superpixel image explanations
    6. lime/clinical/        LIME tabular clinical explanations

Usage:
        python src/xai/run_xai.py --n_samples 20 --run_shap --run_lime
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from lime import lime_image
    from lime import lime_tabular
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from train_with_images import (
    MappedDataset, MultimodalLit, LABEL_MAP, CLINICAL_COLS, CFG, get_val_aug
)
from multimodal_framework import MultimodalClassifier

# ── Constants ──────────────────────────────────────────────────────────────────
CKPT_DEFAULT  = Path("checkpoints/multimodal_real_images/epoch=135-val_loss=0.038-val_acc=0.980.ckpt")
TEST_CSV      = Path("data/processed/csv/mass_test_mapped.csv")
ENC_TRAIN_CSV = Path("data/processed/csv/mass_train_encoded.csv")
ENC_TEST_CSV  = Path("data/processed/csv/mass_test_encoded.csv")
OUT_DIR       = Path("results/xai")

CLASS_NAMES   = [k for k, v in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
# CLASS_NAMES = ['MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK']
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path):
    raw    = torch.load(ckpt_path, map_location='cpu')
    cfg    = raw['hyper_parameters'].get('cfg', CFG)
    arch   = {
        'backbone':           cfg.get('image_backbone', 'densenet201'),
        'clinical_input_dim': cfg.get('clinical_input_dim', 48),
        'fusion_dim':         cfg.get('fusion_dim', 256),
        'num_classes':        cfg.get('num_classes', 3),
        'dropout':            cfg.get('dropout', 0.5),
    }
    base   = MultimodalClassifier(arch)
    dummy  = torch.ones(arch['num_classes'])
    lit    = MultimodalLit(model=base, class_weights=dummy, cfg=cfg)
    lit.load_state_dict(raw['state_dict'], strict=False)
    lit.eval()
    return lit, cfg


# ── Data ───────────────────────────────────────────────────────────────────────
def build_test_loader(batch_size=1):
    test_df  = pd.read_csv(TEST_CSV)
    enc_all  = pd.concat([pd.read_csv(ENC_TRAIN_CSV), pd.read_csv(ENC_TEST_CSV)], ignore_index=True)
    ds = MappedDataset(df=test_df, encoded_df=enc_all,
                       transform=get_val_aug(), is_train=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for DenseNet201 backbone inside MultimodalClassifier.
    Target layer: image_encoder.backbone[0].denseblock4
    """
    def __init__(self, model: MultimodalClassifier):
        self.model      = model
        self.activations = None
        self.gradients   = None

        # Hook onto last dense block (before norm5)
        target = model.image_encoder.backbone[0].denseblock4
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, images: torch.Tensor, clinical: torch.Tensor, class_idx: int):
        self.model.zero_grad()
        logits, _ = self.model(images, clinical)
        logits[:, class_idx].sum().backward()

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1)          # (B, H, W)
        cam     = F.relu(cam)

        # Normalize per sample
        B = cam.size(0)
        cam_flat = cam.view(B, -1)
        cam_min  = cam_flat.min(dim=1)[0].view(B, 1, 1)
        cam_max  = cam_flat.max(dim=1)[0].view(B, 1, 1)
        cam      = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.cpu().numpy()


# ── Attention extractor ────────────────────────────────────────────────────────
def get_attention_weights(model: MultimodalClassifier,
                          images: torch.Tensor,
                          clinical: torch.Tensor):
    """Returns attention_weights from cross-attention fusion."""
    with torch.no_grad():
        _, attn = model(images, clinical)
    # attn shape: (B, num_heads, query_len, key_len) or (B, query_len, key_len)
    # Squeeze to (B, key_len) by averaging over heads and query
    if attn is not None and attn.dim() == 4:
        attn = attn.mean(dim=1).mean(dim=1)   # (B, key_len)
    elif attn is not None and attn.dim() == 3:
        attn = attn.mean(dim=1)               # (B, key_len)
    return attn.cpu().numpy() if attn is not None else None


# ── Plotting helpers ───────────────────────────────────────────────────────────
def save_gradcam(img_tensor: torch.Tensor, cam: np.ndarray,
                 true_label: str, pred_label: str, conf: float,
                 out_path: Path):
    # img_tensor: (3, H, W) normalized
    mean = NORM_MEAN
    std  = NORM_STD
    img  = img_tensor.permute(1, 2, 0).numpy()
    img  = (img * std + mean).clip(0, 1)

    h, w = img.shape[:2]
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h),
                                           Image.BILINEAR)) / 255.0
    heatmap = cm.jet(cam_resized)[:, :, :3]
    overlay = 0.5 * img + 0.5 * heatmap
    overlay = overlay.clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img);           axes[0].set_title('Original');    axes[0].axis('off')
    axes[1].imshow(cam_resized, cmap='jet'); axes[1].set_title('Grad-CAM'); axes[1].axis('off')
    axes[2].imshow(overlay);       axes[2].set_title('Overlay');     axes[2].axis('off')
    correct = '✓' if true_label == pred_label else '✗'
    fig.suptitle(f'True: {true_label} | Pred: {pred_label} ({conf:.1%}) {correct}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_attention_chart(attn: np.ndarray, out_path: Path, title: str):
    """Bar chart of attention weights over spatial positions."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(attn)), attn, color='steelblue')
    ax.set_xlabel('Spatial position (flattened image regions)')
    ax.set_ylabel('Attention weight')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_clinical_importance(importance: np.ndarray, col_names: list, out_path: Path):
    """Horizontal bar chart of mean clinical feature importance."""
    sorted_idx  = np.argsort(importance)
    sorted_imp  = importance[sorted_idx]
    sorted_names = [col_names[i] for i in sorted_idx]

    fig_h = max(6, len(col_names) * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(sorted_imp)), sorted_imp, color='tomato')
    ax.set_yticks(range(len(sorted_imp)))
    ax.set_yticklabels(sorted_names, fontsize=7)
    ax.set_xlabel('Mean |gradient| (proxy importance)')
    ax.set_title('Clinical Feature Importance (Gradient-based)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_shap_importance(importance: np.ndarray, col_names: list, out_path: Path):
    """Horizontal bar chart of mean |SHAP| clinical importance."""
    sorted_idx = np.argsort(importance)
    sorted_imp = importance[sorted_idx]
    sorted_names = [col_names[i] for i in sorted_idx]

    fig_h = max(6, len(col_names) * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(sorted_imp)), sorted_imp, color='seagreen')
    ax.set_yticks(range(len(sorted_imp)))
    ax.set_yticklabels(sorted_names, fontsize=7)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Clinical Feature Importance (SHAP)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized CHW tensor to HWC RGB in [0, 1]."""
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * NORM_STD + NORM_MEAN).clip(0, 1)
    return img


def normalize_images_for_model(images_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert HWC RGB [0,1] to normalized BCHW tensor for model input."""
    if images_hwc.ndim == 3:
        images_hwc = np.expand_dims(images_hwc, axis=0)
    x = np.asarray(images_hwc, dtype=np.float32)
    x = (x - NORM_MEAN[None, None, None, :]) / NORM_STD[None, None, None, :]
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(x).to(device)


def save_lime_image_explanation(
    original_img: np.ndarray,
    explained_img: np.ndarray,
    boundary_overlay: np.ndarray,
    title: str,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(explained_img)
    axes[1].set_title('LIME superpixels')
    axes[1].axis('off')

    axes[2].imshow(boundary_overlay)
    axes[2].set_title('Boundary overlay')
    axes[2].axis('off')

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_lime_tabular_weights(weights: list, out_path: Path, title: str):
    """Save LIME tabular contribution weights as a horizontal bar chart."""
    if not weights:
        return
    feats = [w[0] for w in weights]
    vals = np.array([w[1] for w in weights], dtype=np.float32)
    order = np.argsort(vals)
    feats = [feats[i] for i in order]
    vals = vals[order]
    colors = ['tomato' if v > 0 else 'steelblue' for v in vals]

    fig_h = max(4, 0.45 * len(feats))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel('LIME local contribution')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def build_clinical_predict_fn(
    model: MultimodalClassifier,
    fixed_image: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
):
    """
    Returns predict(clinical_np) -> class probabilities.
    fixed_image shape: (1, 3, H, W)
    clinical_np shape: (N, F) or (F,)
    output shape: (N, C)
    """
    fixed_image = fixed_image.to(device)

    def _predict(clinical_np: np.ndarray) -> np.ndarray:
        x = np.asarray(clinical_np, dtype=np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        outputs = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                end = min(start + batch_size, x.shape[0])
                clinical_t = torch.from_numpy(x[start:end]).to(device)
                images_t = fixed_image.repeat(clinical_t.size(0), 1, 1, 1)
                logits, _ = model(images_t, clinical_t)
                probs = F.softmax(logits, dim=1)
                outputs.append(probs.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    return _predict


def run_shap_clinical(
    model: MultimodalClassifier,
    records: list,
    device: torch.device,
    out_dir: Path,
    shap_background: int,
    shap_nsamples: int,
):
    if not SHAP_AVAILABLE:
        print('SHAP is not available. Install shap to enable --run_shap.')
        return
    if not records:
        print('No records available for SHAP.')
        return

    shap_dir = out_dir / 'shap' / 'clinical'
    shap_dir.mkdir(parents=True, exist_ok=True)

    train_encoded = pd.read_csv(ENC_TRAIN_CSV)
    train_matrix = train_encoded[CLINICAL_COLS].values.astype(np.float32)
    bg_n = min(shap_background, len(train_matrix))
    background = train_matrix[:bg_n]

    clinical_eval = np.stack([r['clinical_np'] for r in records]).astype(np.float32)
    fixed_image = records[0]['image_tensor'].unsqueeze(0).to(device)

    predict_fn = build_clinical_predict_fn(model, fixed_image, device)

    print(f"SHAP shapes: background={background.shape}, eval={clinical_eval.shape}")
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(clinical_eval, nsamples=shap_nsamples)

    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)  # (C, N, F)
        mean_abs = np.abs(stacked).mean(axis=(0, 1))
        class0_values = np.asarray(shap_values[0])
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            if arr.shape[0] == clinical_eval.shape[0]:
                # (N, F, C)
                mean_abs = np.abs(arr).mean(axis=(0, 2))
                class0_values = np.asarray(arr[:, :, 0])
            else:
                # (C, N, F)
                mean_abs = np.abs(arr).mean(axis=(0, 1))
                class0_values = np.asarray(arr[0])
        else:
            mean_abs = np.abs(arr).mean(axis=0)
            class0_values = np.asarray(arr)

    save_shap_importance(mean_abs, CLINICAL_COLS, shap_dir / 'shap_clinical_importance.png')

    # Persist raw SHAP arrays for reproducibility and downstream analysis.
    np.save(shap_dir / 'clinical_eval.npy', clinical_eval)
    np.save(shap_dir / 'shap_values_class0.npy', class0_values)

    # SHAP summary plots (beeswarm + bar)
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            class0_values,
            clinical_eval,
            feature_names=CLINICAL_COLS,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(shap_dir / 'shap_summary_beeswarm.png', dpi=170, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            class0_values,
            clinical_eval,
            feature_names=CLINICAL_COLS,
            plot_type='bar',
            show=False,
        )
        plt.tight_layout()
        plt.savefig(shap_dir / 'shap_summary_bar.png', dpi=170, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP summary plotting failed: {e}")

    # SHAP dependence plots for top features.
    try:
        top_dep_idx = np.argsort(mean_abs)[::-1][: min(5, len(CLINICAL_COLS))]
        for dep_rank, feat_idx in enumerate(top_dep_idx, 1):
            plt.figure(figsize=(9, 6))
            shap.dependence_plot(
                int(feat_idx),
                class0_values,
                clinical_eval,
                feature_names=CLINICAL_COLS,
                interaction_index='auto',
                show=False,
            )
            plt.tight_layout()
            dep_name = str(CLINICAL_COLS[int(feat_idx)]).replace(' ', '_').replace('/', '_')
            plt.savefig(
                shap_dir / f'shap_dependence_{dep_rank:02d}_{dep_name}.png',
                dpi=170,
                bbox_inches='tight',
            )
            plt.close()
    except Exception as e:
        print(f"SHAP dependence plotting failed: {e}")

    # SHAP force plot for a representative sample.
    try:
        base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        sample_idx = 0
        force = shap.force_plot(
            base_value,
            class0_values[sample_idx],
            clinical_eval[sample_idx],
            feature_names=CLINICAL_COLS,
            matplotlib=False,
        )
        shap.save_html(str(shap_dir / 'shap_force_sample0.html'), force)
    except Exception as e:
        print(f"SHAP force plotting failed: {e}")

    top_idx = np.argsort(mean_abs)[::-1][:15]
    rows = []
    for rank, idx in enumerate(top_idx, 1):
        rows.append({'rank': rank, 'feature': CLINICAL_COLS[idx], 'mean_abs_shap': float(mean_abs[idx])})
    pd.DataFrame(rows).to_csv(shap_dir / 'top_features.csv', index=False)
    print(f"Saved SHAP outputs to {shap_dir}")


def run_lime_image(
    model: MultimodalClassifier,
    records: list,
    device: torch.device,
    out_dir: Path,
    lime_num_samples: int,
    lime_num_features: int,
    lime_max_images: int,
):
    if not LIME_AVAILABLE:
        print('LIME or skimage is not available. Install lime to enable --run_lime.')
        return
    if not records:
        print('No records available for LIME image explanations.')
        return

    lime_img_dir = out_dir / 'lime' / 'image'
    lime_img_dir.mkdir(parents=True, exist_ok=True)

    img_explainer = lime_image.LimeImageExplainer(random_state=42)

    for i, rec in enumerate(records[:lime_max_images]):
        image_rgb = denormalize_image(rec['image_tensor'])
        clinical_vec = rec['clinical_np'].astype(np.float32)
        pred_idx = int(rec['pred_idx'])

        def classifier_fn(images_np: np.ndarray) -> np.ndarray:
            x_images = normalize_images_for_model(images_np, device)
            clinical_batch = torch.from_numpy(
                np.repeat(clinical_vec[None, :], x_images.size(0), axis=0).astype(np.float32)
            ).to(device)
            with torch.no_grad():
                logits, _ = model(x_images, clinical_batch)
                probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

        print(
            'LIME image shapes: '
            f"image={image_rgb.shape}, clinical={clinical_vec.shape}, pred_class={pred_idx}"
        )
        explanation = img_explainer.explain_instance(
            image_rgb,
            classifier_fn,
            top_labels=1,
            hide_color=0,
            num_samples=lime_num_samples,
        )

        label_to_show = pred_idx
        if label_to_show not in explanation.local_exp:
            label_to_show = explanation.top_labels[0]

        temp, mask = explanation.get_image_and_mask(
            label_to_show,
            positive_only=False,
            num_features=lime_num_features,
            hide_rest=False,
        )
        temp = np.clip(temp, 0.0, 1.0)
        overlay = mark_boundaries(temp, mask)

        out_path = lime_img_dir / (
            f"{i:04d}_{rec['patient_id']}_{rec['true_name']}_pred{rec['pred_name']}.png"
        )
        title = (
            f"LIME image | True: {rec['true_name']} "
            f"Pred: {rec['pred_name']} ({rec['conf']:.1%})"
        )
        save_lime_image_explanation(image_rgb, temp, overlay, title, out_path)

    print(f"Saved LIME image outputs to {lime_img_dir}")


def run_lime_clinical(
    model: MultimodalClassifier,
    records: list,
    device: torch.device,
    out_dir: Path,
    lime_num_features: int,
    lime_max_images: int,
):
    if not LIME_AVAILABLE:
        print('LIME is not available for tabular explanations.')
        return
    if not records:
        print('No records available for LIME tabular explanations.')
        return

    lime_tab_dir = out_dir / 'lime' / 'clinical'
    lime_tab_dir.mkdir(parents=True, exist_ok=True)

    train_encoded = pd.read_csv(ENC_TRAIN_CSV)
    train_matrix = train_encoded[CLINICAL_COLS].values.astype(np.float32)

    fixed_image = records[0]['image_tensor'].unsqueeze(0).to(device)
    predict_fn = build_clinical_predict_fn(model, fixed_image, device)

    print(f"LIME tabular shapes: train={train_matrix.shape}, fixed_image={tuple(fixed_image.shape)}")
    tab_explainer = lime_tabular.LimeTabularExplainer(
        training_data=train_matrix,
        feature_names=CLINICAL_COLS,
        class_names=CLASS_NAMES,
        mode='classification',
        discretize_continuous=False,
        random_state=42,
    )

    for i, rec in enumerate(records[:lime_max_images]):
        clinical_vec = rec['clinical_np'].astype(np.float32)
        pred_idx = int(rec['pred_idx'])

        exp = tab_explainer.explain_instance(
            data_row=clinical_vec,
            predict_fn=predict_fn,
            num_features=lime_num_features,
            top_labels=1,
        )

        label_to_show = pred_idx
        available = exp.available_labels()
        if label_to_show not in available:
            label_to_show = available[0]

        weights = exp.as_list(label=label_to_show)
        title = (
            f"LIME clinical | PID: {rec['patient_id']} "
            f"True: {rec['true_name']} Pred: {rec['pred_name']}"
        )

        out_path = lime_tab_dir / (
            f"{i:04d}_{rec['patient_id']}_{rec['true_name']}_pred{rec['pred_name']}.png"
        )
        save_lime_tabular_weights(weights, out_path, title)

        html_path = lime_tab_dir / (
            f"{i:04d}_{rec['patient_id']}_{rec['true_name']}_pred{rec['pred_name']}.html"
        )
        exp.save_to_file(str(html_path))

    print(f"Saved LIME clinical outputs to {lime_tab_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples',  type=int,  default=20)
    parser.add_argument('--checkpoint', type=str,  default=str(CKPT_DEFAULT))
    parser.add_argument('--out_dir',    type=str,  default=str(OUT_DIR))
    parser.add_argument('--run_shap',   action='store_true', help='Run SHAP for clinical features')
    parser.add_argument('--run_lime',   action='store_true', help='Run LIME for image and clinical features')
    parser.add_argument('--shap_background', type=int, default=100)
    parser.add_argument('--shap_nsamples',   type=int, default=200)
    parser.add_argument('--lime_num_samples', type=int, default=600)
    parser.add_argument('--lime_num_features', type=int, default=12)
    parser.add_argument('--lime_max_images', type=int, default=5)
    args = parser.parse_args()

    out     = Path(args.out_dir)
    gcam_d  = out / 'gradcam';   gcam_d.mkdir(parents=True, exist_ok=True)
    attn_d  = out / 'attention'; attn_d.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    lit, cfg = load_model(Path(args.checkpoint))
    model    = lit.model.to(device)
    gradcam  = GradCAM(model)

    loader = build_test_loader(batch_size=1)

    # Accumulators for clinical importance
    clinical_grads = []
    records = []
    sample_count   = 0

    for i, batch in enumerate(loader):
        if sample_count >= args.n_samples:
            break

        images   = batch['image'].to(device)
        clinical = batch['clinical_features'].to(device).requires_grad_(True)
        labels   = batch['label']
        pid      = batch['patient_id'][0]

        true_idx  = labels[0].item()
        true_name = CLASS_NAMES[true_idx]

        # Forward for prediction
        with torch.no_grad():
            logits_eval = model(images, clinical.detach())[0]
            probs       = F.softmax(logits_eval, dim=1)[0]
            pred_idx    = probs.argmax().item()
            pred_name   = CLASS_NAMES[pred_idx]
            conf        = probs[pred_idx].item()

        if sample_count == 0:
            print(
                'Core shapes: '
                f"image={tuple(images.shape)}, "
                f"clinical={tuple(clinical.shape)}, "
                f"logits={tuple(logits_eval.shape)}, "
                f"probs={tuple(probs.shape)}"
            )

        # ── Grad-CAM ──────────────────────────────────────────────────────
        try:
            model.zero_grad()
            cam = gradcam(images, clinical.detach(), pred_idx)  # (1, H, W)
            out_path = gcam_d / f"{i:04d}_{pid}_{true_name}_pred{pred_name}.png"
            save_gradcam(images[0].cpu(), cam[0], true_name, pred_name, conf, out_path)
        except Exception as e:
            print(f"  GradCAM failed sample {i}: {e}")

        # ── Attention weights ──────────────────────────────────────────────
        attn = get_attention_weights(model, images, clinical.detach())
        if attn is not None:
            out_path = attn_d / f"{i:04d}_{pid}_{true_name}.png"
            save_attention_chart(
                attn[0], out_path,
                f"Cross-attention  |  True: {true_name}  Pred: {pred_name} ({conf:.1%})"
            )

        # ── Clinical gradient importance ───────────────────────────────────
        try:
            clinical_req = batch['clinical_features'].to(device).requires_grad_(True)
            logits_g, _  = model(images, clinical_req)
            logits_g[:, pred_idx].sum().backward()
            grad = clinical_req.grad[0].abs().cpu().numpy()
            clinical_grads.append(grad)
        except Exception as e:
            print(f"  Clinical grad failed sample {i}: {e}")

        records.append({
            'patient_id': pid,
            'image_tensor': images[0].detach().cpu(),
            'clinical_np': clinical.detach().cpu().numpy()[0].astype(np.float32),
            'true_idx': int(true_idx),
            'pred_idx': int(pred_idx),
            'true_name': true_name,
            'pred_name': pred_name,
            'conf': float(conf),
        })

        sample_count += 1
        if sample_count % 5 == 0:
            print(f"  Processed {sample_count}/{args.n_samples}")

    # ── Clinical importance summary plot ──────────────────────────────────
    if clinical_grads:
        mean_imp = np.stack(clinical_grads).mean(axis=0)
        save_clinical_importance(mean_imp, CLINICAL_COLS, out / 'clinical_importance.png')
        print(f"\nTop 10 most important clinical features:")
        top_idx = np.argsort(mean_imp)[::-1][:10]
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank:2d}. {CLINICAL_COLS[idx]:45s} {mean_imp[idx]:.4f}")

    # ── Optional SHAP / LIME ──────────────────────────────────────────────
    if args.run_shap:
        try:
            run_shap_clinical(
                model=model,
                records=records,
                device=device,
                out_dir=out,
                shap_background=args.shap_background,
                shap_nsamples=args.shap_nsamples,
            )
        except Exception as e:
            print(f"SHAP failed: {e}")

    if args.run_lime:
        try:
            run_lime_image(
                model=model,
                records=records,
                device=device,
                out_dir=out,
                lime_num_samples=args.lime_num_samples,
                lime_num_features=args.lime_num_features,
                lime_max_images=args.lime_max_images,
            )
        except Exception as e:
            print(f"LIME image failed: {e}")

        try:
            run_lime_clinical(
                model=model,
                records=records,
                device=device,
                out_dir=out,
                lime_num_features=args.lime_num_features,
                lime_max_images=args.lime_max_images,
            )
        except Exception as e:
            print(f"LIME clinical failed: {e}")

    print(f"\nDone. Outputs in {out}/")
    print(f"  {gcam_d}/       — {len(list(gcam_d.glob('*.png')))} Grad-CAM images")
    print(f"  {attn_d}/       — {len(list(attn_d.glob('*.png')))} attention charts")
    print(f"  {out}/clinical_importance.png")
    if args.run_shap:
        print(f"  {out / 'shap' / 'clinical'}/")
    if args.run_lime:
        print(f"  {out / 'lime' / 'image'}/")
        print(f"  {out / 'lime' / 'clinical'}/")


if __name__ == '__main__':
    main()
