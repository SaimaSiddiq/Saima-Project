"""
Run real Grad-CAM and SHAP on actual CBIS-DDSM mammogram images.
Saves output images to SaimaProject/results/xai_demo/
Usage:
    cd e:/ThesisProject/SaimaProject
    python ../run_real_xai.py
"""
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent / 'SaimaProject'
SRC         = ROOT / 'src'
CKPT_DIR    = ROOT / 'checkpoints' / 'multimodal_real_images'
TEST_CSV    = ROOT / 'data' / 'processed' / 'csv' / 'mass_test_mapped.csv'
ENC_TRAIN   = ROOT / 'data' / 'processed' / 'csv' / 'mass_train_encoded.csv'
ENC_TEST    = ROOT / 'data' / 'processed' / 'csv' / 'mass_test_encoded.csv'
OUT_DIR     = ROOT / 'results' / 'xai_demo'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))

from train_with_images import CLINICAL_COLS, LABEL_MAP, CFG, get_val_aug
from multimodal_framework import MultimodalClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = [k for k, v in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist()),
])


# ── Load checkpoint ─────────────────────────────────────────────────────────
def load_model():
    ckpts = sorted(CKPT_DIR.glob('*.ckpt'))
    assert ckpts, f'No .ckpt files in {CKPT_DIR}'
    ckpt_path = ckpts[0]
    print(f'Loading checkpoint: {ckpt_path.name}')
    raw    = torch.load(ckpt_path, map_location='cpu')
    hparams = raw.get('hyper_parameters', {})
    cfg    = hparams.get('cfg', CFG)
    arch   = {
        'backbone':           cfg.get('image_backbone', 'densenet201'),
        'clinical_input_dim': cfg.get('clinical_input_dim', len(CLINICAL_COLS)),
        'fusion_dim':         cfg.get('fusion_dim', 256),
        'num_classes':        cfg.get('num_classes', 3),
        'dropout':            cfg.get('dropout', 0.4),
    }
    model = MultimodalClassifier(arch)
    state = raw.get('state_dict', {})
    model_state = {k.replace('model.', '', 1): v for k, v in state.items() if k.startswith('model.')}
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model.to(DEVICE)


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAMHook:
    def __init__(self, layer):
        self._act  = None
        self._grad = None
        self._fwd  = layer.register_forward_hook(lambda m, i, o: setattr(self, '_act', o.detach()))
        self._bwd  = layer.register_full_backward_hook(lambda m, gi, go: setattr(self, '_grad', go[0].detach()))
    def remove(self): self._fwd.remove(); self._bwd.remove()
    def cam(self):
        w = self._grad.mean(dim=(2, 3), keepdim=True)
        c = F.relu((w * self._act).sum(dim=1))[0].cpu().numpy()
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        return c


def generate_gradcam(model, img_t, clin_t, pred_class):
    try:
        target = model.image_encoder.backbone[0].denseblock4
    except Exception:
        target = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                target = m
    hook = GradCAMHook(target)
    model.zero_grad()
    logits, _ = model(img_t, clin_t)
    logits[0, pred_class].backward()
    cam = hook.cam()
    hook.remove()
    return cam


def overlay_cam(img_t, cam):
    img = img_t[0].permute(1, 2, 0).detach().cpu().numpy()
    img = (img * NORM_STD + NORM_MEAN).clip(0, 1)
    h, w = img.shape[:2]
    cam_r = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    heatmap = mpl_cm.jet(cam_r)[:, :, :3]
    overlay = (0.55 * img + 0.45 * heatmap).clip(0, 1)
    return img, cam_r, overlay


# ── Gradient×Input SHAP ──────────────────────────────────────────────────────
def generate_shap(model, img_t, clin_t, pred_class, col_names):
    clin_in = clin_t.clone().requires_grad_(True)
    model.zero_grad()
    logits, _ = model(img_t, clin_in)
    logits[0, pred_class].backward()
    grad = clin_in.grad[0].detach().cpu().numpy()
    inp  = clin_t[0].detach().cpu().numpy()
    attr = grad * inp
    top  = np.argsort(np.abs(attr))[::-1][:12]
    rows = [{'feature': col_names[i], 'value': float(inp[i]), 'shap': float(attr[i]),
             'direction': 'positive' if attr[i] >= 0 else 'negative'} for i in top]
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    model = load_model()
    test_df  = pd.read_csv(TEST_CSV)
    enc_all  = pd.concat([pd.read_csv(ENC_TRAIN), pd.read_csv(ENC_TEST)], ignore_index=True).set_index('patient_id')

    # Pick up to 3 samples whose images exist on disk
    samples = []
    for _, row in test_df.iterrows():
        if Path(row['image_path']).exists():
            samples.append(row)
        if len(samples) == 3:
            break

    if not samples:
        print('ERROR: No images found on disk. Check CSV paths.')
        return

    print(f'Running XAI on {len(samples)} real mammogram images...\n')

    for i, row in enumerate(samples):
        pid      = row['patient_id']
        true_lbl = row['pathology']
        img_path = row['image_path']

        # Image tensor
        pil_img  = Image.open(img_path).convert('RGB')
        img_t    = INFER_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

        # Clinical vector
        if pid in enc_all.index:
            enc_row = enc_all.loc[pid]
            if isinstance(enc_row, pd.DataFrame):
                enc_row = enc_row.iloc[0]
            clin = enc_row[CLINICAL_COLS].values.astype(np.float32)
        else:
            clin = np.zeros(len(CLINICAL_COLS), dtype=np.float32)
        clin_t = torch.from_numpy(clin).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            logits_nc, _ = model(img_t, clin_t)
            probs = F.softmax(logits_nc, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        pred_lbl   = CLASS_NAMES[pred_class]
        conf       = float(probs[pred_class])

        # ── Grad-CAM ──
        cam      = generate_gradcam(model, img_t, clin_t, pred_class)
        orig_img, cam_r, overlay = overlay_cam(img_t, cam)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].imshow(orig_img);                      axes[0].set_title('Original Mammogram'); axes[0].axis('off')
        axes[1].imshow(cam_r, cmap='jet');             axes[1].set_title('Grad-CAM Heatmap');   axes[1].axis('off')
        axes[2].imshow(overlay);                       axes[2].set_title('Overlay');             axes[2].axis('off')
        correct = '✓' if true_lbl == pred_lbl else '✗'
        fig.suptitle(f'Patient {pid}  |  True: {true_lbl}  |  Pred: {pred_lbl} ({conf:.0%})  {correct}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        gcam_path = OUT_DIR / f'gradcam_sample{i+1}_{pid}.png'
        plt.savefig(gcam_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[{i+1}] Saved Grad-CAM → {gcam_path.name}')

        # ── SHAP ──
        shap_rows = generate_shap(model, img_t, clin_t, pred_class, CLINICAL_COLS[:len(clin)])

        feats  = [r['feature'] for r in shap_rows][::-1]
        vals   = [r['shap'] for r in shap_rows][::-1]
        colors = ['#e8365d' if v >= 0 else '#1a8c6a' for v in vals]

        fig2, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(feats))))
        ax.barh(feats, vals, color=colors)
        ax.axvline(0, color='#94a3b8', linewidth=0.8)
        ax.set_xlabel('Gradient × Input Attribution')
        ax.set_title(f'Clinical SHAP  |  Patient {pid}  |  Pred: {pred_lbl} ({conf:.0%})')
        for label in ax.get_yticklabels():
            label.set_fontsize(8)
        plt.tight_layout()
        shap_path = OUT_DIR / f'shap_sample{i+1}_{pid}.png'
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[{i+1}] Saved SHAP    → {shap_path.name}')

        # Print SHAP top-5 to console
        print(f'     Top clinical features driving "{pred_lbl}":')
        for r in shap_rows[:5]:
            sign = '+' if r['direction'] == 'positive' else '-'
            print(f'       {sign}  {r["feature"]:40s}  attr={r["shap"]:+.5f}')
        print()

    print(f'\nAll XAI outputs saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
