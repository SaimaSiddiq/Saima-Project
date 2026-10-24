"""
MammoXplain – Flask ML Backend
================================
Attach your trained models here. Each route is ready to receive
data from the React frontend (via the Node.js gateway).

Install: pip install flask flask-cors torch torchvision pillow shap numpy
Run:     python flask_api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import io
import json
import logging
import re
import sys
from pathlib import Path
import os
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms

app = Flask(__name__)
CORS(app)  # Allow requests from Node.js gateway

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parents[1]
SAIMA_ROOT = BASE_DIR / 'SaimaProject'
SAIMA_SRC = SAIMA_ROOT / 'src'
CHECKPOINT_DIR = SAIMA_ROOT / 'checkpoints' / 'multimodal_real_images'
CHECKPOINT_PATH_ENV = os.environ.get('SAIMA_CHECKPOINT_PATH', '').strip()
ENCODED_TRAIN_CSV = SAIMA_ROOT / 'data' / 'processed' / 'csv' / 'mass_train_encoded.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

CLINICAL_COLS = []
CLINICAL_INDEX = {}
REAL_MODEL = None
REAL_MODEL_META = {}
MODEL_LOAD_ERROR = None


def _normalize_token(value):
    if value is None:
        return ''
    return str(value).strip().upper().replace('–', '-').replace(' ', '_')


def _set_feature(vec, feature_name):
    idx = CLINICAL_INDEX.get(feature_name)
    if idx is not None:
        vec[idx] = 1.0


def _extract_val_metric(path, metric='val_acc'):
    match = re.search(rf'{metric}=([0-9.]+)', path.name)
    if not match:
        return -1.0
    try:
        return float(match.group(1))
    except ValueError:
        return -1.0


def _select_checkpoint():
    if CHECKPOINT_PATH_ENV:
        explicit = Path(CHECKPOINT_PATH_ENV)
        if not explicit.is_absolute():
            explicit = BASE_DIR / explicit
        explicit = explicit.resolve()
        if not explicit.exists():
            raise FileNotFoundError(f'SAIMA_CHECKPOINT_PATH does not exist: {explicit}')
        if explicit.suffix != '.ckpt':
            raise ValueError(f'SAIMA_CHECKPOINT_PATH must point to a .ckpt file: {explicit}')
        return explicit

    candidates = list(CHECKPOINT_DIR.glob('*.ckpt'))
    if not candidates:
        raise FileNotFoundError(f'No checkpoint found in {CHECKPOINT_DIR}')

    by_acc = sorted(candidates, key=lambda p: _extract_val_metric(p, 'val_acc'), reverse=True)
    if _extract_val_metric(by_acc[0], 'val_acc') >= 0:
        return by_acc[0]

    by_f1 = sorted(candidates, key=lambda p: _extract_val_metric(p, 'val_f1'), reverse=True)
    if _extract_val_metric(by_f1[0], 'val_f1') >= 0:
        return by_f1[0]

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _load_clinical_columns():
    if not ENCODED_TRAIN_CSV.exists():
        raise FileNotFoundError(f'Encoded clinical CSV not found: {ENCODED_TRAIN_CSV}')

    df = pd.read_csv(ENCODED_TRAIN_CSV, nrows=1)
    cols = [
        c for c in df.columns
        if c != 'patient_id' and c != 'pathology' and not c.startswith('pathology_')
    ]
    if not cols:
        raise ValueError('No clinical feature columns found in encoded CSV')
    return cols


def _build_clinical_vector(clinical):
    vec = np.zeros(len(CLINICAL_COLS), dtype=np.float32)
    clinical = clinical if isinstance(clinical, dict) else {}

    margin_map = {
        'CIRCUMSCRIBED': 'mass margins_CIRCUMSCRIBED',
        'SPICULATED': 'mass margins_SPICULATED',
        'ILL_DEFINED': 'mass margins_ILL_DEFINED',
        'MICROLOBULATED': 'mass margins_MICROLOBULATED',
        'OBSCURED': 'mass margins_OBSCURED',
    }
    shape_map = {
        'ROUND': 'mass shape_ROUND',
        'OVAL': 'mass shape_OVAL',
        'IRREGULAR': 'mass shape_IRREGULAR',
        'LOBULAR': 'mass shape_LOBULATED',
    }
    density_map = {
        '1': 'breast_density_1',
        '2': 'breast_density_2',
        '3': 'breast_density_3',
        '4': 'breast_density_4',
        'ACR_A': 'breast_density_1',
        'ACR_B': 'breast_density_2',
        'ACR_C': 'breast_density_3',
        'ACR_D': 'breast_density_4',
    }

    margin_key = _normalize_token(clinical.get('massMargin')).replace('-', '_')
    shape_key = _normalize_token(clinical.get('massShape')).replace('-', '_')
    density_raw = _normalize_token(clinical.get('breastDensity')).replace('-', '_')
    density_key = density_raw
    if density_raw:
        acr_match = re.search(r'ACR[_\s]*([ABCD]|[1-4])', density_raw)
        if acr_match:
            token = acr_match.group(1)
            density_key = token if token.isdigit() else f'ACR_{token}'
        else:
            digit_match = re.search(r'([1-4])', density_raw)
            if digit_match:
                density_key = digit_match.group(1)

    if margin_key in margin_map:
        _set_feature(vec, margin_map[margin_key])
    else:
        _set_feature(vec, 'mass margins_UNKNOWN')

    if shape_key in shape_map:
        _set_feature(vec, shape_map[shape_key])
    else:
        _set_feature(vec, 'mass shape_UNKNOWN')

    if density_key in density_map:
        _set_feature(vec, density_map[density_key])

    # Defaults for absent acquisition metadata.
    _set_feature(vec, 'image view_CC')
    _set_feature(vec, 'left or right breast_LEFT')

    # Optional numeric columns if present in encoded schema.
    for numeric_key in ('assessment', 'subtlety'):
        idx = CLINICAL_INDEX.get(numeric_key)
        if idx is not None:
            val = clinical.get(numeric_key, 3)
            try:
                vec[idx] = float(val)
            except (TypeError, ValueError):
                vec[idx] = 3.0

    return vec


def _load_research_model():
    global CLINICAL_COLS, CLINICAL_INDEX, REAL_MODEL, REAL_MODEL_META, MODEL_LOAD_ERROR

    try:
        if str(SAIMA_SRC) not in sys.path:
            sys.path.insert(0, str(SAIMA_SRC))

        from multimodal_framework import MultimodalClassifier, create_model_config

        CLINICAL_COLS = _load_clinical_columns()
        CLINICAL_INDEX = {name: idx for idx, name in enumerate(CLINICAL_COLS)}

        ckpt_path = _select_checkpoint()
        raw = torch.load(ckpt_path, map_location='cpu')
        state = raw.get('state_dict', {})
        hparams = raw.get('hyper_parameters', {})
        cfg = hparams.get('cfg', {})

        clinical_dim = int(cfg.get('clinical_input_dim', len(CLINICAL_COLS)))
        num_classes = int(cfg.get('num_classes', 3))
        backbone = cfg.get('image_backbone', 'densenet201')

        model_cfg = create_model_config(clinical_input_dim=clinical_dim, num_classes=num_classes)
        model_cfg['backbone'] = backbone
        model_cfg['fusion_dim'] = int(cfg.get('fusion_dim', model_cfg.get('fusion_dim', 256)))
        model_cfg['dropout'] = float(cfg.get('dropout', 0.4))

        model = MultimodalClassifier(model_cfg)
        model_state = {k.replace('model.', '', 1): v for k, v in state.items() if k.startswith('model.')}
        missing, unexpected = model.load_state_dict(model_state, strict=False)

        REAL_MODEL = model.to(DEVICE)
        REAL_MODEL.eval()
        REAL_MODEL_META = {
            'checkpoint': str(ckpt_path),
            'checkpoint_source': 'env' if CHECKPOINT_PATH_ENV else 'auto-selected-best',
            'backbone': backbone,
            'num_classes': num_classes,
            'clinical_dim': clinical_dim,
            'missing_keys_count': len(missing),
            'unexpected_keys_count': len(unexpected),
        }
        MODEL_LOAD_ERROR = None
        logger.info('Loaded research model from %s', ckpt_path)
    except Exception as exc:
        REAL_MODEL = None
        REAL_MODEL_META = {}
        MODEL_LOAD_ERROR = str(exc)
        logger.exception('Failed to load research model: %s', exc)


def infer_with_research_model(img_bytes, clinical):
    if REAL_MODEL is None:
        raise RuntimeError(MODEL_LOAD_ERROR or 'Research model is not loaded')

    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    image_tensor = INFER_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    clinical_vec = _build_clinical_vector(clinical)
    model_dim = int(REAL_MODEL_META.get('clinical_dim', len(clinical_vec)))
    if len(clinical_vec) < model_dim:
        clinical_vec = np.pad(clinical_vec, (0, model_dim - len(clinical_vec)), mode='constant')
    elif len(clinical_vec) > model_dim:
        clinical_vec = clinical_vec[:model_dim]

    clinical_tensor = torch.from_numpy(clinical_vec).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = REAL_MODEL(image_tensor, clinical_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    malignant_prob = float(probs[0])
    benign_prob = float(probs[1:].sum()) if probs.shape[0] > 1 else float(1.0 - malignant_prob)
    class_probs = {
        'malignant': round(float(probs[0]), 4),
        'benign': round(float(probs[1]), 4) if probs.shape[0] > 1 else None,
        'benign_without_callback': round(float(probs[2]), 4) if probs.shape[0] > 2 else None,
    }

    features = [
        {
            'name': 'Model Malignant Probability',
            'value': round(malignant_prob, 3),
            'level': 'high' if malignant_prob > 0.7 else 'medium' if malignant_prob > 0.4 else 'low',
        },
        {
            'name': 'Model Benign Probability',
            'value': round(benign_prob, 3),
            'level': 'high' if benign_prob > 0.7 else 'medium' if benign_prob > 0.4 else 'low',
        },
    ]

    return malignant_prob, features, class_probs


def _clamp01(value):
    return float(max(0.0, min(1.0, value)))


def infer_breast_cancer_risk(img_bytes, clinical):
    """
    Lightweight real-time breast-cancer risk estimation from uploaded mammogram.
    Replace this with your trained PyTorch model for production.
    """
    image = Image.open(io.BytesIO(img_bytes)).convert('L')
    image = image.resize((224, 224))
    arr = np.asarray(image, dtype=np.float32) / 255.0

    intensity_mean = float(arr.mean())
    intensity_std = float(arr.std())

    # Simple high-intensity hotspot ratio as a proxy for suspicious regions.
    hotspot_ratio = float((arr > 0.82).mean())

    age_raw = clinical.get('patientAge') if isinstance(clinical, dict) else None
    try:
        age = int(age_raw) if age_raw not in (None, '') else 50
    except (TypeError, ValueError):
        age = 50

    age_factor = _clamp01((age - 35) / 35.0)

    # Weighted score in [0, 1] for binary breast-cancer risk output.
    prob = _clamp01(
        0.35 * hotspot_ratio +
        0.30 * intensity_std +
        0.20 * intensity_mean +
        0.15 * age_factor
    )

    features = [
        {
            'name': 'High-Intensity Region Ratio',
            'value': round(hotspot_ratio, 3),
            'level': 'high' if hotspot_ratio > 0.12 else 'medium' if hotspot_ratio > 0.07 else 'low',
        },
        {
            'name': 'Texture Variability',
            'value': round(intensity_std, 3),
            'level': 'high' if intensity_std > 0.24 else 'medium' if intensity_std > 0.16 else 'low',
        },
        {
            'name': 'Global Intensity Pattern',
            'value': round(intensity_mean, 3),
            'level': 'high' if intensity_mean > 0.62 else 'medium' if intensity_mean > 0.45 else 'low',
        },
        {
            'name': 'Patient Age Signal',
            'value': round(age_factor, 3),
            'level': 'high' if age_factor > 0.7 else 'medium' if age_factor > 0.4 else 'low',
        },
    ]

    class_probs = {
        'malignant': round(float(prob), 4),
        'benign': round(float(1.0 - prob), 4),
        'benign_without_callback': None,
    }
    return prob, features, class_probs


_load_research_model()

# ─── Load models once at startup ──────────────────────────────────────────────
# MODEL_PATHS = {
#     'crossattn': 'models/cross_attention_fusion.pth',
#     'densenet':  'models/densenet201.pth',
#     'resnet':    'models/resnet50.pth',
#     'mobilenet': 'models/mobilenetv2.pth',
#     'inception': 'models/inceptionv3.pth',
# }
# loaded_models = {k: torch.load(v) for k, v in MODEL_PATHS.items()}
# print("[MammoXplain] Models loaded.")


@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint for frontend connection status indicator."""
    return jsonify({
        'status': 'ok',
        'service': 'MammoXplain Flask API',
        'device': str(DEVICE),
        'research_model_loaded': REAL_MODEL is not None,
        'research_model': REAL_MODEL_META,
        'research_model_error': MODEL_LOAD_ERROR,
    })


@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """
    Main diagnosis endpoint.
    Receives:
      - image: multipart file (PNG/JPG/DICOM)
      - model: string (crossattn | densenet | resnet | mobilenet | inception)
      - clinical: JSON string with clinical metadata fields:
          massMargin, massShape, breastDensity, calcType, calcDistribution, patientAge

    Returns:
      - label: 'Malignant' | 'Benign'
      - type: 'malignant' | 'benign'
      - confidence: float (0-100)
      - features: list of {name, value, level} for SHAP feature importances
      - notes: clinical summary string
    """
    image_file = request.files.get('image')
    model_id = request.form.get('model', 'crossattn')
    clinical_raw = request.form.get('clinical', '{}')
    try:
        clinical = json.loads(clinical_raw)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid clinical JSON format'}), 400

    if not image_file:
        return jsonify({'error': 'No image provided'}), 400

    if not image_file.filename:
        return jsonify({'error': 'Empty file name'}), 400

    allowed_suffixes = ('.png', '.jpg', '.jpeg', '.dcm')
    if not image_file.filename.lower().endswith(allowed_suffixes):
        return jsonify({'error': 'Unsupported file type. Use PNG/JPG/JPEG/DICOM'}), 400

    try:
        img_bytes = image_file.read()
        if REAL_MODEL is not None:
            prob, features, class_probs = infer_with_research_model(img_bytes, clinical)
            inference_source = 'saima_checkpoint'
            inference_notes = (
                'Prediction generated by the trained SaimaProject multimodal checkpoint '
                '(malignant vs benign collapse from multiclass output).'
            )
        else:
            prob, features, class_probs = infer_breast_cancer_risk(img_bytes, clinical)
            inference_source = 'heuristic_fallback'
            inference_notes = (
                'Research model unavailable, using fallback estimator. '
                'Check /api/health for model load error.'
            )
    except Exception as exc:
        return jsonify({'error': f'Image processing failed: {exc}'}), 400

    label = 'Malignant' if prob > 0.5 else 'Benign'
    result = {
        'label': label,
        'type': label.lower(),
        'confidence': round(prob * 100, 1),
        'model': model_id,
        'inference_source': inference_source,
        'class_probabilities': class_probs,
        'features': features,
        'notes': inference_notes,
        'received_file': image_file.filename,
        'gradcam_url': None,   # Set to '/api/gradcam/<case_id>' when integrated
        'shap_url': None,      # Set to '/api/shap/<case_id>' when integrated
    }
    return jsonify(result)


@app.route('/api/compare', methods=['POST'])
def compare_models():
    """
    Run the same case through all five models and return
    a comparison table. Body: same as /api/diagnose.
    """
    # Replace with real multi-model inference
    demo_results = [
        {'model': 'Cross-Attention Fusion', 'acc': 96.8, 'auc': 0.981, 'confidence': 91},
        {'model': 'DenseNet201', 'acc': 91.4, 'auc': 0.942, 'confidence': 87},
        {'model': 'ResNet50', 'acc': 89.7, 'auc': 0.921, 'confidence': 84},
        {'model': 'InceptionV3', 'acc': 90.1, 'auc': 0.930, 'confidence': 85},
        {'model': 'MobileNetV2', 'acc': 87.3, 'auc': 0.905, 'confidence': 81},
    ]
    return jsonify({'results': demo_results})


@app.route('/api/gradcam/<case_id>', methods=['GET'])
def gradcam(case_id):
    """Returns Grad-CAM heatmap as base64 PNG for a given case."""
    # Integrate your Grad-CAM generation here
    # cam = generate_gradcam(loaded_models['crossattn'], case_id)
    # buffered = io.BytesIO()
    # cam.save(buffered, format='PNG')
    # img_str = base64.b64encode(buffered.getvalue()).decode()
    # return jsonify({'image': f'data:image/png;base64,{img_str}'})
    return jsonify({'error': 'Grad-CAM not yet integrated'}), 501


@app.route('/api/shap/<case_id>', methods=['GET'])
def shap_report(case_id):
    """Returns full SHAP waterfall data for a given case."""
    return jsonify({'error': 'SHAP endpoint not yet integrated'}), 501


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
