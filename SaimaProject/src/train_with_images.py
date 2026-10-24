"""
Training script using real CBIS-DDSM images.
Uses mass_train_mapped.csv + mass_test_mapped.csv.
Supports: class weights, augmentation, LR scheduling, early stopping.
"""

import sys
import logging
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.functional import f1_score, recall
import pydicom

sys.path.insert(0, str(Path(__file__).parent))
from multimodal_framework import MultimodalClassifier, create_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV   = PROJECT_ROOT / "data/processed/csv/mass_train_mapped.csv"
TEST_CSV    = PROJECT_ROOT / "data/processed/csv/mass_test_mapped.csv"
CKPT_DIR    = PROJECT_ROOT / "checkpoints/multimodal_real_images"
LOG_DIR     = PROJECT_ROOT / "results/logs/multimodal_real_images"
DATASET_SEARCH_ROOTS = [PROJECT_ROOT / "datasets", PROJECT_ROOT / "data"]

LABEL_MAP   = {"MALIGNANT": 0, "BENIGN": 1, "BENIGN_WITHOUT_CALLBACK": 2}
IMG_SIZE    = (320, 320)

CFG = {
    "max_epochs":            120,
    "batch_size":            12,   # reduced: 320x320 images use more GPU memory
    "lr":                    1e-4,
    "weight_decay":          5e-4,
    "early_stop_patience":   30,
    "val_split":             0.15,
    "num_workers":           4,
    "image_backbone":        "densenet201",
    "clinical_input_dim":    45,
    "fusion_dim":            256,
    "num_classes":           3,
    "dropout":               0.5,
    "clinical_dropout_p":    0.5,
    "warmup_image_only_epochs": 30,
    "use_weighted_sampler":  True,
    "label_smoothing":       0.05,
    "loss_type":             "focal",  # focal | ce
    "focal_gamma":           2.0,
    "accumulate_grad_batches": 2,
    "early_stop_monitor":    "val_acc",
}
# ─────────────────────────────────────────────────────────────────────────────

# Clinical feature columns (48-dim one-hot from encoded CSV)
_ENCODED_CSV = Path("data/processed/csv/mass_train_encoded.csv")


def get_clinical_cols() -> list[str]:
    ref = pd.read_csv(_ENCODED_CSV, nrows=1)
    # Exclude patient_id, the raw pathology column, AND any one-hot encoded
    # pathology_* columns — these are the target label and must never be
    # fed as input features (label leakage).
    return [c for c in ref.columns
            if c != "patient_id" and not c.startswith("pathology")]


CLINICAL_COLS = get_clinical_cols()


def get_train_aug():
    return A.Compose([
        A.Rotate(limit=35, p=0.8, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=0,
            p=0.6,
        ),
        A.RandomResizedCrop(
            size=IMG_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            p=0.55,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.GaussNoise(p=0.5, std_range=(0.05, 0.2)),
        A.GaussianBlur(blur_limit=5, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.35, p=0.6),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(24, 64),
            hole_width_range=(24, 64),
            fill=0,
            p=0.35,
        ),
        A.Resize(*IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_aug():
    return A.Compose([
        A.Resize(*IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class MappedDataset(Dataset):
    """Dataset that reads image_path directly from mapped CSV."""

    def __init__(self, df: pd.DataFrame, encoded_df: pd.DataFrame,
                 transform=None, augment=None, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.encoded = encoded_df.set_index("patient_id")
        self.transform = transform
        self.augment = augment
        self.is_train = is_train
        self.clinical_impute = self._build_clinical_impute_values(encoded_df)
        self._path_cache = {}

    @staticmethod
    def _build_clinical_impute_values(encoded_df: pd.DataFrame) -> np.ndarray:
        """Build per-feature imputation values from encoded clinical data."""
        impute = np.zeros(len(CLINICAL_COLS), dtype=np.float32)

        for i, col in enumerate(CLINICAL_COLS):
            if col not in encoded_df.columns:
                continue

            series = pd.to_numeric(encoded_df[col], errors="coerce").dropna()
            if series.empty:
                impute[i] = 0.0
                continue

            values = np.unique(series.values)
            if set(values.tolist()).issubset({0.0, 1.0}):
                # One-hot columns use mode imputation.
                impute[i] = float(series.mode(dropna=True).iloc[0])
            else:
                # Continuous / ordinal columns (e.g., assessment, subtlety) use median.
                impute[i] = float(series.median())

        return impute

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        cache_key = row.get("image_path")
        if isinstance(cache_key, str) and cache_key in self._path_cache:
            return self._path_cache[cache_key]

        raw_path = row.get("image_path")
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path.strip())
            if candidate.exists():
                self._path_cache[cache_key] = candidate
                return candidate

        rel_path = row.get("image file path")
        if isinstance(rel_path, str) and rel_path.strip():
            rel = Path(rel_path.strip())
            search_candidates = [rel, rel.name, rel.parent.name, rel.parent.parent.name]
            for root in DATASET_SEARCH_ROOTS:
                for candidate in search_candidates:
                    abs_candidate = (root / candidate).resolve()
                    if abs_candidate.exists():
                        self._path_cache[cache_key] = abs_candidate
                        return abs_candidate

            for root in DATASET_SEARCH_ROOTS:
                if not root.exists():
                    continue
                for match in root.rglob(rel.name):
                    if match.is_file():
                        self._path_cache[cache_key] = match.resolve()
                        return match.resolve()
                for match in root.rglob(f"*{rel.parent.name}*"):
                    if match.is_file() and match.suffix.lower() in {".dcm", ".png", ".jpg", ".jpeg"}:
                        self._path_cache[cache_key] = match.resolve()
                        return match.resolve()
                for part in [rel.parent.name, rel.parent.parent.name, rel.name]:
                    if not part:
                        continue
                    for match in root.rglob(f"*{part}*"):
                        if match.is_file() and match.suffix.lower() in {".dcm", ".png", ".jpg", ".jpeg"}:
                            self._path_cache[cache_key] = match.resolve()
                            return match.resolve()

        lookup_key = row.get("lookup_key")
        search_terms = []
        if isinstance(lookup_key, str) and lookup_key.strip():
            search_terms.append(lookup_key.strip())
            search_terms.append(lookup_key.split("_")[-1])
        if isinstance(row.get("image file path"), str) and row.get("image file path").strip():
            rel = Path(row.get("image file path").strip())
            if rel.parts:
                search_terms.append(rel.parts[0])
        if isinstance(row.get("patient_id"), str) and row.get("patient_id").strip():
            search_terms.append(row.get("patient_id").strip())

        for term in dict.fromkeys(search_terms):
            if not term:
                continue
            for root in DATASET_SEARCH_ROOTS:
                if not root.exists():
                    continue
                for match in root.rglob(f"*{term}*"):
                    if match.is_dir():
                        for image in sorted(match.rglob("*")):
                            if image.is_file() and image.suffix.lower() in {".dcm", ".png", ".jpg", ".jpeg"}:
                                self._path_cache[cache_key] = image.resolve()
                                return image.resolve()
                    if match.is_file() and match.suffix.lower() in {".dcm", ".png", ".jpg", ".jpeg"}:
                        self._path_cache[cache_key] = match.resolve()
                        return match.resolve()

        raise FileNotFoundError(f"Could not resolve image for patient {row.get('patient_id')}")

    def _load_image(self, path: Path) -> Image.Image:
        if path.suffix.lower() == ".dcm":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dcm = pydicom.dcmread(str(path))
            arr = dcm.pixel_array.astype(np.float32)
            if hasattr(dcm, "PhotometricInterpretation") and dcm.PhotometricInterpretation == "MONOCHROME1":
                arr = arr.max() - arr
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            else:
                arr = np.zeros_like(arr)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=2)
            return Image.fromarray(np.uint8(arr * 255)).convert("RGB")

        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["patient_id"]

        # ── Image ──────────────────────────────────────────────────────────
        img_path = self._resolve_image_path(row)
        img = self._load_image(img_path)
        img_np = np.array(img)

        if self.augment:
            img_tensor = self.augment(image=img_np)["image"]
        elif self.transform:
            img_tensor = self.transform(image=img_np)["image"]
        else:
            img_tensor = transforms.ToTensor()(img)

        # ── Clinical features ───────────────────────────────────────────────
        if pid in self.encoded.index:
            enc_row = self.encoded.loc[pid]
            if isinstance(enc_row, pd.DataFrame):
                enc_row = enc_row.iloc[0]
            clinical = pd.to_numeric(enc_row[CLINICAL_COLS], errors="coerce").values.astype(np.float32)
            nan_mask = np.isnan(clinical)
            if nan_mask.any():
                clinical[nan_mask] = self.clinical_impute[nan_mask]
        else:
            # If a patient_id has no encoded row, fall back to learned imputation priors.
            clinical = self.clinical_impute.copy()

        # Force image reliance: randomly zero clinical 30% of train batches
        if self.is_train and np.random.rand() < CFG["clinical_dropout_p"]:
            clinical = np.zeros_like(clinical)

        label = LABEL_MAP[row["pathology"]]

        return {
            "image":             img_tensor,
            "clinical_features": torch.tensor(clinical),
            "label":             torch.tensor(label, dtype=torch.long),
            "patient_id":        pid,
        }


class WarmupAwareEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores epochs during image-only warmup."""
    def __init__(self, warmup_epochs: int, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            return
        super().on_validation_end(trainer, pl_module)


class FocalCrossEntropyLoss(nn.Module):
    """Cross-entropy focal variant for class imbalance."""

    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def split_train_val_by_patient(df: pd.DataFrame, val_split: float, random_state: int = 42):
    """Split train/val by patient_id to reduce patient-level leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def parse_seed_list(seed_str: str):
    seeds = []
    for part in seed_str.split(','):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if len(seeds) == 0:
        raise ValueError("At least one seed must be provided.")
    return seeds


def assert_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fail fast if identifiers overlap across splits."""
    checks = [
        ("patient_id", train_df["patient_id"], val_df["patient_id"], "train", "val"),
        ("patient_id", train_df["patient_id"], test_df["patient_id"], "train", "test"),
        ("patient_id", val_df["patient_id"], test_df["patient_id"], "val", "test"),
        ("image_path", train_df["image_path"], val_df["image_path"], "train", "val"),
        ("image_path", train_df["image_path"], test_df["image_path"], "train", "test"),
        ("image_path", val_df["image_path"], test_df["image_path"], "val", "test"),
    ]

    for col, a, b, name_a, name_b in checks:
        overlap = set(a.astype(str)) & set(b.astype(str))
        if overlap:
            raise ValueError(
                f"Potential leakage: {len(overlap)} overlapping {col} values between {name_a} and {name_b}."
            )


class MultimodalLit(pl.LightningModule):

    def __init__(self, model, class_weights, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        if cfg.get("loss_type", "focal") == "focal":
            self.criterion = FocalCrossEntropyLoss(
                weight=class_weights,
                gamma=float(cfg.get("focal_gamma", 2.0)),
                label_smoothing=float(cfg.get("label_smoothing", 0.0)),
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=float(cfg.get("label_smoothing", 0.0)),
            )
        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, images, clinical):
        logits, _ = self.model(images, clinical)
        return logits

    def _step(self, batch, stage):
        warmup = self.cfg.get("warmup_image_only_epochs", 0)
        if self.current_epoch < warmup:
            clinical = torch.zeros_like(batch["clinical_features"])
        else:
            clinical = batch["clinical_features"]
        logits = self(batch["image"], clinical)
        loss   = self.criterion(logits, batch["label"])
        preds  = logits.argmax(dim=1)
        acc    = (preds == batch["label"]).float().mean()
        macro_f1 = f1_score(
            preds,
            batch["label"],
            task="multiclass",
            num_classes=int(self.cfg.get("num_classes", 3)),
            average="macro",
        )
        macro_recall = recall(
            preds,
            batch["label"],
            task="multiclass",
            num_classes=int(self.cfg.get("num_classes", 3)),
            average="macro",
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_f1", macro_f1, prog_bar=(stage != "train"), on_epoch=True, on_step=False)
        self.log(f"{stage}_recall", macro_recall, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    def configure_optimizers(self):
        backbone_params = list(self.model.image_encoder.backbone.parameters())
        head_params = (
            list(self.model.image_encoder.feature_proj.parameters()) +
            list(self.model.image_encoder.norm.parameters()) +
            list(self.model.clinical_encoder.parameters()) +
            list(self.model.fusion.parameters()) +
            list(self.model.classifier.parameters())
        )
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.cfg["lr"] * 0.1},  # backbone: 10x lower LR
            {"params": head_params,     "lr": self.cfg["lr"]},
        ], weight_decay=self.cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=30, T_mult=2, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated random seeds, e.g. '42,43,44'.",
    )
    parser.add_argument(
        "--multi_seed_summary_out",
        type=str,
        default="results/multi_seed_summary.json",
        help="Path to save aggregated mean±std metrics across seeds.",
    )
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    log.info(f"Running seeds: {seeds}")

    # ── Load CSVs ──────────────────────────────────────────────────────────
    train_full = pd.read_csv(TRAIN_CSV)
    test_df    = pd.read_csv(TEST_CSV)
    enc_train  = pd.read_csv(_ENCODED_CSV)
    enc_test   = pd.read_csv(Path("data/processed/csv/mass_test_encoded.csv"))
    enc_all    = pd.concat([enc_train, enc_test], ignore_index=True)

    classes = np.array(["MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"])
    all_seed_results = []

    for seed in seeds:
        pl.seed_everything(seed)

        # ── Train / Val split ──────────────────────────────────────────────
        train_df, val_df = split_train_val_by_patient(
            train_full,
            val_split=CFG["val_split"],
            random_state=seed,
        )
        assert_no_overlap(train_df, val_df, test_df)
        log.info(f"[seed={seed}] Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

        # ── Class weights ──────────────────────────────────────────────────
        weights = compute_class_weight("balanced", classes=classes, y=train_df["pathology"].values)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        log.info(f"[seed={seed}] Class weights: {dict(zip(classes, weights.round(3)))}")
        log.info(f"[seed={seed}] Loss type: {CFG['loss_type']} (gamma={CFG['focal_gamma']})")

        # ── Datasets ───────────────────────────────────────────────────────
        train_ds = MappedDataset(train_df, enc_all, augment=get_train_aug(), is_train=True)
        val_ds   = MappedDataset(val_df,   enc_all, augment=get_val_aug(),   is_train=False)
        test_ds  = MappedDataset(test_df,  enc_all, augment=get_val_aug(),   is_train=False)

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                                  shuffle=True,  num_workers=CFG["num_workers"], drop_last=True)

        if CFG.get("use_weighted_sampler", True):
            sample_weights = train_df["pathology"].map(dict(zip(classes, weights))).astype(np.float32).values
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=CFG["batch_size"],
                sampler=sampler,
                shuffle=False,
                num_workers=CFG["num_workers"],
                drop_last=True,
            )

        val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                                  shuffle=False, num_workers=CFG["num_workers"])
        test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"],
                                  shuffle=False, num_workers=CFG["num_workers"])

        # ── Model ──────────────────────────────────────────────────────────
        model_cfg = create_model_config(
            clinical_input_dim=CFG["clinical_input_dim"],
            num_classes=CFG["num_classes"],
        )
        model_cfg["backbone"] = CFG["image_backbone"]
        model_cfg["dropout"] = CFG["dropout"]
        model_cfg["fusion_dim"] = CFG["fusion_dim"]
        model = MultimodalClassifier(model_cfg)
        lit   = MultimodalLit(model, class_weights, CFG)

        # ── Callbacks ──────────────────────────────────────────────────────
        seed_ckpt_dir = CKPT_DIR / f"seed_{seed}"
        seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks = [
            ModelCheckpoint(
                dirpath=str(seed_ckpt_dir),
                filename="epoch={epoch:02d}-val_acc={val_acc:.3f}-val_loss={val_loss:.3f}",
                monitor=CFG["early_stop_monitor"], mode="max", save_top_k=3, auto_insert_metric_name=False,
            ),
            WarmupAwareEarlyStopping(
                warmup_epochs=CFG["warmup_image_only_epochs"],
                monitor=CFG["early_stop_monitor"], patience=CFG["early_stop_patience"], mode="max",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # ── Trainer ────────────────────────────────────────────────────────
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        trainer = pl.Trainer(
            max_epochs=CFG["max_epochs"],
            callbacks=callbacks,
            logger=CSVLogger(str(LOG_DIR), name=f"multimodal_real_seed_{seed}"),
            log_every_n_steps=10,
            accumulate_grad_batches=int(CFG.get("accumulate_grad_batches", 1)),
            gradient_clip_val=1.0,
            deterministic=False,
        )

        log.info("=" * 60)
        log.info(f"Starting training with REAL mammography images (seed={seed})")
        log.info(f"Backbone : {CFG['image_backbone']}")
        log.info(f"Epochs   : {CFG['max_epochs']}  Batch: {CFG['batch_size']}")
        log.info("=" * 60)

        trainer.fit(lit, train_loader, val_loader)

        # ── Test ───────────────────────────────────────────────────────────
        log.info(f"\n[seed={seed}] Running test set evaluation...")
        test_results = trainer.test(lit, test_loader, ckpt_path="best")
        seed_metrics = test_results[0] if test_results else {}
        seed_record = {
            "seed": seed,
            "best_checkpoint": callbacks[0].best_model_path,
            "test_results": seed_metrics,
        }
        all_seed_results.append(seed_record)
        log.info(f"[seed={seed}] Test results: {seed_metrics}")
        log.info(f"[seed={seed}] Best checkpoint: {callbacks[0].best_model_path}")

    # ── Aggregate mean ± std over seeds ────────────────────────────────────
    metric_keys = sorted(
        {
            key
            for rec in all_seed_results
            for key in rec.get("test_results", {}).keys()
            if key.startswith("test_")
        }
    )
    aggregate = {}
    for key in metric_keys:
        vals = [rec["test_results"][key] for rec in all_seed_results if key in rec.get("test_results", {})]
        if len(vals) == 0:
            continue
        arr = np.array(vals, dtype=np.float64)
        aggregate[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "n": int(arr.size),
        }

    log.info("=" * 60)
    log.info("MULTI-SEED SUMMARY (mean ± std)")
    log.info("=" * 60)
    for key in metric_keys:
        if key not in aggregate:
            continue
        m = aggregate[key]["mean"]
        s = aggregate[key]["std"]
        log.info(f"{key:15s}: {m:.4f} ± {s:.4f}")

    out = {
        "config": CFG,
        "seeds": seeds,
        "seed_runs": all_seed_results,
        "aggregate_mean_std": aggregate,
    }
    out_path = Path(args.multi_seed_summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Keep legacy output path for compatibility.
    with open(LOG_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    log.info(f"Saved multi-seed summary: {out_path}")
    log.info("Training complete.")


if __name__ == "__main__":
    main()
