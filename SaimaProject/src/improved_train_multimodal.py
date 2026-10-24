"""
IMPROVED Training script for multimodal breast cancer diagnosis.
Phase 1: Quick Wins Implementation
- Class weights for imbalance
- Increased epochs (100 vs 2)
- Data augmentation
- Learning rate scheduling

Author: Breast Cancer AI Project
Date: May 6, 2026
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torchvision import transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import argparse
from datetime import datetime
import json

# Import our multimodal framework
from multimodal_framework import (
    MultimodalClassifier,
    BreastCancerDataset,
    MultimodalLightningModule,
    create_model_config
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PHASE 1: IMPROVED HYPERPARAMETERS
# ============================================================================

IMPROVED_CONFIG = {
    'max_epochs': 150,                    # INCREASED: From 100 to 150 for better convergence
    'batch_size': 32,                     # INCREASED: From 16 to 32 for better gradients
    'learning_rate': 0.0005,              # DECREASED: From 0.001 to 0.0005 for stability
    'weight_decay': 0.00005,              # DECREASED: From 0.0001 to 0.00005 for light regularization
    'optimizer': 'adam',
    'use_class_weights': True,            # ENABLED: Handles class imbalance
    'use_data_augmentation': True,         # ENABLED: Advanced augmentation
    'use_lr_scheduler': True,              # ENABLED: Learning rate scheduling
    'early_stopping_patience': 25,         # INCREASED: From 15 to 25 for patient convergence
    'image_size': (224, 224),
    'num_workers': 2,                     # CPU-friendly setting
}

# ============================================================================
# PART 1: DATA AUGMENTATION (Medical Imaging Specific)
# ============================================================================

def get_train_transforms(image_size=(224, 224)):
    """
    Advanced data augmentation for medical images - ENHANCED VERSION.
    
    Medical-specific transforms that preserve pathology:
    - Rotation: ±20° (increased from ±15°)
    - Gaussian noise: Simulates imaging variability
    - Elastic deformation: Non-rigid transformations
    - Contrast: Stronger variation
    """
    return A.Compose([
        A.Rotate(limit=20, p=0.7, border_mode=0),
        A.GaussNoise(p=0.5, std_range=(0.05, 0.2)),
        A.GaussianBlur(blur_limit=3, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.ElasticTransform(p=0.3, alpha=30, sigma=5),
        A.Affine(shear=10, p=0.3),
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ], bbox_params=None)


def get_val_transforms(image_size=(224, 224)):
    """Validation transforms (no augmentation, only normalization)."""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


# ============================================================================
# PART 2: CLASS WEIGHT CALCULATION FOR IMBALANCE
# ============================================================================

def calculate_class_weights(y_train):
    """
    Calculate class weights inversely proportional to class frequency.
    Weights ordered to match label_map: BENIGN=0, MALIGNANT=1, BENIGN_WITHOUT_CALLBACK=2
    """
    # Must match BreastCancerDataset.label_map order
    label_map = {'BENIGN': 0, 'MALIGNANT': 1, 'BENIGN_WITHOUT_CALLBACK': 2}
    ordered_classes = np.array(['BENIGN', 'MALIGNANT', 'BENIGN_WITHOUT_CALLBACK'])

    weights = compute_class_weight(
        'balanced',
        classes=ordered_classes,
        y=y_train
    )

    logger.info("=" * 70)
    logger.info("CLASS WEIGHTS (for loss function)")
    logger.info("=" * 70)
    for cls, weight in zip(ordered_classes, weights):
        idx = label_map[cls]
        logger.info(f"  label={idx} ({cls}): weight = {weight:.4f}")
    logger.info("=" * 70)

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# PART 3: IMPROVED DATA LOADING
# ============================================================================

def load_and_preprocess_data(
    data_path, 
    dataset_type='mass', 
    test_size=0.2, 
    val_size=0.1,
    use_augmentation=True
):
    """
    Load and preprocess clinical data with augmentation support.
    """
    logger.info(f"Loading {dataset_type} dataset from {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Log class distribution BEFORE splitting
    logger.info("\n" + "=" * 70)
    logger.info("CLASS DISTRIBUTION (BEFORE SPLIT)")
    logger.info("=" * 70)
    for cls, count in df['pathology'].value_counts().items():
        pct = 100 * count / len(df)
        logger.info(f"{cls:30s}: {count:4d} ({pct:5.1f}%)")
    logger.info("=" * 70 + "\n")

    # Identify feature columns
    feature_cols = [col for col in df.columns if col not in ['patient_id', 'pathology']]

    # Stratified split to preserve class distribution
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df['pathology']
    )

    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size_adjusted, 
        random_state=42,
        stratify=train_val_df['pathology']
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Log class distribution in train set
    logger.info("\nClass distribution in TRAINING set:")
    for cls, count in train_df['pathology'].value_counts().items():
        pct = 100 * count / len(train_df)
        logger.info(f"  {cls:30s}: {count:4d} ({pct:5.1f}%)")

    return train_df, val_df, test_df, feature_cols


def create_data_loaders(
    train_df, 
    val_df, 
    test_df, 
    feature_cols, 
    config,
    use_augmentation=True
):
    """Create PyTorch data loaders with augmentation."""
    
    # Determine transforms
    if use_augmentation:
        train_transform = get_train_transforms(config['image_size'])
        val_transform = get_val_transforms(config['image_size'])
    else:
        val_transform = get_val_transforms(config['image_size'])
        train_transform = val_transform
    
    image_dir = None
    if config.get('image_dir'):
        image_dir_path = Path(config['image_dir'])
        if image_dir_path.exists() and image_dir_path.is_dir():
            image_files = (
                list(image_dir_path.rglob('*.png')) +
                list(image_dir_path.rglob('*.jpg')) +
                list(image_dir_path.rglob('*.jpeg'))
            )
            if image_files:
                logger.info(f"Found {len(image_files)} images in {image_dir_path}")
                image_dir = image_dir_path

    # Create datasets
    train_dataset = BreastCancerDataset(
        clinical_df=train_df,
        image_dir=image_dir,
        transform=train_transform,
        mode='train'
    )
    val_dataset = BreastCancerDataset(
        clinical_df=val_df,
        image_dir=image_dir,
        transform=val_transform,
        mode='val'
    )
    test_dataset = BreastCancerDataset(
        clinical_df=test_df,
        image_dir=image_dir,
        transform=val_transform,
        mode='test'
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0)
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# PART 4: IMPROVED LIGHTNING MODULE WITH CLASS WEIGHTS
# ============================================================================

class ImprovedMultimodalLightningModule(pl.LightningModule):
    """Lightning module with class weights and learning rate scheduling."""
    
    def __init__(self, model, config, class_weights=None):
        super().__init__()
        self.model = model
        self.config = config
        self.automatic_optimization = True
        
        # Loss function WITH CLASS WEIGHTS
        if class_weights is not None:
            logger.info(f"Using class weights: {class_weights}")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            logger.info("Using unweighted cross-entropy loss")
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.val_accuracies = []
        self.test_accuracies = []

    def forward(self, images, clinical_features):
        return self.model(images, clinical_features)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        clinical_features = batch['clinical_features']
        labels = batch['label']

        logits, _ = self.model(images, clinical_features)
        loss = self.criterion(logits, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', (torch.argmax(logits, dim=1) == labels).float().mean(), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        clinical_features = batch['clinical_features']
        labels = batch['label']

        logits, _ = self.model(images, clinical_features)
        loss = self.criterion(logits, labels)

        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', accuracy, on_epoch=True)
        self.val_accuracies.append(accuracy.detach().cpu())

        return loss

    def test_step(self, batch, batch_idx):
        images = batch['image']
        clinical_features = batch['clinical_features']
        labels = batch['label']

        logits, _ = self.model(images, clinical_features)
        loss = self.criterion(logits, labels)

        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()

        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        self.test_accuracies.append(accuracy.detach().cpu())

        return loss

    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        if self.config.get('use_lr_scheduler', False):
            # Cosine annealing learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['max_epochs'],
                eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'monitor': 'val_loss'
                }
            }
        
        return optimizer


# ============================================================================
# PART 5: MAIN TRAINING FUNCTION
# ============================================================================

def train_improved_model(
    data_path='data/processed/csv/mass_train_encoded.csv',
    image_dir='data/processed/images',
    checkpoint_dir='checkpoints/mass_densenet201_improved',
    logs_dir='results/logs/mass_multimodal_improved',
    config=None
):
    """
    Train improved multimodal model with Phase 1 optimizations.
    
    Improvements:
    1. Class weights for imbalance
    2. 100 epochs instead of 2
    3. Data augmentation
    4. Learning rate scheduling
    """
    
    if config is None:
        config = IMPROVED_CONFIG.copy()
    
    config['image_dir'] = image_dir
    
    logger.info("=" * 70)
    logger.info("IMPROVED MULTIMODAL TRAINING - PHASE 1")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\nConfiguration:")
    for key, value in config.items():
        logger.info(f"  {key:30s}: {value}")
    logger.info("=" * 70 + "\n")

    # Load data
    train_df, val_df, test_df, feature_cols = load_and_preprocess_data(
        data_path=data_path,
        use_augmentation=config['use_data_augmentation']
    )

    # Get class weights
    y_train = train_df['pathology'].values
    class_weights = None
    if config['use_class_weights']:
        class_weights = calculate_class_weights(y_train)
        class_weights = class_weights.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        config=config,
        use_augmentation=config['use_data_augmentation']
    )

    # Create model config using clinical feature dimension
    model_config = create_model_config(
        clinical_input_dim=len(feature_cols),
        num_classes=3
    )
    model_config.update({
        'backbone': 'densenet201',
        'pretrained': True,
        'image_feature_dim': 512,
        'clinical_hidden_dims': [256, 128],
        'clinical_feature_dim': 128,
        'fusion_dim': 256
    })
    model = MultimodalClassifier(model_config)

    # Create Lightning module
    lightning_module = ImprovedMultimodalLightningModule(
        model=model,
        config=config,
        class_weights=class_weights
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}',
        save_top_k=3,
        monitor='val_acc',
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        mode='min',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Setup logger
    try:
        tb_logger = TensorBoardLogger(logs_dir, name='improved_model')
    except ModuleNotFoundError:
        logger.warning('TensorBoard is not installed. Falling back to CSVLogger.')
        tb_logger = CSVLogger(logs_dir, name='improved_model')

    # Train
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=tb_logger,
        enable_progress_bar=True,
        accelerator='auto'
    )

    logger.info("\nStarting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Test
    logger.info("\nEvaluating on test set...")
    test_results = trainer.test(lightning_module, test_loader)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Test Results: {test_results}")
    logger.info("=" * 70)

    return model, test_results


# ============================================================================
# PART 6: COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Improved multimodal breast cancer classification training'
    )
    parser.add_argument(
        '--data-path',
        default='data/processed/csv/mass_train_encoded.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--image-dir',
        default='data/processed/images',
        help='Directory containing processed images'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--no-class-weights',
        action='store_true',
        help='Disable class weights'
    )
    parser.add_argument(
        '--no-lr-scheduler',
        action='store_true',
        help='Disable learning rate scheduler'
    )

    args = parser.parse_args()

    # Build config
    config = IMPROVED_CONFIG.copy()
    config['max_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['use_data_augmentation'] = not args.no_augmentation
    config['use_class_weights'] = not args.no_class_weights
    config['use_lr_scheduler'] = not args.no_lr_scheduler

    # Train
    model, results = train_improved_model(
        data_path=args.data_path,
        image_dir=args.image_dir,
        config=config
    )
