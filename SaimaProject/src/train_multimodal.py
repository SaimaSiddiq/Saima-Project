"""
Training script for the multimodal breast cancer diagnosis framework.
Implements training pipeline with data loading, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import argparse
from datetime import datetime

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


def load_and_preprocess_data(data_path, dataset_type='mass', test_size=0.2, val_size=0.1):
    """
    Load and preprocess clinical data for training.

    Args:
        data_path (str): Path to processed CSV data
        dataset_type (str): 'mass' or 'calc' for dataset type
        test_size (float): Test set proportion
        val_size (float): Validation set proportion

    Returns:
        tuple: (train_df, val_df, test_df, feature_columns)
    """
    logger.info(f"Loading {dataset_type} dataset from {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Identify feature columns
    feature_cols = [col for col in df.columns if col not in ['patient_id', 'pathology']]

    # Split data
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['pathology']
    )

    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=42,
        stratify=train_val_df['pathology']
    )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df, feature_cols


def create_data_loaders(train_df, val_df, test_df, feature_cols, config):
    """
    Create PyTorch data loaders.

    Args:
        train_df, val_df, test_df: DataFrames for each split
        feature_cols: List of feature column names
        config: Model configuration

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    transform = None
    image_dir = None
    
    if config.get('image_dir'):
        image_dir_path = Path(config['image_dir'])
        if image_dir_path.exists() and image_dir_path.is_dir():
            image_files = list(image_dir_path.rglob('*.png')) + list(image_dir_path.rglob('*.jpg')) + list(image_dir_path.rglob('*.jpeg'))
            if image_files:
                logger.info(f"Found {len(image_files)} image files in {image_dir_path}")
                image_dir = image_dir_path
                transform = T.Compose([
                    T.Resize(config.get('image_size', (224, 224))),
                    T.CenterCrop(config.get('image_size', (224, 224))),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                logger.warning(f"No image files found in {image_dir_path}. Using dummy images for training.")
        else:
            logger.warning(f"Image directory not found: {image_dir_path}. Using dummy images for training.")

    # Create datasets
    train_dataset = BreastCancerDataset(
        clinical_df=train_df,
        image_dir=image_dir,
        transform=transform,
        mode='train'
    )

    val_dataset = BreastCancerDataset(
        clinical_df=val_df,
        image_dir=image_dir,
        transform=transform,
        mode='val'
    )

    test_dataset = BreastCancerDataset(
        clinical_df=test_df,
        image_dir=image_dir,
        transform=transform,
        mode='test'
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for debugging
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Disable multiprocessing for debugging
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Disable multiprocessing for debugging
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def setup_callbacks_and_logger(experiment_name):
    """
    Setup training callbacks and logger.

    Args:
        experiment_name (str): Name for the experiment

    Returns:
        tuple: (callbacks, logger)
    """
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{experiment_name}',
        filename='{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    # Try to use TensorBoard logger, fallback to CSV logger
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        pl_logger = TensorBoardLogger(
            save_dir='results/logs',
            name=experiment_name
        )
        logging.info("Using TensorBoard logger")
    except ImportError:
        from pytorch_lightning.loggers import CSVLogger
        pl_logger = CSVLogger(
            save_dir='results/logs',
            name=experiment_name
        )
        logging.info("Using CSV logger (TensorBoard not available)")

    return callbacks, pl_logger


def train_model(config, train_loader, val_loader, experiment_name):
    """
    Train the multimodal model.

    Args:
        config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        experiment_name: Name for the experiment

    Returns:
        trainer: Trained PyTorch Lightning trainer
    """
    # Create model
    model = MultimodalLightningModule(config)

    # Setup callbacks and logger
    callbacks, pl_logger = setup_callbacks_and_logger(experiment_name)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices='auto',
        strategy='auto',
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train model
    logging.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    return trainer


def evaluate_model(trainer, test_loader, experiment_name):
    """
    Evaluate the trained model on test set.

    Args:
        trainer: Trained PyTorch Lightning trainer
        test_loader: Test data loader
        experiment_name: Experiment name
    """
    logger.info("Evaluating on test set...")

    # Test the model
    test_results = trainer.test(dataloaders=test_loader, verbose=True)

    # Save test results
    results_path = Path(f'results/reports/{experiment_name}_test_results.txt')
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        f.write(f"Test Results for {experiment_name}\n")
        f.write("=" * 50 + "\n")
        for key, value in test_results[0].items():
            f.write(f"{key}: {value:.4f}\n")

    logger.info(f"Test results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Breast Cancer Model')
    parser.add_argument('--dataset', type=str, default='mass',
                       choices=['mass', 'calc'], help='Dataset type')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'resnet34', 'densenet201', 'inception_v3', 'efficientnet_b0'],
                       help='Backbone model for image encoder')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pretrained backbone weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Do not use pretrained backbone weights')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--num_classes', type=int, default=2, help='Number of target classes')
    parser.add_argument('--image_dir', type=str, default='data/processed/images', help='Directory containing processed images')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    pl.seed_everything(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{args.dataset}_multimodal_{timestamp}"
    else:
        experiment_name = args.experiment_name

    logger.info(f"Starting experiment: {experiment_name}")

    # Load and preprocess data
    data_path = f'data/processed/csv/{args.dataset}_train_encoded.csv'
    train_df, val_df, test_df, feature_cols = load_and_preprocess_data(
        data_path, args.dataset
    )

    # Create model configuration
    # Auto-detect number of classes from unique labels
    unique_labels = train_df['pathology'].nunique()
    num_classes = unique_labels if args.num_classes == 2 and unique_labels > 2 else args.num_classes
    
    config = create_model_config(len(feature_cols), num_classes=num_classes)
    config['backbone'] = args.backbone
    config['batch_size'] = args.batch_size
    config['max_epochs'] = args.max_epochs
    config['learning_rate'] = args.learning_rate
    config['weight_decay'] = args.weight_decay
    config['pretrained'] = args.pretrained
    config['image_dir'] = args.image_dir

    logger.info(f"Detected {num_classes} classes from data: {train_df['pathology'].unique().tolist()}")
    logger.info(f"Model config: {config}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, feature_cols, config
    )

    # Train model
    trainer = train_model(config, train_loader, val_loader, experiment_name)

    # Evaluate model
    evaluate_model(trainer, test_loader, experiment_name)

    logger.info(f"Experiment {experiment_name} completed successfully!")


if __name__ == "__main__":
    # For testing without command line arguments
    import sys

    if len(sys.argv) == 1:
        # Default test run
        logger.info("Running with default parameters for testing...")

        # Set random seeds
        pl.seed_everything(42)
        torch.manual_seed(42)
        np.random.seed(42)

        experiment_name = "mass_multimodal_test"

        # Load data
        data_path = 'data/processed/csv/mass_train_encoded.csv'
        train_df, val_df, test_df, feature_cols = load_and_preprocess_data(
            data_path, 'mass'
        )

        # Create config
        config = create_model_config(len(feature_cols), num_classes=3)
        config['batch_size'] = 8  # Smaller batch for testing
        config['max_epochs'] = 2  # Just a few epochs for testing

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_df, val_df, test_df, feature_cols, config
        )

        # Train model
        trainer = train_model(config, train_loader, val_loader, experiment_name)

        # Evaluate
        evaluate_model(trainer, test_loader, experiment_name)

        logger.info("Test training completed!")

    else:
        main()