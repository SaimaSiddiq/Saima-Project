#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Multimodal Breast Cancer Classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.multimodal_framework import MultimodalLightningModule, create_model_config, BreastCancerDataset


def hyperparameter_tuning(dataset_path, backbone, experiment_base_name, max_epochs=5):
    """
    Perform hyperparameter tuning for a given backbone model.
    """
    # Load data
    df = pd.read_csv(dataset_path)
    feature_cols = [col for col in df.columns if col not in ['patient_id', 'pathology']]
    unique_labels = df['pathology'].nunique()

    # Split data
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['pathology'])
    val_size_adjusted = 0.1 / 0.8
    train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, random_state=42, stratify=train_val_df['pathology'])

    # Hyperparameter search space
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [16, 32]
    weight_decays = [0.0001, 0.00001]

    # Generate all combinations
    param_combinations = list(itertools.product(learning_rates, batch_sizes, weight_decays))

    results = []

    for i, (lr, batch_size, wd) in enumerate(param_combinations):
        experiment_name = f"{experiment_base_name}_hp_{i}"
        print(f"\n=== Training combination {i+1}/{len(param_combinations)}: LR={lr}, BS={batch_size}, WD={wd} ===")

        # Create config
        config = create_model_config(len(feature_cols), num_classes=unique_labels)
        config.update({
            'backbone': backbone,
            'learning_rate': lr,
            'weight_decay': wd,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
        })

        # Create datasets
        train_dataset = BreastCancerDataset(clinical_df=train_df, image_dir=None, mode='train')
        val_dataset = BreastCancerDataset(clinical_df=val_df, image_dir=None, mode='val')
        test_dataset = BreastCancerDataset(clinical_df=test_df, image_dir=None, mode='test')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Create model
        model = MultimodalLightningModule(config)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/{experiment_name}',
            filename='{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min'
        )

        # Logger
        logger = TensorBoardLogger('results/logs', name=experiment_name)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='cpu',
            devices=1,
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger,
            enable_progress_bar=True
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Test
        test_results = trainer.test(model, dataloaders=test_loader, verbose=False)

        # Record results
        result = {
            'experiment': experiment_name,
            'backbone': backbone,
            'learning_rate': lr,
            'batch_size': batch_size,
            'weight_decay': wd,
            'epochs_trained': trainer.current_epoch + 1,
            'best_val_loss': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
            'test_metrics': test_results[0] if test_results else {}
        }

        results.append(result)
        print(f"Completed: Test Acc = {result['test_metrics'].get('test_acc', 'N/A'):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Multimodal Models')
    parser.add_argument('--dataset', type=str, default='mass', choices=['mass', 'calc'],
                       help='Dataset to use (mass or calc)')
    parser.add_argument('--backbones', nargs='+', default=['densenet201', 'resnet50'],
                       help='Backbones to tune')
    parser.add_argument('--max_epochs', type=int, default=5,
                       help='Maximum epochs per experiment')
    parser.add_argument('--experiment_base', type=str, default='hyperparam_tuning',
                       help='Base name for experiments')

    args = parser.parse_args()

    # Dataset path
    dataset_path = f'data/processed/csv/{args.dataset}_train_encoded.csv'

    all_results = []

    for backbone in args.backbones:
        print(f"\n{'='*60}")
        print(f"STARTING HYPERPARAMETER TUNING FOR {backbone.upper()}")
        print(f"{'='*60}")

        results = hyperparameter_tuning(
            dataset_path=dataset_path,
            backbone=backbone,
            experiment_base_name=f"{args.experiment_base}_{args.dataset}_{backbone}",
            max_epochs=args.max_epochs
        )

        all_results.extend(results)

    # Save results
    results_dir = Path('results/hyperparameter_tuning')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f'{args.experiment_base}_{args.dataset}_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create summary table
    summary_data = []
    for result in all_results:
        row = {
            'Backbone': result['backbone'],
            'LR': result['learning_rate'],
            'Batch Size': result['batch_size'],
            'Weight Decay': result['weight_decay'],
            'Test Accuracy': result['test_metrics'].get('test_acc', 0),
            'Test Loss': result['test_metrics'].get('test_loss', 0),
            'Test F1': result['test_metrics'].get('test_f1', 0),
            'Epochs': result['epochs_trained']
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_file = results_dir / f'{args.experiment_base}_{args.dataset}_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    # Print best results
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

    # Best configurations
    for backbone in args.backbones:
        backbone_results = [r for r in all_results if r['backbone'] == backbone]
        if backbone_results:
            best_result = max(backbone_results, key=lambda x: x['test_metrics'].get('test_acc', 0))
            print(f"\nBest {backbone.upper()} configuration:")
            print(f"  Test Accuracy: {best_result['test_metrics'].get('test_acc', 0):.4f}")
            print(f"  Learning Rate: {best_result['learning_rate']}")
            print(f"  Batch Size: {best_result['batch_size']}")
            print(f"  Weight Decay: {best_result['weight_decay']}")


if __name__ == '__main__':
    main()