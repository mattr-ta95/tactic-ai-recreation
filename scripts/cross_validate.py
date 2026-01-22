#!/usr/bin/env python3
"""
K-fold cross-validation for TacticAI.

Usage:
    python scripts/cross_validate.py --config configs/experiments/baseline.yaml --folds 5

    # Quick test with 3 folds
    python scripts/cross_validate.py --folds 3 --epochs 10
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.lightning_module import TacticAILightningModule
from training.data_module import TacticAIDataModule
from utils.config import load_config, flatten_config


def run_fold(fold_idx: int, num_folds: int, config: dict, offline: bool = False) -> dict:
    """Run training for a single fold."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{num_folds}")
    print(f"{'='*60}")

    # Seed for reproducibility
    seed = config.get('seed', 42)
    pl.seed_everything(seed + fold_idx)

    # Create data module for this fold
    data_config = config.get('data', {})
    data_module = TacticAIDataModule(
        data_dir=data_config.get('data_dir', 'data/processed'),
        batch_size=config.get('training', {}).get('batch_size', 32),
        distance_threshold=data_config.get('distance_threshold', 5.0),
        use_enhanced_features=data_config.get('use_enhanced_features', True),
        use_role_features=data_config.get('use_role_features', True),
        use_positional_context=data_config.get('use_positional_context', True),
        use_augmentation=data_config.get('use_augmentation', True),
        fold_idx=fold_idx,
        num_folds=num_folds,
        seed=seed,
    )

    # Create model
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    loss_config = config.get('loss', {})
    scheduler_config = config.get('scheduler', {})

    model = TacticAILightningModule(
        model_type=model_config.get('type', 'gat'),
        node_features=model_config.get('node_features', 14),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 4),
        dropout=model_config.get('dropout', 0.2),
        heads=model_config.get('heads', 4),
        learning_rate=training_config.get('learning_rate', 0.0005),
        use_focal_loss=loss_config.get('use_focal_loss', True),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        label_smoothing=loss_config.get('label_smoothing', 0.1),
        scheduler_factor=scheduler_config.get('factor', 0.5),
        scheduler_patience=scheduler_config.get('patience', 5),
    )

    # Logger
    logging_config = config.get('logging', {})
    group_name = f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if offline:
        logger = CSVLogger(
            save_dir=logging_config.get('save_dir', 'logs'),
            name=f"cv_fold{fold_idx}",
        )
    else:
        logger = WandbLogger(
            project=logging_config.get('project', 'tacticai'),
            name=f"{config.get('experiment_name', 'cv')}_fold{fold_idx}",
            group=group_name,
            config=flatten_config(config),
        )

    # Callbacks
    early_stopping_config = config.get('early_stopping', {})
    callbacks = [
        EarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val/top1_acc'),
            patience=early_stopping_config.get('patience', 10),
            mode=early_stopping_config.get('mode', 'max'),
        ),
        ModelCheckpoint(
            dirpath=f"models/checkpoints/cv_fold{fold_idx}",
            monitor='val/top1_acc',
            mode='max',
            save_top_k=1,
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get('num_epochs', 50),
        accelerator=config.get('hardware', {}).get('accelerator', 'auto'),
        devices=config.get('hardware', {}).get('devices', 1),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        deterministic=config.get('deterministic', True),
    )

    # Train
    trainer.fit(model, data_module)

    # Test on held-out fold
    test_results = trainer.test(model, data_module)

    # Finish W&B run
    if not offline and hasattr(logger, 'experiment'):
        logger.experiment.finish()

    return {
        'fold': fold_idx,
        'test_top1_acc': test_results[0].get('test/top1_acc', 0) if test_results else 0,
        'test_top3_acc': test_results[0].get('test/top3_acc', 0) if test_results else 0,
        'test_top5_acc': test_results[0].get('test/top5_acc', 0) if test_results else 0,
        'best_epoch': trainer.current_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description='K-fold cross-validation for TacticAI')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--offline', action='store_true', help='Disable W&B logging')
    args = parser.parse_args()

    print("=" * 60)
    print("TacticAI K-Fold Cross-Validation")
    print("=" * 60)

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        project_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(project_root, args.config)

    config = load_config(config_path)
    config['seed'] = args.seed

    if args.epochs:
        config.setdefault('training', {})['num_epochs'] = args.epochs

    print(f"Config: {config_path}")
    print(f"Folds: {args.folds}")
    print(f"Seed: {args.seed}")

    # Run all folds
    fold_results = []
    for fold_idx in range(args.folds):
        result = run_fold(fold_idx, args.folds, config, offline=args.offline)
        fold_results.append(result)
        print(f"\nFold {fold_idx + 1} - Test Acc: {result['test_top1_acc']*100:.1f}%")

    # Aggregate results
    top1_accs = [r['test_top1_acc'] for r in fold_results]
    top3_accs = [r['test_top3_acc'] for r in fold_results]
    top5_accs = [r['test_top5_acc'] for r in fold_results]

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {np.mean(top1_accs)*100:.1f}% +/- {np.std(top1_accs)*100:.1f}%")
    print(f"Top-3 Accuracy: {np.mean(top3_accs)*100:.1f}% +/- {np.std(top3_accs)*100:.1f}%")
    print(f"Top-5 Accuracy: {np.mean(top5_accs)*100:.1f}% +/- {np.std(top5_accs)*100:.1f}%")

    # Per-fold breakdown
    print(f"\nPer-fold results:")
    for r in fold_results:
        print(f"  Fold {r['fold']+1}: {r['test_top1_acc']*100:.1f}% (epoch {r['best_epoch']})")

    # Save results
    results = {
        'num_folds': args.folds,
        'seed': args.seed,
        'fold_results': fold_results,
        'summary': {
            'top1_mean': float(np.mean(top1_accs)),
            'top1_std': float(np.std(top1_accs)),
            'top3_mean': float(np.mean(top3_accs)),
            'top3_std': float(np.std(top3_accs)),
            'top5_mean': float(np.mean(top5_accs)),
            'top5_std': float(np.std(top5_accs)),
        },
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path('models/checkpoints/cv_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
