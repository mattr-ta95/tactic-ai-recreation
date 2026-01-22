#!/usr/bin/env python3
"""
PyTorch Lightning training script for TacticAI.

Usage:
    # Basic training with default config
    python scripts/train_lightning.py

    # With specific config
    python scripts/train_lightning.py --config configs/experiments/baseline.yaml

    # With CLI overrides
    python scripts/train_lightning.py --config configs/default.yaml \
        model.hidden_dim=256 training.learning_rate=0.001

    # Disable W&B logging (offline mode)
    python scripts/train_lightning.py --offline
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
)
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.lightning_module import TacticAILightningModule
from training.data_module import TacticAIDataModule
from utils.config import load_config, flatten_config, parse_cli_overrides


def train(config: dict, offline: bool = False):
    """Main training function."""
    print("=" * 70)
    print("TacticAI Lightning Training")
    print("=" * 70)

    # Seed for reproducibility
    seed = config.get('seed', 42)
    pl.seed_everything(seed)

    # Create data module
    data_config = config.get('data', {})
    data_module = TacticAIDataModule(
        data_dir=data_config.get('data_dir', 'data/processed'),
        batch_size=config.get('training', {}).get('batch_size', 32),
        num_workers=config.get('hardware', {}).get('num_workers', 0),
        distance_threshold=data_config.get('distance_threshold', 5.0),
        use_enhanced_features=data_config.get('use_enhanced_features', True),
        use_role_features=data_config.get('use_role_features', True),
        use_positional_context=data_config.get('use_positional_context', True),
        use_knn_edges=data_config.get('use_knn_edges', False),
        knn_k=data_config.get('knn_k', 8),
        use_augmentation=data_config.get('use_augmentation', True),
        val_split=data_config.get('val_split', 0.2),
        test_split=data_config.get('test_split', 0.2),
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
        edge_dim=model_config.get('edge_dim'),
        use_edge_features=training_config.get('use_edge_features', False),
        learning_rate=training_config.get('learning_rate', 0.0005),
        weight_decay=training_config.get('weight_decay', 1e-5),
        use_focal_loss=loss_config.get('use_focal_loss', True),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        label_smoothing=loss_config.get('label_smoothing', 0.1),
        scheduler_factor=scheduler_config.get('factor', 0.5),
        scheduler_patience=scheduler_config.get('patience', 5),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
    )

    print(f"\nModel: {model_config.get('type', 'gat')}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Logger
    logging_config = config.get('logging', {})
    if offline:
        logger = CSVLogger(
            save_dir=logging_config.get('save_dir', 'logs'),
            name=config.get('experiment_name', 'tacticai'),
        )
        print("Using CSV logger (offline mode)")
    else:
        logger = WandbLogger(
            project=logging_config.get('project', 'tacticai'),
            entity=logging_config.get('entity'),
            name=config.get('experiment_name'),
            config=flatten_config(config),
            save_dir=logging_config.get('save_dir', 'logs'),
            log_model=True,
        )
        print(f"Logging to W&B project: {logging_config.get('project', 'tacticai')}")

    # Callbacks
    early_stopping_config = config.get('early_stopping', {})
    checkpoint_config = config.get('checkpoint', {})

    callbacks = [
        EarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val/top1_acc'),
            patience=early_stopping_config.get('patience', 10),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            mode=early_stopping_config.get('mode', 'max'),
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_config.get('dirpath', 'models/checkpoints'),
            filename=checkpoint_config.get('filename', 'epoch={epoch}-val_acc={val/top1_acc:.4f}'),
            monitor=checkpoint_config.get('monitor', 'val/top1_acc'),
            mode=checkpoint_config.get('mode', 'max'),
            save_top_k=checkpoint_config.get('save_top_k', 3),
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    # Try to use RichProgressBar if available
    try:
        callbacks.append(RichProgressBar())
    except:
        pass

    # Trainer
    hardware_config = config.get('hardware', {})
    trainer = pl.Trainer(
        max_epochs=training_config.get('num_epochs', 50),
        accelerator=hardware_config.get('accelerator', 'auto'),
        devices=hardware_config.get('devices', 1),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=logging_config.get('log_every_n_steps', 10),
        deterministic=config.get('deterministic', True),
        enable_progress_bar=True,
    )

    print(f"\nTraining for up to {training_config.get('num_epochs', 50)} epochs")
    print(f"Early stopping: patience={early_stopping_config.get('patience', 10)}")
    print("-" * 70)

    # Train
    trainer.fit(model, data_module)

    # Test
    print("\n" + "=" * 70)
    print("Testing on held-out test set...")
    print("=" * 70)
    test_results = trainer.test(model, data_module)

    # Print results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    if test_results:
        print(f"Test Top-1 Accuracy: {test_results[0].get('test/top1_acc', 0)*100:.1f}%")
        print(f"Test Top-3 Accuracy: {test_results[0].get('test/top3_acc', 0)*100:.1f}%")
        print(f"Test Top-5 Accuracy: {test_results[0].get('test/top5_acc', 0)*100:.1f}%")

    # Finish logging
    if not offline and hasattr(logger, 'experiment'):
        logger.experiment.finish()

    return trainer


def main():
    parser = argparse.ArgumentParser(description='Train TacticAI with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--offline', action='store_true',
                        help='Disable W&B logging (use CSV logger)')
    args, unknown = parser.parse_known_args()

    # Parse CLI overrides (format: key=value)
    overrides = parse_cli_overrides(unknown)

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        # Try relative to project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(project_root, args.config)

    print(f"Loading config from: {config_path}")
    config = load_config(config_path, overrides)

    if overrides:
        print(f"CLI overrides: {overrides}")

    # Train
    train(config, offline=args.offline)


if __name__ == '__main__':
    main()
