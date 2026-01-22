#!/usr/bin/env python3
"""
Run W&B hyperparameter sweep for TacticAI.

Usage:
    # Create sweep
    python scripts/sweep.py --config configs/sweeps/hyperparameter_search.yaml --create

    # Run agent(s)
    python scripts/sweep.py --sweep-id <sweep_id> --count 50

    # Create and immediately run
    python scripts/sweep.py --config configs/sweeps/hyperparameter_search.yaml --create --count 10
"""

import argparse
import yaml
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.lightning_module import TacticAILightningModule
from training.data_module import TacticAIDataModule
from utils.config import load_config


def train_sweep():
    """Training function for W&B sweep agent."""
    # Initialize wandb run
    wandb.init()
    config = dict(wandb.config)

    print(f"\n{'='*60}")
    print("Sweep Run Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # Seed
    pl.seed_everything(42)

    # Create data module with sweep config
    data_module = TacticAIDataModule(
        data_dir='data/processed',
        batch_size=config.get('training.batch_size', 32),
        distance_threshold=config.get('data.distance_threshold', 5.0),
        use_enhanced_features=True,
        use_role_features=True,
        use_positional_context=True,
        use_augmentation=True,
    )

    # Create model with sweep config
    model = TacticAILightningModule(
        model_type='gat',
        node_features=14,
        hidden_dim=config.get('model.hidden_dim', 128),
        num_layers=config.get('model.num_layers', 4),
        dropout=config.get('model.dropout', 0.2),
        heads=config.get('model.heads', 4),
        learning_rate=config.get('training.learning_rate', 0.0005),
        use_focal_loss=True,
        focal_gamma=config.get('loss.focal_gamma', 2.0),
        label_smoothing=config.get('loss.label_smoothing', 0.1),
    )

    # Logger
    logger = WandbLogger(experiment=wandb.run)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val/top1_acc',
            patience=10,
            mode='max',
        ),
        ModelCheckpoint(
            dirpath='models/checkpoints/sweeps',
            monitor='val/top1_acc',
            mode='max',
            save_top_k=1,
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices=1,
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        deterministic=True,
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    test_results = trainer.test(model, data_module)

    # Log final results
    if test_results:
        wandb.log({
            'final/test_top1_acc': test_results[0].get('test/top1_acc', 0),
            'final/test_top3_acc': test_results[0].get('test/top3_acc', 0),
            'final/test_top5_acc': test_results[0].get('test/top5_acc', 0),
        })


def main():
    parser = argparse.ArgumentParser(description='Run W&B hyperparameter sweep')
    parser.add_argument('--config', type=str, help='Sweep config YAML path')
    parser.add_argument('--create', action='store_true', help='Create new sweep')
    parser.add_argument('--sweep-id', type=str, help='Existing sweep ID to join')
    parser.add_argument('--count', type=int, default=50, help='Number of runs')
    parser.add_argument('--project', type=str, default='tacticai', help='W&B project name')
    args = parser.parse_args()

    sweep_id = args.sweep_id

    if args.create:
        if not args.config:
            print("Error: --config required when creating sweep")
            return

        with open(args.config, 'r') as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"\nCreated sweep: {sweep_id}")
        print(f"View at: https://wandb.ai/{args.project}/sweeps/{sweep_id}")

    if sweep_id and args.count > 0:
        print(f"\nStarting {args.count} sweep agents...")
        wandb.agent(sweep_id, function=train_sweep, count=args.count, project=args.project)


if __name__ == '__main__':
    main()
