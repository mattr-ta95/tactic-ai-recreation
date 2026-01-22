"""PyTorch Lightning module for TacticAI GNN training."""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.gnn import get_model
from training.losses import FocalLoss
from training.metrics import TopKAccuracy


class TacticAILightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for TacticAI GNN models.

    Handles:
    - Per-graph loss computation for variable-sized PyG batches
    - Top-k accuracy metrics (top-1, top-3, top-5)
    - Focal loss with label smoothing
    - Learning rate scheduling
    - W&B logging

    Args:
        model_type: Type of model ('simple', 'gat', 'multitask')
        node_features: Number of input node features
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        dropout: Dropout rate
        heads: Number of attention heads (GAT only)
        edge_dim: Edge feature dimension (GAT only, None to disable)
        use_edge_features: Whether to use edge features
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        use_focal_loss: Whether to use focal loss
        focal_gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
        scheduler_factor: LR scheduler reduction factor
        scheduler_patience: LR scheduler patience
    """

    def __init__(
        self,
        model_type: str = 'gat',
        node_features: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        heads: int = 4,
        edge_dim: Optional[int] = None,
        use_edge_features: bool = False,
        learning_rate: float = 0.0005,
        weight_decay: float = 1e-5,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        gradient_clip_val: float = 1.0,
        **kwargs  # Absorb extra config params
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build model using existing factory
        model_kwargs = {
            'node_features': node_features,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
        }
        if model_type == 'gat':
            model_kwargs['heads'] = heads
            if use_edge_features and edge_dim:
                model_kwargs['edge_dim'] = edge_dim

        self.model = get_model(model_type, **model_kwargs)

        # Loss function
        if use_focal_loss:
            self.loss_fn = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        else:
            self.loss_fn = None  # Use F.cross_entropy directly

        # Metrics (separate instances for train/val/test)
        self.train_top1 = TopKAccuracy(k=1)
        self.train_top3 = TopKAccuracy(k=3)
        self.val_top1 = TopKAccuracy(k=1)
        self.val_top3 = TopKAccuracy(k=3)
        self.val_top5 = TopKAccuracy(k=5)
        self.test_top1 = TopKAccuracy(k=1)
        self.test_top3 = TopKAccuracy(k=3)
        self.test_top5 = TopKAccuracy(k=5)

    def forward(self, x, edge_index, batch, edge_attr=None):
        """Forward pass - delegates to wrapped model."""
        if self.hparams.use_edge_features and edge_attr is not None:
            return self.model(x, edge_index, batch, edge_attr=edge_attr)
        return self.model(x, edge_index, batch)

    def _compute_per_graph_loss(self, batch: Batch, out: torch.Tensor):
        """
        Compute loss and predictions for each graph in the batch.

        This is the critical method that handles PyG's flattened batch structure.

        Args:
            batch: PyG Batch object with .batch tensor for graph assignment
            out: Model output logits [total_nodes]

        Returns:
            (loss, predictions_list, labels_list)
        """
        total_loss = 0.0
        predictions = []
        labels = []
        valid_graphs = 0

        for i in range(batch.num_graphs):
            # Get nodes belonging to this graph
            mask = (batch.batch == i)
            graph_logits = out[mask]

            # Get label for this graph
            if batch.y.dim() > 0:
                label = batch.y[i]
            else:
                label = batch.y

            # Convert to int if tensor
            if isinstance(label, torch.Tensor):
                label_val = label.item()
            else:
                label_val = int(label)

            # Skip if label out of bounds
            if label_val >= len(graph_logits):
                continue

            # Compute loss for this graph
            logits_2d = graph_logits.unsqueeze(0)  # [1, num_nodes]
            label_1d = torch.tensor([label_val], device=self.device)  # [1]

            if self.loss_fn is not None:
                graph_loss = self.loss_fn(logits_2d, label_1d)
            else:
                graph_loss = F.cross_entropy(
                    logits_2d, label_1d,
                    label_smoothing=self.hparams.label_smoothing
                )

            total_loss += graph_loss
            predictions.append(graph_logits)
            labels.append(label_val)
            valid_graphs += 1

        if valid_graphs > 0:
            avg_loss = total_loss / valid_graphs
        else:
            avg_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return avg_loss, predictions, labels

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Training step with per-graph loss."""
        # Forward pass
        edge_attr = batch.edge_attr if self.hparams.use_edge_features else None
        out = self(batch.x, batch.edge_index, batch.batch, edge_attr)

        # Compute loss
        loss, preds, labels = self._compute_per_graph_loss(batch, out)

        # Update metrics
        for pred_logits, label in zip(preds, labels):
            self.train_top1.update(pred_logits, label)
            self.train_top3.update(pred_logits, label)

        # Log
        self.log('train/loss', loss, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        self.log('train/top1_acc', self.train_top1.compute(), prog_bar=True)
        self.log('train/top3_acc', self.train_top3.compute())
        self.train_top1.reset()
        self.train_top3.reset()

    def validation_step(self, batch: Batch, batch_idx: int):
        """Validation step."""
        edge_attr = batch.edge_attr if self.hparams.use_edge_features else None
        out = self(batch.x, batch.edge_index, batch.batch, edge_attr)

        loss, preds, labels = self._compute_per_graph_loss(batch, out)

        for pred_logits, label in zip(preds, labels):
            self.val_top1.update(pred_logits, label)
            self.val_top3.update(pred_logits, label)
            self.val_top5.update(pred_logits, label)

        self.log('val/loss', loss, prog_bar=True, batch_size=batch.num_graphs)

    def on_validation_epoch_end(self):
        """Log validation metrics - these drive early stopping and checkpointing."""
        val_acc = self.val_top1.compute()
        self.log('val/top1_acc', val_acc, prog_bar=True)
        self.log('val/top3_acc', self.val_top3.compute())
        self.log('val/top5_acc', self.val_top5.compute())
        self.val_top1.reset()
        self.val_top3.reset()
        self.val_top5.reset()

    def test_step(self, batch: Batch, batch_idx: int):
        """Test step."""
        edge_attr = batch.edge_attr if self.hparams.use_edge_features else None
        out = self(batch.x, batch.edge_index, batch.batch, edge_attr)

        loss, preds, labels = self._compute_per_graph_loss(batch, out)

        for pred_logits, label in zip(preds, labels):
            self.test_top1.update(pred_logits, label)
            self.test_top3.update(pred_logits, label)
            self.test_top5.update(pred_logits, label)

        self.log('test/loss', loss, batch_size=batch.num_graphs)

    def on_test_epoch_end(self):
        """Log final test metrics."""
        self.log('test/top1_acc', self.test_top1.compute())
        self.log('test/top3_acc', self.test_top3.compute())
        self.log('test/top5_acc', self.test_top5.compute())

    def configure_optimizers(self):
        """Configure optimizer with ReduceLROnPlateau scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/top1_acc',
                'interval': 'epoch',
                'frequency': 1,
            }
        }
