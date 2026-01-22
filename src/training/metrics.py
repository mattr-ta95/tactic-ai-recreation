"""Custom metrics for TacticAI training."""

import torch
from torchmetrics import Metric


class TopKAccuracy(Metric):
    """
    Custom metric for top-k accuracy on variable-sized graphs.

    Works with per-graph predictions where each graph has different
    number of candidate receivers.

    Args:
        k: Number of top predictions to consider (default 1)
    """

    def __init__(self, k: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: int):
        """
        Update metric with a single graph prediction.

        Args:
            logits: Prediction logits for one graph [num_nodes]
            target: Ground truth receiver index
        """
        k = min(self.k, len(logits))
        top_k_preds = logits.topk(k).indices.tolist()

        if target in top_k_preds:
            self.correct += 1.0
        self.total += 1

    def compute(self) -> torch.Tensor:
        """Compute the accuracy."""
        return self.correct / self.total if self.total > 0 else torch.tensor(0.0)
