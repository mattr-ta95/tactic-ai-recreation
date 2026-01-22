"""Loss functions for TacticAI training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in receiver prediction.

    Focal loss down-weights easy examples, focusing training on hard examples.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha: Optional class weights tensor
        label_smoothing: Label smoothing factor (default 0.0)
    """

    def __init__(self, gamma: float = 2.0, alpha=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape [batch_size, num_classes]
            targets: Target labels of shape [batch_size]

        Returns:
            Focal loss scalar
        """
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()
