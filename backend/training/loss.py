"""Multiclass Focal Loss implementation for Skinora Acne Severity Classification."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss module for addressing class imbalance and hard negative mining.

    Standard multiclass formulation:
        FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    where p_t is the softmax probability of the true target class.

    Numerically stable implementation using log_softmax and gather.
    When gamma=0 and weight=None, exactly equals standard CrossEntropyLoss (with no label smoothing).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"Focal loss gamma must be non-negative, got {gamma}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"Unsupported reduction: '{reduction}'. Choose 'mean', 'sum', or 'none'."
            )
        self.gamma = float(gamma)
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multiclass focal loss.

        Args:
            inputs: Raw logits tensor of shape (batch_size, num_classes).
            targets: Ground-truth class labels as integer tensor of shape (batch_size,).

        Returns:
            Scalar loss tensor if reduction is 'mean' or 'sum', else per-sample loss tensor.
        """
        if inputs.dim() != 2:
            raise ValueError(
                f"FocalLoss expects 2D inputs of shape (batch_size, num_classes), got {inputs.shape}"
            )

        num_classes = inputs.size(1)

        if targets.dim() == 1:
            targets_idx = targets.unsqueeze(1)
        elif targets.dim() == 2 and targets.size(1) == 1:
            targets_idx = targets
            targets = targets.squeeze(1)
        else:
            raise ValueError(
                f"FocalLoss expects 1D targets of shape (batch_size,), got {targets.shape}"
            )

        if (targets < 0).any() or (targets >= num_classes).any():
            raise ValueError(
                f"Target class indices must be in range [0, {num_classes - 1}]"
            )

        # Numerically stable log-probabilities: log(p)
        log_p = F.log_softmax(inputs, dim=1)

        # Extract log(p_t) for the true class
        log_pt = log_p.gather(dim=1, index=targets_idx).squeeze(1)

        # Softmax probability of the true class: p_t in [0, 1]
        pt = torch.exp(log_pt)

        # Modulating factor: (1 - p_t)^gamma
        # Clamped to [0.0, 1.0] to prevent floating point inaccuracies
        focal_factor = torch.pow(torch.clamp(1.0 - pt, min=0.0, max=1.0), self.gamma)

        # Raw focal loss: -(1 - p_t)^gamma * log(p_t)
        loss = -focal_factor * log_pt

        # Optional class weighting (disabled for E4)
        if self.weight is not None:
            w = self.weight.gather(dim=0, index=targets)
            loss = loss * w

        if self.reduction == "mean":
            if self.weight is not None:
                return loss.sum() / w.sum().clamp(min=1e-8)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return f"FocalLoss(gamma={self.gamma}, reduction='{self.reduction}', weighted={self.weight is not None})"

