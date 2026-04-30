"""
losses.py
Loss factory. CE -> 2-logit head; BCE/Focal -> 1-logit head.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LossType = Literal["ce", "bce", "focal"]


class FocalLoss(nn.Module):
    """Binary focal loss on a 1-logit head."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 1) or (B,); targets: (B,) int64
        logits = logits.view(-1)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()


def build_loss(loss_type: LossType, pos_weight: Optional[float] = None) -> nn.Module:
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    if loss_type == "bce":
        pw = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None
        return _BCEAdapter(nn.BCEWithLogitsLoss(pos_weight=pw))
    if loss_type == "focal":
        return _BinaryAdapter(FocalLoss(gamma=2.0, alpha=0.25))
    raise ValueError(f"Unknown loss_type: {loss_type}")


def num_outputs_for(loss_type: LossType) -> int:
    return 2 if loss_type == "ce" else 1


class _BCEAdapter(nn.Module):
    """Adapts BCEWithLogitsLoss to take int64 labels and 1-logit predictions."""

    def __init__(self, inner: nn.BCEWithLogitsLoss):
        super().__init__()
        self.inner = inner

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.inner.pos_weight is not None:
            self.inner.pos_weight = self.inner.pos_weight.to(logits.device)
        return self.inner(logits.view(-1), targets.float())


class _BinaryAdapter(nn.Module):
    """Wraps any binary loss to forward identically (kept for symmetry / future tweaks)."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.inner(logits, targets)
