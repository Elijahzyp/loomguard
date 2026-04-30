"""
tta.py
Test-time augmentation. Averages RAW LOGITS over views (not probabilities).
Returns logits with the same shape the model normally produces.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader

TTAMode = Literal["none", "4way", "8way"]


def _views(x: torch.Tensor, mode: TTAMode) -> list[torch.Tensor]:
    if mode == "none":
        return [x]
    h = torch.flip(x, dims=[-1])
    v = torch.flip(x, dims=[-2])
    hv = torch.flip(x, dims=[-1, -2])
    if mode == "4way":
        return [x, h, v, hv]
    if mode == "8way":
        r90 = torch.rot90(x, k=1, dims=[-2, -1])
        r180 = torch.rot90(x, k=2, dims=[-2, -1])
        r270 = torch.rot90(x, k=3, dims=[-2, -1])
        return [x, h, v, hv, r90, r180, r270]
    raise ValueError(f"Unknown TTA mode: {mode}")


@torch.no_grad()
def predict_with_tta(model, loader: DataLoader, mode: TTAMode, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (logits, labels). logits is (N, C) where C matches the model head."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        views = _views(images, mode)
        stacked = torch.stack([model(v) for v in views], dim=0)  # (V, B, C) or (V, B, 1)
        avg = stacked.mean(dim=0)
        all_logits.append(avg.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)
