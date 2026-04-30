"""
models.py
Thin timm wrapper. Full fine-tune (no frozen backbone) — matches the baseline.
"""

from __future__ import annotations

import timm
import torch.nn as nn

SUPPORTED = {"resnet18", "efficientnet_b0", "convnext_tiny"}


def build_model(name: str, num_outputs: int, pretrained: bool = True) -> nn.Module:
    if name not in SUPPORTED:
        raise ValueError(f"Unsupported model: {name}. Supported: {sorted(SUPPORTED)}")
    return timm.create_model(name, pretrained=pretrained, num_classes=num_outputs)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
