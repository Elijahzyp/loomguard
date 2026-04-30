"""
data.py
Shared dataset interface for the new pipeline.
Used by classification.train and anomaly.anomaly_efficientad. The frozen
train_patchcore.py keeps its own embedded Dataset and is not migrated here.

Phase 1B will add a parallel KnittedFabricPatchDataset implementing the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_WIDTH = 4096  # AITEX raw image width


class FabricPatchDataset(Protocol):
    """Minimal interface every fabric-patch dataset honors."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int): ...


@dataclass
class PatchSample:
    path: Path
    patch_idx: int
    label: int  # 0 = normal, 1 = defect


class AITEXPatchDataset(Dataset):
    """
    Slices each 4096x256 PNG into 16 horizontal 256x256 patches.
    Matches the slicing used by the frozen baseline so AUROC numbers are comparable.

    transform_kind:
        "albumentations" — transform expects a numpy HWC uint8 image, returns a tensor
                           (used for new pipeline).
        "torchvision"    — transform expects a PIL image, returns a tensor
                           (used by EfficientAD scaffold to match anomalib expectations).
        "none"           — no transform; returns raw PIL image. For inspection only.
    """

    def __init__(
        self,
        root_dir: str | Path,
        patch_size: int = 256,
        transform: Optional[Callable] = None,
        transform_kind: Literal["albumentations", "torchvision", "none"] = "albumentations",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.transform_kind = transform_kind
        self.samples: list[PatchSample] = []

        n_patches = IMAGE_WIDTH // patch_size
        for label_name, label in [("normal", 0), ("defect", 1)]:
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in sorted(label_dir.glob("*.png")):
                for i in range(n_patches):
                    self.samples.append(PatchSample(img_path, i, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        left = s.patch_idx * self.patch_size
        patch = img.crop((left, 0, left + self.patch_size, self.patch_size))

        if self.transform is None or self.transform_kind == "none":
            return patch, s.label

        if self.transform_kind == "albumentations":
            arr = np.array(patch)  # HWC uint8
            out = self.transform(image=arr)
            return out["image"], s.label

        if self.transform_kind == "torchvision":
            return self.transform(patch), s.label

        raise ValueError(f"Unknown transform_kind: {self.transform_kind}")

    def label_counts(self) -> tuple[int, int]:
        n_normal = sum(1 for s in self.samples if s.label == 0)
        n_defect = sum(1 for s in self.samples if s.label == 1)
        return n_normal, n_defect

    def normal_only_subset(self) -> "AITEXPatchDataset":
        """Return a copy of self with only the normal-label samples (for unsupervised methods)."""
        clone = AITEXPatchDataset.__new__(AITEXPatchDataset)
        clone.root_dir = self.root_dir
        clone.patch_size = self.patch_size
        clone.transform = self.transform
        clone.transform_kind = self.transform_kind
        clone.samples = [s for s in self.samples if s.label == 0]
        return clone


def compute_pos_weight(train_dir: str | Path, patch_size: int = 256) -> float:
    """N_normal / N_defect for BCEWithLogitsLoss(pos_weight=...) on the train split."""
    ds = AITEXPatchDataset(train_dir, patch_size=patch_size, transform=None, transform_kind="none")
    n_normal, n_defect = ds.label_counts()
    if n_defect == 0:
        return 1.0
    return float(n_normal) / float(n_defect)
