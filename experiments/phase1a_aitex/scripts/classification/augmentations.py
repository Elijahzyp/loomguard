"""
augmentations.py
Albumentations transform builders for the new pipeline.

Conservative profile is intentionally mild: any rotation outside +-5 degrees on long
fabric strips drags real defects out of the patch and changes labels.
"""

from __future__ import annotations

from typing import Literal

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(profile: Literal["none", "conservative"]) -> A.Compose:
    if profile == "none":
        return A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    if profile == "conservative":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Affine(
                rotate=(-5, 5),
                scale=(0.95, 1.05),
                p=0.3,
                border_mode=cv2.BORDER_REFLECT_101,
                keep_ratio=True,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    raise ValueError(f"Unknown augmentation profile: {profile}")


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
