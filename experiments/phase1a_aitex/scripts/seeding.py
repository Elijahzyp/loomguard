"""
seeding.py
Strict seeding for the new experimental pipeline.
The frozen baseline (train_resnet.py) only seeds torch; this module additionally seeds
numpy, random, cuda, and forces cudnn deterministic mode. Bit-equivalence with the
baseline is therefore not expected — reproducibility tolerance is ±0.01 AUROC.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
