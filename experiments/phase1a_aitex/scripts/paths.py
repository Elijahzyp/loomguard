"""
paths.py
Single source of truth for COLAB_MODE / DATA_ROOT / RESULTS_DIR.
Mirrors the detection used by the frozen baseline scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

COLAB_MODE: bool = os.path.exists("/content")

if COLAB_MODE:
    _BASE = Path("/content/drive/MyDrive/loomguard_data")
    DATA_ROOT: Path = _BASE / "prepared"
    RESULTS_DIR: Path = _BASE / "results"
else:
    DATA_ROOT = Path("data/prepared")
    RESULTS_DIR = Path("results")

RUNS_DIR: Path = RESULTS_DIR / "runs"
SUMMARY_DIR: Path = RESULTS_DIR / "summary"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    import torch
    if COLAB_MODE and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
