"""
eval_resnet18.py
Evaluates a trained ResNet18 binary classifier on AITEX val patches.
Computes AUROC, Defect Recall, FPR, and a full threshold grid.
Run from: experiments/phase1a_aitex/
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_COLAB = os.path.exists("/content")
_BASE = Path("/content/drive/MyDrive/loomguard_data") if _COLAB else Path(".")

CONFIG = {
    "COLAB_MODE": _COLAB,
    "MODEL_PATH": _BASE / "results" / "resnet18_20260429_163108.pth",
    "VAL_DIR": "",  # override: set to a path string to use it; empty = auto ({BASE}/aitex_patches/val)
    "RESULTS_DIR": _BASE / "results",
    "PATCH_SIZE": 256,
    "BATCH_SIZE": 32,
    "SEED": 42,
    "THRESHOLDS": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

_VAL_DIR = Path(CONFIG["VAL_DIR"]) if CONFIG["VAL_DIR"] else _BASE / "aitex_patches" / "val"

IMAGE_WIDTH = 4096


def get_device() -> torch.device:
    if CONFIG["COLAB_MODE"]:
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PatchDataset(Dataset):
    """Slices each 4096×256 image into PATCH_SIZE-wide horizontal strips."""

    def __init__(self, root_dir: Path, patch_size: int, transform=None):
        self.patch_size = patch_size
        self.transform = transform
        self.samples: list[tuple[Path, int, int]] = []
        n_patches = IMAGE_WIDTH // patch_size
        for label_name, label in [("normal", 0), ("defect", 1)]:
            label_dir = root_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in sorted(label_dir.glob("*.png")):
                for i in range(n_patches):
                    self.samples.append((img_path, i, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, patch_idx, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        left = patch_idx * self.patch_size
        img = img.crop((left, 0, left + self.patch_size, self.patch_size))
        if self.transform:
            img = self.transform(img)
        return img, label


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": threshold,
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


def main():
    torch.manual_seed(CONFIG["SEED"])
    device = get_device()
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- path diagnostics ---
    print(f"VAL_DIR: {_VAL_DIR}")
    if not _VAL_DIR.exists():
        raise FileNotFoundError(f"VAL_DIR not found: {_VAL_DIR}\nSet CONFIG['VAL_DIR'] to the correct path.")
    subdirs = [d for d in _VAL_DIR.iterdir() if d.is_dir()]
    for d in sorted(subdirs):
        n = len(list(d.glob("*.png")))
        print(f"  {d.name}/: {n} .png files")
    # ------------------------

    val_dataset = PatchDataset(_VAL_DIR, CONFIG["PATCH_SIZE"], transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    print(f"Val patches: {len(val_dataset)}")

    model = timm.create_model("resnet18", pretrained=False, num_classes=2)
    state_dict = torch.load(CONFIG["MODEL_PATH"], map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {CONFIG['MODEL_PATH']}")

    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    arr_probs = np.array(all_probs, dtype=np.float32)
    arr_labels = np.array(all_labels, dtype=np.int32)

    auroc = float(roc_auc_score(arr_labels, arr_probs))
    print(f"\nAUROC: {auroc:.4f}")

    grid = [metrics_at_threshold(arr_probs, arr_labels, t) for t in CONFIG["THRESHOLDS"]]

    print(f"\n{'Threshold':>10} {'Recall':>8} {'FPR':>8} {'Precision':>10} {'F1':>8}")
    print("-" * 52)
    for row in grid:
        print(
            f"{row['threshold']:>10.1f} {row['recall']:>8.4f} {row['fpr']:>8.4f}"
            f" {row['precision']:>10.4f} {row['f1']:>8.4f}"
        )

    best = max(grid, key=lambda r: r["f1"])
    at05 = next(r for r in grid if r["threshold"] == 0.5)

    results_dir = CONFIG["RESULTS_DIR"]
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    grid_path = results_dir / f"resnet18_threshold_grid_{timestamp}.json"
    with open(grid_path, "w") as f:
        json.dump({"timestamp": timestamp, "auroc": round(auroc, 4), "grid": grid}, f, indent=2)

    results = {
        "timestamp": timestamp,
        "model": "ResNet18",
        "model_path": str(CONFIG["MODEL_PATH"]),
        "n_val_patches": len(val_dataset),
        "auroc": round(auroc, 4),
        "at_threshold_0.5": at05,
        "best_threshold": best,
    }
    results_path = results_dir / f"resnet18_eval_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {results_path}")
    print(f"Grid saved    → {grid_path}")
    print(f"\n@0.5  — Recall: {at05['recall']:.4f} | FPR: {at05['fpr']:.4f} | F1: {at05['f1']:.4f}")
    print(
        f"@best — threshold={best['threshold']} | "
        f"Recall: {best['recall']:.4f} | FPR: {best['fpr']:.4f} | F1: {best['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
