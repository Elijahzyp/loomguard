"""
train_patchcore.py
Runs PatchCore anomaly detection on AITEX patches via anomalib.
PatchCore trains only on normal images (unsupervised / one-class).
Run from: experiments/phase1a_aitex/
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from anomalib.data.dataclasses.torch.image import ImageBatch, ImageItem
from anomalib.engine import Engine
from anomalib.models import Patchcore

CONFIG = {
    "BACKBONE": "resnet18",
    "PATCH_SIZE": 256,
    "SEED": 42,
    "COLAB_MODE": False,
    "DATA_ROOT": Path("data/prepared"),
    # Colab 中将 DATA_ROOT 改为：
    # Path("/content/drive/MyDrive/loomguard_data/prepared")
    # 并将 RESULTS_DIR 改为：
    # Path("/content/drive/MyDrive/loomguard_data/results")
    "RESULTS_DIR": Path("results"),
}

IMAGE_WIDTH = 4096


def get_accelerator() -> tuple[str, int]:
    if CONFIG["COLAB_MODE"]:
        return "gpu", 1
    if torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


class PatchDataset(Dataset):
    """Slices each 4096×256 image into horizontal 256×256 patches."""

    def __init__(
        self,
        dirs: list[tuple[Path, int]],
        patch_size: int,
        transform=None,
    ) -> None:
        self.patch_size = patch_size
        self.transform = transform
        self.samples: list[tuple[Path, int, int]] = []
        n_patches = IMAGE_WIDTH // patch_size
        for directory, label in dirs:
            if not directory.exists():
                continue
            for img_path in sorted(directory.glob("*.png")):
                for i in range(n_patches):
                    self.samples.append((img_path, i, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ImageItem:
        path, patch_idx, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        left = patch_idx * self.patch_size
        patch = img.crop((left, 0, left + self.patch_size, self.patch_size))
        if self.transform:
            patch = self.transform(patch)
        return ImageItem(
            image=patch,
            gt_label=torch.tensor(bool(label)),
            image_path=str(path),
        )


def compute_metrics(scores: list[float], labels: list) -> dict:
    arr_scores = np.array(scores, dtype=np.float32)
    arr_labels = np.array(labels, dtype=np.int32)

    auroc = float(roc_auc_score(arr_labels, arr_scores))

    precision, recall, thresholds = precision_recall_curve(arr_labels, arr_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1[:-1]))
    threshold = float(thresholds[best_idx])

    preds = (arr_scores >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (arr_labels == 1)).sum())
    fp = int(((preds == 1) & (arr_labels == 0)).sum())
    fn = int(((preds == 0) & (arr_labels == 1)).sum())
    tn = int(((preds == 0) & (arr_labels == 0)).sum())

    defect_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "auroc": round(auroc, 4),
        "defect_recall": round(defect_recall, 4),
        "false_positive_rate": round(fpr, 4),
        "threshold": round(threshold, 4),
    }


def main() -> None:
    torch.manual_seed(CONFIG["SEED"])

    accelerator, devices = get_accelerator()
    print(f"Device: {accelerator}")

    data_root = CONFIG["DATA_ROOT"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # PatchCore trains on normal images only
    train_dataset = PatchDataset(
        [(data_root / "train" / "normal", 0)],
        CONFIG["PATCH_SIZE"],
        transform,
    )
    val_dataset = PatchDataset(
        [
            (data_root / "val" / "normal", 0),
            (data_root / "val" / "defect", 1),
        ],
        CONFIG["PATCH_SIZE"],
        transform,
    )

    # anomalib 2.x: DataLoader must use ImageBatch.collate to produce Batch objects
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=ImageBatch.collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=ImageBatch.collate,
    )

    n_train, n_val = len(train_dataset), len(val_dataset)
    print(f"Patches — train (normal only): {n_train}, val: {n_val}")

    model = Patchcore(
        backbone=CONFIG["BACKBONE"],
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        post_processor=False,  # skip built-in score normalization; we compute raw AUROC
        evaluator=False,       # skip built-in metric callbacks
    )

    engine = Engine(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1,             # PatchCore only needs one pass to build memory bank
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,   # memory bank not ready before training; skip sanity check
    )

    # Phase 1: extract features from normal patches → build memory bank
    print("\nBuilding memory bank...")
    t0 = time.time()
    engine.fit(model, train_dataloaders=train_loader)
    elapsed = time.time() - t0
    print(f"Memory bank built in {elapsed:.0f}s")

    if elapsed > 180:
        print("\n⚠️  WARNING: 训练耗时超过 3 分钟，Mac 负载较高。")
        print("   建议切换到 Google Colab（CONFIG 中将 COLAB_MODE 改为 True）。")
        ans = input("   继续本地评估请输入 y，退出请输入 n：").strip().lower()
        if ans != "y":
            sys.exit(0)

    # Phase 2: compute anomaly scores on val set
    print("Evaluating...")
    predictions = engine.predict(model, dataloaders=val_loader, return_predictions=True)

    # Each element of predictions is an ImageBatch with .pred_score and .gt_label
    all_scores: list[float] = []
    all_labels: list = []

    for batch_out in predictions:
        if batch_out is None:
            continue
        # Lightning may wrap in a list on some versions
        if isinstance(batch_out, list):
            batch_out = batch_out[0] if batch_out else None
            if batch_out is None:
                continue
        if hasattr(batch_out, "pred_score") and batch_out.pred_score is not None:
            all_scores.extend(batch_out.pred_score.cpu().tolist())
        if hasattr(batch_out, "gt_label") and batch_out.gt_label is not None:
            all_labels.extend(batch_out.gt_label.cpu().tolist())

    if len(all_scores) != len(all_labels):
        raise RuntimeError(
            f"Score/label count mismatch: {len(all_scores)} scores, "
            f"{len(all_labels)} labels. "
            "Ensure val_loader uses shuffle=False."
        )

    metrics = compute_metrics(all_scores, all_labels)

    results_dir = CONFIG["RESULTS_DIR"]
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"patchcore_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "model": "PatchCore",
        "backbone": CONFIG["BACKBONE"],
        "n_train_patches": n_train,
        "n_val_patches": n_val,
        **metrics,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {results_path}")
    print(
        f"AUROC: {metrics['auroc']:.4f} | "
        f"Defect Recall: {metrics['defect_recall']:.4f} | "
        f"FPR: {metrics['false_positive_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
