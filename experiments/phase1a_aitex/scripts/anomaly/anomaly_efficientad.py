"""
anomaly_efficientad.py
EfficientAD scaffolding parallel to the frozen train_patchcore.py baseline.
Uses the shared scripts.data interface so Phase 1B knitted-fabric data plugs in
identically.

Phase 1A goal: validate that the scaffolding works end-to-end on AITEX. We do not
tune EfficientAD here. If anomalib's EfficientAd cannot be imported or instantiated
on the current environment, raise NotImplementedError with a diagnostic message.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("anomalib").setLevel(logging.WARNING)

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts import paths as P
from scripts.data import AITEXPatchDataset
from scripts.seeding import set_seed

CONFIG = {
    "PATCH_SIZE": 256,
    "MAX_TRAIN_PATCHES": 500,  # mirror PatchCore baseline so timing is comparable
    "SEED": 42,
}


def _try_efficientad_imports() -> tuple:
    """Returns (EfficientAd, Engine, ImageBatch_collate, ImageItem) or raises NotImplementedError."""
    try:
        from anomalib.models import EfficientAd  # noqa: F401
    except Exception as e:
        raise NotImplementedError(
            "anomalib.models.EfficientAd is not importable in this environment.\n"
            f"Underlying error: {type(e).__name__}: {e}\n"
            "TODO: see https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/efficient_ad.html"
        ) from e
    try:
        from anomalib.engine import Engine
        from anomalib.data.dataclasses.torch.image import ImageBatch, ImageItem
    except Exception as e:
        raise NotImplementedError(
            f"anomalib supporting classes import failed: {type(e).__name__}: {e}"
        ) from e
    return EfficientAd, Engine, ImageBatch, ImageItem


def _make_anomalib_dataset(ds: AITEXPatchDataset, ImageItem) -> torch.utils.data.Dataset:
    """Wrap AITEXPatchDataset so it yields anomalib ImageItem objects."""

    class _Adapter(torch.utils.data.Dataset):
        def __init__(self, inner: AITEXPatchDataset):
            self.inner = inner

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, idx: int):
            patch_tensor, label = self.inner[idx]
            sample = self.inner.samples[idx]
            return ImageItem(
                image=patch_tensor,
                gt_label=torch.tensor(bool(label)),
                image_path=str(sample.path),
            )

    return _Adapter(ds)


def _compute_metrics(scores: list[float], labels: list[int]) -> dict:
    a_scores = np.array(scores, dtype=np.float32)
    a_labels = np.array(labels, dtype=np.int32)
    auroc = float(roc_auc_score(a_labels, a_scores))
    precision, recall, thresholds = precision_recall_curve(a_labels, a_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1[:-1]))
    threshold = float(thresholds[best_idx])
    preds = (a_scores >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (a_labels == 1)).sum())
    fp = int(((preds == 1) & (a_labels == 0)).sum())
    fn = int(((preds == 0) & (a_labels == 1)).sum())
    tn = int(((preds == 0) & (a_labels == 0)).sum())
    defect_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "auroc": round(auroc, 4),
        "defect_recall": round(defect_recall, 4),
        "false_positive_rate": round(fpr, 4),
        "threshold": round(threshold, 4),
    }


def _accelerator() -> tuple[str, int]:
    if P.COLAB_MODE and torch.cuda.is_available():
        return "gpu", 1
    if torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


def main() -> None:
    set_seed(CONFIG["SEED"])
    EfficientAd, Engine, ImageBatch, ImageItem = _try_efficientad_imports()

    accelerator, devices = _accelerator()
    print(f"Accelerator: {accelerator}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = AITEXPatchDataset(P.DATA_ROOT / "train", patch_size=CONFIG["PATCH_SIZE"],
                                 transform=transform, transform_kind="torchvision")
    train_ds.samples = [s for s in train_ds.samples if s.label == 0]
    random.seed(CONFIG["SEED"])
    if len(train_ds.samples) > CONFIG["MAX_TRAIN_PATCHES"]:
        train_ds.samples = random.sample(train_ds.samples, CONFIG["MAX_TRAIN_PATCHES"])
    print(f"EfficientAD training on {len(train_ds.samples)} normal patches")

    val_ds = AITEXPatchDataset(P.DATA_ROOT / "val", patch_size=CONFIG["PATCH_SIZE"],
                               transform=transform, transform_kind="torchvision")
    print(f"Val patches: {len(val_ds)}")

    train_loader = DataLoader(_make_anomalib_dataset(train_ds, ImageItem), batch_size=1,
                              shuffle=False, num_workers=0, collate_fn=ImageBatch.collate)
    val_loader = DataLoader(_make_anomalib_dataset(val_ds, ImageItem), batch_size=8,
                            shuffle=False, num_workers=0, collate_fn=ImageBatch.collate)

    try:
        model = EfficientAd(
            imagenet_dir=os.environ.get("EFFICIENTAD_IMAGENETTE_DIR", "./datasets/imagenette"),
        )
    except TypeError:
        # older signatures may not accept imagenet_dir as kwarg
        model = EfficientAd()

    engine = Engine(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    P.ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = P.RUNS_DIR / f"{timestamp}_efficientad_default_seed{CONFIG['SEED']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\nTraining EfficientAD...")
    t0 = time.time()
    try:
        engine.fit(model, train_dataloaders=train_loader)
    except Exception as e:
        raise NotImplementedError(
            f"EfficientAD .fit() failed: {type(e).__name__}: {e}\n"
            "Diagnostic: anomalib's EfficientAd Lightning model differs from PatchCore — its "
            "configure_optimizers() reads self.trainer.datamodule.train_dataloader(), so it "
            "REQUIRES a LightningDataModule (not a bare DataLoader). PatchCore tolerates raw "
            "DataLoaders; EfficientAd does not. To run end-to-end on AITEX, wrap the shared "
            "AITEXPatchDataset in a LightningDataModule subclass (or use anomalib.data.Folder) "
            "and pass datamodule=... instead of train_dataloaders=...\n"
            "Secondary requirement: EfficientAd also expects an imagenette pretraining set; "
            "set EFFICIENTAD_IMAGENETTE_DIR or let anomalib auto-download.\n"
            "TODO (Phase 1B): https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/efficient_ad.html"
        ) from e
    elapsed = time.time() - t0
    print(f"Trained in {elapsed:.0f}s")

    print("Evaluating EfficientAD...")
    predictions = engine.predict(model, dataloaders=val_loader, return_predictions=True)
    all_scores: list[float] = []
    all_labels: list[int] = []
    for batch_out in predictions:
        if batch_out is None:
            continue
        if isinstance(batch_out, list):
            batch_out = batch_out[0] if batch_out else None
            if batch_out is None:
                continue
        if hasattr(batch_out, "pred_score") and batch_out.pred_score is not None:
            all_scores.extend(batch_out.pred_score.cpu().tolist())
        if hasattr(batch_out, "gt_label") and batch_out.gt_label is not None:
            all_labels.extend(batch_out.gt_label.cpu().tolist())

    if len(all_scores) != len(all_labels):
        raise RuntimeError(f"Score/label count mismatch: {len(all_scores)} vs {len(all_labels)}")

    metrics = _compute_metrics(all_scores, all_labels)
    metrics.update({
        "timestamp": timestamp,
        "model": "EfficientAD",
        "model_type": "anomaly",
        "n_train_patches": len(train_ds.samples),
        "n_val_patches": len(val_ds),
        "tag": "efficientad_default",
        "seed": CONFIG["SEED"],
    })
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nResults saved -> {run_dir}/metrics.json")
    print(f"AUROC: {metrics['auroc']:.4f} | Defect Recall: {metrics['defect_recall']:.4f} | "
          f"FPR: {metrics['false_positive_rate']:.4f}")


if __name__ == "__main__":
    main()
