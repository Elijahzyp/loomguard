"""
evaluate.py
Threshold sweep + AUROC + inference timing. Loss-type aware.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

LossType = Literal["ce", "bce", "focal"]

THRESHOLDS = [round(0.05 + 0.05 * i, 2) for i in range(19)]  # 0.05 .. 0.95


def logits_to_defect_probs(logits: torch.Tensor | np.ndarray, loss_type: LossType) -> np.ndarray:
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if loss_type == "ce":
        if logits.ndim == 1 or logits.shape[-1] == 1:
            raise ValueError("CE requires a 2-logit head, got 1-logit shape")
        return torch.softmax(logits, dim=-1)[:, 1].numpy()
    # bce / focal -> 1-logit head -> sigmoid
    return torch.sigmoid(logits.view(-1)).numpy()


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


def recall_at_fpr_le(grid: list[dict], fpr_target: float) -> dict | None:
    eligible = [r for r in grid if r["fpr"] <= fpr_target]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r["recall"])


def time_inference_ms(model: torch.nn.Module, device, image_size: int = 256, warmup: int = 10, runs: int = 50) -> float:
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return round(1000.0 * elapsed / runs, 3)


def compute_metrics(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    loss_type: LossType,
    *,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    n_params: int | None = None,
    extra: dict | None = None,
) -> dict:
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.asarray(labels, dtype=np.int32)
    probs = logits_to_defect_probs(logits, loss_type)

    auroc = float(roc_auc_score(labels, probs))
    grid = [metrics_at_threshold(probs, labels, t) for t in THRESHOLDS]
    at05 = next(r for r in grid if r["threshold"] == 0.5)
    best_f1 = max(grid, key=lambda r: r["f1"])
    r_at_fpr10 = recall_at_fpr_le(grid, 0.10)

    out = {
        "auroc": round(auroc, 4),
        "loss_type": loss_type,
        "threshold_grid": grid,
        "at_threshold_0.5": at05,
        "best_f1_threshold": best_f1,
        "recall_at_fpr_le_0.10": r_at_fpr10,
    }
    if n_params is not None:
        out["params"] = int(n_params)
    if model is not None and device is not None:
        out["inference_ms_per_image"] = time_inference_ms(model, device)
    if extra:
        out.update(extra)
    return out


def metrics_to_markdown(m: dict) -> str:
    lines = []
    lines.append(f"# Run metrics — {m.get('tag', 'unknown')}")
    lines.append("")
    lines.append(f"- AUROC: **{m['auroc']:.4f}**")
    lines.append(f"- loss_type: {m['loss_type']}")
    if "params" in m:
        lines.append(f"- trainable params: {m['params']:,}")
    if "inference_ms_per_image" in m:
        lines.append(f"- inference ms/image (bs=1, warmup=10): {m['inference_ms_per_image']:.3f}")
    lines.append(f"- TTA mode: {m.get('tta_mode', 'none')}")
    lines.append("")
    at05 = m["at_threshold_0.5"]
    lines.append(f"@0.5 — recall {at05['recall']:.4f} | fpr {at05['fpr']:.4f} | f1 {at05['f1']:.4f}")
    bf = m["best_f1_threshold"]
    lines.append(f"@best F1 (t={bf['threshold']}) — recall {bf['recall']:.4f} | fpr {bf['fpr']:.4f} | f1 {bf['f1']:.4f}")
    rfpr = m.get("recall_at_fpr_le_0.10")
    if rfpr is not None:
        lines.append(f"recall @ FPR<=0.10 (t={rfpr['threshold']}) — {rfpr['recall']:.4f}")
    else:
        lines.append("recall @ FPR<=0.10 — no threshold satisfies this constraint")
    lines.append("")
    lines.append("| threshold | recall | fpr | precision | f1 |")
    lines.append("|---|---|---|---|---|")
    for r in m["threshold_grid"]:
        lines.append(f"| {r['threshold']:.2f} | {r['recall']:.4f} | {r['fpr']:.4f} | {r['precision']:.4f} | {r['f1']:.4f} |")
    return "\n".join(lines) + "\n"


def write_metrics(run_dir: Path, metrics: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "metrics.md").write_text(metrics_to_markdown(metrics))
