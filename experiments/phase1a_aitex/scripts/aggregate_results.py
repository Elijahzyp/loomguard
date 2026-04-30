"""
aggregate_results.py
Idempotent scan of results/runs/*/metrics.json into per-experiment summary CSVs.
Run from experiments/phase1a_aitex/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts import paths as P

# Run-dir name format: <YYYYMMDD_HHMMSS>_<tag>_seed<N>
_RUN_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})_(?P<tag>.+)_seed(?P<seed>\d+)$")
_ENS_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})_(?P<tag>.+)_ensemble$")
_ANOM_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})_(?P<tag>.+)_seed(?P<seed>\d+)$")


def _parse_run_name(name: str) -> dict:
    m = _RUN_RE.match(name)
    if m:
        return {"timestamp": m.group("ts"), "tag": m.group("tag"), "seed": int(m.group("seed"))}
    m = _ENS_RE.match(name)
    if m:
        return {"timestamp": m.group("ts"), "tag": m.group("tag"), "seed": -1}
    return {"timestamp": "", "tag": name, "seed": -1}


def _flatten(metrics: dict, parsed: dict) -> dict:
    row = {
        "run": metrics.get("run_name", ""),
        "tag": metrics.get("tag") or parsed["tag"],
        "seed": metrics.get("seed", parsed["seed"]),
        "auroc": metrics.get("auroc"),
        "model_name": metrics.get("model_name", metrics.get("model", "")),
        "loss_type": metrics.get("loss_type", ""),
        "augmentation_profile": metrics.get("augmentation_profile", ""),
        "tta_mode": metrics.get("tta_mode", ""),
        "params": metrics.get("params"),
        "inference_ms_per_image": metrics.get("inference_ms_per_image"),
        "best_auroc_epoch": metrics.get("best_auroc_epoch"),
        "best_acc_epoch": metrics.get("best_acc_epoch"),
        "best_val_auroc_during_training": metrics.get("best_val_auroc_during_training"),
        "best_val_acc_during_training": metrics.get("best_val_acc_during_training"),
    }
    at05 = metrics.get("at_threshold_0.5") or {}
    row["recall_at_0.5"] = at05.get("recall")
    row["fpr_at_0.5"] = at05.get("fpr")
    row["f1_at_0.5"] = at05.get("f1")
    rfpr = metrics.get("recall_at_fpr_le_0.10") or {}
    row["recall_at_fpr_le_0.10"] = rfpr.get("recall")
    row["threshold_at_fpr_le_0.10"] = rfpr.get("threshold")
    bf = metrics.get("best_f1_threshold") or {}
    row["best_f1_threshold"] = bf.get("threshold")
    row["best_f1"] = bf.get("f1")
    # Post-hoc TTA-specific fields (only populated for metrics_tta_*.json rows)
    row["delta_auroc"] = metrics.get("delta_auroc")
    row["inference_ms_per_image_with_tta"] = metrics.get("inference_ms_per_image_with_tta")
    row["source_tag"] = metrics.get("source_tag")
    row["source_auroc"] = metrics.get("source_auroc")
    # Anomaly-detection runs use a flatter schema
    if metrics.get("model_type") == "anomaly" or metrics.get("model") == "EfficientAD":
        row["model_name"] = metrics.get("model", row["model_name"])
        row["defect_recall"] = metrics.get("defect_recall")
        row["fpr_anomaly"] = metrics.get("false_positive_rate")
    return row


def _experiment_group(tag: str) -> str:
    """Map a tag to its experiment-summary group (exp01_aug, exp03_arch, ...)."""
    if tag.startswith("exp01"):
        return "exp01_aug"
    if tag.startswith("exp02"):
        return "exp02_tta"
    if tag.startswith("exp03"):
        return "exp03_arch"
    if tag.startswith("exp04"):
        return "exp04_loss"
    if tag.startswith("exp05"):
        return "exp05_ensemble"
    if "efficientad" in tag.lower() or "patchcore" in tag.lower():
        return "anomaly_detection"
    return "misc"


def main() -> None:
    P.ensure_dirs()
    if not P.RUNS_DIR.exists():
        print(f"No runs/ directory at {P.RUNS_DIR}. Nothing to aggregate.")
        return

    rows_by_group: dict[str, list[dict]] = {}
    for run_dir in sorted(P.RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        # Pick up both metrics.json (primary) and metrics_tta_*.json (post-hoc TTA).
        for metrics_path in sorted(run_dir.glob("metrics*.json")):
            try:
                metrics = json.loads(metrics_path.read_text())
            except Exception as e:
                print(f"  skip {run_dir.name}/{metrics_path.name}: {e}")
                continue
            parsed = _parse_run_name(run_dir.name)
            row = _flatten(metrics, parsed)
            row["run_dir"] = str(run_dir)
            row["metrics_file"] = metrics_path.name
            group = _experiment_group(row["tag"] or parsed["tag"])
            rows_by_group.setdefault(group, []).append(row)

    if not rows_by_group:
        print("No metrics.json found in any run dir.")
        return

    for group, rows in rows_by_group.items():
        df = pd.DataFrame(rows).sort_values(["tag", "seed", "run"])
        out = P.SUMMARY_DIR / f"{group}_comparison.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out}  ({len(df)} runs)")


if __name__ == "__main__":
    main()
