"""
classification/ensemble.py
Average raw val logits across two or more runs and re-evaluate.
Does not retrain anything.

Usage:
    python -m scripts.classification.ensemble \
        --runs results/runs/<run_a> results/runs/<run_b> \
        --tag exp05_ensemble_top2
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts import paths as P
from scripts.classification import evaluate as eval_mod


def _load(run_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    npz = np.load(run_dir / "val_predictions.npz", allow_pickle=False)
    return npz["logits"], npz["labels"], str(npz["loss_type"][0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run dirs containing val_predictions.npz")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    if len(args.runs) < 2:
        raise ValueError("Need at least 2 runs for an ensemble")

    logits_list: list[np.ndarray] = []
    loss_types_per_run: list[str] = []
    labels_ref: np.ndarray | None = None
    for run in args.runs:
        run_dir = Path(run)
        l, lab, lt = _load(run_dir)
        logits_list.append(l)
        loss_types_per_run.append(lt)
        if labels_ref is None:
            labels_ref = lab
        elif not np.array_equal(labels_ref, lab):
            raise RuntimeError(f"Label mismatch between runs (different val splits?). Got {run_dir}.")

    unique_loss_types = set(loss_types_per_run)
    if len(unique_loss_types) > 1:
        # Heterogeneous heads -> convert each to defect probability and average probabilities.
        # Re-encode the average as a 1-logit pseudo-bce head so evaluate.compute_metrics treats it consistently.
        probs = []
        for l, lt in zip(logits_list, loss_types_per_run):
            probs.append(eval_mod.logits_to_defect_probs(l, lt))
        avg_probs = np.mean(np.stack(probs, axis=0), axis=0)
        # Re-encode as a 1-logit "bce" head via inverse sigmoid for downstream evaluate.
        eps = 1e-7
        clipped = np.clip(avg_probs, eps, 1 - eps)
        pseudo_logits = np.log(clipped / (1 - clipped))
        ensembled_logits = pseudo_logits[:, None]
        loss_type = "bce"
        ensemble_mode = "prob_avg (heterogeneous heads)"
    else:
        loss_type = next(iter(unique_loss_types))
        ensembled_logits = np.mean(np.stack(logits_list, axis=0), axis=0)
        ensemble_mode = "logit_avg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.tag}_ensemble"
    P.ensure_dirs()
    run_dir = P.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = eval_mod.compute_metrics(
        ensembled_logits, labels_ref, loss_type,
        extra={
            "tag": args.tag,
            "run_name": run_name,
            "ensemble_inputs": [str(Path(r).resolve()) for r in args.runs],
            "ensemble_mode": ensemble_mode,
            "tta_mode": "n/a",
            "model_name": "ensemble",
        },
    )
    eval_mod.write_metrics(run_dir, metrics)
    np.savez(run_dir / "val_predictions.npz",
             logits=ensembled_logits, labels=labels_ref, loss_type=np.array([loss_type]))
    print(f"\nEnsemble run dir: {run_dir}")
    print(f"AUROC: {metrics['auroc']:.4f}")


if __name__ == "__main__":
    main()
