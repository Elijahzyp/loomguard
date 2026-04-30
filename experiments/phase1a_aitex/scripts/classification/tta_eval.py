"""
tta_eval.py
Post-hoc TTA evaluation. Re-loads a finished classification run, runs the val
split through TTA (logit averaging across views), and writes
metrics_tta_<mode>.json / metrics_tta_<mode>.md / val_predictions_tta_<mode>.npz
into the SAME run dir. The original metrics.json is never touched.

Use case: Exp 2 (test-time augmentation) — train once with --tta-mode none, then
sweep TTA modes post-hoc without retraining.

Usage:
    python -m scripts.classification.tta_eval \
        --run-dir results/runs/20260418_010203_exp01_aug_none_seed42 \
        --tta-mode 4way

    # Cross-machine (Colab vs Mac) — override data root if the original config
    # points at a path that doesn't exist on this machine:
    python -m scripts.classification.tta_eval \
        --run-dir <run> --tta-mode 4way --data-root /content/drive/MyDrive/loomguard_data/prepared

Hard requirements: the run dir must contain model.pth, config.yaml,
val_predictions.npz (used as the label-order safety guard), and metrics.json
(used for delta_auroc).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]  # experiments/phase1a_aitex/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts import paths as P
from scripts.data import AITEXPatchDataset
from scripts.seeding import set_seed
from scripts.classification import augmentations as aug_mod
from scripts.classification import evaluate as eval_mod
from scripts.classification import losses as loss_mod
from scripts.classification import models as model_mod
from scripts.classification import tta as tta_mod


# ---------- helpers ----------

def _load_required(run_dir: Path) -> tuple[dict, dict, np.lib.npyio.NpzFile]:
    """Loads config.yaml, metrics.json, val_predictions.npz. Raises with a clear
    message naming the missing file if any are absent."""
    required = ["model.pth", "config.yaml", "val_predictions.npz", "metrics.json"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"run_dir is missing required files: {missing}.\n"
            f"run_dir = {run_dir}\n"
            "tta_eval requires a complete run dir produced by scripts.classification.train."
        )
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    orig_metrics = json.loads((run_dir / "metrics.json").read_text())
    npz = np.load(run_dir / "val_predictions.npz", allow_pickle=False)
    return cfg, orig_metrics, npz


def _resolve_data_root(args_data_root: str | None, orig_data_root: str | None) -> Path:
    """Resolution order: --data-root > metrics.json data_root > paths.DATA_ROOT.
    Verifies the resolved path exists; otherwise raises with an actionable message."""
    if args_data_root:
        chosen = Path(args_data_root)
        source = "--data-root"
    elif orig_data_root:
        chosen = Path(orig_data_root)
        source = "metrics.json:data_root"
    else:
        chosen = P.DATA_ROOT
        source = "paths.DATA_ROOT"

    if not chosen.exists():
        raise FileNotFoundError(
            f"data_root={chosen} (from {source}) does not exist on this machine.\n"
            f"This run was originally trained with data_root={orig_data_root!r}.\n"
            "Pass --data-root explicitly to point at the val split on the current machine."
        )
    return chosen


def _make_val_dataset(cfg: dict, val_dir: Path, val_tf):
    """Phase-1B-ready dispatch. AITEX is the default; knitted will be added later."""
    dataset_kind = cfg.get("dataset", "aitex")
    if dataset_kind == "aitex":
        return AITEXPatchDataset(
            val_dir, patch_size=cfg["data"]["patch_size"],
            transform=val_tf, transform_kind="albumentations",
        )
    if dataset_kind == "knitted":
        raise NotImplementedError(
            "dataset=knitted requires KnittedFabricPatchDataset (Phase 1B). "
            "Add it to scripts/data.py before running TTA on knitted-fabric runs."
        )
    raise ValueError(f"Unknown dataset kind in config: {dataset_kind!r}")


def _resolve_device(arg_device: str | None) -> torch.device:
    if arg_device:
        return torch.device(arg_device)
    return P.get_device()


def _time_tta_inference_ms(
    model: torch.nn.Module, device: torch.device, mode: str,
    image_size: int = 256, warmup: int = 10, runs: int = 50,
) -> float:
    """Per-image latency for a single forward + V views averaged. Mirrors
    evaluate.time_inference_ms but exercises the TTA forward path."""
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            views = tta_mod._views(x, mode)
            _ = torch.stack([model(v) for v in views], dim=0).mean(dim=0)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            views = tta_mod._views(x, mode)
            _ = torch.stack([model(v) for v in views], dim=0).mean(dim=0)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return round(1000.0 * elapsed / runs, 3)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, write_fn) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=str,
                   help="Path to a completed classification run dir.")
    p.add_argument("--tta-mode", required=True, choices=["4way", "8way"],
                   help="TTA mode. 'none' is the original eval — use train.py for that.")
    p.add_argument("--data-root", type=str, default=None,
                   help="Override DATA_ROOT (e.g. when moving between Mac and Colab).")
    p.add_argument("--bs", type=int, default=None,
                   help="Override val batch size. Defaults to the run's training batch size.")
    p.add_argument("--device", type=str, default=None,
                   help="Override device (cpu/cuda/mps). Default: paths.get_device().")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing metrics_tta_<mode>.* files in the run dir.")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir not found or not a directory: {run_dir}")

    cfg, orig_metrics, npz = _load_required(run_dir)

    # --- output gating ---
    out_metrics_json = run_dir / f"metrics_tta_{args.tta_mode}.json"
    out_metrics_md = run_dir / f"metrics_tta_{args.tta_mode}.md"
    out_predictions = run_dir / f"val_predictions_tta_{args.tta_mode}.npz"
    existing = [pth for pth in (out_metrics_json, out_metrics_md, out_predictions) if pth.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "TTA outputs already exist in run_dir; pass --force to overwrite:\n  "
            + "\n  ".join(str(e) for e in existing)
        )

    # --- seed for deterministic per-eval ops ---
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # --- device ---
    device = _resolve_device(args.device)
    print(f"Device: {device}")

    # --- data ---
    data_root = _resolve_data_root(args.data_root, orig_metrics.get("data_root"))
    val_dir = data_root / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"val split missing: {val_dir}")

    val_tf = aug_mod.get_val_transforms()
    val_ds = _make_val_dataset(cfg, val_dir, val_tf)

    bs = int(args.bs) if args.bs is not None else int(cfg["data"]["batch_size"])
    val_loader = DataLoader(
        val_ds, batch_size=bs,
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        pin_memory=False,
    )
    print(f"Val patches: {len(val_ds)}  bs={bs}")

    # --- model ---
    loss_type = cfg["loss"]["type"]
    n_outputs = loss_mod.num_outputs_for(loss_type)
    model = model_mod.build_model(cfg["model"]["name"], n_outputs,
                                  pretrained=cfg["model"]["pretrained"]).to(device)
    state = torch.load(run_dir / "model.pth", map_location=device)
    model.load_state_dict(state)
    n_params = model_mod.count_params(model)
    print(f"Model: {cfg['model']['name']}  loss_type={loss_type}  params={n_params:,}")

    # --- TTA forward pass ---
    print(f"Running TTA ({args.tta_mode}) over val split...")
    t0 = time.time()
    tta_logits, tta_labels = tta_mod.predict_with_tta(model, val_loader, args.tta_mode, device)
    print(f"TTA pass took {time.time() - t0:.1f}s")

    # --- safety guard: label order must match the original eval pass ---
    original_labels = npz["labels"]
    new_labels = tta_labels.numpy()
    if new_labels.shape != original_labels.shape or not np.array_equal(new_labels, original_labels):
        raise RuntimeError(
            "Label order mismatch between original val_predictions.npz and current TTA pass.\n"
            f"  original shape: {original_labels.shape}  current shape: {new_labels.shape}\n"
            "Causes: dataset content changed, prepare_data was re-run, or DataLoader shuffle is on.\n"
            "tta_eval refuses to write metrics under a different val ordering."
        )

    # --- metrics: skip per-image timing inside compute_metrics; we measure the
    # TTA forward path separately and inject it as inference_ms_per_image_with_tta. ---
    metrics = eval_mod.compute_metrics(
        tta_logits, tta_labels, loss_type,
        n_params=n_params,
        extra={
            # Tag with an exp02_tta_ prefix so aggregate_results.py routes the row
            # into the exp02_tta summary group regardless of the original tag.
            "tag": f"exp02_tta_{orig_metrics.get('tag', 'unknown')}_{args.tta_mode}",
            "run_name": run_dir.name,
            "model_name": cfg["model"]["name"],
            "augmentation_profile": cfg["augmentation"]["profile"],
            "tta_mode": args.tta_mode,
            "seed": seed,
            "source_run_dir": str(run_dir),
            "source_tag": orig_metrics.get("tag"),
            "source_auroc": orig_metrics.get("auroc"),
            "data_root": str(data_root),
            "device": str(device),
        },
    )
    metrics["delta_auroc"] = round(metrics["auroc"] - float(orig_metrics["auroc"]), 4)
    metrics["inference_ms_per_image_with_tta"] = _time_tta_inference_ms(
        model, device, args.tta_mode,
    )

    # --- atomic writes ---
    _atomic_write_text(out_metrics_json, json.dumps(metrics, indent=2))
    _atomic_write_text(out_metrics_md, eval_mod.metrics_to_markdown(metrics))
    def _save_npz(tmp: Path) -> None:
        # np.savez appends '.npz' to a string path; pass a file handle to skip that.
        with open(tmp, "wb") as f:
            np.savez(
                f,
                logits=tta_logits.numpy(),
                labels=tta_labels.numpy(),
                loss_type=np.array([loss_type]),
            )

    _atomic_write_bytes(out_predictions, _save_npz)

    print()
    print(f"=== TTA {args.tta_mode} on {run_dir.name} ===")
    print(f"AUROC: {metrics['auroc']:.4f}  (delta vs original: {metrics['delta_auroc']:+.4f})")
    print(f"Inference ms/image (with TTA): {metrics['inference_ms_per_image_with_tta']:.3f}")
    print(f"Wrote {out_metrics_json.name}, {out_metrics_md.name}, {out_predictions.name}")


if __name__ == "__main__":
    main()
