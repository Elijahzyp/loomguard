"""
classification/train.py
Unified supervised training entrypoint for Exp 1-5 of the Phase 1A roadmap.

Usage:
    python -m scripts.classification.train \
        --config scripts/config/exp01_aug.yaml \
        --seed 42 \
        --tag exp01_aug_run1

CLI overrides take precedence over config file values:
    --aug-profile {none,conservative}
    --tta-mode {none,4way,8way}
    --model-name {resnet18,efficientnet_b0,convnext_tiny}
    --loss-type {ce,bce,focal}
    --epochs N
    --bs N
    --lr FLOAT
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Ensure the repo's scripts/ folder is importable when invoked via `python -m`
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]  # experiments/phase1a_aitex/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts import paths as P
from scripts.data import AITEXPatchDataset, compute_pos_weight
from scripts.seeding import set_seed
from scripts.classification import augmentations as aug_mod
from scripts.classification import evaluate as eval_mod
from scripts.classification import losses as loss_mod
from scripts.classification import models as model_mod
from scripts.classification import tta as tta_mod


# ---------- config ----------

def load_config(config_path: str) -> dict:
    default_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f)
    cfg_path = Path(config_path)
    if cfg_path.resolve() != default_path.resolve():
        with open(cfg_path) as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.aug_profile is not None:
        cfg["augmentation"]["profile"] = args.aug_profile
    if args.tta_mode is not None:
        cfg["tta"]["mode"] = args.tta_mode
    if args.model_name is not None:
        cfg["model"]["name"] = args.model_name
    if args.loss_type is not None:
        cfg["loss"]["type"] = args.loss_type
    if args.epochs is not None:
        cfg["schedule"]["epochs"] = args.epochs
    if args.bs is not None:
        cfg["data"]["batch_size"] = args.bs
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


# ---------- evaluation helpers ----------

def _val_eval(model, val_loader, loss_type: str, device) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (logits, labels, val_auroc, val_acc) without TTA. Cheap per-epoch eval."""
    logits, labels = tta_mod.predict_with_tta(model, val_loader, "none", device)
    probs = eval_mod.logits_to_defect_probs(logits, loss_type)
    lb = labels.numpy().astype(np.int32)
    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(lb, probs))
    preds = (probs >= 0.5).astype(np.int32)
    acc = float((preds == lb).mean())
    return logits.numpy(), lb, auroc, acc


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to experiment YAML (or default.yaml)")
    p.add_argument("--tag", required=True, help="short experiment tag, used in run dir name")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--aug-profile", choices=["none", "conservative"], default=None)
    p.add_argument("--tta-mode", choices=["none", "4way", "8way"], default=None)
    p.add_argument("--model-name", choices=sorted(model_mod.SUPPORTED), default=None)
    p.add_argument("--loss-type", choices=["ce", "bce", "focal"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--bs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--data-root", type=str, default=None,
                   help="Override DATA_ROOT (defaults to scripts.paths.DATA_ROOT)")
    args = p.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args)
    seed = int(cfg["seed"])
    set_seed(seed)

    device = P.get_device()
    print(f"Device: {device}")

    # COLAB_MODE auto-bumps batch size to 64 like the baseline does.
    if P.COLAB_MODE and "batch_size" not in (cfg.get("_user_set") or {}):
        if cfg["data"]["batch_size"] == 32:
            cfg["data"]["batch_size"] = 64

    data_root = Path(args.data_root) if args.data_root else P.DATA_ROOT
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"DATA_ROOT missing splits: {data_root}")

    # Resolve pos_weight from train split if requested
    pos_weight = cfg["loss"].get("pos_weight", "auto")
    if pos_weight == "auto":
        pos_weight = compute_pos_weight(train_dir, patch_size=cfg["data"]["patch_size"])
    pos_weight = float(pos_weight)

    loss_type = cfg["loss"]["type"]
    n_outputs = loss_mod.num_outputs_for(loss_type)

    train_tf = aug_mod.get_train_transforms(cfg["augmentation"]["profile"])
    val_tf = aug_mod.get_val_transforms()

    train_ds = AITEXPatchDataset(train_dir, patch_size=cfg["data"]["patch_size"],
                                 transform=train_tf, transform_kind="albumentations")
    val_ds = AITEXPatchDataset(val_dir, patch_size=cfg["data"]["patch_size"],
                               transform=val_tf, transform_kind="albumentations")

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"],
                              shuffle=True, num_workers=cfg["data"]["num_workers"],
                              pin_memory=False, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"],
                            pin_memory=False)
    print(f"Patches — train: {len(train_ds)} (n_normal={train_ds.label_counts()[0]}, "
          f"n_defect={train_ds.label_counts()[1]}), val: {len(val_ds)}")
    print(f"loss={loss_type}  pos_weight={pos_weight:.4f}  "
          f"model={cfg['model']['name']}  aug={cfg['augmentation']['profile']}  "
          f"tta={cfg['tta']['mode']}  seed={seed}")

    model = model_mod.build_model(cfg["model"]["name"], n_outputs,
                                  pretrained=cfg["model"]["pretrained"]).to(device)
    n_params = model_mod.count_params(model)

    criterion = loss_mod.build_loss(loss_type, pos_weight=pos_weight if loss_type == "bce" else None)
    if cfg["optim"]["optimizer"] != "adam":
        raise NotImplementedError(f"Optimizer {cfg['optim']['optimizer']} not implemented")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["optim"]["lr"]),
                                 weight_decay=float(cfg["optim"]["weight_decay"]))

    # ---------- run dir ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.tag}_seed{seed}"
    P.ensure_dirs()
    run_dir = P.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # ---------- training loop ----------
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_auroc", "secs"])

    best_auroc = -1.0
    best_acc = -1.0
    best_auroc_epoch = -1
    best_acc_epoch = -1
    model_path = run_dir / "model.pth"

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>8}  {'Val AUROC':>10}  {'Time':>6}")
    print("-" * 64)

    for epoch in range(1, int(cfg["schedule"]["epochs"]) + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # quick per-epoch val pass (no TTA, batched logit-only eval)
        val_logits, val_labels, val_auroc, val_acc = _val_eval(model, val_loader, loss_type, device)
        # val_loss for visibility (criterion-based)
        with torch.no_grad():
            vl_total = 0.0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                vl_total += criterion(model(images), labels).item() * images.size(0)
            val_loss = vl_total / len(val_loader.dataset)

        elapsed = time.time() - t0
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                        f"{val_acc:.6f}", f"{val_auroc:.6f}", f"{elapsed:.1f}"])
        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_acc:>7.2%}  {val_auroc:>10.4f}  {elapsed:>5.0f}s")

        # selection: primary = val_auroc (per cfg.selection_metric); also tracking val_acc
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_auroc_epoch = epoch
            torch.save(model.state_dict(), model_path)
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch

    # ---------- final eval (TTA-aware) ----------
    print(f"\nLoading best-by-val-AUROC model (epoch {best_auroc_epoch}) for final eval...")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    final_logits, final_labels = tta_mod.predict_with_tta(model, val_loader, cfg["tta"]["mode"], device)

    metrics = eval_mod.compute_metrics(
        final_logits, final_labels, loss_type,
        model=model, device=device, n_params=n_params,
        extra={
            "tag": args.tag,
            "run_name": run_name,
            "model_name": cfg["model"]["name"],
            "augmentation_profile": cfg["augmentation"]["profile"],
            "tta_mode": cfg["tta"]["mode"],
            "seed": seed,
            "epochs_run": int(cfg["schedule"]["epochs"]),
            "best_auroc_epoch": best_auroc_epoch,
            "best_acc_epoch": best_acc_epoch,
            "best_val_auroc_during_training": round(best_auroc, 4),
            "best_val_acc_during_training": round(best_acc, 4),
            "selection_metric": cfg.get("selection_metric", "val_auroc"),
            "pos_weight": pos_weight,
            "data_root": str(data_root),
            "device": str(device),
        },
    )

    eval_mod.write_metrics(run_dir, metrics)
    np.savez(run_dir / "val_predictions.npz",
             logits=final_logits.numpy(),
             labels=final_labels.numpy(),
             loss_type=np.array([loss_type]))

    print(f"\n=== {run_name} ===")
    print(f"AUROC (final, TTA={cfg['tta']['mode']}): {metrics['auroc']:.4f}")
    print(f"Best val_auroc during training: {best_auroc:.4f} (epoch {best_auroc_epoch})")
    print(f"Best val_acc during training:   {best_acc:.4f} (epoch {best_acc_epoch})")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
