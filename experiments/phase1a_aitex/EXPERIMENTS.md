# Phase 1A AITEX — Experiments Guide

This document describes how to run the **new unified pipeline** (Exp 1–5 + the
EfficientAD anomaly-detection scaffold). The frozen baseline scripts
(`train_resnet.py`, `eval_resnet18.py`, `train_patchcore.py`,
`prepare_data.py`) and `colab_runner.ipynb` cells 1–6 are left byte-identical
and remain the reference points for AUROC ≈ 0.8675 (ResNet18) and ≈ 0.66
(PatchCore).

## Layout

```
scripts/
  prepare_data.py train_resnet.py eval_resnet18.py train_patchcore.py   # FROZEN BASELINES
  paths.py seeding.py data.py aggregate_results.py run_sweep.sh          # shared
  config/  default.yaml exp01..exp05.yaml                                # configs
  classification/  train.py models.py losses.py augmentations.py
                   tta.py evaluate.py ensemble.py
  anomaly/         anomaly_efficientad.py
results/
  runs/<YYYYMMDD_HHMMSS>_<tag>_seed<N>/
    config.yaml train_log.csv val_predictions.npz metrics.json metrics.md model.pth
  summary/<group>_comparison.csv
```

All commands assume `cwd = experiments/phase1a_aitex/`.

## Differences vs the frozen ResNet18 baseline (call out in any comparison)

| Aspect | Frozen baseline (`train_resnet.py`) | New pipeline (`scripts.classification.train`) |
| --- | --- | --- |
| Best-model selection | best `val_acc` across epochs | best `val_auroc` (also logs best `val_acc`) |
| Seeding | `torch.manual_seed(42)` only | `torch + numpy + random + cuda + cudnn deterministic` |
| Augmentation | None | configurable; `none` mirrors baseline within ±0.01 AUROC |
| Eval | separate `eval_resnet18.py` after training | inline at run end + writes `val_predictions.npz` for ensembling |
| Output layout | flat `results/resnet18_<ts>.pth` | per-run dir under `results/runs/` |
| Loss | `CrossEntropyLoss` (2-logit head) | configurable; `bce` is 1-logit + `pos_weight`, `focal` is 1-logit |

Reproducibility tolerance against the baseline is **±0.01 AUROC**, not bit-equivalence.

---

## Exp 1 — Conservative Albumentations augmentation (with no-aug control)

Conservative profile = `HorizontalFlip(0.5)`, `VerticalFlip(0.5)`,
`RandomRotate90(0.5)`, `RandomBrightnessContrast(±0.2, p=0.5)`,
`Affine(rotate=±5°, scale=0.95–1.05, p=0.3, BORDER_REFLECT_101)`.

```bash
# control (no aug — should match frozen baseline within ±0.01 AUROC)
python -m scripts.classification.train \
  --config scripts/config/default.yaml \
  --tag exp01_aug_none --seed 42

# conservative aug
python -m scripts.classification.train \
  --config scripts/config/exp01_aug.yaml \
  --tag exp01_aug_conservative --seed 42
```

## Exp 2 — Test-time augmentation (4-way / 8-way)

Trains identically to the baseline, only the eval-time pass changes. TTA
**averages raw logits** across views; the head's sigmoid/softmax is applied
afterward.

```bash
python -m scripts.classification.train \
  --config scripts/config/exp02_tta.yaml \
  --tag exp02_tta_4way --seed 42

# optional 8-way
python -m scripts.classification.train \
  --config scripts/config/exp02_tta.yaml \
  --tta-mode 8way --tag exp02_tta_8way --seed 42
```

## Exp 3 — Architecture sweep (resnet18 / efficientnet_b0 / convnext_tiny × 3 seeds)

```bash
bash scripts/run_sweep.sh scripts/config/default.yaml exp03_resnet18         resnet18
bash scripts/run_sweep.sh scripts/config/default.yaml exp03_efficientnet_b0  efficientnet_b0
bash scripts/run_sweep.sh scripts/config/default.yaml exp03_convnext_tiny    convnext_tiny
# regenerate summary:
python scripts/aggregate_results.py
# -> results/summary/exp03_arch_comparison.csv
```

## Exp 4 — Loss comparison (bce / ce / focal)

`bce` uses `BCEWithLogitsLoss(pos_weight = N_normal / N_defect)` on a 1-logit head.
`focal` uses γ=2, α=0.25 on a 1-logit head. `ce` is the baseline (2-logit head).

```bash
bash scripts/run_sweep.sh scripts/config/default.yaml exp04_loss_bce   ""  bce
bash scripts/run_sweep.sh scripts/config/default.yaml exp04_loss_ce    ""  ce
bash scripts/run_sweep.sh scripts/config/default.yaml exp04_loss_focal ""  focal
```

## Exp 5 — Two-model logit-average ensemble

```bash
python -m scripts.classification.ensemble \
  --runs results/runs/<best_resnet18_run> results/runs/<best_efficientnet_b0_run> \
  --tag exp05_ensemble_top2

python scripts/aggregate_results.py
```

If the two input runs use **different loss types** (heterogeneous heads), the
ensemble script falls back to averaging defect probabilities (logged as
`ensemble_mode: "prob_avg (heterogeneous heads)"` in `metrics.json`).

## Anomaly detection — EfficientAD scaffold (parallel to PatchCore)

```bash
python -m scripts.anomaly.anomaly_efficientad
```

Phase 1A goal: validate the scaffolding works end-to-end. We do not tune
EfficientAD on AITEX. If anomalib's `EfficientAd` cannot be imported or
instantiated, the script raises a `NotImplementedError` with a clear
diagnostic (commonly: missing imagenette dir; set
`EFFICIENTAD_IMAGENETTE_DIR`). PatchCore's `train_patchcore.py` is left
unchanged.

## Aggregating results

```bash
python scripts/aggregate_results.py
# writes results/summary/{exp01_aug,exp02_tta,exp03_arch,exp04_loss,exp05_ensemble,anomaly_detection,...}_comparison.csv
```

Idempotent — re-runs simply overwrite the CSVs.

## Running the frozen baseline for comparison

```bash
python scripts/train_resnet.py
python scripts/eval_resnet18.py
python scripts/train_patchcore.py
```

These remain the canonical reference; do not modify them.
