# LoomGuard — Claude Code 指南

## 项目概述
基于摄像头的织物缺陷检测 MVP。当前在 Phase 1A (AITEX 公开数据集上验证算法)。

## 工作目录
所有命令在 `experiments/phase1a_aitex/` 下执行。

## 两套代码
- **Frozen baseline** (不要改): `scripts/train_resnet.py`, `scripts/eval_resnet18.py`,
  `scripts/train_patchcore.py`, `scripts/prepare_data.py`
- **统一管线** (活跃开发): `scripts/classification/train.py` 及 `scripts/classification/` 下所有文件

## 关键路径
- 配置: `scripts/config/default.yaml` (所有实验的基准)
- 数据: 本地 `data/prepared/`, Colab 上 `/content/drive/MyDrive/loomguard_data/prepared/`
- 结果: 本地 `results/runs/`, Colab 上 `/content/drive/MyDrive/loomguard_data/results/runs/`
- 路径自动切换: `scripts/paths.py` 通过 `COLAB_MODE = os.path.exists("/content")` 判断

## 训练命令 (统一管线)
```bash
python -m scripts.classification.train \
  --config scripts/config/default.yaml \
  --tag <experiment_tag> --seed 42 --epochs 30 \
  --aug-profile none --model-name resnet18
```

## TTA 后评估
```bash
python -m scripts.classification.tta_eval \
  --run-dir results/runs/<run_dir> --tta-mode 4way
```

## 汇总结果
```bash
python scripts/aggregate_results.py
```

## 已完成的实验 (不需要重跑)
- Frozen baseline: ResNet18 AUROC 0.8675, PatchCore AUROC 0.66
- Exp 1 (augmentation): none vs conservative × 3 seeds — 结论: 无效
- Exp 2 (TTA): 4way/8way × 6 models — 结论: 无效

## 约定
- 改 frozen baseline 文件之前必须问我
- default.yaml 的 epochs 是 10 (frozen baseline 兼容), sweep 通过 --epochs 30 override
- 新实验 tag 格式: exp0N_<描述>
- seed 用 42, 123, 2024 三个
