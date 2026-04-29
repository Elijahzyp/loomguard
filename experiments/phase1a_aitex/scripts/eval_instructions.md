## Colab 执行顺序

1. **Cell 1–3**（clone repo, mount Drive, install deps）
2. **新建 cell**，运行 ResNet18 评估：
   ```
   %run scripts/eval_resnet18.py
   ```
3. **Cell 5**（PatchCore，MAX_TRAIN_PATCHES=500）
4. **Cell 6**（打印完成信息 + 断开 runtime）

## 预计时间

| 步骤 | 时间 |
|------|------|
| eval_resnet18 | ~2 min |
| patchcore 500 patches | ~3–5 min |

## 结果文件

| 文件 | 内容 |
|------|------|
| `resnet18_eval_{timestamp}.json` | AUROC + @0.5 指标 + best-F1 指标 |
| `resnet18_threshold_grid_{timestamp}.json` | 完整 threshold grid（0.1–0.9） |
| `patchcore_{timestamp}.json` | AUROC + Defect Recall + FPR |
