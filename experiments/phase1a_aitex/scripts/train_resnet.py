"""
train_resnet.py
Trains ResNet18 binary classifier (normal vs defect) on AITEX patches.
Run from: experiments/phase1a_aitex/
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

CONFIG = {
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "LR": 1e-4,
    "PATCH_SIZE": 256,
    "SEED": 42,
    "COLAB_MODE": os.path.exists("/content"),
    "DATA_ROOT": Path("/content/drive/MyDrive/loomguard_data/prepared") if os.path.exists("/content") else Path("data/prepared"),
    "RESULTS_DIR": Path("/content/drive/MyDrive/loomguard_data/results") if os.path.exists("/content") else Path("results"),
}

IMAGE_WIDTH = 4096  # AITEX image width


def get_device() -> torch.device:
    if CONFIG["COLAB_MODE"]:
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PatchDataset(Dataset):
    """Slices each 4096×256 image into PATCH_SIZE-wide horizontal strips."""

    def __init__(self, root_dir: str, patch_size: int, transform=None):
        self.patch_size = patch_size
        self.transform = transform
        self.samples: list[tuple[Path, int, int]] = []  # (path, patch_idx, label)

        for label_name, label in [("normal", 0), ("defect", 1)]:
            label_dir = Path(root_dir) / label_name
            if not label_dir.exists():
                continue
            n_patches = IMAGE_WIDTH // patch_size
            for img_path in sorted(label_dir.glob("*.png")):
                for i in range(n_patches):
                    self.samples.append((img_path, i, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, patch_idx, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        left = patch_idx * self.patch_size
        img = img.crop((left, 0, left + self.patch_size, self.patch_size))
        if self.transform:
            img = self.transform(img)
        return img, label


def build_loader(split_dir: str, patch_size: int, batch_size: int, shuffle: bool):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = PatchDataset(split_dir, patch_size=patch_size, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=0, pin_memory=False)
    return loader, len(dataset)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    torch.manual_seed(CONFIG["SEED"])

    if CONFIG["COLAB_MODE"]:
        CONFIG["BATCH_SIZE"] = 64

    device = get_device()
    print(f"Device: {device}")

    data_root = CONFIG["DATA_ROOT"]
    train_loader, n_train = build_loader(
        data_root / "train", CONFIG["PATCH_SIZE"], CONFIG["BATCH_SIZE"], shuffle=True
    )
    val_loader, n_val = build_loader(
        data_root / "val", CONFIG["PATCH_SIZE"], CONFIG["BATCH_SIZE"], shuffle=False
    )
    print(f"Patches — train: {n_train}, val: {n_val}")

    model = timm.create_model("resnet18", pretrained=True, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])

    results_dir = CONFIG["RESULTS_DIR"]
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = results_dir / f"resnet18_{timestamp}.pth"

    best_acc = 0.0
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>8}  {'Time':>6}")
    print("-" * 52)

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_acc:>7.2%}  {elapsed:>5.0f}s")

        if epoch == 1 and elapsed > 180:
            print("\n⚠️  WARNING: 单 epoch 耗时超过 3 分钟，Mac 负载较高。")
            print("   建议切换到 Google Colab（CONFIG 中将 COLAB_MODE 改为 True，device 自动切换到 cuda）。")
            ans = input("   继续本地运行请输入 y，退出请输入 n：").strip().lower()
            if ans != "y":
                sys.exit(0)

    print(f"\nModel saved → {save_path}")
    print(f"Best Val Accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
