"""
prepare_data.py
Splits AITEX dataset into train/val sets and copies files into prepared/ structure.
Run from: experiments/phase1a_aitex/
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

CONFIG = {
    "nodefect_dir": "data/NODefect_images",
    "defect_dir": "data/Defect_images",
    "output_dir": "data/prepared",
    "val_size": 0.2,
    "random_seed": 42,
}


def collect_images(root: str, exclude_pattern: str = "_mask") -> list[Path]:
    """Recursively collect .png files, excluding any with exclude_pattern in filename."""
    root_path = Path(root)
    return [
        p for p in root_path.rglob("*.png")
        if exclude_pattern not in p.name
    ]


def copy_files(files: list[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)


def main():
    cfg = CONFIG
    output = Path(cfg["output_dir"])

    normal_files = collect_images(cfg["nodefect_dir"])
    defect_files = collect_images(cfg["defect_dir"])

    print(f"Found {len(normal_files)} normal images")
    print(f"Found {len(defect_files)} defect images")

    normal_train, normal_val = train_test_split(
        normal_files, test_size=cfg["val_size"], random_state=cfg["random_seed"]
    )
    defect_train, defect_val = train_test_split(
        defect_files, test_size=cfg["val_size"], random_state=cfg["random_seed"]
    )

    splits = {
        output / "train" / "normal": normal_train,
        output / "val"   / "normal": normal_val,
        output / "train" / "defect": defect_train,
        output / "val"   / "defect": defect_val,
    }

    for dest, files in splits.items():
        copy_files(files, dest)
        print(f"  {dest}: {len(files)} files")

    total = sum(len(v) for v in splits.values())
    print(f"\n=== Summary ===")
    print(f"train/normal : {len(normal_train)}")
    print(f"val/normal   : {len(normal_val)}")
    print(f"train/defect : {len(defect_train)}")
    print(f"val/defect   : {len(defect_val)}")
    print(f"Total        : {total}")


if __name__ == "__main__":
    main()
