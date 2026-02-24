"""
Sample 20% of the dataset that contains objects (non-empty labels).
Selection is random across all subfolders (train, val, test).
Copies both images and their corresponding YOLO label files.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# ===========================  EDIT YOUR PATHS HERE  ===========================
# Root can be: (1) a split folder with images/ and labels/ inside (e.g. .../train), or
#             (2) main folder with images/train, labels/train, etc., or (3) main folder with train/images, train/labels, etc.
DATASET_ROOT = r"D:\ship_detection\yolo_obb_minarea\yolo_obb_minarea\train"
OUTPUT_DIR   = r"D:\planet_dataset_making\dataset_20pct_objects"
FRACTION     = 0.20   # 20% of object-containing samples
RANDOM_SEED  = 42     # for reproducibility
# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# ==============================================================================


def get_subfolders(root: Path, base: str) -> list[Path]:
    """Return list of existing subfolders (e.g. train, val, test) under root/base."""
    folder = root / base
    if not folder.is_dir():
        return []
    return [f for f in folder.iterdir() if f.is_dir()]


def has_objects(label_path: Path) -> bool:
    """Return True if the label file exists and has at least one non-empty line."""
    if not label_path.is_file():
        return False
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                return True
    return False


def find_image_for_label(labels_dir: Path, images_dir: Path, stem: str) -> Path | None:
    """Given a label stem, find the corresponding image path (any supported extension)."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def collect_from_flat_split(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Collect (image, label) pairs when images and labels are directly in one folder each (no nested train/val)."""
    pairs: list[tuple[Path, Path]] = []
    for label_path in labels_dir.glob("*.txt"):
        if not has_objects(label_path):
            continue
        stem = label_path.stem
        img_path = find_image_for_label(labels_dir, images_dir, stem)
        if img_path is not None:
            pairs.append((img_path, label_path))
    return pairs


def collect_object_samples(root: Path) -> list[tuple[Path, Path]]:
    """
    Collect all (image_path, label_path) pairs where the label has at least one object.
    Supports three layouts:
    - Layout A (flat): root/images/ and root/labels/ with files directly inside (e.g. .../train/).
    - Layout B (nested): root/images/train/, root/images/val/, root/labels/train/, root/labels/val/.
    - Layout C (splits first): root/train/images/, root/train/labels/, root/val/images/, etc.
    """
    pairs: list[tuple[Path, Path]] = []
    images_root = root / "images"
    labels_root = root / "labels"

    # Layout C: main folder has train/, val/, test/ and each has images/ and labels/ inside
    if not images_root.is_dir():
        for split_dir in sorted(root.iterdir()):
            if not split_dir.is_dir():
                continue
            imgs = split_dir / "images"
            lbls = split_dir / "labels"
            if imgs.is_dir() and lbls.is_dir():
                pairs.extend(collect_from_flat_split(imgs, lbls))
        return pairs

    if not labels_root.is_dir():
        return pairs

    # Check for flat layout: images/ and labels/ contain files directly (no subfolders)
    subdirs_in_images = [f for f in images_root.iterdir() if f.is_dir()]
    if not subdirs_in_images:
        return collect_from_flat_split(images_root, labels_root)

    # Nested layout: images/train, images/val, labels/train, labels/val, etc.
    for img_split_dir in sorted(subdirs_in_images):
        split_name = img_split_dir.name
        lbl_split_dir = labels_root / split_name
        if not lbl_split_dir.is_dir():
            continue
        for label_path in lbl_split_dir.glob("*.txt"):
            stem = label_path.stem
            if not has_objects(label_path):
                continue
            img_path = find_image_for_label(lbl_split_dir, img_split_dir, stem)
            if img_path is not None:
                pairs.append((img_path, label_path))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Sample a fraction of the dataset that contains objects, "
                    "randomly from all subfolders; copy images and labels."
    )
    parser.add_argument("--dataset", type=str, default=DATASET_ROOT,
                        help="Root dataset folder (contains images/ and labels/ with train, val, test).")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output folder for the sampled images and labels.")
    parser.add_argument("--fraction", type=float, default=FRACTION,
                        help="Fraction of object-containing samples to keep (default 0.2 = 20%%).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    root = Path(args.dataset)
    out_dir = Path(args.output)
    fraction = args.fraction
    random.seed(args.seed)

    if not root.is_dir():
        sys.exit(f"Dataset root not found: {root}")

    print("Collecting samples that contain objects from all subfolders …")
    pairs = collect_object_samples(root)
    if not pairs:
        sys.exit("No image+label pairs with objects found. Check that (1) images/ and labels/ exist under the dataset path, "
                 "(2) label .txt files have at least one non-empty line, and (3) image filenames match label stems (e.g. img.png ↔ img.txt).")

    n_total = len(pairs)
    n_keep = max(1, int(round(n_total * fraction)))
    chosen = random.sample(pairs, n_keep)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    print(f"Copying {n_keep} of {n_total} object-containing samples ({100*fraction:.0f}%) to {out_dir} …")
    for img_path, lbl_path in chosen:
        out_img = out_dir / "images" / img_path.name
        out_lbl = out_dir / "labels" / lbl_path.name
        shutil.copy2(img_path, out_img)
        shutil.copy2(lbl_path, out_lbl)

    # Write a simple YOLO-style dataset.yaml for the subset
    yaml_path = out_dir / "dataset.yaml"
    yaml_content = f"""# 20% subset of object-containing samples (random from all splits)
path: {out_dir.resolve()}
train: images
# No val/test split in this subset; use train for training.
names:
  0: ship
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"Done. {n_keep} images and labels written to {out_dir}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
