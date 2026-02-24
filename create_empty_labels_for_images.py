"""
Create an empty .txt label file for each image that does not have a corresponding
label file. Use this for images with no objects so YOLO-style datasets have
one label file per image.
"""

import argparse
import sys
from pathlib import Path

# ===========================  EDIT YOUR PATHS HERE  ===========================
DATASET_ROOT = r"D:\ship_detection\yolo_obb_minarea\yolo_obb_minarea\train\air_cloud"  # or path to train/val/test
# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# ==============================================================================


def get_images_in_dir(images_dir: Path) -> list[Path]:
    """Return list of image paths in the given directory."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(images_dir.glob(f"*{ext}"))
    return sorted(images)


def ensure_empty_label(images_dir: Path, labels_dir: Path, dry_run: bool = False) -> int:
    """
    For each image in images_dir, if no corresponding .txt exists in labels_dir,
    create an empty .txt with the same stem. Returns count of created files.
    """
    created = 0
    for img_path in get_images_in_dir(images_dir):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            if not dry_run:
                label_path.write_text("", encoding="utf-8")
            created += 1
    return created


def process_flat_split(images_dir: Path, labels_dir: Path, dry_run: bool) -> int:
    """Process a single split where images and labels are in one folder each."""
    if not images_dir.is_dir():
        return 0
    labels_dir.mkdir(parents=True, exist_ok=True)
    return ensure_empty_label(images_dir, labels_dir, dry_run)


def process_root(root: Path, dry_run: bool = False) -> int:
    """
    Walk dataset root and create empty labels for images that don't have one.
    Supports:
    - Layout A (flat): root/images/ and root/labels/
    - Layout B (nested): root/images/train/, root/labels/train/, etc.
    - Layout C (splits first): root/train/images/, root/train/labels/, etc.
    """
    total_created = 0

    # Layout C: root/train/, root/val/, root/test/ each with images/ and labels/
    train_dir = root / "train"
    if train_dir.is_dir() and (train_dir / "images").is_dir():
        for split_dir in sorted(root.iterdir()):
            if not split_dir.is_dir():
                continue
            imgs = split_dir / "images"
            lbls = split_dir / "labels"
            if imgs.is_dir():
                n = process_flat_split(imgs, lbls, dry_run)
                if n:
                    print(f"  {split_dir.name}: created {n} empty label(s)")
                total_created += n
        return total_created

    # Layout A or B: root/images/ and root/labels/
    images_root = root / "images"
    labels_root = root / "labels"

    if not images_root.is_dir():
        return total_created

    subdirs = [f for f in images_root.iterdir() if f.is_dir()]
    if not subdirs:
        # Flat: all images directly in images/
        n = process_flat_split(images_root, labels_root, dry_run)
        if n:
            print(f"  (flat): created {n} empty label(s)")
        total_created += n
    else:
        # Nested: images/train, images/val, ...
        for img_split in sorted(subdirs):
            name = img_split.name
            lbl_split = labels_root / name
            n = process_flat_split(img_split, lbl_split, dry_run)
            if n:
                print(f"  {name}: created {n} empty label(s)")
            total_created += n

    return total_created


def main():
    parser = argparse.ArgumentParser(
        description="Create an empty .txt label file for each image that has no label file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_ROOT,
        help="Dataset root (contains images/ and labels/, or train/val/test with images/labels inside).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print how many files would be created, do not write.",
    )
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.is_dir():
        sys.exit(f"Dataset root not found: {root}")

    print(f"Scanning: {root}")
    if args.dry_run:
        print("(dry run – no files will be created)")
    n = process_root(root, dry_run=args.dry_run)
    print(f"Done. Total empty label files created: {n}")


if __name__ == "__main__":
    main()
