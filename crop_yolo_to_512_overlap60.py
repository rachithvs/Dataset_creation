"""
Crop images into 512x512 px tiles with 60% overlap.
Labels are YOLO format: one .txt file per image (same stem as image).
Output: tiled images and corresponding YOLO .txt labels per tile.

Reference: create_yolo_datasets.py (tile logic, padding).
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ===========================  EDIT YOUR PATHS HERE  ===========================
IMAGES_DIR = r"D:\planet_dataset_making\dataset_20pct_objects\images"   # folder containing images
LABELS_DIR = r"D:\planet_dataset_making\dataset_20pct_objects\labels"   # folder containing YOLO .txt (same stem as image)
OUTPUT_DIR = r"D:\planet_dataset_making\dataset_20pct_objects\New folder"  # folder where images/ and labels/ will be created
# ==============================================================================

# Defaults: 512x512 tiles, 60% overlap
TILE_SIZE = 512
OVERLAP_PERCENT = 0.60  # 60% overlap
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def calculate_tile_positions(img_width: int, img_height: int,
                             tile_size: int = TILE_SIZE,
                             overlap_percent: float = OVERLAP_PERCENT):
    """
    Compute (x_offset, y_offset) for each tile with given overlap.
    Overlap 60% => step = tile_size * (1 - 0.6) = 0.4 * tile_size (e.g. 205 px).
    Last tile in each row/column is aligned to image edge so no gap.
    """
    overlap_px = int(tile_size * overlap_percent)
    step = tile_size - overlap_px  # e.g. 512 - 307 = 205

    if img_width <= tile_size and img_height <= tile_size:
        return [(0, 0)], step

    # Build x offsets: step until we cover width; last one = right edge
    x_offs = []
    x = 0
    while x + tile_size <= img_width:
        x_offs.append(x)
        x += step
    if img_width > tile_size and (not x_offs or x_offs[-1] + tile_size < img_width):
        x_offs.append(max(0, img_width - tile_size))

    # Build y offsets the same way
    y_offs = []
    y = 0
    while y + tile_size <= img_height:
        y_offs.append(y)
        y += step
    if img_height > tile_size and (not y_offs or y_offs[-1] + tile_size < img_height):
        y_offs.append(max(0, img_height - tile_size))

    positions = [(xi, yi) for xi in x_offs for yi in y_offs]
    return positions, step


def load_yolo_labels(label_path: Path, img_width: int, img_height: int):
    """
    Load YOLO .txt file. Each line: class_id x_center y_center width height (normalized 0-1).
    Returns list of dicts with pixel coords: {class_id, x_center, y_center, width, height} in pixels.
    """
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except (ValueError, IndexError):
                continue
            # Convert to pixel coordinates
            xc_px = xc * img_width
            yc_px = yc * img_height
            w_px = w * img_width
            h_px = h * img_height
            x1 = xc_px - w_px / 2
            y1 = yc_px - h_px / 2
            x2 = xc_px + w_px / 2
            y2 = yc_px + h_px / 2
            boxes.append({
                "class_id": cid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "xc": xc_px, "yc": yc_px, "w": w_px, "h": h_px,
            })
    return boxes


def box_intersects_tile(box, x_off, y_off, tile_size: int):
    """Check if bbox intersects tile [x_off, x_off+tile_size] x [y_off, y_off+tile_size]."""
    tx2 = x_off + tile_size
    ty2 = y_off + tile_size
    return not (box["x2"] <= x_off or box["x1"] >= tx2 or box["y2"] <= y_off or box["y1"] >= ty2)


def clip_box_to_tile(box, x_off, y_off, tile_size: int):
    """
    Clip bbox to tile region. Returns (x1, y1, x2, y2) in tile-local coordinates,
    or None if no intersection.
    """
    x1 = max(0, box["x1"] - x_off)
    y1 = max(0, box["y1"] - y_off)
    x2 = min(tile_size, box["x2"] - x_off)
    y2 = min(tile_size, box["y2"] - y_off)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def box_to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, tile_size: int) -> str:
    """Convert tile-local box to YOLO line: class_id x_center y_center width height (normalized 0-1)."""
    xc = (x1 + x2) / 2.0 / tile_size
    yc = (y1 + y2) / 2.0 / tile_size
    w = (x2 - x1) / tile_size
    h = (y2 - y1) / tile_size
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def create_tile_with_padding(image: Image.Image, x_off: int, y_off: int,
                             tile_size: int = TILE_SIZE, pad_color: int = 0) -> Image.Image:
    """
    Extract a tile from image; pad with black if tile extends beyond image.
    (Same logic as create_yolo_datasets.create_tile_with_padding.)
    """
    img_w, img_h = image.size
    x1, y1 = x_off, y_off
    x2, y2 = x_off + tile_size, y_off + tile_size

    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        tile = Image.new(image.mode, (tile_size, tile_size), pad_color)
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(img_w, x2)
        src_y2 = min(img_h, y2)
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        src_region = image.crop((src_x1, src_y1, src_x2, src_y2))
        tile.paste(src_region, (dst_x1, dst_y1))
        return tile
    return image.crop((x1, y1, x2, y2))


def process_one_image(
    image_path: Path,
    label_path: Path,
    positions: list,
    tile_size: int,
    min_bbox_px: int,
    out_images_dir: Path,
    out_labels_dir: Path,
    stem: str,
    image_ext: str,
) -> int:
    """
    Load image and labels, create all tiles, write images and YOLO .txt per tile.
    Returns number of tiles written.
    """
    img = Image.open(image_path)
    img_w, img_h = img.size
    boxes = load_yolo_labels(label_path, img_w, img_h)

    count = 0
    for idx, (x_off, y_off) in enumerate(positions):
        tile = create_tile_with_padding(img, x_off, y_off, tile_size)
        tile_name = f"{stem}_tile_{idx}{image_ext}"
        img_out = out_images_dir / tile_name
        lbl_out = out_labels_dir / (tile_name.rsplit(".", 1)[0] + ".txt")

        yolo_lines = []
        for box in boxes:
            if not box_intersects_tile(box, x_off, y_off, tile_size):
                continue
            clipped = clip_box_to_tile(box, x_off, y_off, tile_size)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            if (x2 - x1) < min_bbox_px or (y2 - y1) < min_bbox_px:
                continue
            line = box_to_yolo_line(box["class_id"], x1, y1, x2, y2, tile_size)
            yolo_lines.append(line)

        tile.save(img_out)
        with open(lbl_out, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")
        count += 1
    return count


def find_image_label_pairs(images_dir: Path, labels_dir: Path):
    """Yield (image_path, label_path) for each image that has a matching .txt (or allow missing .txt)."""
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        yield img_path, lbl_path


def main():
    parser = argparse.ArgumentParser(
        description="Crop images into 512x512 tiles with 60%% overlap; labels are YOLO .txt per image."
    )
    parser.add_argument("--images", type=Path, default=Path(IMAGES_DIR),
                        help="Directory containing images.")
    parser.add_argument("--labels", type=Path, default=Path(LABELS_DIR),
                        help="Directory containing YOLO .txt labels (same stem as image).")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_DIR),
                        help="Output root: images/ and labels/ will be created here.")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE,
                        help="Tile size in pixels.")
    parser.add_argument("--overlap", type=float, default=OVERLAP_PERCENT,
                        help="Overlap fraction 0-1 (e.g. 0.6 for 60%%).")
    parser.add_argument("--min-bbox", type=int, default=3,
                        help="Minimum bbox side in pixels to keep in a tile.")
    args = parser.parse_args()

    images_dir = args.images
    labels_dir = args.labels
    out_root = args.output
    tile_size = args.tile_size
    overlap = args.overlap
    min_bbox_px = args.min_bbox

    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    out_images_dir = out_root / "images"
    out_labels_dir = out_root / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(find_image_label_pairs(images_dir, labels_dir))
    if not pairs:
        raise SystemExit("No image files found in --images (or no matching labels in --labels).")

    step = int(tile_size * (1 - overlap))
    print(f"Tile size: {tile_size}x{tile_size}  |  Overlap: {overlap:.0%}  |  Step: {step} px")
    print(f"Output: {out_root}")
    print(f"Processing {len(pairs)} images...")

    total_tiles = 0
    for img_path, lbl_path in tqdm(pairs, desc="Tiling"):
        img = Image.open(img_path)
        w, h = img.size
        img.close()
        positions, _ = calculate_tile_positions(w, h, tile_size, overlap)
        ext = img_path.suffix
        n = process_one_image(
            img_path,
            lbl_path,
            positions,
            tile_size,
            min_bbox_px,
            out_images_dir,
            out_labels_dir,
            img_path.stem,
            ext,
        )
        total_tiles += n

    print(f"Done. {total_tiles} tiles written to {out_images_dir} and {out_labels_dir}.")


if __name__ == "__main__":
    main()
