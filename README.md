# Dataset Creation Scripts

Scripts in this folder help build and manage YOLO-format datasets: tiling large images, creating or sampling labels, and cropping with overlap. Each script has a **path config block** at the top where you can set default input/output paths.

---

## 1. `croping_large_tailes.py`

**Purpose:** Turn geo-referenced **Planet Labs GeoTIFF** images into tiled patches and convert **shapefile** polygon annotations into YOLO bounding boxes. Output is a train/val split with a `dataset.yaml` for YOLO.

**Inputs:**
- **Images:** Folder of GeoTIFF (`.tif`/`.tiff`) images; subfolders are searched.
- **Annotations:** Folder of shapefiles (`.shp`); polygons are reprojected to match each image’s CRS.

**Output:**
- `output/images/train/`, `output/images/val/` — PNG tiles.
- `output/labels/train/`, `output/labels/val/` — one YOLO `.txt` per tile (`class_id x_center y_center width height`, normalized).
- `output/dataset.yaml` — YOLO dataset config.

**Config (edit at top of file):**
- `IMAGES_DIR` — GeoTIFF root folder.
- `ANNOTATIONS_DIR` — Shapefile folder.
- `OUTPUT_DIR` — Dataset output root.
- `TILE_SIZE` — Tile size in pixels (e.g. 512).
- `OVERLAP` — Overlap fraction 0–1 (e.g. 0.2 for 20%).
- `BAND_ORDER` — Comma-separated 1-indexed bands for RGB (e.g. `"1,2,3"`).
- `VAL_SPLIT` — Fraction of source images used for validation.
- `MIN_BBOX` — Minimum bbox side (px) to keep.
- `MAX_BLACK` — Skip tiles where this fraction of pixels is black.

**Usage:**
```bash
python croping_large_tailes.py
# Or override paths:
python croping_large_tailes.py --images "D:\geotiffs" --annotations "D:\shp" --output "D:\out" --tile-size 512 --overlap 0.2
```

**Dependencies:** `geopandas`, `rasterio`, `shapely`, `scikit-learn`, `PyYAML`, `tqdm`, `Pillow`, `numpy`.

---

## 2. `crop_yolo_to_512_overlap60.py`

**Purpose:** Crop **existing** images and their **YOLO `.txt` labels** into **512×512** tiles with **60% overlap**. No shapefiles or GeoTIFFs; works on plain images and per-image label files.

**Inputs:**
- **Images:** Folder of images (e.g. `.png`, `.jpg`, `.tif`).
- **Labels:** Folder of YOLO `.txt` files with the **same base name** as each image (e.g. `img.png` ↔ `img.txt`).

**Output:**
- `output/images/` — 512×512 tiles named like `{stem}_tile_0.png`, `{stem}_tile_1.png`, …
- `output/labels/` — One `.txt` per tile with the same base name; boxes are clipped to the tile and in normalized 0–1.

**Config (edit at top of file):**
- `IMAGES_DIR` — Folder containing images.
- `LABELS_DIR` — Folder containing YOLO `.txt` (same stem as image).
- `OUTPUT_DIR` — Folder where `images/` and `labels/` will be created.

**Usage:**
```bash
python crop_yolo_to_512_overlap60.py
# Or override:
python crop_yolo_to_512_overlap60.py --images "D:\imgs" --labels "D:\lbls" --output "D:\tiles" --overlap 0.6 --min-bbox 3
```

**Options:** `--tile-size`, `--overlap` (default 0.6), `--min-bbox` (drop very small boxes in a tile). Images without a matching `.txt` still get tiled; their tile labels are empty.

---

## 3. `create_empty_labels_for_images.py`

**Purpose:** Create an **empty** `.txt` label file for every image that **does not** already have a corresponding label file. Use this so every image in a YOLO-style dataset has a label file (e.g. “no objects” = empty file).

**Input:** A **dataset root** that follows one of these layouts:
- **Layout A (flat):** `root/images/`, `root/labels/`
- **Layout B (nested):** `root/images/train/`, `root/images/val/`, `root/labels/train/`, `root/labels/val/`
- **Layout C (splits first):** `root/train/images/`, `root/train/labels/`, `root/val/images/`, etc.

**Output:** Empty `.txt` files created in the matching `labels` folder(s) for any image that lacked one (same stem, e.g. `img.png` → `img.txt`).

**Config (edit at top of file):**
- `DATASET_ROOT` — Path to the dataset root (as above).

**Usage:**
```bash
python create_empty_labels_for_images.py
# Or:
python create_empty_labels_for_images.py --dataset "D:\my_dataset"
# Preview only (no files created):
python create_empty_labels_for_images.py --dataset "D:\my_dataset" --dry-run
```

---

## 4. `sample_20pct_with_objects.py`

**Purpose:** **Sample a fraction** (default 20%) of the dataset that has **at least one object** (non-empty label). Randomly selects image+label pairs from all splits and **copies** them to an output folder. Useful for building a smaller subset for training or inspection.

**Input:** Same dataset root layouts as in `create_empty_labels_for_images.py` (flat, nested, or splits-first with `images/` and `labels/`).

**Output:**
- `output/images/` — Copied images.
- `output/labels/` — Copied YOLO `.txt` files.
- `output/dataset.yaml` — Simple YOLO dataset config pointing at the subset.

**Config (edit at top of file):**
- `DATASET_ROOT` — Source dataset root.
- `OUTPUT_DIR` — Where to write the sampled subset.
- `FRACTION` — Fraction of object-containing samples to keep (e.g. 0.2 = 20%).
- `RANDOM_SEED` — For reproducible sampling.

**Usage:**
```bash
python sample_20pct_with_objects.py
# Or:
python sample_20pct_with_objects.py --dataset "D:\full_dataset" --output "D:\subset" --fraction 0.2 --seed 42
```

---

## Typical workflow

1. **GeoTIFF + shapefiles:** Use `croping_large_tailes.py` to tile GeoTIFFs and convert polygons to YOLO → train/val tiles + `dataset.yaml`.
2. **Ensure one label per image:** Run `create_empty_labels_for_images.py` on the dataset root so every image has a `.txt` (empty if no objects).
3. **Subset with objects:** Run `sample_20pct_with_objects.py` to get a random fraction of samples that have objects.
4. **Re-tile with overlap:** Run `crop_yolo_to_512_overlap60.py` on images + labels to get 512×512 tiles with 60% overlap and per-tile YOLO labels.

---

## YOLO label format

All scripts assume or produce **YOLO format** (one `.txt` per image/tile):

- One line per object: `class_id x_center y_center width height`
- All coordinates **normalized to [0, 1]** relative to image width/height.

Supported image extensions in the scripts: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` (and in `crop_yolo_to_512_overlap60.py` also `.bmp`).
