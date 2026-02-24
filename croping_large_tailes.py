"""
Tile geo-referenced Planet Labs GeoTIFF images into 768x768 patches with
configurable overlap, convert shapefile polygon annotations to YOLO bounding-box
format, and split the result into train / val sets.
"""

import argparse
import math
import sys
import yaml
from pathlib import Path

# ===========================  EDIT YOUR PATHS HERE  ===========================
IMAGES_DIR      = r"D:\mosaic_planet_imageray\mosaic_images"       # root folder (subfolders searched automatically)
ANNOTATIONS_DIR = r"D:\new_VARUNA_FIXED\data\test_out"    # folder with .shp files
OUTPUT_DIR      = r"D:\planet_dataset_making\512_planet_dataset"

TILE_SIZE   = 512
OVERLAP     = 0.2       # 60 %
BAND_ORDER  = "1,2,3"   # comma-separated 1-indexed bands (e.g. "3,2,1" for BGR→RGB)
VAL_SPLIT   = 0.2       # 20 % validation
MIN_BBOX    = 5          # ignore boxes smaller than 5 px
MAX_BLACK   = 0.8        # skip tiles where >= 80 % of pixels are black (zero)
# ==============================================================================

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import rowcol, xy
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_band_order(band_str: str) -> list[int]:
    """Parse a comma-separated band string like '3,2,1' into a list of ints."""
    return [int(b.strip()) for b in band_str.split(",")]


def is_mostly_black(arr: np.ndarray, threshold: float) -> bool:
    """Return True if the fraction of all-zero pixels >= *threshold*."""
    black_pixels = np.all(arr == 0, axis=-1).sum()
    return (black_pixels / (arr.shape[0] * arr.shape[1])) >= threshold


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale an array to 0-255 uint8.  Handles 8-bit and 16-bit inputs."""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype in (np.uint16, np.int16):
        return np.clip(arr / 256, 0, 255).astype(np.uint8)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def geo_bbox_for_tile(transform, col_off: int, row_off: int,
                      tile_size: int) -> box:
    """Return a Shapely box in the image's CRS for a pixel tile window."""
    x_min, y_top = xy(transform, row_off, col_off, offset="ul")
    x_max, y_bot = xy(transform, row_off + tile_size,
                       col_off + tile_size, offset="ul")
    return box(min(x_min, x_max), min(y_top, y_bot),
               max(x_min, x_max), max(y_top, y_bot))


def polygon_to_pixel_coords(geom, transform):
    """Convert a Shapely geometry from CRS coords to pixel (col, row) coords."""
    def _to_pixel(x, y, z=None):
        r, c = rowcol(transform, x, y)
        return (c, r)
    return shapely_transform(_to_pixel, geom)


def geom_to_yolo_bbox(geom, tile_col: int, tile_row: int,
                      tile_size: int, min_bbox: int):
    """Convert a Shapely geometry (in pixel coords) to a YOLO bbox line.

    Returns None if the box is smaller than *min_bbox* in either dimension.
    Format: '0 x_center y_center width height' (normalized 0-1).
    """
    minc, minr, maxc, maxr = geom.bounds
    # Shift to tile-local coordinates
    minc -= tile_col
    maxc -= tile_col
    minr -= tile_row
    maxr -= tile_row

    # Clip to tile
    minc = max(0.0, minc)
    minr = max(0.0, minr)
    maxc = min(float(tile_size), maxc)
    maxr = min(float(tile_size), maxr)

    w = maxc - minc
    h = maxr - minr
    if w < min_bbox or h < min_bbox:
        return None

    x_center = (minc + maxc) / 2.0 / tile_size
    y_center = (minr + maxr) / 2.0 / tile_size
    w_norm = w / tile_size
    h_norm = h / tile_size

    return f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def find_tif_files(root: Path) -> list[Path]:
    """Recursively find all .tif / .tiff files under *root* (including subfolders)."""
    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in (".tif", ".tiff")
    )
    return files


def load_annotations(annotation_dir: Path) -> list[gpd.GeoDataFrame]:
    """Recursively load all .shp files from a directory tree."""
    shapefiles = sorted(annotation_dir.rglob("*.shp"))
    if not shapefiles:
        sys.exit(f"No .shp files found in {annotation_dir}")
    gdfs = []
    for shp in shapefiles:
        gdf = gpd.read_file(shp)
        if gdf.empty:
            continue
        gdfs.append(gdf)
    if not gdfs:
        sys.exit("All shapefiles are empty.")
    return gdfs


def reproject_annotations(gdfs: list[gpd.GeoDataFrame],
                          target_crs) -> gpd.GeoDataFrame:
    """Concatenate and reproject all annotation GeoDataFrames to *target_crs*."""
    reprojected = []
    for gdf in gdfs:
        if gdf.crs and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        reprojected.append(gdf)
    merged = gpd.GeoDataFrame(
        gpd.pd.concat(reprojected, ignore_index=True),
        crs=target_crs,
    )
    return merged


def process_image(image_path: Path, annotations: gpd.GeoDataFrame,
                  tile_size: int, stride: int, band_order: list[int],
                  min_bbox: int,
                  max_black: float) -> list[tuple[np.ndarray, list[str], str]]:
    """Tile one GeoTIFF and produce (image_array, label_lines, tile_name) tuples."""

    tiles = []
    with rasterio.open(image_path) as src:
        transform = src.transform
        img_w, img_h = src.width, src.height

        bands = [src.read(b) for b in band_order]
        img_data = np.stack(bands, axis=-1)  # H x W x C
        img_data = normalize_to_uint8(img_data)

        image_box = box(*src.bounds)
        spatial_idx = annotations.sindex
        possible_idx = list(spatial_idx.intersection(src.bounds))
        if possible_idx:
            relevant = annotations.iloc[possible_idx]
        else:
            relevant = annotations.iloc[0:0]

        n_cols = max(1, math.ceil((img_w - tile_size) / stride) + 1)
        n_rows = max(1, math.ceil((img_h - tile_size) / stride) + 1)

        stem = image_path.stem

        for ri in range(n_rows):
            for ci in range(n_cols):
                row_off = min(ri * stride, max(0, img_h - tile_size))
                col_off = min(ci * stride, max(0, img_w - tile_size))

                # Crop (pad with zeros if tile extends past image edge)
                r_end = min(row_off + tile_size, img_h)
                c_end = min(col_off + tile_size, img_w)
                crop = img_data[row_off:r_end, col_off:c_end]

                if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
                    padded = np.zeros((tile_size, tile_size, crop.shape[2]),
                                     dtype=np.uint8)
                    padded[:crop.shape[0], :crop.shape[1]] = crop
                    crop = padded

                if is_mostly_black(crop, max_black):
                    continue

                # Geo-extent of this tile
                tile_box = geo_bbox_for_tile(transform, col_off, row_off,
                                             tile_size)

                # Find intersecting annotations
                labels: list[str] = []
                if not relevant.empty:
                    hit_idx = list(relevant.sindex.intersection(tile_box.bounds))
                    for idx in hit_idx:
                        geom = relevant.geometry.iloc[idx]
                        if geom is None or geom.is_empty:
                            continue
                        clipped = geom.intersection(tile_box)
                        if clipped.is_empty:
                            continue
                        pix_geom = polygon_to_pixel_coords(clipped, transform)
                        line = geom_to_yolo_bbox(pix_geom, col_off, row_off,
                                                 tile_size, min_bbox)
                        if line is not None:
                            labels.append(line)

                tile_name = f"{stem}_r{row_off}_c{col_off}"
                tiles.append((crop, labels, tile_name))

    return tiles


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def save_dataset(all_tiles: dict[str, list[tuple[np.ndarray, list[str], str]]],
                 output_dir: Path, val_split: float):
    """Split tiles per source image and write images + labels to disk."""

    image_names = list(all_tiles.keys())
    if len(image_names) < 2:
        train_names = image_names
        val_names = []
    else:
        train_names, val_names = train_test_split(
            image_names, test_size=val_split, random_state=42
        )

    split_map = {}
    for n in train_names:
        split_map[n] = "train"
    for n in val_names:
        split_map[n] = "val"

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}

    for img_name, tiles in all_tiles.items():
        split = split_map[img_name]
        for crop, labels, tile_name in tiles:
            img_out = output_dir / "images" / split / f"{tile_name}.png"
            lbl_out = output_dir / "labels" / split / f"{tile_name}.txt"

            Image.fromarray(crop).save(img_out)
            with open(lbl_out, "w") as f:
                f.write("\n".join(labels))
                if labels:
                    f.write("\n")

            counts[split] += 1

    return counts


def write_dataset_yaml(output_dir: Path):
    cfg = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["object"],
    }
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create a YOLO dataset from Planet Labs GeoTIFF images "
                    "and shapefile annotations.  Paths default to the values "
                    "set in the config block at the top of this script."
    )
    parser.add_argument("--images", type=str, default=IMAGES_DIR,
                        help="Root folder of GeoTIFF images (subfolders searched).")
    parser.add_argument("--annotations", type=str, default=ANNOTATIONS_DIR,
                        help="Folder of Shapefiles (.shp), searched recursively.")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output dataset folder.")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE,
                        help="Tile size in pixels.")
    parser.add_argument("--overlap", type=float, default=OVERLAP,
                        help="Overlap fraction 0-1.")
    parser.add_argument("--band-order", type=str, default=BAND_ORDER,
                        help="Comma-separated 1-indexed band numbers for RGB.")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT,
                        help="Fraction of source images for validation.")
    parser.add_argument("--min-bbox", type=int, default=MIN_BBOX,
                        help="Minimum bbox size in px to keep.")
    parser.add_argument("--max-black", type=float, default=MAX_BLACK,
                        help="Skip tiles where this fraction of pixels are black.")
    args = parser.parse_args()

    images_dir = Path(args.images)
    ann_dir = Path(args.annotations)
    output_dir = Path(args.output)
    tile_size = args.tile_size
    overlap = args.overlap
    band_order = parse_band_order(args.band_order)
    val_split = args.val_split
    min_bbox = args.min_bbox
    max_black = args.max_black

    stride = int(tile_size * (1 - overlap))
    print(f"Tile size: {tile_size}  |  Overlap: {overlap:.0%}  |  Stride: {stride}")

    if not images_dir.is_dir():
        sys.exit(f"Images directory not found: {images_dir}")
    if not ann_dir.is_dir():
        sys.exit(f"Annotations directory not found: {ann_dir}")

    tif_files = find_tif_files(images_dir)
    if not tif_files:
        sys.exit(f"No .tif/.tiff files found in {images_dir} (searched recursively)")

    print(f"Found {len(tif_files)} GeoTIFF image(s).")
    gdfs = load_annotations(ann_dir)
    print(f"Loaded {len(gdfs)} shapefile(s).")

    all_tiles: dict[str, list[tuple[np.ndarray, list[str], str]]] = {}

    for tif_path in tqdm(tif_files, desc="Processing images"):
        with rasterio.open(tif_path) as src:
            target_crs = src.crs

        annotations = reproject_annotations(gdfs, target_crs)
        tiles = process_image(
            tif_path, annotations, tile_size, stride, band_order, min_bbox,
            max_black
        )
        all_tiles[tif_path.stem] = tiles
        print(f"  {tif_path.name}: {len(tiles)} tiles "
              f"({sum(1 for _, l, _ in tiles if l)} with labels)")

    print("\nSaving dataset …")
    counts = save_dataset(all_tiles, output_dir, val_split)
    write_dataset_yaml(output_dir)

    print(f"\nDone!  train: {counts['train']} tiles  |  val: {counts['val']} tiles")
    print(f"Dataset written to: {output_dir.resolve()}")
    print(f"YAML config: {(output_dir / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()
