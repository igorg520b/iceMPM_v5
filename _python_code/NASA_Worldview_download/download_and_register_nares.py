#!/usr/bin/env python3
"""
Downloads NASA GIBS MODIS true-color imagery for a set of dates (reusing
download_nares_strait.py's WMS downloader) and resamples each one onto
EXACTLY the same pixel grid the C++ simulation's exported rasters use -- same
orthographic projection, same PROJ_TRANSFORM_COEFFS/PROJ_RESIZE_FACTOR as
add_graticule_frame.py's OrthoProjection. That means add_graticule_frame.py
can be pointed at one of these registered images with no changes at all and
the graticule/border will land in exactly the same place as it does on a
simulation snapshot -- for side-by-side comparison figures.

Unlike download_nares_strait.py's own reproject_and_analyze() (which uses
rasterio's generic-purpose ortho warp, picking its own resolution/transform),
this script resamples directly against add_graticule_frame.py's exact pixel
<-> lat/lon formula, so there's no risk of the two ending up on slightly
different grids.

Usage:
    Edit DATES / SIM_JSON below if needed and run:
        python3 download_and_register_nares.py

Raw WMS downloads are cached in RAW_CACHE_DIR (skips re-downloading a date
that's already there). Registered outputs go to OUTPUT_DIR as <date>.png,
ready to use as add_graticule_frame.py's IMAGE_FILE.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from add_graticule_frame import OrthoProjection, DEFAULT_PROJ, box_boundary_loop  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_nares_strait import download_wms_image, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX  # noqa: E402

# --- Defaults, edit as needed -----------------------------------------------
#DATES = ["2024-06-10", "2024-06-13", "2024-06-15", "2024-06-17", "2024-06-19"]
DATES = ["2024-06-13"]
SIM_JSON = None                    # path to simulation.json; None = nares_17k params baked into add_graticule_frame.py
RAW_CACHE_DIR = "nares_wms_cache"  # cached raw WMS downloads (plain lat/lon grid), one per date
OUTPUT_DIR = "nares_registered"    # final images, pixel-registered like the simulation's exports
WMS_IMAGE_WIDTH_PX = 8000          # resolution requested from NASA GIBS, before registration resample
MARGIN_PX = 200                    # extra canvas margin beyond the lat/lon box, in sim-pixel units
# -----------------------------------------------------------------------------


def get_raw_wms_image(date_str):
    """Downloads (or loads from cache) the raw lat/lon-gridded WMS image for a date."""
    cache_path = Path(RAW_CACHE_DIR) / f"{date_str}.png"
    if cache_path.exists():
        print(f"[cache] Using cached WMS download for {date_str}: {cache_path}")
        return Image.open(cache_path).convert("RGB")

    Path(RAW_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    bbox = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    img = download_wms_image(date_str, bbox, WMS_IMAGE_WIDTH_PX, RAW_CACHE_DIR)
    img.save(cache_path)
    return img


def bilinear_sample(src, col, row):
    """Vectorized bilinear sample of src (H,W,3 uint8) at fractional (col,row)
    arrays; out-of-bounds samples come back as (0,0,0) with valid=False."""
    h, w = src.shape[:2]
    col0 = np.floor(col).astype(np.int64)
    row0 = np.floor(row).astype(np.int64)
    col1, row1 = col0 + 1, row0 + 1
    valid = (col0 >= 0) & (row0 >= 0) & (col1 < w) & (row1 < h)

    c0, c1 = np.clip(col0, 0, w - 1), np.clip(col1, 0, w - 1)
    r0, r1 = np.clip(row0, 0, h - 1), np.clip(row1, 0, h - 1)
    fc = (col - col0)[..., None]
    fr = (row - row0)[..., None]

    Ia = src[r0, c0].astype(np.float32)
    Ib = src[r0, c1].astype(np.float32)
    Ic = src[r1, c0].astype(np.float32)
    Id = src[r1, c1].astype(np.float32)
    top = Ia * (1 - fc) + Ib * fc
    bot = Ic * (1 - fc) + Id * fc
    out = top * (1 - fr) + bot * fr
    return out.astype(np.uint8), valid


def register_to_simulation_grid(raw_img, proj):
    """Resamples raw_img (a plain EPSG:4326 lat/lon grid covering
    LAT_MIN..LAT_MAX / LON_MIN..LON_MAX) onto the simulation's orthographic
    pixel grid: output pixel (x, y) is exactly proj.pixel_to_latlon(x, y) --
    the same formula the C++ code and add_graticule_frame.py use, so pixel
    (0, 0) here means the same real-world spot as pixel (0, 0) in a
    simulation snapshot."""
    src = np.array(raw_img)
    src_h, src_w = src.shape[:2]

    # Canvas big enough to hold the whole lat/lon box in simulation-pixel space.
    box = box_boundary_loop(proj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, n=400)
    out_w = int(np.ceil(box[:, 0].max())) + MARGIN_PX
    out_h = int(np.ceil(box[:, 1].max())) + MARGIN_PX

    xs, ys = np.meshgrid(np.arange(out_w), np.arange(out_h))
    lat, lon = proj.pixel_to_latlon(xs, ys)

    src_col = (lon - LON_MIN) / (LON_MAX - LON_MIN) * (src_w - 1)
    src_row = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (src_h - 1)

    out, valid = bilinear_sample(src, src_col, src_row)
    out[~valid] = 0  # outside the downloaded box, or off the visible globe: black
    return out


def main():
    if SIM_JSON:
        proj = OrthoProjection.from_simulation_json(SIM_JSON)
    else:
        proj = OrthoProjection(DEFAULT_PROJ["lat0"], DEFAULT_PROJ["lon0"],
                                DEFAULT_PROJ["resize_factor"], DEFAULT_PROJ["coeffs"])

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for date_str in DATES:
        out_path = Path(OUTPUT_DIR) / f"{date_str}.png"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        raw_img = get_raw_wms_image(date_str)
        registered = register_to_simulation_grid(raw_img, proj)
        Image.fromarray(registered).save(out_path)
        print(f"[done] Wrote {out_path} ({registered.shape[1]}x{registered.shape[0]} px)")


if __name__ == "__main__":
    main()
