#!/usr/bin/env python3
"""
Marks the stereographic projection's tangent point (PROJECTION_CENTER_LAT/LON
in kane_basin_projection.py, i.e. 80N, 69W) with a red '+' on
simulation_regions.jpg, writing a separate *_marked.jpg rather than
overwriting the source image.

Reuses kane_basin_projection.StereoProjection the same way
add_frame_to_initial_state.py does -- extents.json's affine transform is for
the native 20000px raster, so it's built with resize_factor =
extents_width / actual_image_width to match this image's actual width.

Usage:
    python3 mark_projection_center.py
"""
import sys
import json
from pathlib import Path as FilePath

from PIL import Image, ImageDraw

_THIS_DIR = FilePath(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / "image_download"))
from kane_basin_projection import StereoProjection, PROJECTION_CENTER_LAT, PROJECTION_CENTER_LON

Image.MAX_IMAGE_PIXELS = None

# --- Edit these ---------------------------------------------------------------
INPUT_FILE = "../../_paper/fig_initial_state/simulation_regions.jpg"
EXTENTS_JSON = "../image_download/nares_kane_basin_stereo/2024-06-06/extents.json"
OUTPUT_FILE = "../../_paper/fig_initial_state/simulation_regions_marked.jpg"

MARKER_HALF_LENGTH_PX = 25
MARKER_WIDTH_PX = 4
MARKER_COLOR = "red"
# -----------------------------------------------------------------------------


def main():
    with open(EXTENTS_JSON) as f:
        extents = json.load(f)

    with Image.open(INPUT_FILE) as im:
        img = im.convert("RGB")
        img_w, img_h = img.size

    resize_factor = extents["final_image_width"] / img_w
    proj = StereoProjection.from_extents_json(EXTENTS_JSON, resize_factor=resize_factor)
    cx, cy = proj.latlon_to_pixel(PROJECTION_CENTER_LAT, PROJECTION_CENTER_LON)
    print(f"Projection center ({PROJECTION_CENTER_LAT}N, {-PROJECTION_CENTER_LON}W) "
          f"-> pixel ({cx:.1f}, {cy:.1f})")

    draw = ImageDraw.Draw(img)
    s = MARKER_HALF_LENGTH_PX
    draw.line([(cx - s, cy), (cx + s, cy)], fill=MARKER_COLOR, width=MARKER_WIDTH_PX)
    draw.line([(cx, cy - s), (cx, cy + s)], fill=MARKER_COLOR, width=MARKER_WIDTH_PX)

    img.save(OUTPUT_FILE, quality=95)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
