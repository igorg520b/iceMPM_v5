#!/usr/bin/env python3
"""
Adds a lat/lon graticule frame to '_paper/fig_initial_state/simulation_initial_color.jpg'
(the Copernicus/Sentinel-2 Kane Basin mosaic used for the simulation's initial
color, downloaded and reprojected by image_download/kane_basin_stereographic.py,
then downsampled from its native 20000px width to 2000px). Structurally this is
the same recipe as fig1_nares_strait/add_frame_to_basemap_rotated_labels.py --
same rotated-tick-label neatline style, same tuned BORDER_WIDTH/LABEL_FONTSIZE
(see that script's docstring) -- kept as its own script rather than a shared
one because the pixel <-> lat/lon registration here comes from a different
source: kane_basin_stereographic.py's extents.json (pyproj-based
kane_basin_projection.StereoProjection), not a simulation.json georef sidecar.

extents.json was written for the native 20000px-wide raster; the JPEG here is
a 10x downsample of that (same aspect ratio, 2000x1293), so StereoProjection
is built with resize_factor = extents_width / actual_image_width instead of
hardcoding 10, matching whatever downsample ratio the JPEG actually is.

All eight per-date download folders in image_download/nares_kane_basin_stereo/
share the exact same extents.json (same fixed download bbox/resolution for
every date) -- so which date's extents.json is used here does not matter.

Unlike the fig1_nares_strait basemap scripts, LAT_MIN/MAX and LON_MIN/MAX here
match the source image's own native download bbox almost exactly (both come
from kane_basin_projection.py's constants) -- there's no surrounding margin
of real image data for the frame stripes and tick labels to spill into, the
way there was for those basemaps (fetched with extra margin baked in). So the
frame here is drawn onto a canvas padded with real white pixels on all four
sides (sized exactly to what the stripes/labels need, via the same
label_anchor_bounds logic, plus PAD_FRAC breathing room), rather than cropping
into the source image and clamping to [0, img_w]/[0, img_h] like those
scripts do -- that clamp is what was clipping the frame and labels here.

Usage:
    python3 add_frame_to_initial_state.py
"""
import sys
import json
from pathlib import Path as FilePath

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

_THIS_DIR = FilePath(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / "graticule"))
sys.path.insert(0, str(_THIS_DIR.parent / "image_download"))
from add_graticule_frame import (draw_graticule, draw_neatline, draw_outside_mask,
                                  box_boundary_loop, nice_step, nice_ticks, _outward_normal)
from kane_basin_projection import StereoProjection

Image.MAX_IMAGE_PIXELS = None

# --- Edit these ---------------------------------------------------------------
INPUT_FILE = "../../_paper/fig_initial_state/simulation_initial_color.jpg"
EXTENTS_JSON = "../image_download/nares_kane_basin_stereo/2024-06-06/extents.json"
OUTPUT_FILE = "../../_paper/fig_initial_state/simulation_initial_color_framed.png"

# Same download bbox as kane_basin_projection.py's LAT_MIN/LAT_MAX/LON_MIN/LON_MAX
LAT_MIN, LAT_MAX = 77.34, 82.40
LON_MIN, LON_MAX = -85.32, -43.42

SHOW_GRATICULE = False
BORDER_WIDTH = 30.0       # neatline stripe width, pixels -- manually tune this
                          # (and LABEL_FONTSIZE below) to taste; both are also
                          # main() keyword args for programmatic use
LAT_STEP = None
LON_STEP = None
GRATICULE_COLOR = "black"
GRATICULE_LINEWIDTH = 1.0
STRIPE_OUTLINE_LINEWIDTH = 1.0
LABEL_FONTSIZE = 18.0
LON_LABEL_GAP = 25.0      # clearance beyond BORDER_WIDTH for bottom/top (longitude) labels
LAT_LABEL_GAP = 70.0      # clearance beyond BORDER_WIDTH for left/right (latitude) labels
PAD_FRAC = 0.02
DPI = 250
# -----------------------------------------------------------------------------


def label_anchor_bounds(proj, lat_min, lat_max, lon_min, lon_max, lat_step, lon_step,
                         lon_label_offset, lat_label_offset):
    """Bounding box (source-image pixel coords) of every tick label's anchor
    point -- see add_frame_to_basemap_rotated_labels.py's identical helper for
    the full rationale."""
    interior = (0.5 * (lat_min + lat_max), 0.5 * (lon_min + lon_max))
    xs, ys = [], []
    for lon in nice_ticks(lon_min, lon_max, lon_step):
        x, y = proj.latlon_to_pixel(lat_min, lon)
        dx, dy = _outward_normal(proj, True, lat_min, lon, lon_min, lon_max, interior)
        xs.append(x + dx * lon_label_offset)
        ys.append(y + dy * lon_label_offset)
    for lat in nice_ticks(lat_min, lat_max, lat_step):
        x, y = proj.latlon_to_pixel(lat, lon_min)
        dx, dy = _outward_normal(proj, False, lon_min, lat, lat_min, lat_max, interior)
        xs.append(x + dx * lat_label_offset)
        ys.append(y + dy * lat_label_offset)
    return min(xs), max(xs), min(ys), max(ys)


class ShiftedProj:
    """Wraps proj so downstream drawing code (which only calls
    latlon_to_pixel) draws directly in the white-padded canvas's pixel
    space instead of the original image's."""
    def __init__(self, proj, dx, dy):
        self.proj = proj
        self.dx, self.dy = dx, dy

    def latlon_to_pixel(self, lat_deg, lon_deg):
        x, y = self.proj.latlon_to_pixel(lat_deg, lon_deg)
        return x + self.dx, y + self.dy


def main(input_file=INPUT_FILE, extents_json=EXTENTS_JSON, output_file=OUTPUT_FILE,
         border_width=BORDER_WIDTH, label_fontsize=LABEL_FONTSIZE,
         lon_label_gap=LON_LABEL_GAP, lat_label_gap=LAT_LABEL_GAP):
    lon_label_offset = border_width + lon_label_gap
    lat_label_offset = border_width + lat_label_gap

    with open(extents_json) as f:
        extents = json.load(f)

    with Image.open(input_file) as im:
        img_w, img_h = im.size
        img = np.asarray(im.convert("RGB"))

    resize_factor = extents["final_image_width"] / img_w
    proj = StereoProjection.from_extents_json(extents_json, resize_factor=resize_factor)

    lat_step = LAT_STEP or nice_step(LAT_MAX - LAT_MIN)
    lon_step = LON_STEP or nice_step(LON_MAX - LON_MIN)
    box_loop = box_boundary_loop(proj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

    corners_x, corners_y = box_loop[:, 0], box_loop[:, 1]
    label_x0, label_x1, label_y0, label_y1 = label_anchor_bounds(
        proj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_step, lon_step,
        lon_label_offset, lat_label_offset)
    # Text glyphs extend beyond their anchor point (ha/va="center") -- pad a
    # bit further so full label strings clear the canvas, not just their anchors.
    text_margin = label_fontsize * DPI / 72.0

    x0 = min(corners_x.min(), label_x0) - text_margin
    x1 = max(corners_x.max(), label_x1) + text_margin
    y0 = min(corners_y.min(), label_y0) - text_margin
    y1 = max(corners_y.max(), label_y1) + text_margin
    pad_x = (x1 - x0) * PAD_FRAC
    pad_y = (y1 - y0) * PAD_FRAC
    x0, x1 = x0 - pad_x, x1 + pad_x
    y0, y1 = y0 - pad_y, y1 + pad_y

    # How far the requested frame/label extent reaches beyond the source
    # image's own pixel bounds, on each of the four sides -- this is exactly
    # how much real white canvas needs to be added so nothing gets clipped.
    margin_left = max(0, int(np.ceil(-x0)))
    margin_right = max(0, int(np.ceil(x1 - img_w)))
    margin_top = max(0, int(np.ceil(-y0)))       # y0 is the smaller (top) value
    margin_bottom = max(0, int(np.ceil(y1 - img_h)))

    padded = np.full((img_h + margin_top + margin_bottom,
                       img_w + margin_left + margin_right, 3), 255, dtype=img.dtype)
    padded[margin_top:margin_top + img_h, margin_left:margin_left + img_w] = img
    pad_h, pad_w = padded.shape[:2]

    sproj = ShiftedProj(proj, margin_left, margin_top)
    box_loop = box_loop + [margin_left, margin_top]
    xlim = (x0 + margin_left, x1 + margin_left)
    ylim = (y1 + margin_top, y0 + margin_top)  # inverted: row 0 is top

    crop_w = xlim[1] - xlim[0]
    crop_h = abs(ylim[1] - ylim[0])
    fig = plt.figure(figsize=(crop_w / DPI, crop_h / DPI), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(padded, zorder=1, interpolation="none", extent=[0, pad_w, pad_h, 0])

    if SHOW_GRATICULE:
        draw_graticule(ax, sproj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                        lat_step, lon_step, GRATICULE_COLOR, GRATICULE_LINEWIDTH)
    draw_neatline(ax, sproj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                  lat_step, lon_step, border_width,
                  label_fontsize, lon_label_offset, lat_label_offset,
                  STRIPE_OUTLINE_LINEWIDTH, rotate_labels=True)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    draw_outside_mask(ax, box_loop, xlim, ylim)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_file, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {output_file} ({int(round(crop_w))}x{int(round(crop_h))} px, "
          f"margins L{margin_left}/R{margin_right}/T{margin_top}/B{margin_bottom})")


if __name__ == "__main__":
    main()
