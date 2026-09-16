#!/usr/bin/env python3
"""
Adds the striped neatline frame to the Kennedy Channel close-up basemap
(make_basemap_opentopo.py's current extent), for locating the named
islands (Franklin Island, Crozier Island, Ellesmere Island coast) per
the supervisor comment asking where they are.

This is a copy of add_frame_to_basemap_rotated_labels.py pointed at that
close-up PNG, not a new mechanism -- that script's own LAT_MIN/MAX,
LON_MIN/MAX are already independent of what the source PNG covers, as
long as the requested box sits inside it (see its docstring). So to
TRIM the frame smaller than the full downloaded image, just edit
LAT_MIN/LAT_MAX/LON_MIN/LON_MAX below to a tighter box -- the crop
follows the frame extent, not the source PNG's extent. The source PNG
was fetched with padding beyond its own nominal box (basemap_common.py's
pad_frac=0.15, corner_fetch_margin_deg=0.5), so there is real room to
trim inward before you run out of actual image data; how much room
depends on that padding, not a number worth hard-coding here.

Kept as a separate script (rather than editing
add_frame_to_basemap_rotated_labels.py's own constants in place) so
that script still reproduces the paper's Fig. 1b overview map unchanged.

Usage:
    Edit LAT_MIN/LAT_MAX/LON_MIN/LON_MAX (and BORDER_WIDTH/LABEL_FONTSIZE
    etc. if the crop ends up a very different size) below, then:
        python3 add_frame_to_kennedy_channel.py
"""
import sys
from pathlib import Path as FilePath

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(FilePath(__file__).resolve().parent.parent / "graticule"))
from add_graticule_frame import (SimStereoProjection, draw_graticule, draw_neatline,
                                  draw_outside_mask, box_boundary_loop, nice_step,
                                  nice_ticks, _outward_normal)

Image.MAX_IMAGE_PIXELS = None

# Anchored to this script's own directory (not the caller's cwd), same
# reasoning as make_basemap_opentopo.py -- never lands outside
# _python_code/ (in particular, never in _paper/).
_HERE = FilePath(__file__).resolve().parent

# --- Edit these ---------------------------------------------------------------
INPUT_FILE = str(_HERE / "fig_kennedy_channel_basemap_opentopo.png")
GEOREF_FILE = None    # None = INPUT_FILE with .georef.json instead of .png
OUTPUT_FILE = str(_HERE / "fig_kennedy_channel_basemap_opentopo_framed.png")

# Frame extent -- trim this to a box SMALLER than what the source PNG
# covers (currently the make_basemap_opentopo.py fetch box plus its own
# padding) to crop in further. Starts equal to that fetch box; narrow it
# from here.
LAT_MIN, LAT_MAX = 80.43, 80.94
LON_MIN, LON_MAX = -68.2, -65.8

SHOW_GRATICULE = False
BORDER_WIDTH = 55.0       # neatline stripe width, pixels -- scaled for a ~3000px-wide image
LABEL_FONTSIZE = 30.0
LAT_STEP = None  # auto -- latitude tick density is fine as-is
# Auto longitude step gave too many labels along the bottom; doubling it
# halves the tick count. Latitude is left alone (LAT_STEP above).
LON_STEP = 2 * nice_step(LON_MAX - LON_MIN)
GRATICULE_COLOR = "black"
GRATICULE_LINEWIDTH = 1.0
STRIPE_OUTLINE_LINEWIDTH = 1.0
LON_LABEL_GAP = 25.0
LAT_LABEL_GAP = 110.0  # bumped from 70 -- latitude labels were getting cut off on the left
PAD_FRAC = 0.02
DPI = 250
# -----------------------------------------------------------------------------

LON_LABEL_OFFSET = BORDER_WIDTH + LON_LABEL_GAP
LAT_LABEL_OFFSET = BORDER_WIDTH + LAT_LABEL_GAP


def label_anchor_bounds(proj, lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    """Bounding box (source-image pixel coords) of every tick label's anchor
    point -- see add_frame_to_basemap_rotated_labels.py's identical helper
    for the full explanation."""
    interior = (0.5 * (lat_min + lat_max), 0.5 * (lon_min + lon_max))
    xs, ys = [], []
    for lon in nice_ticks(lon_min, lon_max, lon_step):
        x, y = proj.latlon_to_pixel(lat_min, lon)
        dx, dy = _outward_normal(proj, True, lat_min, lon, lon_min, lon_max, interior)
        xs.append(x + dx * LON_LABEL_OFFSET)
        ys.append(y + dy * LON_LABEL_OFFSET)
    for lat in nice_ticks(lat_min, lat_max, lat_step):
        x, y = proj.latlon_to_pixel(lat, lon_min)
        dx, dy = _outward_normal(proj, False, lon_min, lat, lat_min, lat_max, interior)
        xs.append(x + dx * LAT_LABEL_OFFSET)
        ys.append(y + dy * LAT_LABEL_OFFSET)
    return min(xs), max(xs), min(ys), max(ys)


def main():
    georef_path = FilePath(GEOREF_FILE) if GEOREF_FILE else FilePath(INPUT_FILE).with_suffix(".georef.json")
    proj = SimStereoProjection.from_simulation_json(str(georef_path))

    with Image.open(INPUT_FILE) as im:
        img_w, img_h = im.size
        img = np.asarray(im.convert("RGB"))

    lat_step = LAT_STEP or nice_step(LAT_MAX - LAT_MIN)
    lon_step = LON_STEP or nice_step(LON_MAX - LON_MIN)
    box_loop = box_boundary_loop(proj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

    corners_x, corners_y = box_loop[:, 0], box_loop[:, 1]
    label_x0, label_x1, label_y0, label_y1 = label_anchor_bounds(
        proj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, lat_step, lon_step)
    # One glyph's width was not enough clearance for a full rotated label
    # string (e.g. "80.60N") -- widened to ~4 characters' worth.
    text_margin = LABEL_FONTSIZE * DPI / 72.0 * 4

    x0 = min(corners_x.min(), label_x0) - text_margin
    x1 = max(corners_x.max(), label_x1) + text_margin
    y0 = min(corners_y.min(), label_y0) - text_margin
    y1 = max(corners_y.max(), label_y1) + text_margin
    pad_x = (x1 - x0) * PAD_FRAC
    pad_y = (y1 - y0) * PAD_FRAC
    xlim = (max(0, x0 - pad_x), min(img_w, x1 + pad_x))
    ylim = (min(img_h, y1 + pad_y), max(0, y0 - pad_y))  # inverted: row 0 is top

    if xlim[0] <= 0 or xlim[1] >= img_w or ylim[1] <= 0 or ylim[0] >= img_h:
        print("WARNING: requested frame extent (plus labels/padding) reaches the "
              "source image's edge -- it may be clipping into blank/missing data. "
              "Trim LAT_MIN/LAT_MAX/LON_MIN/LON_MAX further inward if so.")

    crop_w = xlim[1] - xlim[0]
    crop_h = abs(ylim[1] - ylim[0])
    fig = plt.figure(figsize=(crop_w / DPI, crop_h / DPI), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(img, zorder=1, interpolation="none", extent=[0, img_w, img_h, 0])

    if SHOW_GRATICULE:
        draw_graticule(ax, proj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                        lat_step, lon_step, GRATICULE_COLOR, GRATICULE_LINEWIDTH)
    draw_neatline(ax, proj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                  lat_step, lon_step, BORDER_WIDTH,
                  LABEL_FONTSIZE, LON_LABEL_OFFSET, LAT_LABEL_OFFSET,
                  STRIPE_OUTLINE_LINEWIDTH, rotate_labels=True)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    draw_outside_mask(ax, box_loop, xlim, ylim)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUTPUT_FILE, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {OUTPUT_FILE} ({int(round(crop_w))}x{int(round(crop_h))} px)")


if __name__ == "__main__":
    main()
