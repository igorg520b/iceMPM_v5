#!/usr/bin/env python3
"""
Variant of add_frame_to_basemap.py (this directory) with tick labels rotated
to follow the local tangent of the frame curve, matching Fig. 1b's own style
in Spatial_continuity_of_measured_seawater_and_tracer.pdf (the "84N", "82N"
etc. labels there tilt to track the border rather than staying horizontal).

Kept as a separate script rather than folding into add_frame_to_basemap.py
since that one already works well -- the rotation is opt-in on
add_graticule_frame.py's draw_neatline (rotate_labels=True), so the
existing script's own behavior/output is completely unaffected by this file.

Otherwise identical to add_frame_to_basemap.py -- same extent/style knobs,
same .georef.json sidecar loading, same label-anchor-bounds crop sizing.
See that script's docstring for the general usage notes (editing extents,
widening beyond the source PNG's coverage, etc.).

Usage:
    Edit INPUT_FILE/OUTPUT_FILE and the extent/style constants below, then:
        python3 add_frame_to_basemap_rotated_labels.py
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

# --- Edit these ---------------------------------------------------------------
INPUT_FILE = "fig1_nares_basemap_opentopo.png"     # or fig1_nares_basemap_osm.png
GEOREF_FILE = None    # None = INPUT_FILE with .georef.json instead of .png (the sidecar
                       # make_basemap_osm.py/make_basemap_opentopo.py already wrote)
OUTPUT_FILE = "fig1_nares_basemap_opentopo_framed_rotated.png"

LAT_MIN, LAT_MAX = 77.60, 84.00   # frame extent -- independent of what the source PNG covers,
LON_MIN, LON_MAX = -85.0, -50.0   # as long as it's within that (currently matches Fig. 1b)

SHOW_GRATICULE = False
BORDER_WIDTH = 30.0       # neatline stripe width, pixels -- scaled for a ~3000px-wide image
                          # (add_graticule_frame.py's own defaults assume a much bigger source)
LAT_STEP = None
LON_STEP = None
GRATICULE_COLOR = "black"
GRATICULE_LINEWIDTH = 1.0
STRIPE_OUTLINE_LINEWIDTH = 1.0
LABEL_FONTSIZE = 18.0     # also scaled down from add_graticule_frame.py's 41 -- see that
                          # script's own module docstring
LON_LABEL_GAP = 25.0      # clearance beyond BORDER_WIDTH for bottom/top (longitude) labels
LAT_LABEL_GAP = 70.0      # clearance beyond BORDER_WIDTH for left/right (latitude) labels --
                          # bigger than LON_LABEL_GAP since those labels are wider strings
PAD_FRAC = 0.02           # extra aesthetic breathing room beyond the label bounding box,
                          # as a fraction of box size (small -- the label-bbox margin below
                          # already does the real work of not clipping anything)
DPI = 250
# -----------------------------------------------------------------------------

LON_LABEL_OFFSET = BORDER_WIDTH + LON_LABEL_GAP
LAT_LABEL_OFFSET = BORDER_WIDTH + LAT_LABEL_GAP


def label_anchor_bounds(proj, lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    """Bounding box (source-image pixel coords) of every tick label's anchor
    point, using the exact same interior-relative outward-normal logic
    draw_neatline itself uses -- so the crop is sized from where the labels
    actually land, not a guessed padding constant that has to be re-tuned
    whenever the extent, font size, or label-offset changes. Rotation
    doesn't move the anchor, so this bound is unaffected by rotate_labels;
    the text_margin added in main() below is generous enough to also cover
    a rotated string's slightly different bounding box."""
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
    # Text glyphs extend beyond their anchor point (ha/va="center") -- pad a
    # bit further so full label strings clear the crop, not just their anchors.
    text_margin = LABEL_FONTSIZE * DPI / 72.0

    x0 = min(corners_x.min(), label_x0) - text_margin
    x1 = max(corners_x.max(), label_x1) + text_margin
    y0 = min(corners_y.min(), label_y0) - text_margin
    y1 = max(corners_y.max(), label_y1) + text_margin
    pad_x = (x1 - x0) * PAD_FRAC
    pad_y = (y1 - y0) * PAD_FRAC
    xlim = (max(0, x0 - pad_x), min(img_w, x1 + pad_x))
    ylim = (min(img_h, y1 + pad_y), max(0, y0 - pad_y))  # inverted: row 0 is top

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
