#!/usr/bin/env python3
"""
Variant of add_frame_to_basemap_rotated_labels.py (this directory) that also
draws a box connecting four named capes marking the southern and northern
ends of Nares Strait -- Cape Lawrence/Cape Jackson (south) and Cape
Baird/Cape Morton (north), coordinates as given by the user (converted from
degrees-minutes to decimal degrees below). No text labels are drawn at the
corners -- per the user's own plan to add labels manually afterward.

Kept as a separate script (duplicating the small render pipeline, same
pattern add_frame_to_basemap_rotated_labels.py itself followed) rather than
adding a flag to the already-working scripts, so their own output stays
completely unaffected.

Usage:
    Edit INPUT_FILE/OUTPUT_FILE and the extent/style constants below, then:
        python3 add_capes_box_overlay.py
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
INPUT_FILE = "fig1_nares_basemap_osm.png"     # or fig1_nares_basemap_opentopo.png
GEOREF_FILE = None    # None = INPUT_FILE with .georef.json instead of .png (the sidecar
                       # make_basemap_osm.py/make_basemap_opentopo.py already wrote)
OUTPUT_FILE = "fig1_nares_basemap_osm_capes_box.png"

LAT_MIN, LAT_MAX = 77.60, 84.00   # frame extent -- independent of what the source PNG covers,
LON_MIN, LON_MAX = -90.0, -50.0   # as long as it's within that (currently matches Fig. 1b)

# Nares Strait endpoint capes, degrees-minutes -> decimal degrees, in polygon
# order (south-west -> north-west -> north-east -> south-east -> closes back
# to the first point): west/Ellesmere side capes first, east/Greenland side
# second, so the drawn box doesn't self-intersect.
CAPES_DM = [
    ("Cape Lawrence", (80, 25.2), (69, 13.2)),   # south, Ellesmere Island
    ("Cape Baird",    (81, 22.2), (64, 25.8)),   # north, Ellesmere Island
    ("Cape Morton",   (81, 12.0), (62, 48.0)),   # north, Greenland
    ("Cape Jackson",  (80,  9.6), (67,  0.0)),   # south, Greenland
]
BOX_COLOR = "red"
BOX_LINEWIDTH = 2.5
BOX_MARKER_SIZE = 8.0

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


def dm_to_decimal(deg, minutes):
    return deg + minutes / 60.0


def label_anchor_bounds(proj, lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    """Same as in add_frame_to_basemap_rotated_labels.py -- see there for the
    full rationale. Duplicated here rather than imported since these scripts
    are each meant to stand alone."""
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


def draw_capes_box(ax, proj):
    pts = []
    for name, (lat_d, lat_m), (lon_d, lon_m) in CAPES_DM:
        lat = dm_to_decimal(lat_d, lat_m)
        lon = -dm_to_decimal(lon_d, lon_m)  # all four capes are west longitudes
        pts.append(proj.latlon_to_pixel(lat, lon))
    pts.append(pts[0])  # close the loop
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=BOX_COLOR, linewidth=BOX_LINEWIDTH, zorder=4,
            marker="o", markersize=BOX_MARKER_SIZE, solid_capstyle="round")


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
    draw_capes_box(ax, proj)
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
