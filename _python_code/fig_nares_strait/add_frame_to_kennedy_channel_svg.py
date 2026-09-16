#!/usr/bin/env python3
"""
Experimental variant of add_frame_to_kennedy_channel.py producing a final
SVG that stays a *vector* frame/labels over a *small, JPEG-compressed*
basemap raster -- see add_frame_to_basemap_svg.py's docstring for the full
explanation (short version: draw_neatline's frame stripes and tick labels
are already real matplotlib vector artists; only the basemap ax.imshow
raster needs to be swapped down to a downscaled JPEG after the fact).

Also bumps LON_LABEL_GAP by roughly one font-height so the bottom-edge
longitude labels clear the striped neatline.

Writes to svg_test/ (this directory) rather than overwriting the existing
fig_kennedy_channel_basemap_opentopo_framed.png. Three files come out per
run: the plain .png, matplotlib's own .svg (large, PNG-embedded -- kept for
comparison), and _vector.svg (the deliverable).

Usage:
    python3 add_frame_to_kennedy_channel_svg.py
"""
import base64
import re
import sys
from io import BytesIO
from pathlib import Path as FilePath

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

sys.path.insert(0, str(FilePath(__file__).resolve().parent.parent / "graticule"))
from add_graticule_frame import (SimStereoProjection, draw_graticule, draw_neatline,
                                  draw_outside_mask, box_boundary_loop, nice_step,
                                  nice_ticks, _outward_normal)

Image.MAX_IMAGE_PIXELS = None

_HERE = FilePath(__file__).resolve().parent
_OUT_DIR = _HERE / "svg_test"

# --- Edit these ---------------------------------------------------------------
INPUT_FILE = str(_HERE / "fig_kennedy_channel_basemap_opentopo.png")
GEOREF_FILE = None
OUTPUT_STEM = str(_OUT_DIR / "fig_kennedy_channel_basemap_opentopo_framed")  # .png/.svg appended

LAT_MIN, LAT_MAX = 80.43, 80.94
LON_MIN, LON_MAX = -68.2, -65.8

SHOW_GRATICULE = False
BORDER_WIDTH = 55.0
LABEL_FONTSIZE = 30.0
LAT_STEP = None
LON_STEP = 2 * nice_step(LON_MAX - LON_MIN)
GRATICULE_COLOR = "black"
GRATICULE_LINEWIDTH = 1.0
STRIPE_OUTLINE_LINEWIDTH = 1.0
LON_LABEL_GAP = 25.0 + LABEL_FONTSIZE * 250.0 / 72.0  # + ~one font-height at DPI=250, to clear the frame
LAT_LABEL_GAP = 110.0
PAD_FRAC = 0.02
DPI = 250

RASTER_JPEG_WIDTH = 1200
RASTER_JPEG_QUALITY = 80
# -----------------------------------------------------------------------------

LON_LABEL_OFFSET = BORDER_WIDTH + LON_LABEL_GAP
LAT_LABEL_OFFSET = BORDER_WIDTH + LAT_LABEL_GAP


def mask_outside_white(img, polygon_pts):
    """See add_frame_to_basemap_svg.py's identical helper for the full
    explanation."""
    img = img.convert("RGB")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon_pts], fill=255)
    white = Image.new("RGB", img.size, (255, 255, 255))
    return Image.composite(img, white, mask)


def embed_downscaled_jpeg(svg_path, cropped_img, jpeg_width, quality, out_path):
    """See add_frame_to_basemap_svg.py's identical helper for the full
    explanation."""
    w, h = cropped_img.size
    new_h = round(h * jpeg_width / w)
    resized = cropped_img.convert("RGB").resize((jpeg_width, new_h), Image.LANCZOS)
    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    svg_text = FilePath(svg_path).read_text()
    new_text, n = re.subn(
        r'data:image/png;base64,\s*[A-Za-z0-9+/=\s]+?(?="\s+id=)',
        "data:image/jpeg;base64,\n" + b64,
        svg_text, count=1)
    if n != 1:
        raise RuntimeError(f"expected exactly one PNG <image> payload in {svg_path}, found {n}")
    FilePath(out_path).write_text(new_text)


def label_anchor_bounds(proj, lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
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
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    text_margin = LABEL_FONTSIZE * DPI / 72.0 * 4

    x0 = min(corners_x.min(), label_x0) - text_margin
    x1 = max(corners_x.max(), label_x1) + text_margin
    y0 = min(corners_y.min(), label_y0) - text_margin
    y1 = max(corners_y.max(), label_y1) + text_margin
    pad_x = (x1 - x0) * PAD_FRAC
    pad_y = (y1 - y0) * PAD_FRAC
    xlim = (max(0, x0 - pad_x), min(img_w, x1 + pad_x))
    ylim = (min(img_h, y1 + pad_y), max(0, y0 - pad_y))

    if xlim[0] <= 0 or xlim[1] >= img_w or ylim[1] <= 0 or ylim[0] >= img_h:
        print("WARNING: requested frame extent (plus labels/padding) reaches the "
              "source image's edge -- it may be clipping into blank/missing data.")

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
    fig.savefig(OUTPUT_STEM + ".png", dpi=DPI)
    fig.savefig(OUTPUT_STEM + ".svg")
    plt.close(fig)

    with Image.open(INPUT_FILE) as im_full:
        cropped = im_full.crop((round(xlim[0]), round(ylim[1]), round(xlim[1]), round(ylim[0])))
    box_loop_local = box_loop - [xlim[0], ylim[1]]
    cropped = mask_outside_white(cropped, box_loop_local)
    embed_downscaled_jpeg(OUTPUT_STEM + ".svg", cropped, RASTER_JPEG_WIDTH,
                          RASTER_JPEG_QUALITY, OUTPUT_STEM + "_vector.svg")

    print(f"Wrote {OUTPUT_STEM}.png, .svg, and _vector.svg ({int(round(crop_w))}x{int(round(crop_h))} px)")


if __name__ == "__main__":
    main()
