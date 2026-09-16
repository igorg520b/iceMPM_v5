#!/usr/bin/env python3
"""
Overlay a lat/lon graticule and a striped cartographic neatline border onto a
full-resolution raster exported from the 'visualizer' tool (Tools > Render
Full Grid Snapshot).

The exported raster has row 0 at the top and is pixel-for-pixel aligned with
the projection parameters (PROJ_LAT_0, PROJ_LON_0, PROJ_RESIZE_FACTOR,
PROJ_TRANSFORM_COEFFS) stored in the run's simulation.json -- the same
projection math used internally by simulation/projection.cpp's
Projection::ProjectPixel/InverseProjectPixel. As of the switch to a general/
oblique STEREOGRAPHIC projection (Snyder, spherical, k0=1, tangent at
PROJ_LAT_0/PROJ_LON_0 -- see projection.h), SimStereoProjection below
reimplements that same formula (no pyproj/cartopy dependency needed), not
the older orthographic one. simulation.json's ParentConfig chain (e.g.
nares_20k_06/simulation.json inheriting PROJ_* from base_config.json) is
followed automatically, matching config_loader.cpp.

Alternatively, set EXTENTS_JSON to a Kane Basin stereographic run's
extents.json (produced by image_download/kane_basin_stereographic.py) to
overlay a graticule on that dataset instead; pixel <-> lat/lon registration
then comes from kane_basin_projection.StereoProjection (pyproj + the affine
transform saved in extents.json) -- a different, independently-maintained
stereographic implementation for that separate dataset pipeline, which
happens to be the same projection family (see projection.h's docstring).

Usage:
    Everything is a plain constant below -- edit whatever you need and run:
        python3 add_graticule_frame.py

Only needed for a handful of one-off paper figures -- not wired into the
C++/Qt GUI on purpose, see project discussion.
"""
import json
import math
import sys
from pathlib import Path as FilePath

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib import patheffects

Image.MAX_IMAGE_PIXELS = None  # source rasters are legitimately huge (e.g. 20000x12927)

# --- Defaults, edit as needed -----------------------------------------------
IMAGE_FILE = "../../_input_data/nares_20k_06/color_06_20k.png"
SIM_JSON = "../../_input_data/nares_20k_06/simulation.json"  # ParentConfig chain resolved automatically
EXTENTS_JSON = None       # path to a kane_basin_stereographic.py run's extents.json (stereographic,
                          # Kane Basin). Takes priority over SIM_JSON/DEFAULT_PROJ when set.
OUTPUT_FILE = "nares_intro_graticule.png"
#LAT_MIN, LAT_MAX = 77.60, 81.80
#LON_MIN, LON_MAX = -78.9, -62.6
LAT_MIN, LAT_MAX = 77.60, 84.00
LON_MIN, LON_MAX = -75.9, -55.0
BORDER_WIDTH = 90.0      # neatline stripe width, pixels
LAT_STEP = None          # degrees between gridlines/stripe ticks; None = auto (e.g. 3.0 to match GMT-style figures)
LON_STEP = None          # degrees; None = auto (e.g. 5.0)

SHOW_GRATICULE = False    # gridlines are optional; the striped neatline frame is always drawn

GRATICULE_COLOR = "black"
GRATICULE_LINEWIDTH = 1.0
STRIPE_OUTLINE_LINEWIDTH = 1.0  # contour line drawn around each B/W neatline stripe
LABEL_FONTSIZE = 41.0
PAD_FRAC_LEFT = 0.20     # extra crop margin, as a fraction of box width/height, one knob per side
PAD_FRAC_RIGHT = 0.10
PAD_FRAC_TOP = 0.005
PAD_FRAC_BOTTOM = 0.03
DPI = 250                # controls text/line crispness only -- output pixel size always
                          # matches the source raster's native resolution for the cropped region

# How far tick labels sit beyond the box edge (pixels). Must clear
# BORDER_WIDTH or labels land on top of the striped neatline. Increase
# LON_LABEL_OFFSET to push bottom/top (longitude) labels further down/up;
# increase LAT_LABEL_OFFSET to push left/right (latitude) labels further out.
LON_LABEL_OFFSET = BORDER_WIDTH + 120.0
LAT_LABEL_OFFSET = BORDER_WIDTH - 310.0
# -----------------------------------------------------------------------------

R_EARTH = 6371000.0  # matches Projection::R (sphere, not WGS84 ellipsoid)

# Fallback if SIM_JSON is also None -- the nares_17k dataset's old orthographic
# run (kept only for backward compatibility with older figures made from it).
DEFAULT_PROJ = dict(
    lat0=79.87,
    lon0=-64.37,
    resize_factor=5.0,
    coeffs=[11.796660738708361, 0.0, -499251.71124660445,
            0.0, -11.796660738708361, 336065.7037738611],
)


def load_merged_config(path):
    """Recursively resolves a simulation.json's ParentConfig chain, child
    keys overriding parent keys -- mirrors config_loader.cpp's
    LoadMergedConfigDocument exactly (including relative-to-child-file
    path resolution)."""
    path = FilePath(path).resolve()
    with open(path) as f:
        doc = json.load(f)
    if "ParentConfig" not in doc:
        return doc
    parent_path = (path.parent / doc["ParentConfig"]).resolve()
    merged = load_merged_config(parent_path)
    merged.update(doc)
    return merged


class SimStereoProjection:
    """Python port of Projection::ProjectPixel/InverseProjectPixel
    (simulation/projection.cpp) -- general/oblique stereographic, Snyder's
    spherical formula, k0=1. Pixel coords are top-left origin, (x=column,
    y=row), matching the exported raster exactly."""

    def __init__(self, lat0, lon0, resize_factor, coeffs, R=R_EARTH):
        self.phi0 = math.radians(lat0)
        self.lam0 = math.radians(lon0)
        self.R = R
        self.resize = resize_factor
        self.a, self.b, self.c, self.d, self.e, self.f = coeffs

    @classmethod
    def from_simulation_json(cls, path):
        doc = load_merged_config(path)
        return cls(doc["PROJ_LAT_0"], doc["PROJ_LON_0"],
                    doc["PROJ_RESIZE_FACTOR"], doc["PROJ_TRANSFORM_COEFFS"])

    def pixel_to_latlon(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        px, py = x * self.resize, y * self.resize
        x_geo = self.a * px + self.b * py + self.c
        y_geo = self.d * px + self.e * py + self.f
        rho = np.hypot(x_geo, y_geo)

        with np.errstate(invalid="ignore", divide="ignore"):
            c = 2.0 * np.arctan2(rho, 2.0 * self.R)
            cos_c, sin_c = np.cos(c), np.sin(c)
            phi = np.arcsin(cos_c * math.sin(self.phi0) +
                             (y_geo * sin_c * math.cos(self.phi0)) / rho)
            lam = self.lam0 + np.arctan2(
                x_geo * sin_c,
                rho * math.cos(self.phi0) * cos_c - y_geo * math.sin(self.phi0) * sin_c)

        zero_mask = rho == 0
        phi = np.where(zero_mask, self.phi0, phi)
        lam = np.where(zero_mask, self.lam0, lam)
        return np.degrees(phi), np.degrees(lam)

    def latlon_to_pixel(self, lat_deg, lon_deg):
        phi = np.radians(np.asarray(lat_deg, dtype=np.float64))
        lam = np.radians(np.asarray(lon_deg, dtype=np.float64))
        cos_c = math.sin(self.phi0) * np.sin(phi) + math.cos(self.phi0) * np.cos(phi) * np.cos(lam - self.lam0)
        k = 2.0 / (1.0 + cos_c)

        x_geo = self.R * k * np.cos(phi) * np.sin(lam - self.lam0)
        y_geo = self.R * k * (math.cos(self.phi0) * np.sin(phi) -
                              math.sin(self.phi0) * np.cos(phi) * np.cos(lam - self.lam0))
        x_geo = x_geo - self.c
        y_geo = y_geo - self.f

        det = self.a * self.e - self.b * self.d
        px = (self.e * x_geo - self.b * y_geo) / det
        py = (-self.d * x_geo + self.a * y_geo) / det
        return px / self.resize, py / self.resize


def nice_step(span, target_lines=6):
    """Pick a 'round' degree step giving roughly target_lines gridlines
    across span (degrees). Candidates go down to 1 arcminute."""
    candidates = [1/60, 2/60, 5/60, 10/60, 15/60, 20/60, 30/60,
                  1, 2, 3, 4, 5, 10, 15, 20, 30]
    raw = span / target_lines
    best = min(candidates, key=lambda c: abs(c - raw))
    return best


def nice_ticks(lo, hi, step):
    """Gridline/tick positions snapped to round multiples of step (e.g. step=3
    gives ..., 75, 78, 81, 84, ... regardless of lo), matching GMT-style maps
    instead of ticks offset from an arbitrary lo."""
    start = math.ceil(lo / step - 1e-9) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def format_deg(value, step, pos_letter, neg_letter):
    letter = pos_letter if value >= 0 else neg_letter
    value = abs(value)
    if step < 1.0 - 1e-9:
        deg = int(math.floor(value + 1e-9))
        minutes = round((value - deg) * 60)
        if minutes == 60:
            deg += 1
            minutes = 0
        return f"{deg}°{minutes:02d}'{letter}" if deg else f"{minutes}'{letter}"
    return f"{value:g}°{letter}"


def sample_curve(proj, fixed_is_lat, fixed_val, var_lo, var_hi, n=400):
    t = np.linspace(var_lo, var_hi, n)
    if fixed_is_lat:
        x, y = proj.latlon_to_pixel(np.full_like(t, fixed_val), t)
    else:
        x, y = proj.latlon_to_pixel(t, np.full_like(t, fixed_val))
    return np.column_stack([x, y])


def draw_graticule(ax, proj, lat_min, lat_max, lon_min, lon_max, lat_step, lon_step,
                    color="black", linewidth=0.35):
    for lat in nice_ticks(lat_min, lat_max, lat_step):
        pts = sample_curve(proj, True, lat, lon_min, lon_max)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, zorder=2)
    for lon in nice_ticks(lon_min, lon_max, lon_step):
        pts = sample_curve(proj, False, lon, lat_min, lat_max)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, zorder=2)


def offset_curve(curve_pts, offset):
    """Offset a polyline perpendicular to its local tangent by `offset`."""
    tangents = np.gradient(curve_pts, axis=0)
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    norms[norms == 0] = 1.0
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]]) / norms[:, None]
    return curve_pts + normals * offset


def polygon_signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def box_boundary_loop(proj, lat_min, lat_max, lon_min, lon_max, n=300):
    """Closed polyline tracing the requested lat/lon box's perimeter."""
    bottom = sample_curve(proj, True, lat_min, lon_min, lon_max, n)
    right = sample_curve(proj, False, lon_max, lat_min, lat_max, n)
    top = sample_curve(proj, True, lat_max, lon_max, lon_min, n)
    left = sample_curve(proj, False, lon_min, lat_max, lat_min, n)
    return np.vstack([bottom, right, top, left])


def draw_outside_mask(ax, box_loop, xlim, ylim):
    """Paint everything in the visible axes outside the box white, so only
    the requested region (plus the neatline drawn on top of it) is visible."""
    x0, x1 = xlim
    y0, y1 = ylim
    outer = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    hole = np.vstack([box_loop, box_loop[:1]])

    if polygon_signed_area(outer) * polygon_signed_area(hole) > 0:
        hole = hole[::-1]

    verts = np.vstack([outer, hole])
    codes = ([Path.MOVETO] + [Path.LINETO] * (len(outer) - 2) + [Path.CLOSEPOLY] +
              [Path.MOVETO] + [Path.LINETO] * (len(hole) - 2) + [Path.CLOSEPOLY])
    patch = PathPatch(Path(verts, codes), facecolor="white", edgecolor="none", zorder=3)
    ax.add_patch(patch)


def edge_breakpoints(lo, hi, step):
    """Major breakpoints along an edge: the box's own endpoints plus every
    round gridline tick in between, sorted -- so stripe boundaries can be
    anchored on the same positions as the graticule lines."""
    return sorted(set([round(lo, 10), round(hi, 10)] + nice_ticks(lo, hi, step)))


def draw_neatline(ax, proj, lat_min, lat_max, lon_min, lon_max,
                   lat_step, lon_step, border_width, label_fontsize,
                   lon_label_offset, lat_label_offset, stripe_outline_linewidth=0.3,
                   rotate_labels=False):
    edges = [
        dict(fixed_is_lat=True, fixed_val=lat_min, lo=lon_min, hi=lon_max, step=lon_step),
        dict(fixed_is_lat=True, fixed_val=lat_max, lo=lon_min, hi=lon_max, step=lon_step),
        dict(fixed_is_lat=False, fixed_val=lon_min, lo=lat_min, hi=lat_max, step=lat_step),
        dict(fixed_is_lat=False, fixed_val=lon_max, lo=lat_min, hi=lat_max, step=lat_step),
    ]

    # One solid-color stripe per gridline interval (including the ragged
    # first/last partial interval at each box edge), alternating black/white
    # -- stripe boundaries land exactly on the same ticks as the graticule,
    # with no extra mid-interval split.
    all_polys = []
    for e in edges:
        breakpoints = edge_breakpoints(e["lo"], e["hi"], e["step"])
        for idx, (b0, b1) in enumerate(zip(breakpoints[:-1], breakpoints[1:])):
            seg = sample_curve(proj, e["fixed_is_lat"], e["fixed_val"], b0, b1, n=60)
            outer = offset_curve(seg, border_width)
            quad = np.vstack([outer, seg[::-1]])
            all_polys.append((quad, idx % 2 == 0))

    patches, colors = [], []
    for quad, is_black in all_polys:
        patches.append(Polygon(quad, closed=True))
        colors.append("black" if is_black else "white")
    pc = PatchCollection(patches, facecolor=colors, edgecolor="black",
                          linewidth=stripe_outline_linewidth, zorder=5)
    ax.add_collection(pc)

    # Tick labels at major graticule intersections with the border. White
    # halo (not just a higher zorder) because near sharp corners the offset
    # curve's own miter can bulge past a label's nominal clearance -- a
    # black glyph sitting on a black stripe segment there would otherwise
    # just vanish, same color as what's behind it.
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    interior = (0.5 * (lat_min + lat_max), 0.5 * (lon_min + lon_max))
    for lon in nice_ticks(lon_min, lon_max, lon_step):
        x, y = proj.latlon_to_pixel(lat_min, lon)
        dx, dy = _outward_normal(proj, True, lat_min, lon, lon_min, lon_max, interior)
        angle = _label_rotation_deg(proj, True, lat_min, lon, lon_min, lon_max) if rotate_labels else 0.0
        ax.text(x + dx * lon_label_offset, y + dy * lon_label_offset,
                format_deg(lon, lon_step, "E", "W"),
                fontsize=label_fontsize, ha="center", va="center", zorder=6,
                rotation=angle, rotation_mode="anchor", path_effects=halo)
    for lat in nice_ticks(lat_min, lat_max, lat_step):
        x, y = proj.latlon_to_pixel(lat, lon_min)
        dx, dy = _outward_normal(proj, False, lon_min, lat, lat_min, lat_max, interior)
        angle = _label_rotation_deg(proj, False, lon_min, lat, lat_min, lat_max, extra_deg=-90.0) if rotate_labels else 0.0
        ax.text(x + dx * lat_label_offset, y + dy * lat_label_offset,
                format_deg(lat, lat_step, "N", "S"),
                fontsize=label_fontsize, ha="center", va="center", zorder=6,
                rotation=angle, rotation_mode="anchor", path_effects=halo)


def _outward_normal(proj, fixed_is_lat, fixed_val, at, lo, hi, interior):
    """Unit normal to the border curve at parameter `at`, pointing away from
    the map interior (used to offset tick labels outward). `interior` is a
    (lat, lon) point known to be inside the box -- rotating the tangent by a
    fixed +/-90 degrees only gives "away from interior" consistently for
    two of the four edges (which two depends on parameterization direction,
    not easily guessed), so instead we just check against a real interior
    point and flip if the naive rotation pointed the wrong way."""
    eps = (hi - lo) * 1e-4 or 1e-4
    p0 = np.array(proj.latlon_to_pixel(fixed_val, at - eps) if fixed_is_lat
                  else proj.latlon_to_pixel(at - eps, fixed_val))
    p1 = np.array(proj.latlon_to_pixel(fixed_val, at + eps) if fixed_is_lat
                  else proj.latlon_to_pixel(at + eps, fixed_val))
    tangent = p1 - p0
    norm = np.hypot(*tangent) or 1.0
    nx, ny = -tangent[1] / norm, tangent[0] / norm

    edge_pt = np.array(proj.latlon_to_pixel(fixed_val, at) if fixed_is_lat
                        else proj.latlon_to_pixel(at, fixed_val))
    interior_pt = np.array(proj.latlon_to_pixel(*interior))
    if np.dot((nx, ny), interior_pt - edge_pt) > 0:
        nx, ny = -nx, -ny
    return (nx, ny)


def _label_rotation_deg(proj, fixed_is_lat, fixed_val, at, lo, hi, extra_deg=0.0):
    """Text-rotation angle (degrees, matplotlib's on-screen convention) that
    makes a label follow the border curve's local tangent at `at`, matching
    Fig. 1b's style -- rather than sitting horizontal regardless of how the
    frame curves at that point. `extra_deg` is applied on top of that
    tangent angle (e.g. -90 for a label family that should run perpendicular
    to its edge instead of along it -- latitude labels along the near-vertical
    left/right edges read more easily rotated this way than tilted steeply
    parallel to the edge). Flipped 180 degrees whenever the result would
    otherwise render the string upside-down or right-to-left."""
    eps = (hi - lo) * 1e-4 or 1e-4
    p0 = np.array(proj.latlon_to_pixel(fixed_val, at - eps) if fixed_is_lat
                  else proj.latlon_to_pixel(at - eps, fixed_val))
    p1 = np.array(proj.latlon_to_pixel(fixed_val, at + eps) if fixed_is_lat
                  else proj.latlon_to_pixel(at + eps, fixed_val))
    tangent = p1 - p0
    # Flip y: image pixel rows increase downward, but matplotlib's text
    # rotation angle is measured in the usual math sense (CCW, y-up).
    angle = math.degrees(math.atan2(-tangent[1], tangent[0])) + extra_deg
    angle = (angle + 180) % 360 - 180  # normalize to (-180, 180]
    if angle > 90 or angle < -90:
        angle += 180
    return angle


def make_figure(image_file=IMAGE_FILE, output_file=OUTPUT_FILE):
    """Runs the full pipeline for one image_file -> output_file pair, using
    all the other module-level constants above (LAT_MIN, BORDER_WIDTH, etc.)
    as-is. Exposed as a function so a batch driver can loop over many
    input/output filenames without touching the command line."""
    if EXTENTS_JSON:
        # kane_basin_projection.py (pyproj-based) lives in _python_code/image_download,
        # a sibling of this script's own graticule/ directory -- imported lazily so the
        # common SIM_JSON codepath below doesn't need pyproj installed at all.
        sys.path.insert(0, str(FilePath(__file__).resolve().parent.parent / "image_download"))
        from kane_basin_projection import StereoProjection
        proj = StereoProjection.from_extents_json(EXTENTS_JSON)
    elif SIM_JSON:
        proj = SimStereoProjection.from_simulation_json(SIM_JSON)
    else:
        proj = SimStereoProjection(DEFAULT_PROJ["lat0"], DEFAULT_PROJ["lon0"],
                                   DEFAULT_PROJ["resize_factor"], DEFAULT_PROJ["coeffs"])

    lat_step = LAT_STEP or nice_step(LAT_MAX - LAT_MIN)
    lon_step = LON_STEP or nice_step(LON_MAX - LON_MIN)

    # Compute the crop bbox (in source-image pixel coords) up front, purely
    # from the projection -- no need to touch the image itself for this.
    box_loop = box_boundary_loop(proj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    corners_x, corners_y = box_loop[:, 0], box_loop[:, 1]
    x0, x1 = corners_x.min(), corners_x.max()
    y0, y1 = corners_y.min(), corners_y.max()
    # Left/right margins have to clear the latitude labels sitting on the left
    # edge; top/bottom margins have to clear the longitude labels on the
    # bottom edge -- so each starts from the relevant label offset, then adds
    # its own independent PAD_FRAC_* on top.
    pad_left = (x1 - x0) * PAD_FRAC_LEFT + LAT_LABEL_OFFSET + 60
    pad_right = (x1 - x0) * PAD_FRAC_RIGHT + LAT_LABEL_OFFSET + 60
    pad_top = (y1 - y0) * PAD_FRAC_TOP + LON_LABEL_OFFSET + 60
    pad_bottom = (y1 - y0) * PAD_FRAC_BOTTOM + LON_LABEL_OFFSET + 60
    xlim = (x0 - pad_left, x1 + pad_right)
    ylim = (y1 + pad_bottom, y0 - pad_top)  # inverted: row 0 is top

    # Crop with PIL BEFORE decoding into a matplotlib/numpy float array --
    # source rasters here are huge (e.g. 20000x12927), and plt.imread's
    # float32/64 conversion of the FULL image would multiply memory use
    # several-fold over just decoding the (much smaller) cropped region.
    with Image.open(image_file) as im:
        img_w, img_h = im.size
        crop_left = max(0, int(math.floor(xlim[0])))
        crop_right = min(img_w, int(math.ceil(xlim[1])))
        crop_upper = max(0, int(math.floor(ylim[1])))   # ylim[1] is the smaller (top) value
        crop_lower = min(img_h, int(math.ceil(ylim[0])))
        cropped = im.crop((crop_left, crop_upper, crop_right, crop_lower))
        img = np.asarray(cropped)

    # Shift all pixel-space geometry so it's relative to the crop's own
    # origin, matching where imshow will actually place img[0, 0].
    box_loop = box_loop - [crop_left, crop_upper]
    xlim = (xlim[0] - crop_left, xlim[1] - crop_left)
    ylim = (ylim[0] - crop_upper, ylim[1] - crop_upper)

    crop_w_px = xlim[1] - xlim[0]
    crop_h_px = abs(ylim[1] - ylim[0])
    fig = plt.figure(figsize=(crop_w_px / DPI, crop_h_px / DPI), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])  # fill the whole canvas, no margins
    ax.imshow(img, zorder=1, interpolation="none",
              extent=[0, cropped.size[0], cropped.size[1], 0])

    class ShiftedProj:
        """Wraps proj so sample_curve/draw_* (which only know about
        latlon_to_pixel) draw in the crop's shifted pixel space."""
        def latlon_to_pixel(self, lat_deg, lon_deg):
            x, y = proj.latlon_to_pixel(lat_deg, lon_deg)
            return x - crop_left, y - crop_upper

    sproj = ShiftedProj()

    if SHOW_GRATICULE:
        draw_graticule(ax, sproj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                        lat_step, lon_step, GRATICULE_COLOR, GRATICULE_LINEWIDTH)
    draw_neatline(ax, sproj, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                  lat_step, lon_step, BORDER_WIDTH,
                  LABEL_FONTSIZE, LON_LABEL_OFFSET, LAT_LABEL_OFFSET,
                  STRIPE_OUTLINE_LINEWIDTH)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Fill everything outside the requested box with white (page background)
    draw_outside_mask(ax, box_loop, xlim, ylim)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_file, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {output_file} ({int(round(crop_w_px))}x{int(round(crop_h_px))} px)")


def main():
    make_figure(IMAGE_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
