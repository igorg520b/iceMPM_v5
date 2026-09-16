"""
Reproduces, in code, the manual fill-in process used on
rasterize_glo12_thickness.py's output: open it, overlay land_mask.png to
see which no-data pixels are actually open water (not land), and paint the
nearest real color over those gaps -- plus erase the white boundary "frame"
around the GLO12 data footprint first, since otherwise it would get treated
as real (near-maximum) thickness.

Two earlier, wrong approaches, kept here as a record of what didn't work
and why:

1. Filling gaps on GLO12's own coarse (~1/12 deg, here 60x502) lat/lon grid
   before reprojecting. At this latitude (~80N) a grid that looks square in
   degrees is nowhere near square on the ground (1 deg of longitude is only
   cos(latitude) as long as 1 deg of latitude), so plain nearest-neighbor
   fill on that grid extends real data far too aggressively east-west. This
   is fixable with a sampling= correction, but it's still the wrong layer
   to work at (see next point).

2. Nearest-filling every no-data destination pixel from its single nearest
   real-data pixel, with no distance limit, directly on the (correct,
   isotropic) destination grid. This still produced strange, "feathered"
   results -- verified numerically correct at the pixel level, not a bug --
   because GLO12's true data footprint here is a long, thin, *branching*
   shape (ice within a winding strait). Euclidean nearest-neighbor doesn't
   know the channel bends: two points can be close in a straight line while
   being on different arms of the channel (across a headland, say), and the
   boundary between their fill "zones" is exactly the feathered, branching
   crack pattern that showed up. A Photoshop brush never has this problem
   because it only ever samples color *locally* -- it never reaches across
   to a distant arm of the channel.

This script reproduces that local-only behavior with a real distance cap
(MAX_FILL_DISTANCE_KM below): a no-data pixel only gets filled if a
real-data pixel is within that ground distance; anything farther away is
left as before (black/THICKNESS_MIN), same as rasterize_glo12_thickness.py
already does for genuinely-out-of-reach areas. This also directly matches
the actual goal -- filling narrow channels GLO12's coarse grid didn't
resolve -- rather than extrapolating across the whole basin.

Steps:
1. Render the thickness value at every destination pixel via a *masked*
   bilinear blend of the 4 nearest GLO12 grid corners: invalid corners are
   excluded from the blend and the remaining weights renormalized, rather
   than either blending in an invalid corner's huge fill-sentinel value
   (the original white-frame bug) or discarding the pixel outright whenever
   any single corner is invalid (far too strict once tried -- GLO12's real
   data is sparse enough here that almost every destination pixel's
   footprint touches at least one invalid corner). A pixel only counts as
   having no data when *none* of its 4 corners are valid.
2. Load land_mask.png (black = water, white = land, same convention as
   preparer/data_preparer.cpp's ProcessMaskLayer) purely to *report* how
   many of the remaining no-data pixels are open water GLO12 simply has no
   coverage for (what you'd actually go looking for with the brush tool),
   vs. land.
3. Nearest-fill every no-data pixel within MAX_FILL_DISTANCE_KM of real
   data, via scipy.ndimage.distance_transform_edt on the destination
   image's own (isotropic) pixel grid, using PROJ_TRANSFORM_COEFFS's own
   meters-per-pixel scale to convert the cap to pixels. Pixels with no real
   data within range are left at the old black/THICKNESS_MIN convention.

Land pixels within range of real coastal data get filled the same as
open-water gaps (not forced to a hard 0), for the same resize-safety
reason as before: a sharp 0-vs-real-value discontinuity right at the
coastline would contaminate real coastal values if this PNG is later
resized. Land far from any real data (the interior of Ellesmere Island /
Greenland) stays black, same as always -- there's nothing local to extend
there anyway.

Output convention (same as rasterize_glo12_thickness.py): black (0) =
THICKNESS_MIN meters or still no data, white (255) = THICKNESS_MAX meters,
linearly interpolated.

No command-line arguments -- edit the constants below and run directly.
"""
import os
import time
from datetime import datetime, timedelta

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

Image.MAX_IMAGE_PIXELS = None

# --- Edit these as needed ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THICKNESS_NC_PATH = "/home/s2/Projects-CUDA/_netcdf_2025/glo12v4_kane_basin_ice_thickness.nc"
LAND_MASK_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "_input_data", "land_mask.png")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "thickness_data_2025-05-29_filled.png")

# Output image size -- must match footprint_mask.png / land_mask.png / ice_mask.png
IMAGE_WIDTH = 20000
IMAGE_HEIGHT = 12926

SIMULATION_START_DATE = "2025-05-29"  # nearest available GLO12 frame to this date is used

# Kane Basin stereographic projection -- matches PROJ_LAT_0/PROJ_LON_0/
# PROJ_TRANSFORM_COEFFS in _input_data/nares_2k/simulation.json, and
# kane_basin_projection.py's PROJECTION_CENTER_LAT/LON/EARTH_RADIUS_M.
# PROJ_TRANSFORM_COEFFS[0]/[4] also double as this image's meters-per-pixel
# scale (used below to convert MAX_FILL_DISTANCE_KM into pixels).
PROJ_LAT_0 = 80.0
PROJ_LON_0 = -69.0
PROJ_TRANSFORM_COEFFS = [49.86, 0.0, -392875.8347788442, 0.0, -49.85843583197958, 348669.83948254184]
EARTH_RADIUS_M = 6371000.0

THICKNESS_MIN = 0.8   # meters -> black (0)
THICKNESS_MAX = 3.0   # meters -> white (255)

LAND_MASK_THRESHOLD = 128  # red channel < this = water, matching ProcessMaskLayer's default

# How far a no-data pixel may reach for a nearby real value. Deliberately
# small -- this is meant to cover a narrow unresolved channel or the
# immediate coastal fringe, not extrapolate across the whole basin (see
# module docstring for why an unbounded fill doesn't work here).
MAX_FILL_DISTANCE_KM = 10.0

ROW_CHUNK = 500  # rows of output processed per batch, keeps memory bounded


def inverse_stereographic(gx, gy, lat0_deg, lon0_deg, R):
    """Spherical oblique stereographic inverse (Snyder, k0=1).
    gx, gy: projected coordinates in meters (numpy arrays).
    Returns (lat_deg, lon_deg) numpy arrays."""
    phi0 = np.radians(lat0_deg)
    lam0 = np.radians(lon0_deg)

    rho = np.hypot(gx, gy)
    c = 2.0 * np.arctan2(rho, 2.0 * R)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.arcsin(cos_c * np.sin(phi0) + np.where(rho > 0, gy * sin_c * np.cos(phi0) / rho, 0.0))
        lam = lam0 + np.arctan2(gx * sin_c, rho * np.cos(phi0) * cos_c - gy * np.sin(phi0) * sin_c)

    at_center = rho == 0.0
    phi = np.where(at_center, phi0, phi)
    lam = np.where(at_center, lam0, lam)

    return np.degrees(phi), np.degrees(lam)


def masked_bilinear_sample(field, valid, lat, lon, lat0, dlat, lon0, dlon):
    """Bilinear sample of field that excludes invalid corners from the
    blend entirely, renormalizing weights over whichever corners *are*
    valid, instead of either (a) blending in an invalid corner's raw value
    (the white-frame bug) or (b) requiring all 4 corners valid to accept
    any value at all (too strict -- see module docstring).

    A destination pixel is only reported as having no data (second return
    value) when *none* of its 4 corners are valid, or it falls outside the
    source grid entirely."""
    n_lat, n_lon = field.shape

    fy = (lat - lat0) / dlat
    fx = (lon - lon0) / dlon
    in_bounds = (fy >= 0) & (fy <= n_lat - 1) & (fx >= 0) & (fx <= n_lon - 1)

    y0 = np.clip(np.floor(fy).astype(np.int64), 0, n_lat - 2)
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, n_lon - 2)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = np.clip(fy - y0, 0.0, 1.0)
    wx = np.clip(fx - x0, 0.0, 1.0)

    w00, w01, w10, w11 = (1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx
    v00, v01, v10, v11 = field[y0, x0], field[y0, x1], field[y1, x0], field[y1, x1]
    m00, m01, m10, m11 = valid[y0, x0], valid[y0, x1], valid[y1, x0], valid[y1, x1]

    weight_sum = w00 * m00 + w01 * m01 + w10 * m10 + w11 * m11
    value_sum = w00 * v00 * m00 + w01 * v01 * m01 + w10 * v10 * m10 + w11 * v11 * m11

    has_data = (weight_sum > 0) & in_bounds
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(has_data, value_sum / np.where(weight_sum > 0, weight_sum, 1.0), 0.0)

    return value, ~has_data


def nearest_time_index(times_hours, target_date):
    base = datetime(1950, 1, 1)
    target_hours = (target_date - base).total_seconds() / 3600.0
    idx = int(np.argmin(np.abs(times_hours - target_hours)))
    return idx


def load_land_mask(path, width, height):
    """Loads land_mask.png (black = water, white = land, red channel,
    same convention as preparer/data_preparer.cpp's ProcessMaskLayer), and
    resizes with nearest-neighbor if its resolution doesn't exactly match
    ours -- nearest, not bilinear, so the land/water boundary stays crisp
    rather than growing a gray fringe."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        print(f"land_mask.png is {img.size}, resizing (nearest) to {(width, height)}")
        img = img.resize((width, height), Image.Resampling.NEAREST)
    red = np.array(img)[:, :, 0]
    return red < LAND_MASK_THRESHOLD  # True = water


def main():
    t_start = time.time()
    target_date = datetime.strptime(SIMULATION_START_DATE, "%Y-%m-%d")
    dst_w, dst_h = IMAGE_WIDTH, IMAGE_HEIGHT
    a, b, c, d, e, ff = PROJ_TRANSFORM_COEFFS

    with h5py.File(THICKNESS_NC_PATH, "r") as nc:
        lat_grid = nc["latitude"][:].astype(np.float64)
        lon_grid = nc["longitude"][:].astype(np.float64)
        times_hours = nc["time"][:].astype(np.float64)
        t_idx = nearest_time_index(times_hours, target_date)
        actual_date = datetime(1950, 1, 1) + timedelta(hours=float(times_hours[t_idx]))
        print(f"Requested date {target_date.date()} -> nearest available frame {actual_date} (index {t_idx})")

        # Same packed-int16-vs-unpacked-float32 handling as
        # rasterize_glo12_thickness.py -- see that script's comment for the
        # full explanation. Defaults to a no-op for unpacked files.
        sithick_var = nc["sithick"]
        raw = sithick_var[t_idx].astype(np.float64)
        scale_factor = float(sithick_var.attrs["scale_factor"][0]) if "scale_factor" in sithick_var.attrs else 1.0
        add_offset = float(sithick_var.attrs["add_offset"][0]) if "add_offset" in sithick_var.attrs else 0.0
        if "_FillValue" in sithick_var.attrs:
            src_invalid = (raw == sithick_var.attrs["_FillValue"][0])
        else:
            src_invalid = np.zeros_like(raw, dtype=bool)
        src_invalid |= ~np.isfinite(raw)
        src_valid = ~src_invalid
        sithick = raw * scale_factor + add_offset
        sithick[src_invalid] = 0.0  # value is irrelevant where invalid; excluded from the blend below regardless

    lat0_grid, dlat = float(lat_grid[0]), float(lat_grid[1] - lat_grid[0])
    lon0_grid, dlon = float(lon_grid[0]), float(lon_grid[1] - lon_grid[0])

    print("Step 1: rendering thickness + validity onto the destination grid (masked bilinear, invalid corners excluded)...")
    dest_value = np.zeros((dst_h, dst_w), dtype=np.float32)
    dest_invalid = np.zeros((dst_h, dst_w), dtype=bool)

    px = np.arange(dst_w, dtype=np.float64) + 0.5
    for y0 in range(0, dst_h, ROW_CHUNK):
        y1 = min(y0 + ROW_CHUNK, dst_h)
        py = (np.arange(y0, y1, dtype=np.float64) + 0.5)[:, None]
        pxg = px[None, :]

        gx = a * pxg + b * py + c
        gy = d * pxg + e * py + ff

        lat, lon = inverse_stereographic(gx, gy, PROJ_LAT_0, PROJ_LON_0, EARTH_RADIUS_M)
        value, invalid = masked_bilinear_sample(sithick, src_valid, lat, lon, lat0_grid, dlat, lon0_grid, dlon)

        dest_value[y0:y1, :] = value.astype(np.float32)
        dest_invalid[y0:y1, :] = invalid

    n_invalid = int(dest_invalid.sum())
    print(f"  {dest_invalid.size} pixels, {n_invalid} ({100.0 * n_invalid / dest_invalid.size:.1f}%) with no real GLO12 data")

    print("Step 2: loading land_mask.png (reporting only)...")
    is_water = load_land_mask(LAND_MASK_PATH, dst_w, dst_h)
    missing_over_water = dest_invalid & is_water
    print(f"  {int(missing_over_water.sum())} pixels are open water with no GLO12 data")

    print(f"Step 3: nearest-fill no-data pixels within {MAX_FILL_DISTANCE_KM} km of real data...")
    t_fill = time.time()
    meters_per_px = abs(a)
    max_fill_px = MAX_FILL_DISTANCE_KM * 1000.0 / meters_per_px
    print(f"  ({meters_per_px:.2f} m/px -> {max_fill_px:.1f} px radius)")
    # Isotropic destination pixel grid, unlike GLO12's own lat/lon grid, so
    # plain index distance is already proportional to real ground distance.
    dist, nearest_idx = distance_transform_edt(dest_invalid, return_distances=True, return_indices=True)
    filled_value = dest_value[tuple(nearest_idx)]
    do_fill = dest_invalid & (dist <= max_fill_px)
    final_value = np.where(do_fill, filled_value, dest_value)
    still_missing_over_water = missing_over_water & ~do_fill
    print(f"  filled {int(do_fill.sum())} of {int(n_invalid)} no-data pixels "
          f"({int((missing_over_water & do_fill).sum())} of them open water); "
          f"{int(still_missing_over_water.sum())} open-water pixels remain out of range, left black")
    print(f"  done in {time.time() - t_fill:.1f}s")

    tmin, tmax = THICKNESS_MIN, THICKNESS_MAX
    frac = np.clip((final_value - tmin) / (tmax - tmin), 0.0, 1.0)
    out = np.round(frac * 255.0).astype(np.uint8)
    # Pixels that stayed invalid (out of fill range) keep the old
    # black/no-data convention, same as rasterize_glo12_thickness.py.
    out = np.where(dest_invalid & ~do_fill, 0, out).astype(np.uint8)

    Image.fromarray(out, mode="L").save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH} ({dst_w}x{dst_h}, thickness range [{tmin}, {tmax}] m -> [0, 255])")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
