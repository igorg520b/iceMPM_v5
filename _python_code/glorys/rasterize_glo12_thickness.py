"""
Rasterizes GLO12 sea-ice thickness (sithick) into a grayscale PNG in the
Kane Basin stereographic image space, for use as an ImageThicknessMask
(see preparer/parameterparser.h ThicknessFrom/ThicknessTo).

Output convention matches ThicknessFrom/ThicknessTo: black (0) = THICKNESS_MIN
meters, white (255) = THICKNESS_MAX meters, linearly interpolated and clamped.
Cells with no GLO12 coverage (land / fill value) are written as black (0),
same as picking THICKNESS_MIN there -- consistent with the "display zero
where GLO12 has no data" convention used elsewhere in this project.

Output pixel grid follows the same pixel<->lat/lon convention as
kane_basin_projection.StereoProjection (top-left origin, pixel (x+0.5,y+0.5)
through the affine transform) and PROJ_TRANSFORM_COEFFS in simulation.json,
so it lines up exactly with footprint_mask.png, color_2k.png, land_mask.png,
ice_mask.png etc.

No pyproj/rasterio dependency (not installed in this environment) -- the
spherical oblique stereographic inverse (Snyder, k0=1) is implemented
directly, matching "+proj=stere +lat_0=.. +lon_0=.. +R=.. +k=1".

No command-line arguments -- edit the constants below and run directly
(e.g. from PyCharm's Run button).
"""
import os
from datetime import datetime, timedelta

import h5py
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --- Edit these as needed ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THICKNESS_NC_PATH = os.path.join(SCRIPT_DIR, "glo12v4_kane_basin_ice_thickness.nc")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "thickness_data_2025.png")

# Output image size -- must match footprint_mask.png / land_mask.png / ice_mask.png
IMAGE_WIDTH = 20000
IMAGE_HEIGHT = 12926

SIMULATION_START_DATE = "2025-05-30"  # nearest available GLO12 frame to this date is used

# Kane Basin stereographic projection -- matches PROJ_LAT_0/PROJ_LON_0/
# PROJ_TRANSFORM_COEFFS in _input_data/nares_2k/simulation.json, and
# kane_basin_projection.py's PROJECTION_CENTER_LAT/LON/EARTH_RADIUS_M.
PROJ_LAT_0 = 80.0
PROJ_LON_0 = -69.0
PROJ_TRANSFORM_COEFFS = [49.86, 0.0, -392875.8347788442, 0.0, -49.85843583197958, 348669.83948254184]
EARTH_RADIUS_M = 6371000.0

THICKNESS_MIN = 0.8   # meters -> black (0)
THICKNESS_MAX = 3.0   # meters -> white (255)

FILL_THRESHOLD = 1.0e30  # sithick _FillValue is ~9.97e36; anything this big is "no data"

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

    # rho == 0 maps exactly to the projection center
    at_center = rho == 0.0
    phi = np.where(at_center, phi0, phi)
    lam = np.where(at_center, lam0, lam)

    return np.degrees(phi), np.degrees(lam)


def bilinear_sample(field, lat, lon, lat0, dlat, lon0, dlon):
    """Bilinear sample of a regular-grid 2D field (lat-major, lon-minor) at
    arbitrary (lat, lon) points. Points outside the grid are clamped to the
    edge (their validity is decided separately by the caller)."""
    n_lat, n_lon = field.shape

    fy = (lat - lat0) / dlat
    fx = (lon - lon0) / dlon

    y0 = np.clip(np.floor(fy).astype(np.int64), 0, n_lat - 2)
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, n_lon - 2)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = np.clip(fy - y0, 0.0, 1.0)
    wx = np.clip(fx - x0, 0.0, 1.0)

    v00 = field[y0, x0]
    v01 = field[y0, x1]
    v10 = field[y1, x0]
    v11 = field[y1, x1]

    top = v00 * (1 - wx) + v01 * wx
    bot = v10 * (1 - wx) + v11 * wx
    value = top * (1 - wy) + bot * wy

    out_of_bounds = (fy < 0) | (fy > n_lat - 1) | (fx < 0) | (fx > n_lon - 1)
    return value, out_of_bounds


def nearest_time_index(times_hours, target_date):
    base = datetime(1950, 1, 1)
    target_hours = (target_date - base).total_seconds() / 3600.0
    idx = int(np.argmin(np.abs(times_hours - target_hours)))
    return idx


def main():
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

        # Some GLO12 products (e.g. the GLORYS12V1 "my" reanalysis) store
        # sithick as packed int16 (scale_factor/add_offset) with a small
        # integer _FillValue (e.g. -32767), instead of unpacked float32 with
        # a huge (~1e36) fill sentinel like the ANFC product this script was
        # originally written against. h5py reads raw HDF5 values and applies
        # neither convention automatically (that's a netCDF-library-level
        # decode, not an HDF5-level one), so both are handled explicitly
        # here: unpack via scale_factor/add_offset if present (a no-op 1.0/
        # 0.0 default otherwise), and remap this file's actual fill value to
        # the same huge-sentinel convention FILL_THRESHOLD already checks
        # below, so the rest of the pipeline doesn't need to change.
        sithick_var = nc["sithick"]
        raw = sithick_var[t_idx].astype(np.float64)
        scale_factor = float(sithick_var.attrs["scale_factor"][0]) if "scale_factor" in sithick_var.attrs else 1.0
        add_offset = float(sithick_var.attrs["add_offset"][0]) if "add_offset" in sithick_var.attrs else 0.0
        is_fill = (raw == sithick_var.attrs["_FillValue"][0]) if "_FillValue" in sithick_var.attrs else np.zeros_like(raw, dtype=bool)
        sithick = raw * scale_factor + add_offset
        sithick[is_fill] = FILL_THRESHOLD * 10.0

    lat0_grid, dlat = float(lat_grid[0]), float(lat_grid[1] - lat_grid[0])
    lon0_grid, dlon = float(lon_grid[0]), float(lon_grid[1] - lon_grid[0])

    out = np.zeros((dst_h, dst_w), dtype=np.uint8)
    tmin, tmax = THICKNESS_MIN, THICKNESS_MAX

    px = np.arange(dst_w, dtype=np.float64) + 0.5
    for y0 in range(0, dst_h, ROW_CHUNK):
        y1 = min(y0 + ROW_CHUNK, dst_h)
        py = (np.arange(y0, y1, dtype=np.float64) + 0.5)[:, None]
        pxg = px[None, :]

        gx = a * pxg + b * py + c
        gy = d * pxg + e * py + ff

        lat, lon = inverse_stereographic(gx, gy, PROJ_LAT_0, PROJ_LON_0, EARTH_RADIUS_M)
        value, out_of_bounds = bilinear_sample(sithick, lat, lon, lat0_grid, dlat, lon0_grid, dlon)

        invalid = out_of_bounds | (value >= FILL_THRESHOLD) | ~np.isfinite(value)
        value = np.where(invalid, tmin, value)

        frac = np.clip((value - tmin) / (tmax - tmin), 0.0, 1.0)
        out[y0:y1, :] = np.round(frac * 255.0).astype(np.uint8)

    Image.fromarray(out, mode="L").save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH} ({dst_w}x{dst_h}, thickness range [{tmin}, {tmax}] m -> [0, 255])")


if __name__ == "__main__":
    main()
