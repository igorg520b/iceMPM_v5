"""
Shared Kane Basin stereographic projection parameters and helpers.

Single source of truth for the geographic area, projection center, and
output-folder layout used by kane_basin_stereographic.py (download +
reproject), generate_data_mask.py (offline mask generation), and
add_graticule_frame.py (figure overlays) -- so all three always agree on the
same projection.
"""
import json
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from PIL import Image
from rasterio.transform import Affine, from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

# --- Geographic area (Nares Strait) ---
LAT_MIN, LAT_MAX = 77.34, 82.40
LON_MIN, LON_MAX = -85.32, -43.42

# --- Stereographic projection center: Kane Basin ---
PROJECTION_CENTER_LAT = 80.00
PROJECTION_CENTER_LON = -69.00
EARTH_RADIUS_M = 6371000.0

# --- Processing parameters ---
RESOLUTION = 40          # download/warp resolution, meters/pixel
TARGET_WIDTH_PX = 20000  # final image width after rescale, pixels

# --- Output folders (relative to wherever the scripts are run from) ---
TILES_BASE_FOLDER = "nares_tiles"
BASE_OUTPUT_FOLDER = "nares_kane_basin_stereo"


def proj4_string():
    return f"+proj=stere +lat_0={PROJECTION_CENTER_LAT} +lon_0={PROJECTION_CENTER_LON} +R={EARTH_RADIUS_M}"


def crs():
    return pyproj.CRS.from_proj4(proj4_string())


class StereoProjection:
    """pixel (x=col, y=row, top-left origin) <-> (lat, lon) mapping for a
    Kane Basin stereographic raster, built directly from the CRS + affine
    transform that produced it (extents.json). Exposes the same
    pixel_to_latlon(x, y) / latlon_to_pixel(lat, lon) interface as
    add_graticule_frame's OrthoProjection, so the (projection-agnostic)
    graticule/neatline drawing code works with either."""

    def __init__(self, transform_coeffs, proj4=None, resize_factor=1.0):
        # resize_factor > 1 adapts extents.json's affine transform (saved for
        # the native TARGET_WIDTH_PX raster) to a further-downsampled copy of
        # that same image, e.g. resize_factor=10 for a 2000px image made from
        # a 20000px original: new_pixel * resize_factor == old_pixel, so only
        # the linear part of the transform (a, b, d, e) scales; the
        # translation (c, f) is unchanged.
        a, b, c, d, e, f = transform_coeffs[:6]
        self.transform = Affine(a * resize_factor, b * resize_factor, c,
                                 d * resize_factor, e * resize_factor, f)
        self.crs = pyproj.CRS.from_proj4(proj4 or proj4_string())
        self._to_wgs84 = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self._from_wgs84 = pyproj.Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    @classmethod
    def from_extents_json(cls, path, resize_factor=1.0):
        with open(path) as f:
            doc = json.load(f)
        return cls(doc["affine_transform"], doc.get("projection_proj4"), resize_factor=resize_factor)

    def pixel_to_latlon(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        gx, gy = self.transform * (x + 0.5, y + 0.5)
        lon, lat = self._to_wgs84.transform(gx, gy)
        return np.asarray(lat), np.asarray(lon)

    def latlon_to_pixel(self, lat_deg, lon_deg):
        lat = np.asarray(lat_deg, dtype=np.float64)
        lon = np.asarray(lon_deg, dtype=np.float64)
        gx, gy = self._from_wgs84.transform(lon, lat)
        px, py = ~self.transform * (gx, gy)
        return px - 0.5, py - 0.5


def warp_to_stereographic(array, extents, base_dir, resolution=None):
    """Reprojects a WGS84 raster (H, W) or (H, W, C) covering `extents`
    (lon_min/lat_min/lon_max/lat_max) into the Kane Basin stereographic CRS,
    pinned at `resolution` m/pixel (default RESOLUTION) instead of letting
    GDAL pick its own default -- so this warp is a faithful resample at the
    resolution actually requested, not an arbitrary up/downsample. Shared by
    kane_basin_stereographic.py (real imagery) and generate_data_mask.py
    (synthetic all-white footprint) so both warp identically.

    Returns (dst_crs, dst_transform, warped_array)."""
    resolution = resolution or RESOLUTION
    array = np.asarray(array)
    single_band = array.ndim == 2
    if single_band:
        array = array[:, :, None]
    height, width, n_bands = array.shape
    dst_crs = crs()
    temp_tif_path = Path(base_dir) / "_tmp_wgs84_source.tif"

    try:
        res_lon = (extents['lon_max'] - extents['lon_min']) / width
        res_lat = (extents['lat_max'] - extents['lat_min']) / height
        src_transform = from_origin(extents['lon_min'], extents['lat_max'], res_lon, res_lat)
        with rasterio.open(
                temp_tif_path, "w", driver="GTiff", height=height, width=width,
                count=n_bands, dtype=array.dtype, crs="EPSG:4326", transform=src_transform
        ) as dst:
            for i in range(n_bands): dst.write(array[:, :, i], i + 1)

        with rasterio.open(temp_tif_path) as src:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=resolution
            )
            dst_array = np.zeros((n_bands, dst_height, dst_width), dtype=src.meta["dtype"])
            for i in range(n_bands):
                reproject(source=rasterio.band(src, i + 1), destination=dst_array[i],
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.bilinear)

        dst_array = np.transpose(dst_array, (1, 2, 0))
        if single_band:
            dst_array = dst_array[:, :, 0]
        return dst_crs, dst_transform, dst_array
    finally:
        if temp_tif_path.exists(): temp_tif_path.unlink()


def rescale_to_width(array, transform, target_width_px, resample=Image.Resampling.LANCZOS):
    """Downsamples array (H, W) or (H, W, C) so its width matches
    target_width_px (aspect ratio preserved) and returns the updated affine
    transform. target_width_px must be <= the source width -- this is meant
    to always be a genuine downsample (real averaging), never an upsample
    that would fabricate resolution that wasn't actually there.

    Returns (resized_array, new_transform, new_width, new_height)."""
    src_height, src_width = array.shape[:2]
    if target_width_px > src_width:
        raise ValueError(
            f"target_width_px ({target_width_px}) exceeds the source width "
            f"({src_width}); this would upsample and fabricate resolution."
        )
    scale = target_width_px / src_width
    target_height_px = max(1, round(src_height * scale))

    resized = Image.fromarray(array).resize((target_width_px, target_height_px), resample)
    resized_array = np.array(resized)

    sx = src_width / target_width_px
    sy = src_height / target_height_px
    new_transform = transform * Affine.scale(sx, sy)
    return resized_array, new_transform, target_width_px, target_height_px
