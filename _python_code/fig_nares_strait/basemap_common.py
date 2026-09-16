#!/usr/bin/env python3
"""
Shared projection/tile-fetch/reproject machinery for this directory's
make_basemap_*.py scripts -- factored out once a second XYZ-tile source
(OpenTopoMap, alongside the original CartoDB "Positron, no labels") needed
the exact same corner-padding fix, to avoid two copies drifting apart.

Also writes a georeference sidecar JSON next to each output PNG, shaped
exactly like a simulation.json's PROJ_* fields (same keys
SimStereoProjection.from_simulation_json in
_python_code/graticule/add_graticule_frame.py already reads) -- so
add_frame_to_basemap.py (this directory) can add the striped neatline frame
to any of these outputs later, reusing that existing loader unchanged.
"""
import io
import json
import math
import sys
import time
from pathlib import Path as FilePath
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import Affine
from pyproj import Transformer

sys.path.insert(0, str(FilePath(__file__).resolve().parent.parent / "graticule"))
from add_graticule_frame import SimStereoProjection  # noqa: E402, re-exported for convenience

R_EARTH = 6371000.0   # matches simulation/projection.h's Projection::R (sphere, not WGS84)
WEB_MERCATOR = CRS.from_epsg(3857)
R_MERC = 6378137.0    # Web Mercator's own sphere radius (EPSG:3857), unrelated to R_EARTH


def lonlat_to_tile_xy(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xt = (lon + 180.0) / 360.0 * n
    yt = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xt, yt


def fetch_tile(tile_url, subdomains, user_agent, z, x, y, n, retries=3):
    s = subdomains[(x + y) % len(subdomains)]
    url = tile_url.format(s=s, z=z, x=x % n, y=y)
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": user_agent}, timeout=20)
            r.raise_for_status()
            return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise


def fetch_xyz_mosaic(tile_url, subdomains, user_agent, lat_min, lat_max, lon_min, lon_max, zoom):
    """Fetches every tile covering the bbox from a standard {z}/{x}/{y} XYZ
    source, stitches into one array, and returns it with its EPSG:3857
    affine transform."""
    n = 2 ** zoom
    lat_max_c = min(lat_max, 85.05)  # Web Mercator's own valid range
    x0f, y0f = lonlat_to_tile_xy(lon_min, lat_max_c, zoom)  # top-left corner
    x1f, y1f = lonlat_to_tile_xy(lon_max, lat_min, zoom)    # bottom-right corner
    tx0, tx1 = int(math.floor(x0f)), int(math.floor(x1f))
    ty0, ty1 = int(math.floor(y0f)), int(math.floor(y1f))

    tiles_x = list(range(tx0, tx1 + 1))
    tiles_y = list(range(ty0, ty1 + 1))
    mosaic = np.zeros((len(tiles_y) * 256, len(tiles_x) * 256, 3), dtype=np.uint8)

    def _get(ix, iy):
        return ix, iy, fetch_tile(tile_url, subdomains, user_agent, zoom, tiles_x[ix], tiles_y[iy], n)

    jobs = [(ix, iy) for iy in range(len(tiles_y)) for ix in range(len(tiles_x))]
    print(f"Fetching {len(jobs)} tiles at zoom {zoom} from {tile_url.split('/')[2]}...")
    t0 = time.time()
    completed = 0
    report_every = max(1, len(jobs) // 20)  # ~20 progress lines total
    with ThreadPoolExecutor(max_workers=8) as ex:
        for ix, iy, arr in ex.map(lambda p: _get(*p), jobs):
            mosaic[iy * 256:(iy + 1) * 256, ix * 256:(ix + 1) * 256] = arr
            completed += 1
            if completed % report_every == 0 or completed == len(jobs):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - completed) / rate if rate > 0 else float("inf")
                print(f"  {completed}/{len(jobs)} tiles ({100*completed/len(jobs):.0f}%), "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    merc_span = 2 * math.pi * R_MERC
    px_size = merc_span / n / 256
    x_origin = -merc_span / 2 + tx0 * 256 * px_size
    y_origin = merc_span / 2 - ty0 * 256 * px_size
    transform = Affine(px_size, 0, x_origin, 0, -px_size, y_origin)
    return mosaic, transform


def latlon_box_meters_extent(transformer, lat_min, lat_max, lon_min, lon_max, n=200):
    """Bounding box, in the target CRS's meters, of a lat/lon rectangle's
    perimeter (sampled, not just the 4 corners -- the perimeter is curved
    in the projected plane)."""
    lats = np.concatenate([np.full(n, lat_min), np.linspace(lat_min, lat_max, n),
                           np.full(n, lat_max), np.linspace(lat_max, lat_min, n)])
    lons = np.concatenate([np.linspace(lon_min, lon_max, n), np.full(n, lon_max),
                           np.linspace(lon_max, lon_min, n), np.full(n, lon_min)])
    x, y = transformer.transform(lons, lats)
    return x.min(), x.max(), y.min(), y.max()


def stereo_box_to_latlon_extent(inverse_transformer, xmin, xmax, ymin, ymax, n=200):
    """Inverse of latlon_box_meters_extent: given a RECTANGLE in the
    stereographic plane's meters, returns the lat/lon bounding box its own
    perimeter covers -- always at least as large as the nominal lat/lon box
    that produced it, since a lat/lon box's curved perimeter sits strictly
    inside its own meters bounding rectangle. Used so tile-fetching covers
    the OUTPUT rectangle's corners, not just the nominal box (otherwise the
    corners have no source data and render black)."""
    xs = np.concatenate([np.full(n, xmin), np.linspace(xmin, xmax, n),
                         np.full(n, xmax), np.linspace(xmax, xmin, n)])
    ys = np.concatenate([np.linspace(ymin, ymax, n), np.full(n, ymax),
                         np.linspace(ymax, ymin, n), np.full(n, ymin)])
    lon, lat = inverse_transformer.transform(xs, ys)
    return lat.min(), lat.max(), lon.min(), lon.max()


def build_stereo_crs(lat0, lon0, R=R_EARTH):
    return CRS.from_proj4(f"+proj=stere +lat_0={lat0} +lon_0={lon0} +R={R} +units=m +no_defs")


def make_basemap(tile_url, subdomains, user_agent, lat0, lon0, lat_min, lat_max, lon_min, lon_max,
                  output_width_px, zoom, output_file, pad_frac=0.15, corner_fetch_margin_deg=0.5, dpi=250):
    """Shared pipeline: fetch an XYZ tile source, reproject into our
    stereographic CRS centered at (lat0, lon0), save the PNG plus a
    georeference sidecar JSON (same PROJ_* shape as simulation.json)."""
    dst_crs = build_stereo_crs(lat0, lon0)
    to_stereo = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    to_latlon = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)

    xmin, xmax, ymin, ymax = latlon_box_meters_extent(to_stereo, lat_min, lat_max, lon_min, lon_max)
    pad = max(xmax - xmin, ymax - ymin) * pad_frac
    xmin, xmax, ymin, ymax = xmin - pad, xmax + pad, ymin - pad, ymax + pad

    px_size = (xmax - xmin) / output_width_px
    height_px = int(round((ymax - ymin) / px_size))
    dst_transform = Affine(px_size, 0, xmin, 0, -px_size, ymax)

    fetch_lat_min, fetch_lat_max, fetch_lon_min, fetch_lon_max = stereo_box_to_latlon_extent(
        to_latlon, xmin, xmax, ymin, ymax)
    fetch_lat_min -= corner_fetch_margin_deg
    fetch_lat_max += corner_fetch_margin_deg
    fetch_lon_min -= corner_fetch_margin_deg
    fetch_lon_max += corner_fetch_margin_deg
    print(f"Nominal box: lat [{lat_min},{lat_max}] lon [{lon_min},{lon_max}]")
    print(f"Fetch box (covers rectangle corners + margin): "
          f"lat [{fetch_lat_min:.2f},{fetch_lat_max:.2f}] lon [{fetch_lon_min:.2f},{fetch_lon_max:.2f}]")

    mosaic, src_transform = fetch_xyz_mosaic(tile_url, subdomains, user_agent,
                                             fetch_lat_min, fetch_lat_max, fetch_lon_min, fetch_lon_max, zoom)

    dst = np.zeros((height_px, output_width_px, 3), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=mosaic[:, :, band],
            destination=dst[:, :, band],
            src_transform=src_transform,
            src_crs=WEB_MERCATOR,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    fig = plt.figure(figsize=(output_width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(dst, zorder=1, interpolation="none", extent=[0, output_width_px, height_px, 0])
    ax.set_xlim(0, output_width_px)
    ax.set_ylim(height_px, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)

    # Georeference sidecar -- same PROJ_* keys as simulation.json, so
    # add_frame_to_basemap.py can load it via SimStereoProjection.from_simulation_json
    # unchanged.
    sidecar = {
        "PROJ_LAT_0": lat0,
        "PROJ_LON_0": lon0,
        "PROJ_RESIZE_FACTOR": 1.0,
        "PROJ_TRANSFORM_COEFFS": [px_size, 0.0, xmin, 0.0, -px_size, ymax],
    }
    sidecar_path = FilePath(output_file).with_suffix(".georef.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Wrote {output_file} ({output_width_px}x{height_px} px) and {sidecar_path.name}")
    return output_file, sidecar_path
