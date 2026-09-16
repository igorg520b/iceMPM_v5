import sys
import os
import math
import json
from pathlib import Path

# --- Third-party library imports ---
# Ensure you have these installed:
# pip install sentinelhub-py imageio Pillow numpy pyproj rasterio

from sentinelhub import (
    SHConfig, MimeType, CRS, BBox, SentinelHubRequest, DataCollection,
    BBoxSplitter, bbox_to_dimensions
)
from imageio import imwrite
from PIL import Image
import numpy as np
import pyproj
import rasterio
from rasterio.transform import from_origin, Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling

import kane_basin_projection as kbp

# ==============================================================================
# --- 1. SCRIPT CONFIGURATION ---
# ==============================================================================

# --- Geographic area, projection center, download/target resolution, and
# output folders all live in kane_basin_projection.py so this script,
# generate_data_mask.py, and add_graticule_frame.py stay in agreement. ---
LAT_MIN, LAT_MAX = kbp.LAT_MIN, kbp.LAT_MAX
LON_MIN, LON_MAX = kbp.LON_MIN, kbp.LON_MAX
PROJECTION_CENTER_LAT = kbp.PROJECTION_CENTER_LAT
PROJECTION_CENTER_LON = kbp.PROJECTION_CENTER_LON
RESOLUTION = kbp.RESOLUTION            # meters/pixel, see kane_basin_projection.py
TARGET_WIDTH_PX = kbp.TARGET_WIDTH_PX  # final image width, pixels
TILES_BASE_FOLDER = kbp.TILES_BASE_FOLDER
BASE_OUTPUT_FOLDER = kbp.BASE_OUTPUT_FOLDER

# --- Time Parameter: only this single date is downloaded ---
TIME_INTERVAL = ("2025-06-30", "2025-06-30")

# --- Technical Parameters ---
TILE_SIZE_PX = 1024

# --- Post-Processing Options ---
SAVE_TILE_METADATA = True

# --- Copernicus Data Space Ecosystem Credentials ---
# Not hardcoded here -- loaded from sentinelhub's own saved profile
# (~/.config/sentinelhub/config.toml). Set it up once with:
#   sentinelhub.config --sh_client_id YOUR_ID --sh_client_secret YOUR_SECRET \
#       --sh_base_url https://sh.dataspace.copernicus.eu \
#       --sh_token_url https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
# See _python_code/README.md.


# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def download_tiles(config, tiles_folder, time_interval):
    """Downloads tiles for the time interval, reusing any already-cached tile
    fragments found in tiles_folder from a previous run."""
    print("--- STEP 1: Starting Tile Download ---")
    os.makedirs(tiles_folder, exist_ok=True)
    full_bbox = BBox(bbox=[LON_MIN, LAT_MIN, LON_MAX, LAT_MAX], crs=CRS.WGS84)
    full_size = bbox_to_dimensions(full_bbox, resolution=RESOLUTION)
    print(f"Full area dimensions at {RESOLUTION}m resolution: {full_size} pixels")

    n_tiles_x = math.ceil(full_size[0] / TILE_SIZE_PX)
    n_tiles_y = math.ceil(full_size[1] / TILE_SIZE_PX)
    bbox_splitter = BBoxSplitter([full_bbox], crs=CRS.WGS84, split_shape=(n_tiles_x, n_tiles_y))
    tile_bboxes, tile_info = bbox_splitter.get_bbox_list(), bbox_splitter.get_info_list()
    print(f"Splitting into a {n_tiles_x} x {n_tiles_y} grid. Total tiles: {len(tile_bboxes)}")

    evalscript = """
    //VERSION=3
    // Continuous, smooth true-colour mapping from B2, B3, B4
    // Parameters are set at the top; no discontinuities.

    // ---------- USER-TWEAKABLE COEFFICIENTS ------------------
    var globalGamma  = 0.9;   // overall gamma applied after gain
    var gammaR       = 1.0;   // per-channel gamma
    var gammaG       = 0.9;
    var gammaB       = 0.9;
    var gain         = 0.95;   // linear scaling before gamma
    // ---------------------------------------------------------

    function setup() {
      return {
        input: ["B02", "B03", "B04", "dataMask"],
        output: { bands: 4 }
      };
    }

    // smooth gamma (power-law) without any clamping or conditionals
    function gamma(x, g) {
      return Math.pow(x, g);
    }

    function evaluatePixel(s) {
      var b = s.B02 * gain;
      var g = s.B03 * gain;
      var r = s.B04 * gain;

      r = gamma(r, gammaR * globalGamma);
      g = gamma(g, gammaG * globalGamma);
      b = gamma(b, gammaB * globalGamma);

      return [r, g, b, s.dataMask];
    }
    """

    metadata_list = []
    for i, (tile_bbox, info) in enumerate(zip(tile_bboxes, tile_info)):
        row, col = info['index_y'], info['index_x']
        tile_pixel_size = bbox_to_dimensions(tile_bbox, resolution=RESOLUTION)
        filename = f"tile_{row}_{col}.png"
        filepath = os.path.join(tiles_folder, filename)

        if os.path.exists(filepath):
            print(f"-> Tile [{row}, {col}] already cached, reusing ({i + 1}/{len(tile_bboxes)}).")
        else:
            print(f"-> Downloading tile [{row}, {col}] ({i + 1}/{len(tile_bboxes)})...")
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from(
                            "s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                        ),
                        time_interval=time_interval,
                        other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                bbox=tile_bbox, size=tile_pixel_size, config=config
            )

            try:
                image_tile = request.get_data()[0]
                imwrite(filepath, image_tile)
            except Exception as e:
                print(f"   ERROR downloading tile [{row}, {col}]: {e}")
                continue

        tile_metadata = {
            'filename': filename, 'row': row, 'col': col,
            'bbox_wgs84': list(tile_bbox), 'pixel_size': tile_pixel_size
        }
        metadata_list.append(tile_metadata)

    print("--- Tile Download Complete ---\n")
    return metadata_list


def save_tile_metadata(metadata, output_folder):
    """Saves the per-tile metadata to metadata.json."""
    print("--- STEP 2: Saving Per-Tile Metadata ---")
    metadata_path = os.path.join(output_folder, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"   Per-tile metadata saved to {metadata_path}\n")


def stitch_tiles(metadata, tiles_folder):
    """Stitches individual cached tiles into a single large image."""
    print("--- STEP 3: Stitching Tiles ---")
    lon_min = min(m["bbox_wgs84"][0] for m in metadata)
    lat_min = min(m["bbox_wgs84"][1] for m in metadata)
    lon_max = max(m["bbox_wgs84"][2] for m in metadata)
    lat_max = max(m["bbox_wgs84"][3] for m in metadata)
    extents = {'lon_min': lon_min, 'lat_min': lat_min, 'lon_max': lon_max, 'lat_max': lat_max}

    resolutions = [((m["bbox_wgs84"][2] - m["bbox_wgs84"][0]) / m["pixel_size"][0],
                    (m["bbox_wgs84"][3] - m["bbox_wgs84"][1]) / m["pixel_size"][1]) for m in metadata]
    ref_res_lon = min(abs(r[0]) for r in resolutions)
    ref_res_lat = min(abs(r[1]) for r in resolutions)

    final_width = int(round((lon_max - lon_min) / ref_res_lon))
    final_height = int(round((lat_max - lat_min) / ref_res_lat))
    print(f"Stitched image size: {final_width} x {final_height} pixels")

    stitched_image = Image.new("RGB", (final_width, final_height))

    def lon_to_x(lon):
        return int(round((lon - lon_min) / ref_res_lon))

    def lat_to_y(lat):
        return int(round((lat_max - lat) / ref_res_lat))

    for m in metadata:
        lon0, lat0, lon1, lat1 = m["bbox_wgs84"]
        x0, y0 = lon_to_x(lon0), lat_to_y(lat1)
        x1, y1 = lon_to_x(lon1), lat_to_y(lat0)
        tile_img = Image.open(os.path.join(tiles_folder, m['filename'])).convert("RGB")
        target_size = (x1 - x0, y1 - y0)
        if tile_img.size != target_size:
            tile_img = tile_img.resize(target_size, Image.Resampling.BILINEAR)
        stitched_image.paste(tile_img, (x0, y0))

    print("--- Tile Stitching Complete ---\n")
    return stitched_image, extents


def reproject_to_stereographic(stitched_image, extents, base_dir):
    """Reprojects the WGS84 mosaic to a Stereographic projection centered on
    Kane Basin. The output resolution is pinned to RESOLUTION (m/pixel)
    instead of letting GDAL pick its own default, so this warp is a
    faithful resample at the resolution actually captured, not an arbitrary
    up/downsample."""
    print("--- STEP 4: Reprojecting to Stereographic View (Kane Basin) ---")
    stitched_array = np.array(stitched_image)
    height, width, _ = stitched_array.shape
    src_crs = "EPSG:4326"
    dst_crs = kbp.crs()
    temp_tif_path = Path(base_dir) / "temp_stitched_platecarree.tif"

    try:
        res_lon, res_lat = (extents['lon_max'] - extents['lon_min']) / width, (
                    extents['lat_max'] - extents['lat_min']) / height
        src_transform = from_origin(extents['lon_min'], extents['lat_max'], res_lon, res_lat)
        with rasterio.open(
                temp_tif_path, "w", driver="GTiff", height=height, width=width,
                count=3, dtype=stitched_array.dtype, crs=src_crs, transform=src_transform
        ) as dst:
            for i in range(3): dst.write(stitched_array[:, :, i], i + 1)

        with rasterio.open(temp_tif_path) as src:
            stereo_transform, stereo_width, stereo_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=RESOLUTION
            )
            stereo_array = np.zeros((3, stereo_height, stereo_width), dtype=src.meta["dtype"])
            for i in range(3):
                reproject(source=rasterio.band(src, i + 1), destination=stereo_array[i],
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=stereo_transform, dst_crs=dst_crs, resampling=Resampling.bilinear)

        stereo_array = np.transpose(stereo_array, (1, 2, 0))
        print(f"   Reprojected size at {RESOLUTION} m/pixel: {stereo_width} x {stereo_height} pixels")
        print("--- Reprojection Complete ---\n")
        return dst_crs, stereo_transform, stereo_array
    finally:
        if os.path.exists(temp_tif_path): os.remove(temp_tif_path)


def rescale_to_target_width(stereo_array, stereo_transform, target_width_px, base_dir):
    """Downsamples the reprojected image so its width matches
    target_width_px (aspect ratio preserved) and updates the affine
    transform to match. Since the source was reprojected at RESOLUTION
    m/pixel (finer than the target), this is a genuine downsample -
    real averaging, not pixel 'stretching' / fabricated resolution."""
    print("--- STEP 5: Rescaling to Target Width ---")
    src_height, src_width, _ = stereo_array.shape
    if target_width_px > src_width:
        raise ValueError(
            f"TARGET_WIDTH_PX ({target_width_px}) exceeds the reprojected width "
            f"({src_width}) at RESOLUTION={RESOLUTION}m/px; this would upsample "
            f"and fabricate resolution. Lower TARGET_WIDTH_PX or lower RESOLUTION."
        )
    scale = target_width_px / src_width
    target_height_px = max(1, round(src_height * scale))

    resized = Image.fromarray(stereo_array).resize(
        (target_width_px, target_height_px), Image.Resampling.LANCZOS
    )
    resized_array = np.array(resized)

    sx = src_width / target_width_px
    sy = src_height / target_height_px
    final_transform = stereo_transform * Affine.scale(sx, sy)

    out_png_path = Path(base_dir) / "final_stereographic_image.png"
    Image.fromarray(resized_array).save(out_png_path)
    print(f"   Rescaled {src_width} x {src_height} -> {target_width_px} x {target_height_px} pixels")
    print(f"   Final stereographic image saved to {out_png_path}")
    print("--- Rescaling Complete ---\n")
    return final_transform, target_width_px, target_height_px


def save_final_metadata(extents, stereo_crs, final_transform, final_image_width, final_image_height, output_folder):
    """
    Saves the final projection metadata, including everything needed to later
    reconstruct the pixel (i, j) <-> (lat, lon) mapping: the proj4 string of
    the stereographic projection, its center point, the final affine
    transform (post-rescale), and pixel-scale estimates.
    """
    print("--- STEP 6: Saving Final Image & Projection Metadata ---")

    geod = pyproj.Geod(ellps="WGS84")

    # Ground distance in meters along the lat_min parallel, used as an
    # authoritative resolution estimate (matches the convention used by the
    # orthographic download script).
    lon_min, lon_max = extents['lon_min'], extents['lon_max']
    reference_lat = extents['lat_min']
    _, _, distance_meters = geod.inv(lon_min, reference_lat, lon_max, reference_lat)
    custom_resolution = distance_meters / final_image_width

    # Ground distance spanned by one pixel at the image center, computed
    # directly from the final affine transform (local cross-check, since
    # scale varies across a stereographic projection away from its center).
    px_cx, px_cy = final_image_width // 2, final_image_height // 2
    x0_c, y0_c = rasterio.transform.xy(final_transform, px_cy, px_cx, offset="center")
    x1_c, y1_c = rasterio.transform.xy(final_transform, px_cy, px_cx + 1, offset="center")
    to_wgs84 = pyproj.Transformer.from_crs(stereo_crs, "EPSG:4326", always_xy=True)
    lon0_c, lat0_c = to_wgs84.transform(x0_c, y0_c)
    lon1_c, lat1_c = to_wgs84.transform(x1_c, y1_c)
    _, _, center_dist_m = geod.inv(lon0_c, lat0_c, lon1_c, lat1_c)

    extents['resolution_meters_per_pixel'] = round(custom_resolution, 2)
    extents['approx_meters_per_pixel_at_center'] = round(center_dist_m, 2)
    extents['download_resolution_meters_per_pixel'] = RESOLUTION
    extents['target_width_px'] = TARGET_WIDTH_PX
    extents['projection'] = 'stereographic'
    extents['projection_proj4'] = stereo_crs.to_proj4()
    extents['projection_center_lat'] = PROJECTION_CENTER_LAT
    extents['projection_center_lon'] = PROJECTION_CENTER_LON
    extents['final_image_width'] = final_image_width
    extents['final_image_height'] = final_image_height
    # Affine coefficients (a, b, c, d, e, f) mapping pixel (col, row) ->
    # (x, y) in the stereographic CRS (meters): x = a*col + b*row + c,
    # y = d*col + e*row + f. Combined with projection_proj4, this fully
    # reconstructs (i, j) -> (lon, lat):
    #   x, y = Affine(*affine_transform) * (i + 0.5, j + 0.5)
    #   lon, lat = pyproj.Transformer.from_crs(projection_proj4, "EPSG:4326", always_xy=True).transform(x, y)
    extents['affine_transform'] = list(final_transform)

    print(f"   Ground distance along {reference_lat}° parallel: {distance_meters:,.2f} meters")
    print(f"   Full image width: {final_image_width:,} pixels")
    print(f"   Final calculated pixel size (width-based): {custom_resolution:.2f} meters/pixel")
    print(f"   Final calculated pixel size (center-based): {center_dist_m:.2f} meters/pixel")

    extents_path = os.path.join(output_folder, 'extents.json')
    with open(extents_path, 'w') as f:
        json.dump(extents, f, indent=4)
    print(f"   Final image & projection metadata saved to {extents_path}\n")


# ==============================================================================
# --- 3. PROCESSING PIPELINE ---
# ==============================================================================

def process_time_interval(config, time_interval, tiles_base_folder, base_output_folder):
    """Executes the full download-and-processing pipeline for one date."""
    start_date = time_interval[0]
    # Persistent tile cache: kept across runs so re-running the script reuses
    # already-downloaded fragments instead of re-downloading them.
    tiles_folder = os.path.join(tiles_base_folder, start_date)
    # Stitched/projected output for this run.
    current_output_folder = os.path.join(base_output_folder, start_date)
    os.makedirs(current_output_folder, exist_ok=True)
    print(f"Tile cache directory: {tiles_folder}")
    print(f"Output directory for this run: {current_output_folder}")

    tile_metadata = download_tiles(config, tiles_folder, time_interval)
    if not tile_metadata:
        print(f"No tiles were downloaded for {time_interval}. Aborting.")
        return

    if SAVE_TILE_METADATA:
        save_tile_metadata(tile_metadata, tiles_folder)

    stitched_image, geo_extents = stitch_tiles(tile_metadata, tiles_folder)

    stereo_crs, stereo_transform, stereo_array = reproject_to_stereographic(
        stitched_image, geo_extents, current_output_folder
    )

    final_transform, final_width, final_height = rescale_to_target_width(
        stereo_array, stereo_transform, TARGET_WIDTH_PX, current_output_folder
    )

    save_final_metadata(geo_extents, stereo_crs, final_transform, final_width, final_height, current_output_folder)


# ==============================================================================
# --- 4. MAIN EXECUTION SCRIPT ---
# ==============================================================================

if __name__ == "__main__":
    try:
        # sh_client_id/sh_client_secret/sh_base_url/sh_token_url come from the
        # saved default profile (~/.config/sentinelhub/config.toml) -- see the
        # "Copernicus Data Space Ecosystem Credentials" comment above.
        sh_config = SHConfig()
        if not sh_config.sh_client_id or not sh_config.sh_client_secret:
            raise RuntimeError(
                "No Sentinel Hub credentials in the saved profile -- run "
                "`sentinelhub.config --sh_client_id ... --sh_client_secret ...` first "
                "(see _python_code/README.md)."
            )
        print("Sentinel Hub configuration is set correctly.")
    except Exception as e:
        print(f"Error setting up Sentinel Hub configuration: {e}");
        sys.exit(1)

    process_time_interval(sh_config, TIME_INTERVAL, TILES_BASE_FOLDER, BASE_OUTPUT_FOLDER)

    print(f"\nProcessing complete for {TIME_INTERVAL[0]}.")
    print(f"Tile fragments cached in: '{TILES_BASE_FOLDER}'")
    print(f"Final stereographic output in: '{BASE_OUTPUT_FOLDER}'")
