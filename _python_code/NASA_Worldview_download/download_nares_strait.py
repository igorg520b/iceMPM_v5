import os
import sys
import math
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pyproj
import rasterio
from rasterio.transform import from_origin, Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image, ImageEnhance
from owslib.wms import WebMapService

# ==============================================================================
# --- 1. SCRIPT CONFIGURATION ---
# ==============================================================================

# Nares Strait - Updated to match Copernicus script exactly
LAT_MIN, LAT_MAX = 77.34, 82.40
LON_MIN, LON_MAX = -85.32, -43.42

TIME_INTERVALS = [
    ("2024-06-08", "2024-06-20") # Set to a single day for testing high resolution
]

BASE_OUTPUT_FOLDER = "nares_strait_imagery_ortho"
IMAGE_WIDTH_PX = 8000  # High horizontal resolution from WMS for true 4K output

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def download_wms_image(date_str, bbox, width, output_folder):
    """Downloads a single composite image from NASA GIBS via WMS."""
    print(f"--- STEP 1: Downloading WMS Imagery for {date_str} ---")
    
    # Calculate height based on aspect ratio in degrees (EPSG:4326)
    lon_span = bbox[2] - bbox[0]
    lat_span = bbox[3] - bbox[1]
    height = int(width / (lon_span / lat_span))
    
    print(f"Requesting image of size {width} x {height} pixels...")
    
    wms_url = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?'
    wms = WebMapService(wms_url, version='1.1.1')
    layer_name = 'MODIS_Terra_CorrectedReflectance_TrueColor'
    
    img_data = wms.getmap(
        layers=[layer_name],
        styles=['default'],
        srs='EPSG:4326',
        bbox=bbox,
        size=(width, height),
        format='image/png',
        time=date_str
    )
    
    temp_img_path = os.path.join(output_folder, f"temp_downloaded_wms_{date_str}.png")
    with open(temp_img_path, 'wb') as out:
        out.write(img_data.read())
        
    print(f"   [DONE] Saved temporary WMS image to {temp_img_path}")
    
    # Apply gamma correction to darken midtones while preserving whites
    img = Image.open(temp_img_path).convert("RGB")
    gamma = 1.8
    lut = [int(255 * (i / 255.0) ** gamma) for i in range(256)]
    img_corrected = img.point(lut * 3)
    print(f"   [DONE] Applied gamma correction (gamma={gamma})\n")
    
    return img_corrected

def reproject_and_analyze(downloaded_image, extents, date_str, base_dir):
    """Reprojects the image to Orthographic and saves it."""
    print("--- STEP 2: Reprojecting to Orthographic View ---")
    image_array = np.array(downloaded_image)
    height, width, _ = image_array.shape
    
    lon0 = (extents['lon_min'] + extents['lon_max']) / 2
    lat0 = (extents['lat_min'] + extents['lat_max']) / 2
    
    src_crs = "EPSG:4326"
    dst_crs = pyproj.CRS.from_proj4(f"+proj=ortho +lat_0={lat0} +lon_0={lon0} +R=6371000")
    
    temp_tif_path = Path(base_dir) / f"temp_platecarree_{date_str}.tif"

    try:
        res_lon = (extents['lon_max'] - extents['lon_min']) / width
        res_lat = (extents['lat_max'] - extents['lat_min']) / height
        src_transform = from_origin(extents['lon_min'], extents['lat_max'], res_lon, res_lat)
        
        with rasterio.open(
                temp_tif_path, "w", driver="GTiff", height=height, width=width,
                count=3, dtype=image_array.dtype, crs=src_crs, transform=src_transform
        ) as dst:
            for i in range(3): 
                dst.write(image_array[:, :, i], i + 1)

        with rasterio.open(temp_tif_path) as src:
            ortho_transform, ortho_width, ortho_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            
            # Reduce output dimensions by a factor of 2 to get ~4K width
            ortho_width = ortho_width // 2
            ortho_height = ortho_height // 2
            ortho_transform = ortho_transform * Affine.scale(2.0, 2.0)
            
            ortho_array = np.zeros((3, ortho_height, ortho_width), dtype=src.meta["dtype"])
            
            for i in range(3):
                reproject(
                    source=rasterio.band(src, i + 1), 
                    destination=ortho_array[i],
                    src_transform=src.transform, 
                    src_crs=src.crs,
                    dst_transform=ortho_transform, 
                    dst_crs=dst_crs, 
                    resampling=Resampling.bilinear
                )

        ortho_array = np.transpose(ortho_array, (1, 2, 0))
        out_png_path = Path(base_dir) / f"nares_strait_terra_{date_str}_ortho.png"
        Image.fromarray(ortho_array).save(out_png_path)
        print(f"   [DONE] Final orthographic image saved to {out_png_path}")
        print("--- Reprojection Complete ---\n")
        return out_png_path
    finally:
        if os.path.exists(temp_tif_path): 
            os.remove(temp_tif_path)

def process_date(date_str, base_output_folder):
    """Executes the pipeline for a single date."""
    os.makedirs(base_output_folder, exist_ok=True)
    print(f"Output directory for {date_str}: {base_output_folder}")

    bbox = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    extents = {'lon_min': LON_MIN, 'lat_min': LAT_MIN, 'lon_max': LON_MAX, 'lat_max': LAT_MAX}

    # 1. Download WMS Image
    try:
        downloaded_image = download_wms_image(date_str, bbox, IMAGE_WIDTH_PX, base_output_folder)
    except Exception as e:
        print(f"[-] Error downloading WMS image for {date_str}: {e}")
        return

    # 2. Reproject and Analyze
    try:
        reproject_and_analyze(downloaded_image, extents, date_str, base_output_folder)
    except Exception as e:
        print(f"[-] Error during reprojection for {date_str}: {e}")
        return
    
    # Cleanup temp WMS image
    temp_wms = os.path.join(base_output_folder, f"temp_downloaded_wms_{date_str}.png")
    if os.path.exists(temp_wms):
        os.remove(temp_wms)

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    for start, end in TIME_INTERVALS:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            print(f"\n{'=' * 80}\nProcessing date: {date_str}\n{'=' * 80}")
            process_date(date_str, BASE_OUTPUT_FOLDER)
            current_date += timedelta(days=1)

    print(f"\n[DONE] Full processing pipeline complete! [DONE]")
    print(f"All outputs are located in: '{BASE_OUTPUT_FOLDER}'")
