#!/usr/bin/env python3
"""
Builds a binary (black/white, no gray) footprint mask for the Kane Basin
stereographic projection: white = inside the downloaded lat/lon bounding box
(LAT_MIN/MAX, LON_MIN/MAX in kane_basin_projection.py), black = the warp's
empty fill outside it.

This is a purely geometric property of the bbox + projection + resolution/
target width -- it does NOT depend on any date's actual imagery, so it is
unaffected by clouds, coverage gaps, or dark open water (unlike thresholding
a real downloaded image). Starts from a synthetic all-white WGS84 raster and
warps it with the exact same pinned-resolution reprojection + rescale
(kane_basin_projection.warp_to_stereographic / rescale_to_width) that
kane_basin_stereographic.py uses for real imagery, so this mask lines up
pixel-for-pixel with any date's final_stereographic_image.png (as long as
RESOLUTION/TARGET_WIDTH_PX haven't changed since).

Fully offline: never imports sentinelhub, never touches the network.

Usage:
    python3 generate_data_mask.py [--threshold N] [--output PATH]
"""
import argparse
import os

import numpy as np
import pyproj
from PIL import Image

import kane_basin_projection as kbp

# This is our own generated raster (up to TARGET_WIDTH_PX wide), not an
# untrusted upload, so disable Pillow's decompression bomb guard rather than
# have it reject legitimately large output.
Image.MAX_IMAGE_PIXELS = None

DEFAULT_THRESHOLD = 128  # midpoint between synthetic black (0) and white (255)
DEFAULT_OUTPUT = os.path.join(kbp.BASE_OUTPUT_FOLDER, "footprint_mask.png")


def generate_footprint_mask(output_path, threshold=DEFAULT_THRESHOLD):
    extents = {
        'lon_min': kbp.LON_MIN, 'lat_min': kbp.LAT_MIN,
        'lon_max': kbp.LON_MAX, 'lat_max': kbp.LAT_MAX,
    }

    # Source WGS84 raster: 100% white, at the same resolution the real
    # download pipeline warps from.
    width_px, height_px = wgs84_source_size(kbp.RESOLUTION)
    print(f"Synthetic all-white source: {width_px} x {height_px} px at {kbp.RESOLUTION} m/px")
    white_array = np.full((height_px, width_px), 255, dtype=np.uint8)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    _, stereo_transform, stereo_array = kbp.warp_to_stereographic(white_array, extents, output_dir)
    print(f"Warped footprint: {stereo_array.shape[1]} x {stereo_array.shape[0]} px")

    resized_array, _, final_w, final_h = kbp.rescale_to_width(stereo_array, stereo_transform, kbp.TARGET_WIDTH_PX)
    print(f"Rescaled to target width: {final_w} x {final_h} px")

    mask = np.where(resized_array > threshold, 255, 0).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(output_path)
    coverage_pct = 100.0 * (mask == 255).mean()
    print(f"Footprint mask saved to {output_path} ({coverage_pct:.1f}% white)")
    return mask


def wgs84_source_size(resolution_m):
    """Pixel size of a WGS84 raster covering the Nares Strait bbox at
    resolution_m meters/pixel, using geodesic distances along the bottom
    edge and left edge as the width/height reference."""
    geod = pyproj.Geod(ellps="WGS84")
    _, _, width_m = geod.inv(kbp.LON_MIN, kbp.LAT_MIN, kbp.LON_MAX, kbp.LAT_MIN)
    _, _, height_m = geod.inv(kbp.LON_MIN, kbp.LAT_MIN, kbp.LON_MIN, kbp.LAT_MAX)
    width_px = max(2, round(width_m / resolution_m))
    height_px = max(2, round(height_m / resolution_m))
    return width_px, height_px


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                         help=f"Binary cutoff 0-255 applied after warp/rescale blending (default {DEFAULT_THRESHOLD}).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help=f"Output PNG path (default {DEFAULT_OUTPUT}).")
    args = parser.parse_args()

    generate_footprint_mask(args.output, args.threshold)


if __name__ == "__main__":
    main()
