#!/usr/bin/env python3
"""
OpenTopoMap base map, close-up on the islands in Kennedy Channel
(Franklin Island, Crozier Island, the Ellesmere Island coast).
Originally written for the paper's introduction figure at a much wider
extent (same as Fig. 1b, see make_basemap_osm.py's docstring; that
extent is kept below as a comment) -- reused here for the close-up map,
per the supervisor comment asking to locate these islands. Genuine
terrain/hillshade detail with real Arctic coverage (confirmed by
testing several free tile terrain sources -- Esri's
World_Hillshade/World_Terrain_Base are SRTM-based and have NO data at
all above ~60N, so they 404/render blank here; OpenTopoMap blends in
other DEM sources that do cover the Arctic).

Unlike make_basemap_osm.py's CartoDB source, OpenTopoMap tiles have text
labels baked in (place names, etc.) -- there is no no-label variant of it.
Kept as a separate script/output specifically because of that tradeoff
(detail + labels vs. no bathymetry + no labels).

Uses this project's own stereographic projection (see basemap_common.py
for the shared fetch/reproject/corner-padding pipeline), but temporarily
re-centered on Franklin Island rather than the simulation's own center
(base_config.json's PROJ_LAT_0/PROJ_LON_0 = 80N, 69W, kept below as a
comment) -- see the PROJ_LAT_0/PROJ_LON_0 comment below for why.
Writes a .georef.json sidecar alongside the PNG -- see
add_frame_to_basemap.py to add the striped neatline frame afterward.

Usage:
    python3 make_basemap_opentopo.py
"""
from pathlib import Path as FilePath
from basemap_common import make_basemap

# original center, matches the simulation (base_config.json's PROJ_LAT_0/PROJ_LON_0):
# PROJ_LAT_0 = 80.0
# PROJ_LON_0 = -69.0

# temporary center for the Kennedy Channel close-up: Franklin Island
# (80 deg 38' N, 66 deg 48' W), so the stereographic projection's local
# north-south/east-west directions line up with the image axes near the
# islands of interest -- off-center (the original 80N/69W), the frame
# comes out visibly tilted since that alignment only holds exactly at
# the projection's own center.
PROJ_LAT_0 = 80.633
PROJ_LON_0 = -66.800

# original extent, matches Fig. 1b (paper's overview map):
# LAT_MIN, LAT_MAX = 77.60, 84.00
# LON_MIN, LON_MAX = -90.0, -50.0

# close-up extent: islands in Kennedy Channel
LAT_MIN, LAT_MAX = 80.43, 80.94
LON_MIN, LON_MAX = -68.2, -65.8

OUTPUT_WIDTH_PX = 3000
# ZOOM = 7   # was tuned for the original ~450 km-wide extent above
ZOOM = 12  # much smaller close-up extent needs a higher zoom for real detail

TILE_URL = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
SUBDOMAINS = "abc"
USER_AGENT = "iceMPM_v5-fig1-nares-strait/1.0 (research figure prep)"

# Anchored to this script's own directory (not the caller's cwd), so this
# can never land in _paper/ or anywhere else outside _python_code/.
OUTPUT_FILE = str(FilePath(__file__).resolve().parent / "fig_kennedy_channel_basemap_opentopo.png")


def main():
    make_basemap(TILE_URL, SUBDOMAINS, USER_AGENT, PROJ_LAT_0, PROJ_LON_0,
                 LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, OUTPUT_WIDTH_PX, ZOOM, OUTPUT_FILE)


if __name__ == "__main__":
    main()
