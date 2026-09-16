"""
Downloads ocean current (uo, vo) and sea-ice thickness (sithick) for a date
range that predates the ANFC operational product's coverage window --
download_glorys_all.py's dataset_ids (cmems_mod_glo_phy_anfc_*) only go back
to 2022-06-01, confirmed by hitting CoordinatesOutOfDatasetBounds when
requesting an earlier date (e.g. 2018).

For anything earlier, Copernicus Marine's historical archive is a separate
product line: the GLORYS12V1 multi-year reanalysis
(GLOBAL_MULTIYEAR_PHY_001_030, covering 1993 to roughly the present, updated
yearly). Its single daily-mean dataset, cmems_mod_glo_phy_my_0.083deg_P1D-m,
already bundles both ocean current and sea-ice fields together (confirmed
via the product's own PUM, section "Details of datasets") -- unlike the
ANFC catalog, which splits these across separate dataset_ids. So this
script makes one subset() call instead of three, and writes one output file
that can be pointed at by BOTH "GLO12Data" and "GLO12ThicknessData" in a
simulation config (opening the same file twice for independent read-only
access is fine).

Two real differences from the ANFC-based pipeline, not just a dataset_id
swap:

1. Resolution: this dataset is daily-mean only (P1D-m). There is no hourly
   reanalysis equivalent, so ocean current forcing here has coarser time
   resolution than the 2024 hindcast's hourly (PT1H-m) currents.

2. No tidal currents: GLORYS12V1's own PUM states plainly, "Tidal
   constituents: Not taken into account." Tides are added into the ANFC
   *operational* system specifically (cmems_mod_glo_phy_anfc_merged-uv_PT1H-i,
   ANFC-only, same 2022-06-01 start), so there is no historical-reanalysis
   substitute available here. This script does not attempt to download a
   tidal-current file at all -- just omit (or set to "") the "GLO12Tides"
   key in the project's simulation.json; the simulation is robust to this
   (see SimParams::ParseFile's GLO12Tides handling).

Also note: the "ist" (ice surface temperature) variable download_glorys_all
also fetches from the ANFC ice dataset does not exist in this reanalysis
product's variable list (thetao, so, uo, vo, zos, mlotst, bottomT, siconc,
sithick, usi, vsi) -- not fetched here. It is unused by the C++ simulation
code regardless (only "sithick" is read from the thickness file).

Usage:
    python3 download_glorys12v1_reanalysis.py
"""
import sys
from pathlib import Path

import copernicusmarine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "image_download"))
import kane_basin_projection as kbp  # noqa: E402

# Kane Basin / Nares Strait region (same bbox as the satellite-imagery pipeline)
LAT_MIN, LAT_MAX = kbp.LAT_MIN, kbp.LAT_MAX
LON_MIN, LON_MAX = kbp.LON_MIN, kbp.LON_MAX

# 2018 breakup event: the breakup itself is May 29, but the simulation needs
# to start running somewhat earlier than that, so this range starts May 20
# rather than June 1.
START_DATETIME = "2018-05-20T00:00:00"
END_DATETIME = "2018-06-30T00:00:00"

SURFACE_DEPTH = 0.49402499198913574  # shallowest model level, same convention as download_glorys_all.py

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES = ["uo", "vo", "sithick"]
OUTPUT_FILENAME = "glo12v1_my_kane_basin_reanalysis.nc"

print("Kane Basin region:")
print(f"  Lat: {LAT_MIN} to {LAT_MAX}")
print(f"  Lon: {LON_MIN} to {LON_MAX}")
print(f"  Time: {START_DATETIME} to {END_DATETIME}")
print(f"  Dataset: {DATASET_ID} (daily mean, GLORYS12V1 multi-year reanalysis)")

print("\nDownloading ocean current (uo, vo) + sea-ice thickness (sithick)...")
copernicusmarine.subset(
    dataset_id=DATASET_ID,
    variables=VARIABLES,
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    start_datetime=START_DATETIME,
    end_datetime=END_DATETIME,
    minimum_depth=SURFACE_DEPTH,
    maximum_depth=SURFACE_DEPTH,  # ignored for sithick, which has no depth axis
    output_filename=OUTPUT_FILENAME,
    force_download=True,
)
print(f"Saved '{OUTPUT_FILENAME}' (uo, vo, sithick).")

print("\nNo tidal-current file was downloaded: GLORYS12V1 does not model tides")
print("(see this script's module docstring). Point both \"GLO12Data\" and")
print(f"\"GLO12ThicknessData\" at '{OUTPUT_FILENAME}' in the project's config, and")
print("leave \"GLO12Tides\" unset (or \"\").")
