import sys
from pathlib import Path

import copernicusmarine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "image_download"))
import kane_basin_projection as kbp  # noqa: E402

# Kane Basin / Nares Strait region (same bbox as the satellite-imagery pipeline)
LAT_MIN, LAT_MAX = kbp.LAT_MIN, kbp.LAT_MAX
LON_MIN, LON_MAX = kbp.LON_MIN, kbp.LON_MAX

START_DATETIME = "2025-05-01T00:00:00"
END_DATETIME = "2025-06-30T00:00:00"

SURFACE_DEPTH = 0.49402499198913574  # shallowest model level, used as "surface" (matches download_glorys.py)

print("Kane Basin region:")
print(f"  Lat: {LAT_MIN} to {LAT_MAX}")
print(f"  Lon: {LON_MIN} to {LON_MAX}")
print(f"  Time: {START_DATETIME} to {END_DATETIME}")

# 1. Ocean current (uo, vo) -- hourly-mean GLO12v4 dataset, surface depth level.
print("\nDownloading ocean current...")
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
    variables=["uo", "vo"],
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    start_datetime=START_DATETIME,
    end_datetime=END_DATETIME,
    minimum_depth=SURFACE_DEPTH,
    maximum_depth=SURFACE_DEPTH,
    output_filename="glo12v4_kane_basin_currents.nc",
    force_download=True,
)
print("Saved 'glo12v4_kane_basin_currents.nc'.")

# 2. Tidal current (utide, vtide) -- GLO12v4's SMOC (Surface Merged Ocean
# Current) dataset, same underlying product, decomposed to isolate the tidal
# drift component. Surface-only, no depth dimension.
print("\nDownloading tidal current...")
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
    variables=["utide", "vtide"],
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    start_datetime=START_DATETIME,
    end_datetime=END_DATETIME,
    output_filename="glo12v4_kane_basin_tidal_currents.nc",
    force_download=True,
)
print("Saved 'glo12v4_kane_basin_tidal_currents.nc'.")

# 3. Sea ice thickness (sithick) + sea ice surface temperature (ist) -- both
# only available on GLO12v4's daily-mean dataset, not the hourly one.
print("\nDownloading sea ice thickness + sea ice surface temperature...")
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
    variables=["sithick", "ist"],
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    start_datetime=START_DATETIME,
    end_datetime=END_DATETIME,
    output_filename="glo12v4_kane_basin_ice_thickness.nc",
    force_download=True,
)
print("Saved 'glo12v4_kane_basin_ice_thickness.nc' (sithick + ist).")

print("\nAll downloads complete.")
