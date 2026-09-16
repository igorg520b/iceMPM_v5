import sys
from pathlib import Path

import copernicusmarine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "image_download"))
import kane_basin_projection as kbp  # noqa: E402

# Kane Basin / Nares Strait region (same bbox as the satellite-imagery pipeline)
LAT_MIN, LAT_MAX = kbp.LAT_MIN, kbp.LAT_MAX
LON_MIN, LON_MAX = kbp.LON_MIN, kbp.LON_MAX

# Same period as the GLORYS (GLO12v4) download, for direct comparison.
START_DATETIME = "2024-06-07T00:00:00"
END_DATETIME = "2024-06-24T00:00:00"  # exclusive upper bound -> covers through end of June 23

SURFACE_DEPTH = 0.0  # TOPAZ5's shallowest model level is exactly 0 m

print("Kane Basin region:")
print(f"  Lat: {LAT_MIN} to {LAT_MAX}")
print(f"  Lon: {LON_MIN} to {LON_MAX}")
print(f"  Time: {START_DATETIME} to {END_DATETIME}")

# TOPAZ5 (Arctic Ocean Physics Analysis and Forecast, met.no/HYCOM+CICE) --
# ocean current (vxo, vyo) and sea ice thickness (sithick) are both on the
# same hourly-instantaneous dataset, unlike GLORYS which needed two separate
# datasets. Native grid is a 6km north-pole stereographic projection, but
# copernicusmarine's subset() regrids to a regular lat/lon grid on request,
# same as GLORYS.
print("\nDownloading ocean current + sea ice thickness...")
copernicusmarine.subset(
    dataset_id="cmems_mod_arc_phy_anfc_6km_detided_PT1H-i",
    variables=["vxo", "vyo", "sithick"],
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    start_datetime=START_DATETIME,
    end_datetime=END_DATETIME,
    minimum_depth=SURFACE_DEPTH,
    maximum_depth=SURFACE_DEPTH,
    output_filename="topaz5_kane_basin_currents_ice.nc",
    force_download=True,
)
print("Saved 'topaz5_kane_basin_currents_ice.nc'.")

print("\nDownload complete.")
