import cdsapi

DATASET = "reanalysis-carra-single-levels"

DOMAIN = "west_domain"
LEVEL_TYPE = "surface_or_atmosphere"
PRODUCT_TYPE = "analysis"
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "skin_temperature",      # surface temperature
    "sea_ice_thickness",
]
TIMES = [
    "00:00", "03:00", "06:00",
    "09:00", "12:00", "15:00",
    "18:00", "21:00",
]

# CDS requests take a
# single day list applied to every month in the request, so months with
# different day counts (or a partial final month like June here) are
# downloaded as separate requests.
YEAR = "2025"
MONTHLY_DAY_RANGES = {
    "05": range(1, 32),   # full May
    "06": range(1, 31),   # June
}

# Credentials come from ~/.cdsapirc (url/key), not hardcoded here.
client = cdsapi.Client()


for month, day_range in MONTHLY_DAY_RANGES.items():
    request = {
        "domain": DOMAIN,
        "level_type": LEVEL_TYPE,
        "variable": VARIABLES,
        "product_type": PRODUCT_TYPE,
        "time": TIMES,
        "year": [YEAR],
        "month": [month],
        "day": [f"{d:02d}" for d in day_range],
        "data_format": "netcdf",
    }
    target = f"carra1_{YEAR}{month}.nc"
    print(f"Requesting {YEAR}-{month}, days {day_range.start:02d}-{day_range.stop - 1:02d} -> {target}")
    client.retrieve(DATASET, request).download(target)
