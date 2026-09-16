# Getting and checking the simulation's input data

This is the short version, for a new user who just wants to download the
satellite imagery and reanalysis data the simulation needs, and check it
looks right before running `preparer`. It covers three scripts and their
viewers — nothing else. For every other script in `_python_code/`
(figure generation, calibration, alternate/legacy data sources), see
[`README.md`](README.md) in this directory.

## Setup

```bash
cd _python_code
source .venv/bin/activate   # pre-built venv; pip-install the same packages if starting fresh
```

You'll need free API credentials for two or three services, depending on
which of the scripts below you use. None of the scripts hold credentials
themselves — each library reads them from its own config file outside the
repo, so nothing secret is ever committed.

- **CDS / CARRA1** (atmosphere): account at
  https://cds.climate.copernicus.eu, then get your API key from your
  profile page and write `~/.cdsapirc`:
  ```
  url: https://cds.climate.copernicus.eu/api
  key: YOUR-UID:YOUR-API-KEY
  ```
- **Copernicus Marine / GLO12** (ocean): account at
  https://data.marine.copernicus.eu, then run once:
  ```bash
  copernicusmarine login
  ```
- **Copernicus Data Space Ecosystem / Sentinel Hub** (satellite imagery):
  create an OAuth client at https://dataspace.copernicus.eu (Sentinel Hub
  dashboard → User Settings → OAuth clients), then run once:
  ```bash
  sentinelhub.config --sh_client_id YOUR_ID --sh_client_secret YOUR_SECRET \
      --sh_base_url https://sh.dataspace.copernicus.eu \
      --sh_token_url https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
  ```

## 1. Downloading satellite imagery

`image_download/kane_basin_stereographic.py` fetches a Sentinel-2 true-color
mosaic for one date, reprojects it into the simulation's own stereographic
projection, and writes:

- `final_stereographic_image.png` (20000 px wide) — the source basemap
  image the simulation's initial ice state/color image is built from
- `final_stereographic_image_2000px.png` — a downsampled copy, handy for
  quick viewing
- `extents.json` — the pixel↔lat/lon affine transform sidecar

under `image_download/nares_kane_basin_stereo/<date>/`. Edit
`TIME_INTERVAL` near the top of the script for a different date, then run
it directly (`python3 kane_basin_stereographic.py`).

## 2. Downloading reanalysis data (CARRA1 + GLO12)

The date range downloaded must cover the period the simulation actually
models: `PreSimulationStartDate` through `SimulationEndDate` in the
simulation's config (see the top-level `README.md`'s
[Configuration](../README.md#configuration-json-files) section). Both
scripts below are already set to the paper's own hindcast period (May 1 –
June 30, 2024); edit their date variables if you want a different period,
and keep the two in sync with each other and with your `simulation.json`.

- **`carra1/download_carra1_data.py`** — CARRA1 atmospheric reanalysis (10 m
  wind u/v, skin temperature, sea ice thickness) over the west Arctic
  domain, via the CDS API. Writes one `.nc` file per month
  (`carra1_202405.nc`, `carra1_202406.nc`, ...) to the directory you run it
  from. Edit `YEAR`/`MONTHLY_DAY_RANGES` for a different period.

- **`glorys/download_glorys_all.py`** — GLO12v4 (Mercator Ocean) ocean data
  over the Kane Basin/Nares Strait bounding box, via `copernicusmarine`.
  Writes three files: `glo12v4_kane_basin_currents.nc` (surface current),
  `glo12v4_kane_basin_tidal_currents.nc` (tidal current), and
  `glo12v4_kane_basin_ice_thickness.nc` (sea ice thickness + surface
  temperature). Edit `START_DATETIME`/`END_DATETIME` for a different
  period.

Run either script directly (`python3 download_carra1_data.py`, etc.); each
prints progress as it downloads. Put the resulting `.nc` files wherever you
like — they don't need to be inside the repo — then point your
`simulation.json`/`base_config.json` at them as described in the top-level
README.

## 3. Visualizing the downloaded data

Both viewers below are interactive (sliders/buttons through time) and need
a display; they overlay the reanalysis data on the real basemap so you can
sanity-check alignment and coverage before running `preparer`.

- **`carra1/visualize_carra1.py`** — viewer for a downloaded CARRA1 `.nc`
  file (wind, temperature, ice thickness).
- **`glorys/visualize_glorys_all.py`** — combined viewer for all three
  GLO12 files at once (currents, tidal currents, ice thickness/
  temperature). This is the current/preferred GLO12 viewer.

Edit the file path near the top of either script to point at your
downloaded `.nc` file(s), then run it directly.

## Next step

Once you have imagery + `.nc` files, wire them into a `simulation.json` and
run `preparer` — see the top-level [`README.md`](../README.md#running-the-pipeline-end-to-end).
