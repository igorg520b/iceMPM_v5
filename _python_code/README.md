# `_python_code/` — data preparation and figure scripts

This directory holds every Python script used to (a) fetch the raw
reanalysis/imagery data the simulation needs, (b) turn it into the inputs
`preparer`/`simulation` actually read, and (c) build the paper's figures
from simulation output. Almost none of these take command-line arguments —
each script configures itself via plain variable assignments near the top
("Configuration — edit and rerun"); open the script, edit the constants,
run it directly (`python3 script.py`).

This file is aimed at a scientist trying to **reproduce the simulation's
inputs and the paper's key results from scratch**. The three sections under
"Reproducing the pipeline" below are the ones that matter for that; the
directory-by-directory reference after it documents everything else for
completeness.

Just want to download the imagery/reanalysis data and check it before
running `preparer`, without the full reference? See
[`GETTING_DATA.md`](GETTING_DATA.md) instead — it covers only the download
scripts and their viewers.

## Environment

A virtual environment already exists at `_python_code/.venv`. Activate it
before running anything here:

```bash
cd _python_code
source .venv/bin/activate
```

Its key packages: `cdsapi`, `copernicusmarine`, `sentinelhub`, `h5py`,
`numpy`, `matplotlib`, `pyproj`, `rasterio`, `cartopy`, `xarray`, `Pillow`.
If you're setting up a fresh environment instead, install those with pip.

### Credentials (one-time setup, per machine)

**The download scripts need API credentials that are personal to whoever
runs them.** None are included in this repo or in these scripts — you must
register your own (free) account with each service below and get your own
key/login. None of the scripts hardcode credentials — each library loads
them from its own saved config file outside the repo, so nothing secret
ever gets committed.

- **CDS / CARRA1** (`cdsapi`, used by `carra1/download_carra1_data.py`):
  create an account at https://cds.climate.copernicus.eu, get your API key
  from your profile page, then write `~/.cdsapirc`:
  ```
  url: https://cds.climate.copernicus.eu/api
  key: YOUR-UID:YOUR-API-KEY
  ```
- **Copernicus Marine / GLO12 & TOPAZ5** (`copernicusmarine`, used by
  `glorys/download_glorys_all.py` and `topaz5/download_topaz5.py`): create
  an account at https://data.marine.copernicus.eu, then run once:
  ```bash
  copernicusmarine login
  ```
  which prompts for username/password and stores a token under
  `~/.copernicusmarine/`.
- **Copernicus Data Space Ecosystem / Sentinel Hub imagery**
  (`sentinelhub`, used by `image_download/kane_basin_stereographic.py`):
  create an OAuth client at https://dataspace.copernicus.eu (Sentinel Hub
  dashboard → User Settings → OAuth clients), then run once:
  ```bash
  sentinelhub.config --sh_client_id YOUR_ID --sh_client_secret YOUR_SECRET \
      --sh_base_url https://sh.dataspace.copernicus.eu \
      --sh_token_url https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
  ```
  which saves everything to `~/.config/sentinelhub/config.toml`.

## Reproducing the pipeline

### 1. Obtaining CARRA1 (atmosphere) and GLO12 (ocean) data

**The downloaded date range must cover the period the simulation actually
models — check this yourself before running these scripts.** The
simulation config (`base_config.json` / your run's `simulation.json`)
defines this via `PreSimulationStartDate` (start of the thermal spin-up
lead-in — earlier than the simulation itself, needed so the ice's internal
temperature profile has time to equilibrate before `SimulationStartDate`),
`SimulationStartDate`, and `SimulationEndDate`. The `.nc` files need to
span at least `[PreSimulationStartDate, SimulationEndDate]`, or the
simulation will run out of forcing data partway through (or start with an
un-equilibrated thermal state). E.g. the values used for this project's
own hindcast were `PreSimulationStartDate = 2024-05-01`,
`SimulationEndDate = 2024-06-29` — matching `download_carra1_data.py`'s
`YEAR`/`MONTHLY_DAY_RANGES` (May 1 – June 29) and
`download_glorys_all.py`'s `START_DATETIME`/`END_DATETIME` (June 1 – 30;
GLO12 alone doesn't need the May lead-in since only CARRA1 drives the
thermal spin-up) below. If you change the modeled period, edit both the
config dates and these scripts' date ranges to match — the two are not
linked automatically.

- `carra1/download_carra1_data.py` — downloads CARRA1 reanalysis (10 m
  wind u/v, skin temperature, sea ice thickness) over the west Arctic
  domain via the CDS API, one `.nc` file per month
  (`carra1_202405.nc`, `carra1_202406.nc`, ...). Edit `YEAR` /
  `MONTHLY_DAY_RANGES` for a different period.
- `glorys/download_glorys_all.py` — downloads GLO12v4 (Mercator Ocean)
  surface ocean current (`glo12v4_kane_basin_currents.nc`), tidal current
  (`glo12v4_kane_basin_tidal_currents.nc`), and sea ice thickness +
  surface temperature (`glo12v4_kane_basin_ice_thickness.nc`) over the
  Kane Basin/Nares Strait bounding box (from
  `image_download/kane_basin_projection.py`), via `copernicusmarine`. Edit
  `START_DATETIME`/`END_DATETIME` for a different period.
- `topaz5/download_topaz5.py` — same idea but for TOPAZ5 (met.no/HYCOM+CICE
  Arctic reanalysis), used only as an independent cross-check of GLO12, not
  as a simulation input. Optional.

Each of these prints progress and writes its `.nc` file(s) into the
directory you run it from — run them from wherever you want the data to
land (see the next section).

### 2. Obtaining Copernicus imagery (initial ice state / basemaps)

- `image_download/kane_basin_projection.py` — not run directly; shared
  constants (bounding box, projection center 80°N/69°W, output resolution)
  imported by the three scripts below, so they always agree on the same
  geographic footprint and projection.
- `image_download/kane_basin_stereographic.py` — the main imagery
  downloader. Fetches Sentinel-2 true-color tiles for one date (edit
  `TIME_INTERVAL` near the top) from the Copernicus Data Space Ecosystem
  via `sentinelhub`, reprojects them into the project's own stereographic
  projection, and writes `final_stereographic_image.png` (20000 px wide)
  plus a downsampled `final_stereographic_image_2000px.png` and an
  `extents.json` sidecar (the pixel↔lat/lon affine transform) under
  `image_download/nares_kane_basin_stereo/<date>/`. This is the source
  image the simulation's initial ice state / color image is built from,
  and the ground-truth basemap several other figure scripts align against.
  Requires the Sentinel Hub credentials above.
- `image_download/generate_data_mask.py` — offline (no network/imagery
  needed): builds a black/white footprint mask of the download bounding
  box in the same projection/resolution, for use alongside the real
  imagery.
- `NASA_Worldview_download/download_nares_strait.py` and
  `download_and_register_nares.py` — an alternative MODIS/GIBS-based
  imagery source (different projection convention, orthographic). Not the
  path used for the paper's actual figures/simulation input; kept for
  reference. Requires `owslib`, which is **not** currently installed in
  `.venv` — install it separately if you need this path.

### 3. Wiring the downloaded `.nc` files into a simulation config

The scripts above just download data to wherever you run them from — they
don't know about the simulation's directory layout. To actually use the
data:

1. Put all the downloaded `.nc` files in one directory the simulation
   machine can read (anywhere is fine — it does **not** need to be inside
   the git repo; e.g. `base_config.json`'s own setup points at a sibling
   folder like `/home/.../_netcdf/`).
2. Point the relevant simulation config (typically `base_config.json`,
   inherited by every `simulation.json` via `"ParentConfig"`) at those
   files:
   ```json
   "CARRA1Data": ["/path/to/_netcdf/carra1_202405.nc", "/path/to/_netcdf/carra1_202406.nc"],
   "GLO12Data": "/path/to/_netcdf/glo12v4_kane_basin_currents.nc",
   "GLO12Tides": "/path/to/_netcdf/glo12v4_kane_basin_tidal_currents.nc",
   "GLO12ThicknessData": "/path/to/_netcdf/glo12v4_kane_basin_ice_thickness.nc"
   ```
   `CARRA1Data` takes a list (one or more monthly files, chronological
   order); the three `GLO12*` keys each take a single file. Paths may be
   absolute, or relative to the `.json` file's own directory — unlike
   `RuntimeDirectory`/output paths, these are resolved against the config
   file's location, not the run's output directory.
3. (Optional) `glorys/rasterize_glo12_thickness.py` turns GLO12's
   `sithick` field into a grayscale `ImageThicknessMask` PNG (black =
   `ThicknessFrom`, white = `ThicknessTo`, matching
   `preparer`'s `ThicknessFrom`/`ThicknessTo` convention) if you want
   GLO12-derived initial ice thickness instead of a uniform value.

After that, `preparer`/`simulation` will load the CARRA1/GLO12 data
directly from those paths — no further Python step is needed.

### 4. Producing the p–q plots and the region-diagnostics CSV plot

These both start from a completed (or in-progress) simulation run's
output, not from raw downloaded data.

- **p–q hexbin plots** (`pq_calibration/`) — visualize the trial stress
  state (pressure `p` vs. deviatoric stress `q`) the simulation reaches,
  against the fracture envelope:
  - `plot_pq_hexbin_single.py` — one snapshot, one panel. Edit
    `SIMULATION_JSON` to point at the run's `simulation.json`, and either
    `SNAPSHOT_FILE` (exact filename, e.g. `"f01188.h5"`),
    `SNAPSHOT_NUMBER` (just the integer frame number — resolved to the
    closest snapshot actually on disk), or leave both `None` for the
    latest snapshot found. Best used on a run with `"AllowFracture": false`
    (see the script's docstring) so the sampled (p, q) cloud isn't
    already clipped back onto the envelope. Writes
    `pq_hexbin_single.{svg,pdf,png}` next to the script — never copies
    into `_paper/`.
  - `plot_pq_hexbin_envelope.py` — same idea, multi-panel grid across
    several snapshots at once, for watching the cloud evolve over time.
    Writes into its own `hexbin_envelope/` subfolder.
  - `plot_new_piecewise_curves_interactive.py` — interactive
    (slider-driven) exploration of the piecewise-linear fracture/yield
    curve shape itself (not tied to any simulation output); needs a
    GUI-capable matplotlib backend and a display.

- **Region diagnostics CSV plot** (`fig_diagnostics/plot_region_diagnostics.py`)
  — first export the CSV from a running/finished simulation: open the
  `visualizer` tool, configure an `AnalysisRegionMask` for the region you
  want, then **File → Export Region Diagnostics**. That writes a CSV with
  per-frame masked-region drag stress, ice/current speed, ice strength,
  and fractured fraction. Then edit `CSV_PATH` at the top of
  `plot_region_diagnostics.py` to point at that CSV and run it — it
  produces a four-panel time series (`{OUT_NAME}.{svg,pdf,png}`, next to
  the script). It does not copy its output into `_paper/` — copy manually
  if you want to update the paper's own figure.

## Directory-by-directory script reference

### `carra1/`
- `download_carra1_data.py` — see above.
- `visualize_carra1.py` — interactive viewer (sliders/buttons) for a
  downloaded CARRA1 `.nc` file, overlaid on the real Kane Basin
  stereographic basemap for alignment checking.

### `glorys/` (GLO12 ocean/ice data)
- `download_glorys_all.py` — see above.
- `visualize_glorys_all.py` — interactive viewer combining currents, tidal
  currents, and ice thickness/temperature from the three GLO12 `.nc`
  files, overlaid on the real basemap. The current/preferred viewer.
- `visualize_glorys_currents.py`, `visualize_glorys_ice.py` — older,
  single-quantity viewers with a hardcoded orthographic projection
  (pre-dating `kane_basin_projection.py`); superseded by
  `visualize_glorys_all.py`, kept for reference.
- `rasterize_glo12_thickness.py` — see above (builds an
  `ImageThicknessMask` PNG from `sithick`).

### `topaz5/`
- `download_topaz5.py`, `visualize_topaz5.py` — TOPAZ5 download/viewer,
  same pattern as `glorys/`. Cross-check dataset only, not a simulation
  input.

### `image_download/`
- `kane_basin_projection.py`, `kane_basin_stereographic.py`,
  `generate_data_mask.py` — see above.
- `NASA_Worldview_download/download_nares_strait.py`,
  `download_and_register_nares.py` — see above (alternative MODIS
  imagery path, not used for the paper).

### `pq_calibration/`
- `plot_pq_hexbin_single.py`, `plot_pq_hexbin_envelope.py`,
  `plot_new_piecewise_curves_interactive.py` — see above.

### `fig_diagnostics/`
- `plot_region_diagnostics.py` — see above. `plot_region_diagnostics.nb`
  is the original Mathematica version this Python script replaced (kept
  for reference; mixes quantities of very different scale on one axis,
  which the Python version fixes with four separate panels).

### `fig_fracture_and_yield/`
- `fracture_yield_figures_mpl.py` — the paper's Figures 2/3: the
  piecewise-linear intact-fracture envelope and yield surface in p–q
  space, matching `kernels.cu`'s `PlasticProjection`/
  `Q_From_Yield_Surface` exactly. A from-scratch matplotlib
  recreation of `ice_fracture_yield_curves.nb` (Mathematica) in the
  paper's house style.
- `cf_softening_figure.py` — cohesion-softening curve
  `c_f(gamma_p)` (paper eq. 23) vs. accumulated plastic shear strain.

### `fig_thermal_profile/`
- `make_thermal_figure.py` — two-panel thermal-model figure built from a
  real exported CARRA1 surface-temperature time series (one lat/lon
  point): (a) surface forcing vs. reconstructed mid-depth temperature for
  two reference thicknesses, (b) full vertical profiles at representative
  dates. Uses `solve_heat_equation_point.py`'s solver.
- `make_strength_figure.py` — companion figure carrying the same
  reconstructed temperature through to estimated flexural strength vs.
  time (brine volume → Timco & O'Brien 1994), matching
  `partition_kernel_compute_ice_strength`'s exact quadrature.

### `fig_initial_state/`
- `add_frame_to_initial_state.py` — adds the striped lat/lon graticule
  frame to the initial-state color image (downsampled Sentinel-2 mosaic).
- `mark_projection_center.py` — marks the projection's tangent point
  (80°N, 69°W) with a red "+" on a copy of the simulation-regions image.

### `fig_nares_strait/` (introduction/overview basemap figure)
- `basemap_common.py` — shared tile-fetch/reproject machinery (not run
  directly).
- `make_basemap_osm.py`, `make_basemap_opentopo.py` — build a basemap PNG
  (OSM-no-labels or OpenTopoMap terrain) at the paper's Fig. 1b extent,
  writing a `.georef.json` sidecar.
- `add_frame_to_basemap.py`, `add_frame_to_basemap_rotated_labels.py` —
  add the striped neatline/graticule frame to one of those basemaps
  (rotated-label variant matches the reference paper's figure style).
- `add_capes_box_overlay.py` — variant that also draws a box connecting
  four named capes at the strait's north/south ends.

### `fig_results_comparison/`
- `crop_results_images.py` — crops matching simulation/satellite image
  pairs (June 8, June 10, June 19) to the same physical region for the
  paper's Results section.

### `graticule/`
- `add_graticule_frame.py` — general-purpose lat/lon graticule + striped
  neatline overlay for a full-resolution raster exported from
  `visualizer` (Tools → Render Full Grid Snapshot), using the run's own
  `simulation.json` projection parameters. Not wired into the C++/Qt GUI
  on purpose.
- `batch_process_figures.py` — runs the above over a batch of images, then
  shrinks each with `vips`.

### Top-level scripts
- `solve_heat_equation_point.py` — standalone single-point 1D vertical
  ice heat-conduction solver (3-sine-mode analytical reduction), for
  exploration/validation only — not the runtime model used by
  `preparer`/`gplate`/`cplate`. Input: a CSV from `preparer`'s "Export
  Temperature Time Series" tool.
- `plot_spinup_brine_profiles.py`, `plot_spinup_points_on_map.py` — debug
  viewers for `DataPreparer::ThermalSpinUpSave`'s console output (paste
  the printed line into `RAW_DATA_STRING`): per-point temperature/brine/
  strength profiles, and a map of sampled point locations.

### `mathematica/`
Original Mathematica notebooks (`.nb`) for the fracture/yield surfaces and
thermal spin-up, superseded by the matplotlib scripts above but kept for
reference/cross-checking.

### `_archived/`
Old scripts no longer in active use (`combined_4_multiple_dates.py`).
