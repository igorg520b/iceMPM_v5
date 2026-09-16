import sys
import datetime
from pathlib import Path as FilePath

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from PIL import Image
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.colors import LinearSegmentedColormap
from rasterio.transform import Affine

sys.path.insert(0, str(FilePath(__file__).resolve().parent.parent / "image_download"))
import kane_basin_projection as kbp  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # our own generated basemap, not an untrusted upload

# --- Configuration ---
CURRENTS_FILE = "glo12v4_kane_basin_currents.nc"    # uo, vo (hourly)
TIDAL_FILE = "glo12v4_kane_basin_tidal_currents.nc"  # utide, vtide (hourly)
ICE_FILE = "glo12v4_kane_basin_ice_thickness.nc"     # sithick, ist (daily)

# Real satellite-imagery basemap, in the exact Kane Basin stereographic
# projection recorded by kane_basin_stereographic.py -- same ground-truth
# basemap approach used by visualize_carra1.py.
BASEMAP_DIR = FilePath(__file__).resolve().parent.parent / "image_download" / "nares_kane_basin_stereo" / "2024-06-08"
BASEMAP_IMAGE = BASEMAP_DIR / "final_stereographic_image_2000px.png"
BASEMAP_EXTENTS_JSON = BASEMAP_DIR / "extents.json"

# Same fixed color scale as visualize_carra1.py's surface temperature panel,
# for direct visual comparison between the two datasets.
TEMP_C_MIN, TEMP_C_MAX = -12.0, 0.0
TEMP_CMAP = LinearSegmentedColormap.from_list("temp_blue_yellow", ["#08106b", "#ffe800"])

# Fixed sea ice thickness color scale, rainbow (blue to red), non-translucent.
ICE_THICKNESS_MIN, ICE_THICKNESS_MAX = 0.1, 3.0


def format_time(t_val):
    if hasattr(t_val, 'values'):
        t_val = t_val.values
    return str(t_val)[:16]


def load_basemap_extent():
    import json
    with open(BASEMAP_EXTENTS_JSON) as f:
        doc = json.load(f)
    transform = Affine(*doc["affine_transform"])
    orig_w, orig_h = doc["final_image_width"], doc["final_image_height"]
    x_min, y_max = transform * (0, 0)
    x_max, y_min = transform * (orig_w, orig_h)
    return [x_min, x_max, y_min, y_max]


def visualize():
    # 1. Setup projection -- same Kane Basin stereographic basemap as visualize_carra1.py.
    projection = ccrs.Stereographic(
        central_latitude=kbp.PROJECTION_CENTER_LAT,
        central_longitude=kbp.PROJECTION_CENTER_LON,
        globe=ccrs.Globe(ellipse=None, semimajor_axis=kbp.EARTH_RADIUS_M, semiminor_axis=kbp.EARTH_RADIUS_M),
    )
    img_extent = load_basemap_extent()
    basemap_img = Image.open(BASEMAP_IMAGE)

    # 2. Load GLORYS data. uo/vo and utide/vtide share the same hourly time
    # axis and grid, so merge them; sea ice thickness/temperature is daily,
    # on its own time axis, so it's kept separate and matched to the nearest day.
    print(f"Loading {CURRENTS_FILE} and {TIDAL_FILE}...")
    ds_currents = xr.open_dataset(CURRENTS_FILE).isel(depth=0)
    ds_tide = xr.open_dataset(TIDAL_FILE).isel(depth=0)
    # The two files were downloaded separately and their lat/lon coordinate
    # arrays differ at float32 rounding noise (~1e-5 deg) despite being the
    # same requested grid -- snap tide's coords onto currents' before
    # merging, or xr.merge silently outer-joins them into a bogus, much
    # larger, mostly-NaN grid instead of raising an error.
    ds_tide = ds_tide.assign_coords(latitude=ds_currents.latitude, longitude=ds_currents.longitude)
    ds = xr.merge([ds_currents, ds_tide])
    print(f"Loading {ICE_FILE}...")
    ice_ds = xr.open_dataset(ICE_FILE)
    ice_ds = ice_ds.assign_coords(latitude=ds_currents.latitude, longitude=ds_currents.longitude)

    uo_da, vo_da = ds['uo'], ds['vo']
    utide_da, vtide_da = ds['utide'], ds['vtide']
    sithick_da = ice_ds['sithick']
    ist_da = ice_ds['ist']  # sea ice surface temperature
    ocean_speed_da = np.sqrt(uo_da ** 2 + vo_da ** 2)
    tide_speed_da = np.sqrt(utide_da ** 2 + vtide_da ** 2)

    times = ds['time'].values          # hourly master time axis
    ice_times = ice_ds['time'].values  # daily
    num_times = len(times)

    def nearest_ice_idx(time_idx):
        return int(np.argmin(np.abs(ice_times - times[time_idx])))

    lons = ds['longitude'].values  # 1D, regular grid
    lats = ds['latitude'].values   # 1D

    lat_mask = (lats >= kbp.LAT_MIN) & (lats <= kbp.LAT_MAX)
    lon_mask = (lons >= kbp.LON_MIN) & (lons <= kbp.LON_MAX)

    def data_range(da):
        full = da.values
        masked = full[:, lat_mask, :][:, :, lon_mask]
        return float(np.nanmin(masked)), float(np.nanmax(masked))

    # panels: label -> dict(get(idx)->2D array, cmap, scale_mode, quiver (u_da,v_da) or None)
    panels = {
        "Ocean current U (m/s)": dict(
            get=lambda idx: uo_da.isel(time=idx).values, cmap='RdBu_r', scale_mode='diverging', quiver=None, da=uo_da),
        "Ocean current V (m/s)": dict(
            get=lambda idx: vo_da.isel(time=idx).values, cmap='RdBu_r', scale_mode='diverging', quiver=None, da=vo_da),
        "Ocean current speed (m/s)": dict(
            get=lambda idx: ocean_speed_da.isel(time=idx).values, cmap='Blues', scale_mode='sequential',
            quiver=(uo_da, vo_da), da=ocean_speed_da),
        "Tidal current U (m/s)": dict(
            get=lambda idx: utide_da.isel(time=idx).values, cmap='RdBu_r', scale_mode='diverging', quiver=None, da=utide_da),
        "Tidal current V (m/s)": dict(
            get=lambda idx: vtide_da.isel(time=idx).values, cmap='RdBu_r', scale_mode='diverging', quiver=None, da=vtide_da),
        "Tidal current speed (m/s)": dict(
            get=lambda idx: tide_speed_da.isel(time=idx).values, cmap='Blues', scale_mode='sequential',
            quiver=(utide_da, vtide_da), da=tide_speed_da),
        "Sea ice thickness (m)": dict(
            get=lambda idx: sithick_da.isel(time=nearest_ice_idx(idx)).values, cmap='rainbow', scale_mode='fixed',
            quiver=None, da=sithick_da, fixed_range=(ICE_THICKNESS_MIN, ICE_THICKNESS_MAX), alpha=1.0),
        "Sea ice surface temp (°C)": dict(
            get=lambda idx: ist_da.isel(time=nearest_ice_idx(idx)).values, cmap=TEMP_CMAP, scale_mode='fixed',
            quiver=None, da=ist_da, fixed_range=(TEMP_C_MIN, TEMP_C_MAX)),
    }
    labels = list(panels.keys())

    ranges = {}
    for label, p in panels.items():
        if p['scale_mode'] == 'fixed':
            ranges[label] = p['fixed_range']
            continue
        dmin, dmax = data_range(p['da'])
        if p['scale_mode'] == 'diverging':
            vabs = max(abs(dmin), abs(dmax))
            ranges[label] = (-vabs, vabs)
        else:
            ranges[label] = (dmin, dmax)

    # 3. Figure layout: nearly full-window map, two rows of buttons to
    # switch fields (8 of them), a data-visibility checkbox, zoom-reset and
    # histogram buttons, and a time slider.
    fig = plt.figure(figsize=(11, 11))
    ax = plt.axes([0.05, 0.15, 0.9, 0.79], projection=projection)
    ax.set_extent(img_extent, crs=projection)
    ax.imshow(basemap_img, origin='upper', extent=img_extent, transform=projection, zorder=0)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--', zorder=4)
    gl.top_labels = False
    gl.right_labels = False

    quiver_skip = 4  # GLORYS grid is coarser (~7km) than CARRA's, needs less thinning
    state = {"label": labels[0], "time_idx": 0, "mesh": None, "cbar": None, "quiver": None}
    visibility = {"data": True}

    def apply_visibility():
        if state["mesh"] is not None:
            state["mesh"].set_visible(visibility["data"])
        if state["quiver"] is not None:
            state["quiver"].set_visible(visibility["data"])
        if state["cbar"] is not None:
            state["cbar"].ax.set_visible(visibility["data"])
        fig.canvas.draw_idle()

    def draw(label, time_idx):
        p = panels[label]
        vmin, vmax = ranges[label]
        data_slice = p['get'](time_idx)

        if state["mesh"] is not None:
            state["mesh"].remove()
        if state["quiver"] is not None:
            state["quiver"].remove()
            state["quiver"] = None

        mesh = ax.pcolormesh(
            lons, lats, data_slice, transform=ccrs.PlateCarree(),
            cmap=p['cmap'], vmin=vmin, vmax=vmax, shading='auto', alpha=p.get('alpha', 0.7), zorder=1
        )
        state["mesh"] = mesh

        if p['quiver'] is not None:
            u_da, v_da = p['quiver']
            u_vals = u_da.isel(time=time_idx).values[::quiver_skip, ::quiver_skip]
            v_vals = v_da.isel(time=time_idx).values[::quiver_skip, ::quiver_skip]
            X_q, Y_q = np.meshgrid(lons[::quiver_skip], lats[::quiver_skip])
            state["quiver"] = ax.quiver(
                X_q, Y_q, u_vals, v_vals, transform=ccrs.PlateCarree(),
                color='black', alpha=0.8, width=0.002, zorder=2
            )

        if state["cbar"] is None:
            state["cbar"] = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
        else:
            state["cbar"].update_normal(mesh)
        state["cbar"].set_label(label)

        t_val = times[time_idx]
        if label in ("Sea ice thickness (m)", "Sea ice surface temp (°C)"):
            ice_t = ice_times[nearest_ice_idx(time_idx)]
            ax.set_title(f"GLO12v4 - {format_time(t_val)} (ice: nearest day {format_time(ice_t)})")
        else:
            ax.set_title(f"GLO12v4 - {format_time(t_val)}")
        apply_visibility()

    # Two rows of buttons to switch fields (8 panels, 4x2 grid).
    n = len(labels)
    ncols = 4
    btn_w = 0.9 / ncols
    row_y = [0.105, 0.058]  # top row, bottom row
    buttons = []
    for idx, label in enumerate(labels):
        row = idx // ncols
        col = idx % ncols
        bax = fig.add_axes([0.05 + col * btn_w, row_y[row], btn_w - 0.01, 0.04])
        btn = Button(bax, label, color='0.85', hovercolor='0.7')
        buttons.append(btn)

    def make_on_click(label):
        def on_click(event):
            state["label"] = label
            for b, lbl in zip(buttons, labels):
                b.ax.set_facecolor('0.6' if lbl == label else '0.85')
            draw(label, state["time_idx"])
        return on_click

    for btn, label in zip(buttons, labels):
        btn.on_clicked(make_on_click(label))
    buttons[0].ax.set_facecolor('0.6')

    ax_slider = fig.add_axes([0.15, 0.005, 0.7, 0.02])
    slider = Slider(ax_slider, 'Time', 0, max(num_times - 1, 0), valinit=0, valstep=1)
    slider.valtext.set_text(format_time(times[0]))  # show date/hour instead of the raw index

    def on_slider(val):
        idx = int(val)
        state["time_idx"] = idx
        slider.valtext.set_text(format_time(times[idx]))
        draw(state["label"], idx)

    slider.on_changed(on_slider)

    ax_check = fig.add_axes([0.80, 0.945, 0.19, 0.04])
    check = CheckButtons(ax_check, ['GLORYS data'], actives=[True])

    def on_check(label):
        visibility['data'] = not visibility['data']
        apply_visibility()

    check.on_clicked(on_check)

    # Zoom reset -- the toolbar's magnifying-glass tool only zooms in via a
    # rubber-band rectangle; this restores the original full-basemap extent
    # in one click (equivalent to the toolbar's Home button).
    ax_reset = fig.add_axes([0.80, 0.895, 0.19, 0.04])
    btn_reset = Button(ax_reset, 'Reset zoom', color='0.85', hovercolor='0.7')

    def on_reset_zoom(event):
        ax.set_extent(img_extent, crs=projection)
        fig.canvas.draw_idle()

    btn_reset.on_clicked(on_reset_zoom)

    # Histogram of the currently displayed field's values (current time
    # slice, cropped to the Kane Basin bbox, NaNs dropped) -- opens in a
    # separate window, reused across clicks.
    ax_hist_btn = fig.add_axes([0.80, 0.845, 0.19, 0.04])
    btn_hist = Button(ax_hist_btn, 'Histogram', color='0.85', hovercolor='0.7')
    hist_state = {"fig": None, "ax": None}

    def on_histogram(event):
        label = state["label"]
        data_slice = panels[label]['get'](state["time_idx"])
        masked = data_slice[lat_mask, :][:, lon_mask]
        valid = masked[~np.isnan(masked)]

        if hist_state["fig"] is None or not plt.fignum_exists(hist_state["fig"].number):
            hist_state["fig"], hist_state["ax"] = plt.subplots(figsize=(6, 4))

        hax = hist_state["ax"]
        hax.clear()
        if valid.size:
            hax.hist(valid, bins=40, color='steelblue', edgecolor='black')
        hax.set_xlabel(label)
        hax.set_ylabel('Count')
        t_val = times[state["time_idx"]]
        hax.set_title(f"{label}\n{format_time(t_val)}  (n={valid.size})")
        hist_state["fig"].tight_layout()
        hist_state["fig"].canvas.draw_idle()
        hist_state["fig"].show()

    btn_hist.on_clicked(on_histogram)

    # Live hover readout of the ice/current quantities at the nearest
    # GLORYS grid cell under the cursor, regardless of which panel is
    # currently displayed.
    def nearest_grid_index(lon, lat):
        i = int(np.argmin(np.abs(lons - lon)))
        j = int(np.argmin(np.abs(lats - lat)))
        return j, i

    hover_text = ax.annotate(
        "", xy=(0.01, 0.99), xycoords='axes fraction', fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), zorder=12
    )

    def on_hover(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            if hover_text.get_text():
                hover_text.set_text("")
                fig.canvas.draw_idle()
            return

        hover_lon, hover_lat = ccrs.PlateCarree().transform_point(event.xdata, event.ydata, src_crs=projection)
        j, i = nearest_grid_index(hover_lon, hover_lat)

        ice_idx = nearest_ice_idx(state["time_idx"])
        ice_val = float(sithick_da.isel(time=ice_idx).values[j, i])
        ist_val = float(ist_da.isel(time=ice_idx).values[j, i])
        ocean_speed_val = float(ocean_speed_da.isel(time=state["time_idx"]).values[j, i])
        tide_speed_val = float(tide_speed_da.isel(time=state["time_idx"]).values[j, i])
        ice_str = "nan (open water/no ice)" if np.isnan(ice_val) else f"{ice_val:.3f} m"
        ist_str = "nan (open water/no ice)" if np.isnan(ist_val) else f"{ist_val:.2f} °C"

        hover_text.set_text(
            f"lat={hover_lat:.3f}, lon={hover_lon:.3f}\n"
            f"Sea ice thickness: {ice_str}\n"
            f"Sea ice surface temp: {ist_str}\n"
            f"Ocean current speed: {ocean_speed_val:.3f} m/s\n"
            f"Tidal current speed: {tide_speed_val:.3f} m/s"
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    draw(state["label"], state["time_idx"])
    print("Visualization ready. Hover over the map to inspect values. Showing plot...")
    plt.show()


if __name__ == "__main__":
    visualize()
