import os
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NC_FILE = "carra1_202406.nc"

# Real satellite-imagery basemap, in the exact Kane Basin stereographic
# projection recorded by kane_basin_stereographic.py. Using this instead of
# cartopy's generalized/medium-resolution Natural Earth coastlines as ground
# truth for alignment checks: it's authoritative pixel-for-pixel because its
# projection (proj4 + affine transform) is exactly recorded in extents.json,
# rather than being an independently-sourced vector coastline that may not
# line up with CARRA to better than a few km.
BASEMAP_DIR = FilePath(__file__).resolve().parent.parent / "image_download" / "nares_kane_basin_stereo" / "2024-06-08"
BASEMAP_IMAGE = BASEMAP_DIR / "final_stereographic_image_2000px.png"
BASEMAP_EXTENTS_JSON = BASEMAP_DIR / "extents.json"

# Candidate variable names per quantity -- CDS converts CARRA GRIB to NetCDF
# via cfgrib, which uses ecCodes "shortName"s; these can vary slightly by
# CARRA data version, so try a few known aliases and use whichever is present.
VAR_CANDIDATES = {
    "u": ["u10", "10u"],
    "v": ["v10", "10v"],
    "ice_thickness": ["sithick", "icetk", "sit"],
    "surface_temperature": ["skt", "stl1"],
}

# Fixed surface-temperature color scale, in Celsius, clipped outside range.
TEMP_C_MIN, TEMP_C_MAX = -12.0, 0.0
TEMP_CMAP = LinearSegmentedColormap.from_list("temp_blue_yellow", ["#08106b", "#ffe800"])


def pick_variable(ds, quantity):
    for name in VAR_CANDIDATES[quantity]:
        if name in ds.variables:
            return name
    raise KeyError(
        f"Could not find a variable for '{quantity}' (tried {VAR_CANDIDATES[quantity]}). "
        f"Available variables: {list(ds.data_vars)}"
    )


def format_time(t_val):
    if hasattr(t_val, 'values'):
        t_val = t_val.values
    try:
        if isinstance(t_val, np.datetime64):
            return str(t_val)[:16]
        if np.issubdtype(type(t_val), np.integer) or np.issubdtype(type(t_val), np.floating):
            if t_val > 1e10:
                t_val = t_val / 1e9  # ns to s
            return datetime.datetime.fromtimestamp(t_val).strftime('%Y-%m-%d %H:%M')
        return str(t_val)[:16]
    except Exception:
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
    # 1. Setup projection -- the exact Kane Basin stereographic projection
    # the satellite-imagery basemap was rendered in (kane_basin_projection.py).
    projection = ccrs.Stereographic(
        central_latitude=kbp.PROJECTION_CENTER_LAT,
        central_longitude=kbp.PROJECTION_CENTER_LON,
        globe=ccrs.Globe(ellipse=None, semimajor_axis=kbp.EARTH_RADIUS_M, semiminor_axis=kbp.EARTH_RADIUS_M),
    )
    img_extent = load_basemap_extent()
    basemap_img = Image.open(BASEMAP_IMAGE)

    # 2. Load CARRA data
    print(f"Loading {NC_FILE}...")
    try:
        ds = xr.open_dataset(NC_FILE)
    except Exception:
        ds = xr.open_dataset(NC_FILE, decode_times=False)

    if 'expver' in ds.dims:
        ds = ds.isel(expver=0)

    var_u = pick_variable(ds, "u")
    var_v = pick_variable(ds, "v")
    var_ice = pick_variable(ds, "ice_thickness")
    var_temp = pick_variable(ds, "surface_temperature")

    u_da = ds[var_u]
    v_da = ds[var_v]
    ice_da = ds[var_ice]
    temp_c_da = ds[var_temp] - 273.15
    speed_da = np.sqrt(u_da ** 2 + v_da ** 2)

    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
    if time_dim not in ds.dims:
        dims = [d for d in u_da.dims if d not in ('x', 'y', 'latitude', 'longitude')]
        time_dim = dims[0] if dims else None
    times = ds[time_dim] if time_dim else np.array([0])
    num_times = len(times) if time_dim else 1

    def get_slice(da, idx):
        return da.isel({time_dim: idx}) if time_dim else da

    if 'longitude' in ds:
        ds['longitude'] = xr.where(ds['longitude'] > 180, ds['longitude'] - 360, ds['longitude'])
    lons = ds['longitude'].values if 'longitude' in ds else None
    lats = ds['latitude'].values if 'latitude' in ds else None

    # Restrict color-scale statistics + quiver overlay to points inside the
    # Kane Basin bbox, not the whole CARRA west_domain.
    bbox_mask = (
        (lats >= kbp.LAT_MIN) & (lats <= kbp.LAT_MAX) &
        (lons >= kbp.LON_MIN) & (lons <= kbp.LON_MAX)
    )

    # scale mode: 'diverging' = symmetric about 0; 'sequential' = actual
    # [min, max] of the data; 'fixed' = hard-coded [vmin, vmax], clipped.
    panels = {
        "10m U wind (m/s)": (u_da, 'RdBu_r', 'diverging', False, None),
        "10m V wind (m/s)": (v_da, 'RdBu_r', 'diverging', False, None),
        "Wind speed (m/s)": (speed_da, 'Blues', 'sequential', True, None),
        f"Sea ice thickness ({ice_da.attrs.get('units', 'm')})": (ice_da, 'Blues', 'sequential', False, None),
        "Surface temperature (°C)": (temp_c_da, TEMP_CMAP, 'fixed', False, (TEMP_C_MIN, TEMP_C_MAX)),
    }
    labels = list(panels.keys())

    def data_range(da):
        full_data = da.values if num_times > 1 else get_slice(da, 0).values
        masked = full_data[..., bbox_mask]
        return float(np.nanmin(masked)), float(np.nanmax(masked))

    ranges = {}
    for label, (da, cmap, scale_mode, show_quiver, fixed_range) in panels.items():
        if scale_mode == 'fixed':
            ranges[label] = fixed_range
            continue
        dmin, dmax = data_range(da)
        if scale_mode == 'diverging':
            vabs = max(abs(dmin), abs(dmax))
            ranges[label] = (-vabs, vabs)
        else:
            ranges[label] = (dmin, dmax)

    # 3. Figure layout: nearly full-window map, a row of buttons to switch
    # fields, checkboxes to toggle the CARRA/contour overlays, and a time slider.
    fig = plt.figure(figsize=(11, 11))
    ax = plt.axes([0.05, 0.14, 0.9, 0.8], projection=projection)
    ax.set_extent(img_extent, crs=projection)
    ax.imshow(basemap_img, origin='upper', extent=img_extent, transform=projection, zorder=0)

    # No separate coastline "contour" overlay: cartopy's built-in Natural
    # Earth coastlines turned out to be several km off from the real
    # satellite basemap in this remote high-Arctic area (a known limitation
    # of that medium-resolution, non-survey-grade dataset here) -- verified
    # by direct comparison, while CARRA and the basemap photo do agree with
    # each other. The photo itself is the accurate ground truth, so it's
    # kept as the sole geographic reference. Gridlines are pure lat/lon math
    # (not an external vector dataset), so they stay on as a lightweight,
    # always-accurate reference frame.
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--', zorder=4)
    gl.top_labels = False
    gl.right_labels = False

    quiver_skip = (slice(None, None, 8), slice(None, None, 8))
    state = {"label": labels[0], "time_idx": 0, "mesh": None, "cbar": None, "quiver": None}
    visibility = {"carra": True}

    def apply_visibility():
        if state["mesh"] is not None:
            state["mesh"].set_visible(visibility["carra"])
        if state["quiver"] is not None:
            state["quiver"].set_visible(visibility["carra"])
        if state["cbar"] is not None:
            state["cbar"].ax.set_visible(visibility["carra"])
        fig.canvas.draw_idle()

    def draw(label, time_idx):
        da, cmap, scale_mode, show_quiver, _ = panels[label]
        vmin, vmax = ranges[label]
        data_slice = get_slice(da, time_idx).values

        if state["mesh"] is not None:
            state["mesh"].remove()
        if state["quiver"] is not None:
            state["quiver"].remove()
            state["quiver"] = None

        mesh = ax.pcolormesh(
            lons, lats, data_slice, transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto', alpha=0.7, zorder=1
        )
        state["mesh"] = mesh

        if show_quiver:
            u_vals = get_slice(u_da, time_idx).values[quiver_skip]
            v_vals = get_slice(v_da, time_idx).values[quiver_skip]
            X_q = lons[quiver_skip]
            Y_q = lats[quiver_skip]
            state["quiver"] = ax.quiver(
                X_q, Y_q, u_vals, v_vals, transform=ccrs.PlateCarree(),
                color='black', alpha=0.8, width=0.002, zorder=2
            )

        if state["cbar"] is None:
            state["cbar"] = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
        else:
            state["cbar"].update_normal(mesh)
        state["cbar"].set_label(label)

        t_val = times[time_idx] if num_times > 1 else times[0]
        ax.set_title(f"CARRA1 - {format_time(t_val)}")
        apply_visibility()

    # Button row to switch fields (compact, doesn't eat into map space)
    n = len(labels)
    btn_w = 0.9 / n
    buttons = []
    for i, label in enumerate(labels):
        bax = fig.add_axes([0.05 + i * btn_w, 0.045, btn_w - 0.01, 0.05])
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

    # Non-tracking: while dragging, only update the date label (cheap); defer
    # the actual (expensive) redraw until the mouse button is released.
    def on_slider(val):
        idx = int(val)
        state["time_idx"] = idx
        slider.valtext.set_text(format_time(times[idx]))

    def on_release(event):
        if event.inaxes == ax_slider:
            draw(state["label"], state["time_idx"])

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Overlay toggle: view the raw basemap photo alone, or with CARRA data on top.
    ax_check = fig.add_axes([0.80, 0.945, 0.19, 0.04])
    check = CheckButtons(ax_check, ['CARRA1 data'], actives=[True])

    def on_check(label):
        visibility['carra'] = not visibility['carra']
        apply_visibility()

    check.on_clicked(on_check)

    # Histogram of the currently displayed field's values (current time
    # slice, cropped to the Kane Basin bbox, NaNs dropped) -- opens in a
    # separate window, reused across clicks.
    ax_hist_btn = fig.add_axes([0.80, 0.895, 0.19, 0.04])
    btn_hist = Button(ax_hist_btn, 'Histogram', color='0.85', hovercolor='0.7')
    hist_state = {"fig": None, "ax": None}

    def on_histogram(event):
        label = state["label"]
        da, cmap, scale_mode, show_quiver, _ = panels[label]
        data_slice = get_slice(da, state["time_idx"]).values
        masked = data_slice[bbox_mask]
        valid = masked[~np.isnan(masked)]

        if hist_state["fig"] is None or not plt.fignum_exists(hist_state["fig"].number):
            hist_state["fig"], hist_state["ax"] = plt.subplots(figsize=(6, 4))

        hax = hist_state["ax"]
        hax.clear()
        if valid.size:
            hax.hist(valid, bins=40, color='steelblue', edgecolor='black')
        hax.set_xlabel(label)
        hax.set_ylabel('Count')
        t_val = times[state["time_idx"]] if num_times > 1 else times[0]
        hax.set_title(f"{label}\n{format_time(t_val)}  (n={valid.size})")
        hist_state["fig"].tight_layout()
        hist_state["fig"].canvas.draw_idle()
        hist_state["fig"].show()

    btn_hist.on_clicked(on_histogram)

    # Debug aid: click anywhere on the map to compare (a) the lat/lon the
    # AXES/PROJECTION assigns to that pixel (what the basemap photo and
    # coastline are drawn against) with (b) the lat/lon CARRA itself reports
    # for its nearest grid cell there (what the data is drawn against).
    click_artists = {"click_marker": None, "grid_marker": None, "text": None}

    def on_map_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None or event.button != 1:
            return

        click_lon, click_lat = ccrs.PlateCarree().transform_point(
            event.xdata, event.ydata, src_crs=projection
        )

        dist2 = (lons - click_lon) ** 2 + (lats - click_lat) ** 2
        j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
        grid_lat, grid_lon = float(lats[j, i]), float(lons[j, i])

        da, cmap, scale_mode, show_quiver, _ = panels[state["label"]]
        val = float(get_slice(da, state["time_idx"]).values[j, i])

        dlat_km = (grid_lat - click_lat) * 111.0
        dlon_km = (grid_lon - click_lon) * 111.0 * np.cos(np.radians(click_lat))
        offset_km = float(np.hypot(dlat_km, dlon_km))

        print(f"--- click ---")
        print(f"  clicked pixel  -> lat={click_lat:.4f}, lon={click_lon:.4f}  (basemap photo / coastline reference)")
        print(f"  nearest CARRA grid cell (j={j}, i={i}) -> lat={grid_lat:.4f}, lon={grid_lon:.4f}, "
              f"{state['label']}={val:.3f}")
        print(f"  grid cell is {offset_km:.2f} km from the click (should be <~2 km, half the 2.5km grid spacing)")

        for key in ("click_marker", "grid_marker", "text"):
            if click_artists[key] is not None:
                click_artists[key].remove()
                click_artists[key] = None

        click_artists["click_marker"] = ax.plot(
            click_lon, click_lat, marker='x', color='red', markersize=12, mew=3,
            transform=ccrs.PlateCarree(), zorder=10
        )[0]
        click_artists["grid_marker"] = ax.plot(
            grid_lon, grid_lat, marker='+', color='lime', markersize=14, mew=3,
            transform=ccrs.PlateCarree(), zorder=10
        )[0]
        click_artists["text"] = ax.annotate(
            f"x click: {click_lat:.3f}, {click_lon:.3f}\n"
            f"+ CARRA grid pt: {grid_lat:.3f}, {grid_lon:.3f}\n"
            f"separation: {offset_km:.2f} km",
            xy=(0.01, 0.01), xycoords='axes fraction', fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), zorder=11
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_map_click)

    # Live hover readout: always shows sea ice thickness + surface temperature
    # (+ wind speed) at the nearest CARRA grid cell under the cursor,
    # regardless of which field is currently displayed -- lets you inspect
    # the raw thickness numbers directly (e.g. to judge how noisy/rough the
    # field is) without needing to switch panels or click.
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

        hover_lon, hover_lat = ccrs.PlateCarree().transform_point(
            event.xdata, event.ydata, src_crs=projection
        )
        dist2 = (lons - hover_lon) ** 2 + (lats - hover_lat) ** 2
        j, i = np.unravel_index(np.argmin(dist2), dist2.shape)

        ice_val = float(get_slice(ice_da, state["time_idx"]).values[j, i])
        temp_val = float(get_slice(temp_c_da, state["time_idx"]).values[j, i])
        speed_val = float(get_slice(speed_da, state["time_idx"]).values[j, i])
        ice_str = "nan (land/no ice)" if np.isnan(ice_val) else f"{ice_val:.3f} m"

        hover_text.set_text(
            f"lat={hover_lat:.3f}, lon={hover_lon:.3f}\n"
            f"Sea ice thickness: {ice_str}\n"
            f"Surface temp: {temp_val:.2f} °C\n"
            f"Wind speed: {speed_val:.2f} m/s"
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    draw(state["label"], state["time_idx"])
    print("Visualization ready. Hover to inspect values, click to check lat/lon alignment. Showing plot...")
    plt.show()


if __name__ == "__main__":
    visualize()
