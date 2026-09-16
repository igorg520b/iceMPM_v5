import sys
from pathlib import Path as FilePath

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from PIL import Image
from matplotlib.widgets import Slider, Button, CheckButtons
from rasterio.transform import Affine

sys.path.insert(0, str(FilePath(__file__).resolve().parent.parent / "image_download"))
import kane_basin_projection as kbp  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # our own generated basemap, not an untrusted upload

# --- Configuration ---
DATA_FILE = "topaz5_kane_basin_currents_ice.nc"  # vxo, vyo, sithick (hourly)

# Real satellite-imagery basemap, in the exact Kane Basin stereographic
# projection recorded by kane_basin_stereographic.py -- same ground-truth
# basemap approach used by visualize_carra1.py / visualize_glorys_all.py.
BASEMAP_DIR = FilePath(__file__).resolve().parent.parent / "image_download" / "nares_kane_basin_stereo" / "2024-06-08"
BASEMAP_IMAGE = BASEMAP_DIR / "final_stereographic_image_2000px.png"
BASEMAP_EXTENTS_JSON = BASEMAP_DIR / "extents.json"

# Fixed sea ice thickness color scale, rainbow (blue to red), non-translucent
# -- same as visualize_glorys_all.py, for direct visual comparison.
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
    # 1. Setup projection -- same Kane Basin stereographic basemap as the other scripts.
    projection = ccrs.Stereographic(
        central_latitude=kbp.PROJECTION_CENTER_LAT,
        central_longitude=kbp.PROJECTION_CENTER_LON,
        globe=ccrs.Globe(ellipse=None, semimajor_axis=kbp.EARTH_RADIUS_M, semiminor_axis=kbp.EARTH_RADIUS_M),
    )
    img_extent = load_basemap_extent()
    basemap_img = Image.open(BASEMAP_IMAGE)

    # 2. Load TOPAZ5 data. Ocean current and ice thickness are all on the
    # same hourly grid/time axis (unlike GLORYS, no merging or daily/hourly
    # matching needed).
    print(f"Loading {DATA_FILE}...")
    ds = xr.open_dataset(DATA_FILE)

    vxo_da, vyo_da = ds['vxo'], ds['vyo']
    sithick_da = ds['sithick']
    speed_da = np.sqrt(vxo_da ** 2 + vyo_da ** 2)

    times = ds['time'].values
    num_times = len(times)

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
            get=lambda idx: vxo_da.isel(time=idx).values, cmap='RdBu_r', scale_mode='diverging', quiver=None, da=vxo_da),
        "Ocean current V (m/s)": dict(
            get=lambda idx: vyo_da.isel(time=idx).values, cmap='RdBu_r', scale_mode='diverging', quiver=None, da=vyo_da),
        "Ocean current speed (m/s)": dict(
            get=lambda idx: speed_da.isel(time=idx).values, cmap='Blues', scale_mode='sequential',
            quiver=(vxo_da, vyo_da), da=speed_da),
        "Sea ice thickness (m)": dict(
            get=lambda idx: sithick_da.isel(time=idx).values, cmap='rainbow', scale_mode='fixed',
            quiver=None, da=sithick_da, fixed_range=(ICE_THICKNESS_MIN, ICE_THICKNESS_MAX), alpha=1.0),
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

    # 3. Figure layout: nearly full-window map, one row of buttons to switch
    # fields (4 of them), a data-visibility checkbox, zoom-reset and
    # histogram buttons, and a time slider.
    fig = plt.figure(figsize=(11, 11))
    ax = plt.axes([0.05, 0.15, 0.9, 0.79], projection=projection)
    ax.set_extent(img_extent, crs=projection)
    ax.imshow(basemap_img, origin='upper', extent=img_extent, transform=projection, zorder=0)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.6, linestyle='--', zorder=4)
    gl.top_labels = False
    gl.right_labels = False

    quiver_skip = 4  # TOPAZ5 grid is coarser (~6km) than CARRA's, needs less thinning
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

        ax.set_title(f"TOPAZ5 - {format_time(times[time_idx])}")
        apply_visibility()

    # One row of buttons to switch fields.
    n = len(labels)
    btn_w = 0.9 / n
    buttons = []
    for i, label in enumerate(labels):
        bax = fig.add_axes([0.05 + i * btn_w, 0.058, btn_w - 0.01, 0.05])
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

    ax_check = fig.add_axes([0.80, 0.945, 0.19, 0.04])
    check = CheckButtons(ax_check, ['TOPAZ5 data'], actives=[True])

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
        hax.set_title(f"{label}\n{format_time(times[state['time_idx']])}  (n={valid.size})")
        hist_state["fig"].tight_layout()
        hist_state["fig"].canvas.draw_idle()
        hist_state["fig"].show()

    btn_hist.on_clicked(on_histogram)

    # Live hover readout of ice thickness + current speed at the nearest
    # TOPAZ5 grid cell under the cursor, regardless of which panel is
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

        ice_val = float(sithick_da.isel(time=state["time_idx"]).values[j, i])
        speed_val = float(speed_da.isel(time=state["time_idx"]).values[j, i])
        ice_str = "nan (open water/no ice)" if np.isnan(ice_val) else f"{ice_val:.3f} m"

        hover_text.set_text(
            f"lat={hover_lat:.3f}, lon={hover_lon:.3f}\n"
            f"Sea ice thickness: {ice_str}\n"
            f"Ocean current speed: {speed_val:.3f} m/s"
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    draw(state["label"], state["time_idx"])
    print("Visualization ready. Hover over the map to inspect values. Showing plot...")
    plt.show()


if __name__ == "__main__":
    visualize()
