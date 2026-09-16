#!/usr/bin/env python3
"""
Python replacement for plot_region_diagnostics.nb -- reads
region_diagnostics.csv (visualizer's "Export Region Diagnostics", File
menu) and plots it as four separate panels, one quantity family per
panel, instead of the notebook's single mixed-scale plot (avg/max levels
and wind/ocean speed sharing one axis despite being orders of magnitude
apart).

Panels, top to bottom:
    1. Wind and current drag stress (traction), mean over the masked
       region, Pa. Both are genuinely the same unit and a comparable
       order of magnitude (see PPMainWindow::exportRegionDiagnostics_triggered,
       visualizer_mainwindow.cpp -- same quadratic drag law the
       simulation itself uses, mass-weighted so zero-mass nodes don't
       contribute), so sharing one axis is correct here, unlike the
       notebook's speed/strength mixing.
    2. Mean ice speed alongside mean ocean current speed, m/s -- both on the
       same axis (not a dual/twin axis) since they are the same order of
       magnitude and the point is to compare them directly.
    3. Mean ice strength, kPa (ice_strength_mean is exported in Pa).
    4. Fractured fraction, as a percent of the masked region's mass.

Time axis: calendar dates (e.g. "Jun 06"), same DateFormatter convention
as fig_thermal_profile/make_thermal_figure.py and make_strength_figure.py,
rather than elapsed hours -- easier to line up against the observed
June 10 breakup.

Same Times New Roman / adjustable-FONT_SIZE house style as the other
paper figure scripts. Configuration is via the plain variable assignments
below (no command-line arguments) -- edit these and rerun.
"""
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------------------------------------------------
# Style -- Times New Roman throughout, matching make_thermal_figure.py /
# make_strength_figure.py / fracture_yield_figures_mpl.py. FONT_SIZE is
# the one knob to turn for a bigger/smaller figure -- everything else
# (legend, ticks) scales off it.
# ----------------------------------------------------------------------
FONT_SIZE = 15
LEGEND_FONT_SIZE = 15

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['mathtext.cal'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# --- Configuration -- edit and rerun ---
CSV_PATH = Path(__file__).resolve().parent / "region_diagnostics_2025.csv"

OUT_DIR = Path(__file__).resolve().parent
OUT_NAME = "region_diagnostics_test_run2"   # -> {OUT_NAME}.{svg,pdf,png}
# Note: unlike the paper's own figure run, this output is NOT copied into
# _paper/figures/ -- this CSV is a separate (20M-point, Wisteria) diagnostic
# run, and copying it would silently overwrite the paper's existing
# region_diagnostics_timeseries.{svg,pdf,png}, which come from the 82M-point
# production run.

FIGSIZE = (8.5, 10.0)   # inches, one column x four stacked panels
PNG_DPI = 200

# Panel label (a)/(b)/(c)/(d) position, in axes-fraction coordinates.
# X is negative to sit outside (left of) the axes frame; more negative
# moves it further left. Y=1.0 is the top of the axes.
PANEL_LABEL_X = -0.045
PANEL_LABEL_Y = 0.98

# Background shading for the three breakup states, identified automatically
# from fractured_fraction's rate of change -- see find_transition_window().
COLOR_INTACT = '#e2ffff'
COLOR_TRANSITION = '#fee2e2'
COLOR_FRAGMENTED = '#fbf4e1'

# Line colors.
COLOR_WIND = '#a67a13'
COLOR_CURRENT = '#3a6b9e'   # muted mid blue -- less saturated than pure '#0000ff'
COLOR_FRACTURED = '#b83732'
COLOR_ICE_SPEED = 'black'

# State-name labels drawn onto the fractured-fraction panel's background,
# explaining what the two colors mean. Position is in axes-fraction
# coordinates (0,0 = bottom-left, 1,1 = top-right of that panel) so it can
# be moved independently of the data -- the default sits each label
# roughly centered in its own background region for the June 6 test run,
# but that split point moves with wherever find_transition_window() lands
# for a different CSV, so re-check/move these if the label ends up
# straddling the transition band.
STATE_LABEL_FONT_SIZE = FONT_SIZE * 0.9
STATE_LABEL_COLOR = '#333333'
LABEL_INTACT_TEXT = 'Mostly intact'
LABEL_INTACT_XY = (0.13, 0.90)
LABEL_FRAGMENTED_TEXT = 'Fractured'
LABEL_FRAGMENTED_XY = (0.62, 0.20)

# Smoothing window (hours) applied to d(fractured_fraction)/dt before
# locating the transition -- suppresses per-sample jitter (~0.001% steps)
# that would otherwise register as spurious slope. Converted to a sample
# count using the CSV's actual (uniform) sampling interval.
SMOOTH_WINDOW_HOURS = 0.5

# The transition's start/end are defined as where the smoothed rate first
# rises above, and later falls back below, this fraction of its own peak
# value (a standard rise/fall-time convention) -- not a fixed percent
# threshold, since the total amount of fragmentation varies run to run.
REL_RATE_THRESHOLD = 0.05

# If the peak smoothed rate never exceeds this (percent of masked mass per
# hour), the run never shows a clear breakup transition (e.g. a
# fracture-disabled test run) -- skip the shading rather than flag a
# meaningless "transition" in near-flat noise.
MIN_PEAK_RATE_PCT_PER_HOUR = 0.05

# Column indices in region_diagnostics.csv, matching
# PPMainWindow::exportRegionDiagnostics_triggered (visualizer_mainwindow.cpp):
# frame,epoch_utc,datetime_utc,ice_strength_mean,ice_strength_max,
# wind_speed_mean,wind_speed_max,current_speed_mean,current_speed_max,
# ice_speed_mean,ice_speed_max,fractured_fraction,
# wind_drag_mean,wind_drag_max,current_drag_mean,current_drag_max
COL_EPOCH = 1
COL_ICE_STRENGTH_MEAN = 3
COL_CURRENT_SPEED_MEAN = 7
COL_ICE_SPEED_MEAN = 9
COL_FRACTURED_FRACTION = 11
COL_WIND_DRAG_MEAN = 12
COL_CURRENT_DRAG_MEAN = 14


def load_csv(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    return data


def to_datetimes(epoch_seconds):
    return [datetime.datetime.fromtimestamp(int(t), tz=datetime.timezone.utc) for t in epoch_seconds]


def find_transition_window(epoch, fractured_pct):
    """Locates the main breakup transition in fractured_pct, purely from
    its own rate of change -- no hardcoded dates or fraction values, so
    this works unchanged on any run's CSV, not just this particular one.

    Returns (start_idx, end_idx), the indices bracketing the transition
    (intact is [0, start_idx), transition is [start_idx, end_idx],
    fragmented is (end_idx, -1]), or None if no clear transition is found
    (e.g. a fracture-disabled run, where fractured_pct stays near zero).

    Method: smooth d(fractured_pct)/dt over SMOOTH_WINDOW_HOURS to
    suppress per-sample jitter, find its peak (the steepest point of the
    rise), then walk outward from that peak in both directions until the
    smoothed rate drops below REL_RATE_THRESHOLD of its own peak value --
    a standard rise/fall-time convention, self-scaling to however much
    fragmentation this particular run produces.
    """
    dt_s = np.median(np.diff(epoch))
    window_samples = max(1, int(round(SMOOTH_WINDOW_HOURS * 3600.0 / dt_s)))
    if window_samples % 2 == 0:
        window_samples += 1  # odd, so the box filter doesn't shift the signal

    t_hours = (epoch - epoch[0]) / 3600.0
    rate = np.gradient(fractured_pct, t_hours)  # percent per hour

    box = np.ones(window_samples) / window_samples
    rate_smooth = np.convolve(rate, box, mode='same')

    peak_idx = int(np.argmax(rate_smooth))
    peak_val = rate_smooth[peak_idx]
    if peak_val < MIN_PEAK_RATE_PCT_PER_HOUR:
        return None

    threshold = REL_RATE_THRESHOLD * peak_val

    start_idx = peak_idx
    while start_idx > 0 and rate_smooth[start_idx] > threshold:
        start_idx -= 1

    end_idx = peak_idx
    n = len(rate_smooth)
    while end_idx < n - 1 and rate_smooth[end_idx] > threshold:
        end_idx += 1

    return start_idx, end_idx


def add_panel_label(ax, letter):
    # Outside the axes frame (PANEL_LABEL_X < 0) -- no data behind it, so
    # no background box needed; clip_on=False keeps it from being cut off
    # by the axes' own clip box (savefig(bbox_inches='tight') still
    # expands the figure to include it).
    ax.text(PANEL_LABEL_X, PANEL_LABEL_Y, f'({letter})', transform=ax.transAxes,
             fontsize=FONT_SIZE, fontweight='bold', va='top', ha='right',
             clip_on=False, zorder=10)


def add_state_background(ax, dates, transition_window):
    if transition_window is None:
        ax.axvspan(dates[0], dates[-1], color=COLOR_INTACT, zorder=0)
        return
    start_idx, end_idx = transition_window
    ax.axvspan(dates[0], dates[start_idx], color=COLOR_INTACT, zorder=0)
    ax.axvspan(dates[start_idx], dates[end_idx], color=COLOR_TRANSITION, zorder=0)
    ax.axvspan(dates[end_idx], dates[-1], color=COLOR_FRAGMENTED, zorder=0)


def add_state_text_labels(ax):
    # Axes-fraction positions (LABEL_*_XY above), so these stay put
    # relative to the panel regardless of figure size -- reposition those
    # constants directly if a label ends up overlapping the data or
    # straddling the transition band for a different CSV/run.
    ax.text(*LABEL_INTACT_XY, LABEL_INTACT_TEXT, transform=ax.transAxes,
             fontsize=STATE_LABEL_FONT_SIZE, color=STATE_LABEL_COLOR,
             style='italic', ha='center', va='top', zorder=5)
    ax.text(*LABEL_FRAGMENTED_XY, LABEL_FRAGMENTED_TEXT, transform=ax.transAxes,
             fontsize=STATE_LABEL_FONT_SIZE, color=STATE_LABEL_COLOR,
             style='italic', ha='center', va='top', zorder=5)


def style_panel(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=FONT_SIZE * 0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def main():
    data = load_csv(CSV_PATH)
    dates = to_datetimes(data[:, COL_EPOCH])

    wind_drag = data[:, COL_WIND_DRAG_MEAN]
    current_drag = data[:, COL_CURRENT_DRAG_MEAN]
    ice_speed = data[:, COL_ICE_SPEED_MEAN]
    current_speed = data[:, COL_CURRENT_SPEED_MEAN]
    ice_strength_kpa = data[:, COL_ICE_STRENGTH_MEAN] / 1e3
    fractured_pct = data[:, COL_FRACTURED_FRACTION] * 100.0

    transition_window = find_transition_window(data[:, COL_EPOCH], fractured_pct)
    if transition_window is not None:
        start_idx, end_idx = transition_window
        print(f"Transition window: {dates[start_idx]} (start) to {dates[end_idx]} (end), "
              f"fractured_fraction {fractured_pct[start_idx]:.2f}% -> {fractured_pct[end_idx]:.2f}%")
    else:
        print("No clear transition found (fractured_fraction stays flat) -- shading the whole run as intact.")

    fig, (ax_drag, ax_speed, ax_strength, ax_fractured) = plt.subplots(
        4, 1, figsize=FIGSIZE, sharex=True)

    for ax, letter in zip((ax_drag, ax_speed, ax_strength, ax_fractured), 'abcd'):
        add_state_background(ax, dates, transition_window)
        add_panel_label(ax, letter)

    ax_drag.plot(dates, wind_drag, color=COLOR_WIND, lw=1.6, label='Wind', zorder=3)
    ax_drag.plot(dates, current_drag, color=COLOR_CURRENT, lw=1.6, label='Ocean current', zorder=3)
    style_panel(ax_drag, 'Drag stress (Pa)')
    ax_drag.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)

    ax_speed.plot(dates, ice_speed, color=COLOR_ICE_SPEED, lw=1.6, label='Ice', zorder=3)
    ax_speed.plot(dates, current_speed, color=COLOR_CURRENT, lw=1.1, ls='--', label='Ocean current', zorder=3)
    style_panel(ax_speed, 'Speed (m/s)')
    ax_speed.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)

    ax_strength.plot(dates, ice_strength_kpa, color='black', lw=1.6, zorder=3)
    style_panel(ax_strength, 'Ice strength (kPa)')

    ax_fractured.plot(dates, fractured_pct, color=COLOR_FRACTURED, lw=1.6, zorder=3)
    style_panel(ax_fractured, 'Fractured (%)')
    add_state_text_labels(ax_fractured)
    ax_fractured.set_xlabel('Date (UTC)', fontsize=FONT_SIZE)

    ax_fractured.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()
    fig.tight_layout()

    for ext, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": PNG_DPI})):
        out_path = OUT_DIR / f"{OUT_NAME}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", **kwargs)
        print(f"Wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
