#!/usr/bin/env python3
"""
Two-panel thermal-model figure for the paper, built from the REAL exported
CARRA1 surface-temperature series (../carra1_temp_80.0000_-69.0000.csv,
80N/69W, May 1 - Jun 29 2024) rather than the idealized piecewise-linear
worked example in heat_note_expanded.pdf -- reuses the exact solver in
../solve_heat_equation_point.py (the (R)/(U) formulas from
../heat_equation_spec.md), just imported rather than duplicated.

Times New Roman / pdf.fonttype 42 styling matches
fig2_3_fracture_and_yield/fracture_yield_figures_mpl.py, so this figure
looks consistent with Figures 4/5 in the paper rather than using
matplotlib's default font.

Panel (a): surface forcing T_s(t) alongside the reconstructed temperature at
the MID-DEPTH (z = H/2, spelled out explicitly in the legend -- not just
"H=..." -- since H alone reads ambiguously against panel (b), where H is the
column's full depth) for two reference thicknesses spanning the hindcast's
range (thin, H_THIN, and thick, H_THICK). Point of the whole section: thin
ice tracks the surface almost instantly, thick ice lags by weeks, because
each mode's relaxation time scales as H^2.

Panel (b): full vertical profiles T(z) at a few representative real dates
for one representative thickness (H_PROFILE) -- the interior "cold bulge"
shape (the lag) directly, rather than just its effect at one depth.

Usage:
    python3 make_thermal_figure.py
"""
import sys
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solve_heat_equation_point import (load_csv, solve_amplitudes,
                                        reconstruct_T_series, reconstruct_profile)

# --- Style -- matches fig2_3_fracture_and_yield/fracture_yield_figures_mpl.py ---
FONT_SIZE = 14           # base size (axis labels, tick labels) -- tune and rerun
LEGEND_FONT_SIZE = 14    # legend entries, both panels -- separate since legends are denser
TITLE_FONT_SIZE = 14     # panel titles, e.g. "(a) Forcing and mid-depth response" --
                         # separate because axes.titlesize defaults to a relative 'large'
                         # (~1.2x FONT_SIZE), which is why it looked oversized before this
                         # was its own knob; set explicitly on both set_title() calls below.

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
CSV_PATH = Path(__file__).resolve().parent.parent / "carra1_temp_80.0000_-69.0000.csv"
N_MODES = 5
KAPPA_ICE = 1.1e-6      # m^2/s
T_BOTTOM = -1.8         # deg C

H_THIN = 0.8            # m -- thin end of the hindcast's h0 range (Table 2)
H_THICK = 3.0           # m -- thick end
H_PROFILE = 3.0         # m -- thickness used for the panel (b) vertical profiles

# Line colors for H_THIN/H_THICK -- deliberately not matplotlib's default
# tab:orange/tab:blue (same pair used in make_strength_figure.py's
# sigma_thin/sigma_thick, so the two figures read as a matched set).
COLOR_THIN = '#c1440e'
COLOR_THICK = '#1a5276'

# May 1 (spin-up start), Jun 1, Jun 10 (observed breakup), Jun 20 -- deliberately
# NOT claimed as "the simulation end" anywhere (in code or caption): the actual
# hindcast length is a separate decision that may end up shorter.
PROFILE_DATES = ["2024-05-01", "2024-06-01", "2024-06-10", "2024-06-20"]

# Onset of the observed breakup, marked on panel (a) per the supervisor
# comment asking to show when the event occurs -- satellite imagery shows
# no motion before June 10, and visible cracking/motion from Kennedy
# Channel to Smith Sound that day. The specific time of day is not known,
# so this is placed at noon as a neutral marker for the date, not a
# claimed instant (the panel (a) time axis spans ~2 months, so sub-day
# precision would not be visible anyway).
BREAKUP_ONSET = "2024-06-10 12:00"

# One hand-picked color per PROFILE_DATES entry (same index order) -- not an
# auto colormap (viridis, etc.): those are exactly the "everyone recognizes
# this" look this figure is trying to avoid, same reasoning as COLOR_THIN/
# COLOR_THICK above. Roughly cool-to-warm as the season progresses, June 10
# (the observed breakup date) picked to stand out against the other three.
PROFILE_COLORS = ['#264653', '#2a9d8f', '#e76f51', '#8b2e2e']

OUT_DIR = Path(__file__).resolve().parent
PAPER_FIGURES_DIR = Path(__file__).resolve().parents[2] / "_paper" / "figures"
# Split into two separate files (panel a / panel b) rather than one combined
# figure -- lets the two be laid out independently when typesetting the paper.
OUT_FILE_A = "fig_thermal_profile_a.pdf"
OUT_FILE_B = "fig_thermal_profile_b.pdf"


def to_datetimes(epoch_seconds):
    return [datetime.datetime.fromtimestamp(int(t), tz=datetime.timezone.utc) for t in epoch_seconds]


def nearest_index(timestamps, date_str):
    target = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    target_epoch = int(target.timestamp())
    return int(np.argmin(np.abs(timestamps - target_epoch)))


def save_fig(fig, out_file):
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_file)
    fig.savefig(OUT_DIR / out_file.replace('.pdf', '.png'), dpi=200)
    print(f"Wrote {OUT_DIR / out_file}")

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(OUT_DIR / out_file, PAPER_FIGURES_DIR / out_file)
    print(f"Copied to {PAPER_FIGURES_DIR / out_file}")
    plt.close(fig)


def main():
    timestamps, temps = load_csv(CSV_PATH)
    dates = to_datetimes(timestamps)

    Ts_thin, A_thin = solve_amplitudes(timestamps, temps, H_THIN, N_MODES, KAPPA_ICE)
    Ts_thick, A_thick = solve_amplitudes(timestamps, temps, H_THICK, N_MODES, KAPPA_ICE)
    mid_thin = reconstruct_T_series(H_THIN / 2, Ts_thin, A_thin, H_THIN, T_BOTTOM)
    mid_thick = reconstruct_T_series(H_THICK / 2, Ts_thick, A_thick, H_THICK, T_BOTTOM)

    # Panel (b) uses its own solve (different H -> different tau_n, A) at H_PROFILE
    Ts_prof, A_prof = solve_amplitudes(timestamps, temps, H_PROFILE, N_MODES, KAPPA_ICE)
    z = np.linspace(0.0, H_PROFILE, 101)

    # --- Panel (a): forcing + mid-depth response, two thicknesses ---
    fig_a, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(dates, temps, color='0.55', lw=1.0, ls='--', label=r'Surface $T_{\mathrm{s}}(t)$ (CARRA1)')
    ax1.plot(dates, mid_thin, color=COLOR_THIN, lw=1.6,
              label=f'$H$={H_THIN:g} m, mid-depth ($z$={H_THIN/2:g} m)')
    ax1.plot(dates, mid_thick, color=COLOR_THICK, lw=1.6,
              label=f'$H$={H_THICK:g} m, mid-depth ($z$={H_THICK/2:g} m)')
    ax1.axhline(T_BOTTOM, color='navy', ls=':', lw=0.8, label=f'Ocean bottom ($T_{{\\mathrm{{b}}}}$={T_BOTTOM:g}$^\\circ$C)')
    onset_dt = datetime.datetime.strptime(BREAKUP_ONSET, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
    ax1.axvline(onset_dt, color='0.2', ls='-', lw=1.2, label='Observed breakup onset (June 10)')
    ax1.set_ylabel('Temperature ($^\\circ$C)')
    ax1.set_title('(a) Forcing and mid-depth response', fontsize=TITLE_FONT_SIZE)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.grid(alpha=0.3)
    ax1.legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
    fig_a.autofmt_xdate()
    save_fig(fig_a, OUT_FILE_A)

    # --- Panel (b): full vertical profiles at representative dates ---
    fig_b, ax2 = plt.subplots(figsize=(5.0, 4.5))
    for k, date_str in enumerate(PROFILE_DATES):
        idx = nearest_index(timestamps, date_str)
        profile = reconstruct_profile(z, Ts_prof[idx], A_prof[idx, :], H_PROFILE, T_BOTTOM)
        ax2.plot(profile, z, color=PROFILE_COLORS[k], lw=1.8, label=date_str)
    ax2.axvline(T_BOTTOM, color='navy', ls=':', lw=0.8)
    ax2.set_xlabel('Temperature ($^\\circ$C)')
    ax2.set_ylabel('Depth (m), 0 = surface')
    ax2.set_title(f'(b) Vertical profiles, $H$={H_PROFILE:g} m', fontsize=TITLE_FONT_SIZE)
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3)
    ax2.legend(loc='lower left', fontsize=LEGEND_FONT_SIZE)
    save_fig(fig_b, OUT_FILE_B)


if __name__ == "__main__":
    main()
