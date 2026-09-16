#!/usr/bin/env python3
"""
Estimated flexural strength vs. time, companion to make_thermal_figure.py's
Figure 2 -- same real CARRA1 series, same two reference thicknesses, but
carrying the reconstructed temperature all the way through to strength
instead of stopping at temperature.

Brine volume (Frankenstein & Garner 1967) and flexural strength (Timco &
O'Brien 1994) match kernels.cu's brine_volume_frankenstein_garner /
flexural_strength_timco_obrien exactly (same formulas, same constants).
The depth average matches partition_kernel_compute_ice_strength's actual
quadrature exactly too: N_DEPTH=8, midpoints of 8 equal sub-layers,
z_k = (k+0.5)*H/8, plain unweighted average -- a midpoint rule, not
trapezoidal (see kernels.cu:715-724).

Salinity: IceSalinity = 6.0 ppt, base_config.json's actual configured value
(flat, not thickness-dependent -- the real simulation does not differentiate
salinity by thickness, so this figure does not either).

Usage:
    python3 make_strength_figure.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solve_heat_equation_point import load_csv, solve_amplitudes, reconstruct_profile
from make_thermal_figure import (to_datetimes, H_THIN, H_THICK, N_MODES, KAPPA_ICE, T_BOTTOM, CSV_PATH,
                                  COLOR_THIN, COLOR_THICK)

# --- Style -- matches make_thermal_figure.py / fracture_yield_figures_mpl.py ---
FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
TITLE_FONT_SIZE = 14

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
ICE_SALINITY = 6.0       # ppt -- base_config.json's IceSalinity, flat/uniform
N_DEPTH = 8               # midpoint rule, matches kernels.cu exactly (not trapezoidal)

OUT_DIR = Path(__file__).resolve().parent
PAPER_FIGURES_DIR = Path(__file__).resolve().parents[2] / "_paper" / "figures"
OUT_FILE = "fig_strength_vs_time.pdf"


def brine_volume_frankenstein_garner(T, S):
    """Matches kernels.cu's brine_volume_frankenstein_garner exactly."""
    return S * (49.185 / np.abs(T) + 0.532) * 1e-3


def flexural_strength_timco_obrien(vb):
    """Matches kernels.cu's flexural_strength_timco_obrien exactly. Returns MPa."""
    return 1.76 * np.exp(-5.88 * np.sqrt(vb))


def strength_series(timestamps, temps, H, salinity):
    """Reconstructed depth-averaged flexural strength at every timestamp in
    the series, for one reference thickness H -- the exact same 8-point
    midpoint quadrature as partition_kernel_compute_ice_strength."""
    Ts, A = solve_amplitudes(timestamps, temps, H, N_MODES, KAPPA_ICE)
    z = (np.arange(N_DEPTH) + 0.5) * H / N_DEPTH   # midpoints of N_DEPTH equal sub-layers
    sigma = np.empty(len(timestamps))
    for i in range(len(timestamps)):
        T_at_depths = reconstruct_profile(z, Ts[i], A[i, :], H, T_BOTTOM)
        vb = brine_volume_frankenstein_garner(T_at_depths, salinity)
        sigma[i] = flexural_strength_timco_obrien(vb).mean()
    return sigma


def main():
    timestamps, temps = load_csv(CSV_PATH)
    dates = to_datetimes(timestamps)

    sigma_thin = strength_series(timestamps, temps, H_THIN, ICE_SALINITY)
    sigma_thick = strength_series(timestamps, temps, H_THICK, ICE_SALINITY)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(dates, sigma_thin, color=COLOR_THIN, lw=1.6, label=f'$H$={H_THIN:g} m')
    ax.plot(dates, sigma_thick, color=COLOR_THICK, lw=1.6, label=f'$H$={H_THICK:g} m')
    ax.set_ylabel(r'Flexural strength $\sigma_{\mathrm{f}}$ (MPa)')
    ax.set_title('Depth-averaged flexural strength vs. time', fontsize=TITLE_FONT_SIZE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(OUT_DIR / OUT_FILE)
    fig.savefig(OUT_DIR / OUT_FILE.replace('.pdf', '.png'), dpi=200)
    print(f"Wrote {OUT_DIR / OUT_FILE}")

    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(OUT_DIR / OUT_FILE, PAPER_FIGURES_DIR / OUT_FILE)
    print(f"Copied to {PAPER_FIGURES_DIR / OUT_FILE}")


if __name__ == "__main__":
    main()
