#!/usr/bin/env python3
"""
Solve the 1D vertical ice heat-conduction problem for a single point, using
the 3-sine-mode analytical reduction described in heat_equation_spec.md
(section 1, formulas (R) and (U)).

Input: a CSV produced by preparer's "Export Temperature Time Series" tool
(Tools menu), with columns `timestamp_utc, epoch_unix, temperature_C` --
the CARRA1 surface temperature at one lat/lon location, one row per frame.

Output: three plots --
  1. Raw CARRA1 surface temperature vs. time, with the fixed ocean/bottom
     boundary temperature (Tb) shown for reference.
  2. Vertical ice temperature profiles T(z) at weekly intervals.
  3. Ice temperature vs. time at 1/4, 1/2, and 3/4 of the ice thickness.

This is a standalone single-point solver for exploration/validation only --
NOT the runtime simulation model (see heat_equation_spec.md, which reserves
the actual per-point/per-frame implementation for `preparer` spin-up and
`gplate`/`cplate` for a later task).

Configuration is via the plain variable assignments below (no command-line
arguments) -- edit these and rerun.
"""
import csv
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Configuration -- edit and rerun ---
CSV_PATH = "carra1_temp_80.0000_-69.0000.csv"   # CSV from preparer's "Export Temperature Time Series" tool
THICKNESS_M = 2.3       # m, ice thickness (point's reference thickness, held fixed)
N_MODES = 5             # number of sine modes in the (R)/(U) reduction
KAPPA_ICE = 1.1e-6      # m^2/s, ice thermal diffusivity
T_BOTTOM = -1.8         # deg C, fixed ocean/bottom boundary temperature
OUTDIR = None            # directory to save plots; None = alongside the CSV
SHOW_PLOTS = True       # open interactive windows in addition to saving PNGs

SECONDS_PER_WEEK = 7 * 86400


def load_csv(path):
    """Reads a preparer-exported temperature time series. Returns
    (timestamps [int64 seconds, ascending], temps_C [float64])."""
    timestamps, temps = [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(int(row['epoch_unix']))
            temps.append(float(row['temperature_C']))
    timestamps = np.array(timestamps, dtype=np.int64)
    temps = np.array(temps, dtype=np.float64)
    order = np.argsort(timestamps)
    return timestamps[order], temps[order]


def solve_amplitudes(timestamps, temps, H, n_modes, kappa):
    """
    Advances the per-point state (Ts_old, A_1..A_n) across the full time
    series using the exact closed-form update (U) -- see
    heat_equation_spec.md section 1. Unlike the runtime model (nominally
    hourly steps), this uses each CSV row's ACTUAL elapsed time as dt_th;
    (U) is exact for any step size, so no fixed cadence is required.

    Returns (Ts, A): Ts is (T,) -- just `temps` echoed back, one entry per
    row; A is (T, n_modes) -- the mode amplitudes recorded at each row's
    timestamp (A[0, :] = 0, matching heat_equation_spec.md section 6's
    "A_n = 0 at window start").
    """
    n = len(timestamps)
    A = np.zeros((n, n_modes))
    Ts = temps.copy()

    modes = np.arange(1, n_modes + 1)
    tau = (H * H) / (kappa * np.pi**2 * modes**2)   # tau_n, seconds
    b = 2.0 / (modes * np.pi)                         # b_n

    for i in range(1, n):
        dt = float(timestamps[i] - timestamps[i - 1])
        if dt <= 0:
            raise ValueError(f"Non-increasing/duplicate timestamp at row {i} "
                              f"({timestamps[i - 1]} -> {timestamps[i]})")
        r = (Ts[i] - Ts[i - 1]) / dt
        A_eq = -r * b * tau
        A[i, :] = (A[i - 1, :] - A_eq) * np.exp(-dt / tau) + A_eq

    return Ts, A


def reconstruct_T_series(z, Ts, A, H, Tb):
    """(R) evaluated at a FIXED depth z across a full (T,)/(T, n_modes)
    time series -- used for the "temperature at 1/4, 1/2, 3/4 thickness
    vs. time" plot."""
    modes = np.arange(1, A.shape[-1] + 1)
    sine_sum = A @ np.sin(modes * np.pi * z / H)   # (T,)
    return Ts + (Tb - Ts) * (z / H) + sine_sum


def reconstruct_profile(z, Ts, A, H, Tb):
    """(R) evaluated over an array of depths z at a SINGLE instant (scalar
    Ts, (n_modes,) A) -- used for the vertical-profile plot."""
    modes = np.arange(1, A.shape[-1] + 1)
    sine_sum = np.sin(np.outer(z, modes) * np.pi / H) @ A   # (n_z,)
    return Ts + (Tb - Ts) * (z / H) + sine_sum


def to_datetimes(epoch_seconds):
    return [datetime.datetime.fromtimestamp(int(t), tz=datetime.timezone.utc) for t in epoch_seconds]


def plot_raw_temperature(timestamps, temps, Tb, out_path, show):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(to_datetimes(timestamps), temps, color='steelblue', lw=1.2, label='CARRA1 surface temperature')
    ax.axhline(Tb, color='navy', ls='--', lw=1.0, label=f'Ocean / bottom boundary ({Tb:g} °C)')
    ax.set_xlabel('Date (UTC)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('CARRA1 surface temperature')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if not show:
        plt.close(fig)


def plot_weekly_profiles(timestamps, Ts, A, H, Tb, out_path, show):
    z = np.linspace(0.0, H, 101)
    week_marks = np.arange(timestamps[0], timestamps[-1] + 1, SECONDS_PER_WEEK)
    idxs = np.searchsorted(timestamps, week_marks)
    idxs = np.clip(idxs, 0, len(timestamps) - 1)

    fig, ax = plt.subplots(figsize=(6.5, 7))
    cmap = plt.get_cmap('viridis')
    for k, idx in enumerate(idxs):
        color = cmap(k / max(len(idxs) - 1, 1))
        profile = reconstruct_profile(z, Ts[idx], A[idx, :], H, Tb)
        date_label = to_datetimes([timestamps[idx]])[0].strftime('%Y-%m-%d')
        ax.plot(profile, z, color=color, lw=1.5, label=date_label)

    ax.axvline(Tb, color='navy', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Depth z (m), 0 = surface')
    ax.set_title(f'Vertical ice temperature profiles (weekly), H = {H:g} m')
    ax.invert_yaxis()  # surface at top, ocean bottom at bottom of the plot
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if not show:
        plt.close(fig)


def plot_depth_time_series(timestamps, Ts, A, H, Tb, out_path, show):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dates = to_datetimes(timestamps)

    ax.plot(dates, Ts, color='0.6', lw=0.8, ls='--', label='Surface (Ts)')
    for frac, color in ((0.25, 'tab:orange'), (0.5, 'tab:green'), (0.75, 'tab:red')):
        series = reconstruct_T_series(frac * H, Ts, A, H, Tb)
        ax.plot(dates, series, color=color, lw=1.4, label=f'z = {frac:g}·H')
    ax.axhline(Tb, color='navy', ls='--', lw=0.8, label=f'Ocean / bottom ({Tb:g} °C)')

    ax.set_xlabel('Date (UTC)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Ice temperature vs. time at 1/4, 1/2, 3/4 thickness, H = {H:g} m')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if not show:
        plt.close(fig)


def main():
    csv_path = Path(CSV_PATH)
    outdir = Path(OUTDIR) if OUTDIR else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    timestamps, temps = load_csv(csv_path)
    if len(timestamps) < 2:
        raise ValueError(f"Need at least 2 rows in {csv_path}, got {len(timestamps)}")

    print(f"Loaded {len(timestamps)} frames from {csv_path} "
          f"({to_datetimes([timestamps[0]])[0]} .. {to_datetimes([timestamps[-1]])[0]})")
    print(f"H={THICKNESS_M:g} m, modes={N_MODES}, kappa={KAPPA_ICE:g} m^2/s, Tb={T_BOTTOM:g} C")

    Ts, A = solve_amplitudes(timestamps, temps, THICKNESS_M, N_MODES, KAPPA_ICE)

    plot_raw_temperature(timestamps, temps, T_BOTTOM, outdir / f"{stem}_raw_temperature.png", SHOW_PLOTS)
    plot_weekly_profiles(timestamps, Ts, A, THICKNESS_M, T_BOTTOM,
                          outdir / f"{stem}_weekly_profiles.png", SHOW_PLOTS)
    plot_depth_time_series(timestamps, Ts, A, THICKNESS_M, T_BOTTOM,
                            outdir / f"{stem}_depth_time_series.png", SHOW_PLOTS)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
