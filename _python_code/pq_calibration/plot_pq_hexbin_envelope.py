#!/usr/bin/env python3
"""
Plots the per-point (p, q) stress-state hexbin density for each snapshot,
with the fracture curves drawn on top -- to see how the observed stress
cloud sits relative to the yield surface. Hexbin only (no histograms, no
CSV) -- see plot_pq_snapshot.py for the fuller analysis this was split off
from.

Background: simulation/kernels.cu's partition_kernel_g2p normally writes
idx_stored_P/idx_stored_Q (rows 22/23 of the "pts_data" HDF5 dataset) only
once per point, at the step it first cracks. For this to show the full
population's current stress state (not just the few points that already
broke), that write needs to be temporarily changed to unconditional -- see
the "TEMPORARY" comment in kernels.cu's partition_kernel_g2p.

Two piecewise-linear curves, both from PlasticProjection/Q_From_Yield_Surface
in simulation/kernels.cu (post the ellipse -> piecewise-linear redesign):
- intact envelope: the "does this point crack at all" boundary. Tensile leg
  from (-IceTensileFailureStrength, 0) to (0, IceShearStrength); compression-
  shear leg IceShearStrength + tan(IceFractureAngle)*p for p>=0. No upper
  cutoff (the old IceCompressiveStrength/pmax hard cutoff was removed).
- fractured/yield surface: governs an already-cracked point -- min of a
  steep Drucker-Prager line from DP_threshold_p and a shallower second leg,
  meeting at a kink whose height is Cf * (where that DP line would otherwise
  cross the intact envelope's compression-shear leg). No upper cutoff either.
  See Q_From_Yield_Surface for the exact formula.

SIMULATION_JSON below is loaded through its ParentConfig chain (same
inheritance base_config.json -> simulation.json uses at runtime), so the
curve parameters (IceShearStrength, IceTensileFailureStrength,
IceFractureAngle, Cf, IceYieldFrictionAngle, DP_phi, DP_threshold_p) are
read directly from the config rather than hand-copied constants. Cf is used
as configured (c_f0) -- the Cf(gamma_p) softening in PlasticProjection is
per-point/runtime state, not something this static plot can reproduce.

Always plots the UseLiveIceStrength=false curve (fixed IceShearStrength/
IceTensileFailureStrength), regardless of what the config actually sets --
idx_xi is a per-point, time-varying live quantity with no single
representative value to plot a static curve against. main() prints a
warning if the resolved config's UseLiveIceStrength is actually true, since
the plotted curve won't then match what the simulation actually used.

Each snapshot gets two separate figures -- (a) all points against both
curves, (b) fractured points only (split on SimParams::status_cracked)
against the yield surface -- rather than one combined figure, for easier
typesetting later. Same Times New Roman / vector-output house style as
plot_pq_hexbin_single.py, no title (no timestamp, no point count).

Note: idx_stored_P/Q hold the *trial* (p, q) -- computed before
PlasticProjection runs, every step (see the "TEMPORARY" comment) -- so
points routinely land outside the envelope; that's the elastic predictor
that triggers the projection, not the post-projection state. Don't expect
the cloud to sit strictly inside the curve.

Configuration is via the plain variable assignments below (no command-line
arguments) -- edit these and rerun.
"""
import glob
import json
import os
import re

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Style -- Times New Roman throughout, matching plot_pq_hexbin_single.py /
# fracture_yield_figures_mpl.py / make_strength_figure.py. FONT_SIZE is the
# one knob to turn for a bigger/smaller figure -- everything else scales
# off it.
# ----------------------------------------------------------------------
FONT_SIZE = 16
LEGEND_FONT_SIZE = 13

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
#SIMULATION_JSON = "/home/s2/Projects-CUDA/iceMPM_v5/_input_data/nares_10k_06/simulation.json"
SIMULATION_JSON = "/home/s2/Projects-CUDA/iceMPM_v5/_input_data/nares_20k_06_Wisteria/simulation.json"

# Snapshot directory is derived from SIMULATION_JSON's RuntimeDirectory/ProjectName
# (see resolve_snapshot_dir) -- override here if you want a different location.
SNAPSHOT_DIR = None

OUTPUT_DIR = "."

# Fixed (p, q) extent for the hexbin plots, shared across all snapshots so
# the density plots are directly comparable to each other. q has no
# negative values (it's a deviatoric stress magnitude), so its range starts
# at 0. Adjust these based on the actual data being analyzed.
HEXBIN_P_RANGE = (-150000, 300000)
HEXBIN_Q_RANGE = (0, 250000)

# Exclude points that have left the modeled domain (SimParams::status_disabled,
# parameters_sim.h) -- their stress state is stale and not physically meaningful.
EXCLUDE_DISABLED = True

# Row indices into the "pts_data" dataset, matching SimParams::PtArrIdx in
# simulation/parameters_sim.h -- update these if the enum ever changes.
IDX_UTILITY_DATA = 0
IDX_STORED_P = 22
IDX_STORED_Q = 23
STATUS_CRACKED = 0x20000
STATUS_DISABLED = 0x40000

HEXBIN_GRIDSIZE = 150
FIGSIZE = (6.5, 5.5)   # inches
PNG_DPI = 250

HEXBIN_DIR = os.path.join(OUTPUT_DIR, "hexbin_envelope")


def load_config_chain(path):
    # Same inheritance rule as SimParams::ParseFile/ParameterParser at runtime:
    # ParentConfig (an absolute path) supplies defaults, the child's own keys
    # override them. Recurses in case the parent itself has a ParentConfig.
    with open(path) as f:
        cfg = json.load(f)
    if "ParentConfig" in cfg:
        merged = load_config_chain(cfg["ParentConfig"])
        merged.update(cfg)
        return merged
    return cfg


def resolve_snapshot_dir(cfg):
    runtime_dir = cfg.get("RuntimeDirectory")
    project_name = cfg.get("ProjectName")
    if not runtime_dir or not project_name:
        raise ValueError("Config is missing RuntimeDirectory/ProjectName -- set SNAPSHOT_DIR manually instead.")
    return os.path.join(runtime_dir, "input", project_name, "snapshots") + "/"


def find_snapshots(directory):
    paths = sorted(glob.glob(os.path.join(directory, "*.h5")))
    result = []
    for path in paths:
        m = re.search(r"(\d+)\.h5$", os.path.basename(path))
        if m and int(m.group(1)) == 0:
            continue
        result.append(path)
    return result


def load_pq(path):
    with h5py.File(path, "r") as f:
        pts = f["pts_data"]
        # utility_data is written via __longlong_as_double (kernels.cu) -- a bit-level reinterpret,
        # not a numeric value -- so recovering it needs .view(), not .astype() (a value conversion).
        utility = np.ascontiguousarray(pts[IDX_UTILITY_DATA, :]).view(np.uint64)
        p = pts[IDX_STORED_P, :].astype(np.float64)
        q = pts[IDX_STORED_Q, :].astype(np.float64)
        simulation_time_s = float(pts.attrs["SimulationTime"])

    if EXCLUDE_DISABLED:
        keep = (utility & STATUS_DISABLED) == 0
        p, q, utility = p[keep], q[keep], utility[keep]

    return p, q, utility, simulation_time_s


def intact_envelope(p, q0, negT, fracture_angle_deg):
    # PlasticProjection's piecewise-linear intact envelope (UseLiveIceStrength=false):
    # tensile leg for p<=0, compression-shear leg (unbounded above) for p>=0.
    tan_fa = np.tan(np.radians(fracture_angle_deg))
    q_tensile = (q0 / -negT) * (p - negT)
    q_compression = q0 + tan_fa * p
    q = np.where(p <= 0, q_tensile, q_compression)
    return np.where(p < negT, np.nan, q)


def yield_surface(p, q0, fracture_angle_deg, cf, dp_threshold_p, dp_phi_deg, yield_friction_deg):
    # Q_From_Yield_Surface + PlasticProjection's crossover-point calculation:
    # steep DP leg from dp_threshold_p, up to q_crossover = cf * (height where
    # that DP line would otherwise cross the intact envelope's compression-shear
    # leg), then a shallower second leg anchored at that crossover point.
    dp_tan_phi = np.tan(np.radians(dp_phi_deg))
    tan_fa = np.tan(np.radians(fracture_angle_deg))
    tan_yf = np.tan(np.radians(yield_friction_deg))
    p_intersect = (q0 + dp_tan_phi * dp_threshold_p) / (dp_tan_phi - tan_fa)
    q_intersect = q0 + tan_fa * p_intersect
    q_crossover = q_intersect * cf
    p_crossover = dp_threshold_p + q_crossover / dp_tan_phi
    q_steep = (p - dp_threshold_p) * dp_tan_phi
    q_second_leg = q_crossover + tan_yf * (p - p_crossover)
    q = np.minimum(q_steep, q_second_leg)
    return np.where(p < dp_threshold_p, np.nan, q)


def build_panel(p, q, params, show_intact, show_fractured):
    # One hexbin density panel, in kPa / Times New Roman house style, no
    # title -- matches plot_pq_hexbin_single.py's look.
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if p.size == 0:
        ax.text(0.5, 0.5, "no points", ha="center", va="center", transform=ax.transAxes)
    else:
        hexbin_extent = (*(v / 1e3 for v in HEXBIN_P_RANGE), *(v / 1e3 for v in HEXBIN_Q_RANGE))
        hb = ax.hexbin(p / 1e3, q / 1e3, gridsize=HEXBIN_GRIDSIZE, extent=hexbin_extent,
                        bins="log", mincnt=1, cmap="viridis")
        # Rasterize just this collection (not the whole figure): with this
        # many hexagons, matplotlib's PDF/SVG backends can silently drop
        # patches (a known large-PolyCollection vector-output bug, confirmed
        # via plot_pq_hexbin_single.py -- see that script's comment).
        hb.set_rasterized(True)
        fig.colorbar(hb, ax=ax, label=r"$\log_{10}$(count)")

    negT = -params["IceTensileFailureStrength"]

    if show_intact:
        p_lo = max(negT, HEXBIN_P_RANGE[0])
        p_curve = np.linspace(p_lo, HEXBIN_P_RANGE[1], 400)
        q_curve = intact_envelope(p_curve, params["IceShearStrength"], negT, params["IceFractureAngle"])
        ax.plot(p_curve / 1e3, q_curve / 1e3, color="red", lw=1.8, label="intact envelope")

    if show_fractured:
        p_lo_frac = max(params["DP_threshold_p"], HEXBIN_P_RANGE[0])
        p_curve_frac = np.linspace(p_lo_frac, HEXBIN_P_RANGE[1], 400)
        q_curve_frac = yield_surface(p_curve_frac, params["IceShearStrength"], params["IceFractureAngle"],
                                      params["Cf"], params["DP_threshold_p"], params["DP_phi"],
                                      params["IceYieldFrictionAngle"])
        ax.plot(p_curve_frac / 1e3, q_curve_frac / 1e3, color="orange", lw=1.8, linestyle="--",
                label="fractured/yield surface")

    ax.set_xlim(v / 1e3 for v in HEXBIN_P_RANGE)
    ax.set_ylim(v / 1e3 for v in HEXBIN_Q_RANGE)
    ax.set_xlabel("Pressure, $p$ (kPa)")
    ax.set_ylabel("Deviatoric stress, $q$ (kPa)")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, framealpha=0.9)

    fig.tight_layout()
    return fig


def save_panel_fig(fig, directory, name):
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": PNG_DPI})):
        out_path = os.path.join(directory, f"{name}.{ext}")
        fig.savefig(out_path, bbox_inches="tight", **kwargs)
        print(f"Wrote {out_path}")
    plt.close(fig)


def process_snapshot(path, params):
    stem = os.path.splitext(os.path.basename(path))[0]

    p, q, utility, simulation_time_s = load_pq(path)
    simulation_time_h = simulation_time_s / 3600.0
    print(f"{stem}: loaded {p.size} points  (t={simulation_time_h:.3f} h)")

    is_cracked = (utility & STATUS_CRACKED) != 0
    p_frac, q_frac = p[is_cracked], q[is_cracked]

    fig_a = build_panel(p, q, params, show_intact=True, show_fractured=True)
    save_panel_fig(fig_a, HEXBIN_DIR, f"{stem}_a")

    fig_b = build_panel(p_frac, q_frac, params, show_intact=False, show_fractured=True)
    save_panel_fig(fig_b, HEXBIN_DIR, f"{stem}_b")


def main():
    cfg = load_config_chain(SIMULATION_JSON)
    params = {k: cfg[k] for k in ("IceShearStrength", "IceTensileFailureStrength", "IceFractureAngle",
                                    "Cf", "IceYieldFrictionAngle", "DP_phi", "DP_threshold_p")}

    print(f"Config: {SIMULATION_JSON}")
    if cfg.get("UseLiveIceStrength", False):
        print("WARNING: this config has UseLiveIceStrength=true -- the simulation used per-point live idx_xi, "
              "but this script always plots the fixed-constant (UseLiveIceStrength=false) curve below. "
              "The plotted curve won't match what the simulation actually used.")
    print("Curve parameters (UseLiveIceStrength=false): " +
          ", ".join(f"{k}={v:.6g}" for k, v in params.items()))

    snapshot_dir = SNAPSHOT_DIR or resolve_snapshot_dir(cfg)
    print(f"Snapshot directory: {snapshot_dir}")
    print(f"Hexbin range: p={HEXBIN_P_RANGE}, q={HEXBIN_Q_RANGE}")

    os.makedirs(HEXBIN_DIR, exist_ok=True)
    snapshots = find_snapshots(snapshot_dir)
    print(f"Found {len(snapshots)} snapshot(s) to process (excluding the 0th)")

    for path in snapshots:
        process_snapshot(path, params)


if __name__ == "__main__":
    main()
