#!/usr/bin/env python3
"""
Single-panel version of plot_pq_hexbin_envelope.py, for the paper (setup.tex,
subsec:envelope_params, "% insert p-q plot" marker) -- one snapshot, one
p-q hexbin density plot, with the fracture envelope drawn on top, in the
same Times New Roman / vector-output house style as fracture_yield_figures_mpl.py
and make_strength_figure.py.

Intended data source: a no-fracture run ("AllowFracture": false in the
config, e.g. nares_5k_10_no_fracture/simulation.json), during a high-wind
period. AllowFracture=false means PlasticProjection (kernels.cu) never
runs, so points never actually crack; idx_stored_P/idx_stored_Q are still
written every step regardless (the unconditional "TEMPORARY" dump, see
kernels.cu around partition_kernel_g2p), so the sampled (p, q) cloud is the
genuine trial stress state the ice reaches under wind/current forcing, with
nothing clipped back onto the envelope by the plasticity model. That is
exactly the distribution setup.tex's envelope-parameter discussion refers
to ("the p-q stress states our own simulation reaches before any fracture
occurs").

Two output images, matching plot_pq_hexbin_envelope.py's convention:
- {OUT_NAME}_a: all (non-disabled) points, against both the intact envelope
  (red -- eq:qfrac_def in method.tex, tensile leg + flat compressive leg +
  vertical closure at p_compressive, no angle-dependent term, matching the
  current, final piecewise-linear calibration) and the yield surface
  (orange dashed -- Q_From_Yield_Surface in kernels.cu, for points that
  have already cracked).
- {OUT_NAME}_b: cracked points only (SimParams::status_cracked), against
  the yield surface only. Empty/near-empty on a genuine no-fracture run
  (AllowFracture=false) or an early, still-intact snapshot.

Both curves use the MEDIAN idx_xi across the sampled (non-disabled) points
at this snapshot as a representative scale (UseLiveIceStrength=true in
PlasticProjection scales q0/negT/p_compressive off the per-point xi the
same way); main() also prints the actual min/median/max so you can judge
how much the true per-point envelope actually varies. The yield surface's
Cf is used as configured (c_f0) -- the Cf(gamma_p) softening in
PlasticProjection is per-point/runtime state, not something this static
plot can reproduce (same simplification as plot_pq_hexbin_envelope.py).

Envelope scale (q_shear, p_tensile, p_compressive) is derived from idx_xi
(the raw, per-point, per-time depth-averaged flexural strength) through
IceStrengthFactor/IceStrengthFactorTensile/IceStrengthFactorCompressive,
exactly as in eq:strength_factors -- but unlike those old fixed-strength
scripts, idx_xi is NOT a single config constant here, it is a live field
that varies point to point. The single overlay curve below uses the
MEDIAN idx_xi across the sampled (non-disabled) points at this snapshot as
a representative scale; main() also prints the actual min/median/max so
you can judge how much the true per-point envelope actually varies.

Configuration is via the plain variable assignments below (no command-line
arguments) -- edit these and rerun.
"""
import glob
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Style -- Times New Roman throughout, matching fracture_yield_figures_mpl.py /
# cf_softening_figure.py / make_strength_figure.py. FONT_SIZE is the one
# knob to turn for a bigger/smaller figure -- everything else scales off it.
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
SIMULATION_JSON = "/home/s2/Projects-CUDA/iceMPM_v5/_input_data/nares_5k_10_no_fracture/simulation.json"

# Snapshot directory is derived from SIMULATION_JSON's RuntimeDirectory/ProjectName
# (see resolve_snapshot_dir) -- override here if you want a different location.
SNAPSHOT_DIR = None

# Which snapshot to plot -- exact filename (e.g. "f00037.h5"), or None to
# fall back to SNAPSHOT_NUMBER below.
SNAPSHOT_FILE = 'f00036.h5'

# Alternative to SNAPSHOT_FILE: just the integer frame number (e.g. 1188).
# Resolved to the closest snapshot actually present on disk (exact match if
# it exists). Ignored if SNAPSHOT_FILE is set. Leave both None to use the
# latest snapshot found (excluding f00000.h5, the pre-forcing initial state).
SNAPSHOT_NUMBER = None

OUT_DIR = Path(__file__).resolve().parent
OUT_NAME = "pq_hexbin_single"   # -> pq_hexbin_single_a.{svg,pdf,png}, _b.{svg,pdf,png}
# Note: no longer copied into _paper/figures/ -- that would silently
# overwrite the paper's own figure. Copy manually if you actually want to
# update it there.

FIGSIZE = (6.5, 5.5)   # inches
PNG_DPI = 250

# Fixed (p, q) extent, in Pa -- matches plot_pq_hexbin_envelope.py's range so
# the two are directly comparable. q has no negative values (it's a
# deviatoric stress magnitude), so its range starts at 0.
HEXBIN_P_RANGE = (-150000, 300000)
HEXBIN_Q_RANGE = (0, 200000)
HEXBIN_GRIDSIZE = 150

# Exclude points that have left the modeled domain (SimParams::status_disabled,
# parameters_sim.h) -- their stress state is stale and not physically meaningful.
EXCLUDE_DISABLED = True

# Row indices into the "pts_data" dataset, matching SimParams::PtArrIdx in
# simulation/parameters_sim.h -- update these if the enum ever changes.
IDX_UTILITY_DATA = 0
IDX_XI = 13
IDX_STORED_P = 22
IDX_STORED_Q = 23
STATUS_CRACKED = 0x20000
STATUS_DISABLED = 0x40000


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


def find_nearest_snapshot(directory, number):
    # Picks the snapshot whose frame number is closest to `number` (exact
    # match if present); ties broken toward the lower number.
    snapshots = find_snapshots(directory)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {directory}")
    numbered = [(int(re.search(r"(\d+)\.h5$", os.path.basename(p)).group(1)), p) for p in snapshots]
    best = min(numbered, key=lambda t: (abs(t[0] - number), t[0]))
    if best[0] != number:
        print(f"NOTE: snapshot f{number:05d}.h5 not found -- using nearest available, f{best[0]:05d}.h5")
    return best[1]


def load_pq_xi(path):
    with h5py.File(path, "r") as f:
        pts = f["pts_data"]
        # utility_data is written via __longlong_as_double (kernels.cu) -- a bit-level
        # reinterpret, not a numeric value -- so recovering it needs .view(), not .astype().
        utility = np.ascontiguousarray(pts[IDX_UTILITY_DATA, :]).view(np.uint64)
        p = pts[IDX_STORED_P, :].astype(np.float64)
        q = pts[IDX_STORED_Q, :].astype(np.float64)
        xi = pts[IDX_XI, :].astype(np.float64)
        simulation_time_s = float(pts.attrs["SimulationTime"])

    if EXCLUDE_DISABLED:
        keep = (utility & STATUS_DISABLED) == 0
        p, q, xi, utility = p[keep], q[keep], xi[keep], utility[keep]

    return p, q, xi, utility, simulation_time_s


def intact_envelope(p, q_shear, p_tensile, p_compressive):
    # eq:qfrac_def (method.tex): tensile leg from (-p_tensile, 0) to (0, q_shear),
    # flat compressive leg at q_shear up to p_compressive. No angle-dependent
    # term -- the current, final calibration has IceFractureAngle=0.
    q_tensile = (q_shear / p_tensile) * (p + p_tensile)
    q = np.where(p <= 0, q_tensile, q_shear)
    return np.where((p < -p_tensile) | (p > p_compressive), np.nan, q)


def yield_surface(p, q0, fracture_angle_deg, cf, dp_threshold_p, dp_phi_deg, yield_friction_deg):
    # Q_From_Yield_Surface + PlasticProjection's crossover-point calculation
    # (kernels.cu), ported verbatim from plot_pq_hexbin_envelope.py: steep DP
    # leg from dp_threshold_p, up to q_crossover = cf * (height where that DP
    # line would otherwise cross the intact envelope's compression-shear
    # leg), then a shallower second leg anchored at that crossover point. No
    # upper cutoff (the model's separate p_crush cap on cracked material's
    # pressure is not represented in this curve -- same simplification as
    # plot_pq_hexbin_envelope.py).
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


def build_panel(p, q, q_shear, p_tensile, p_compressive, yield_params, show_intact, show_fractured):
    # One hexbin density panel, in kPa / Times New Roman house style --
    # matches plot_pq_hexbin_envelope.py's build_panel, plus the median-xi
    # intact envelope this script already had.
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if p.size == 0:
        ax.text(0.5, 0.5, "no points", ha="center", va="center", transform=ax.transAxes)
    else:
        hb = ax.hexbin(p / 1e3, q / 1e3, gridsize=HEXBIN_GRIDSIZE,
                        extent=(*(v / 1e3 for v in HEXBIN_P_RANGE), *(v / 1e3 for v in HEXBIN_Q_RANGE)),
                        bins="log", mincnt=1, cmap="viridis")
        # Rasterize just this collection (not the whole figure): with this many
        # hexagons, matplotlib's PDF/SVG backends can silently drop patches
        # (a known large-PolyCollection vector-output bug) -- confirmed on this
        # exact script, reproduced twice, PNG unaffected since it's already
        # raster. Axes/text/legend stay vector.
        hb.set_rasterized(True)
        fig.colorbar(hb, ax=ax, label=r"$\log_{10}$(count)")

    if show_intact:
        p_curve = np.linspace(-p_tensile, p_compressive, 400) / 1e3
        q_curve = intact_envelope(p_curve * 1e3, q_shear, p_tensile, p_compressive) / 1e3
        # $\xi$ renders as a broken glyph in the custom Times New Roman mathtext
        # fontset (no Greek coverage), so the legend uses the paper's own
        # notation for this quantity instead, $\bar\sigma_{\mathrm f}$.
        ax.plot(p_curve, q_curve, color="red", lw=1.8,
                label="fracture envelope\n(median $\\bar\\sigma_f$)")
        # vertical closure at the compressive threshold, matching fracture_yield_figures_mpl.py's Figure 2
        ax.plot([p_compressive / 1e3, p_compressive / 1e3], [q_shear / 1e3, 0], color="red", lw=1.8)

    if show_fractured:
        p_lo_frac = max(yield_params["dp_threshold_p"], HEXBIN_P_RANGE[0])
        p_curve_frac = np.linspace(p_lo_frac, HEXBIN_P_RANGE[1], 400)
        q_curve_frac = yield_surface(p_curve_frac, q_shear, yield_params["fracture_angle_deg"],
                                      yield_params["cf"], yield_params["dp_threshold_p"],
                                      yield_params["dp_phi_deg"], yield_params["yield_friction_deg"])
        ax.plot(p_curve_frac / 1e3, q_curve_frac / 1e3, color="orange", lw=1.8, linestyle="--",
                label="yield surface\n(median $\\bar\\sigma_f$)")

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


def save_panel_fig(fig, suffix):
    for ext, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": PNG_DPI})):
        out_path = OUT_DIR / f"{OUT_NAME}_{suffix}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", **kwargs)
        print(f"Wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


def main():
    cfg = load_config_chain(SIMULATION_JSON)
    print(f"Config: {SIMULATION_JSON}")
    if not cfg.get("AllowFracture", True):
        print("NOTE: this config has AllowFracture=false -- there is no cracked population, "
              "so panel _b (fractured points only) will be empty.")

    snapshot_dir = SNAPSHOT_DIR or resolve_snapshot_dir(cfg)
    print(f"Snapshot directory: {snapshot_dir}")

    if SNAPSHOT_FILE is not None:
        path = os.path.join(snapshot_dir, SNAPSHOT_FILE)
    elif SNAPSHOT_NUMBER is not None:
        path = find_nearest_snapshot(snapshot_dir, SNAPSHOT_NUMBER)
    else:
        snapshots = find_snapshots(snapshot_dir)
        if not snapshots:
            raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")
        path = snapshots[-1]
    print(f"Snapshot: {path}")

    p, q, xi, utility, simulation_time_s = load_pq_xi(path)
    simulation_time_h = simulation_time_s / 3600.0
    print(f"Loaded {p.size} points  (t={simulation_time_h:.3f} h)")

    xi_min, xi_med, xi_max = np.min(xi), np.median(xi), np.max(xi)
    print(f"idx_xi (raw flexural strength, Pa): min={xi_min:.4g}, median={xi_med:.4g}, max={xi_max:.4g}")
    print("Envelope/yield-surface overlays below use the median; per-point curves vary across this range.")

    k_s = cfg["IceStrengthFactor"]
    k_t = cfg["IceStrengthFactorTensile"]
    k_c = cfg["IceStrengthFactorCompressive"]
    q_shear = k_s * xi_med
    p_tensile = k_t * xi_med
    p_compressive = k_c * xi_med
    print(f"Envelope (median xi): q_shear={q_shear:.4g} Pa, p_tensile={p_tensile:.4g} Pa, "
          f"p_compressive={p_compressive:.4g} Pa")

    yield_params = {
        "fracture_angle_deg": cfg["IceFractureAngle"],
        "cf": cfg["Cf"],
        "dp_threshold_p": cfg["DP_threshold_p"],
        "dp_phi_deg": cfg["DP_phi"],
        "yield_friction_deg": cfg["IceYieldFrictionAngle"],
    }
    print("Yield surface params (c_f used as configured, c_f0 -- no per-point softening history): " +
          ", ".join(f"{k}={v:.6g}" for k, v in yield_params.items()))

    is_cracked = (utility & STATUS_CRACKED) != 0
    p_frac, q_frac = p[is_cracked], q[is_cracked]
    print(f"{is_cracked.sum()} of {p.size} points are cracked")

    # No cracked population to compare the yield surface against on a genuine
    # no-fracture run (AllowFracture=false) -- suppress that curve on panel a
    # rather than draw it floating over an all-intact cloud. Fig 18-style
    # fractured runs still get both curves.
    allow_fracture = cfg.get("AllowFracture", True)

    fig_a = build_panel(p, q, q_shear, p_tensile, p_compressive, yield_params,
                         show_intact=True, show_fractured=allow_fracture)
    print(f"Panel a (all points): max p={p.max() / 1e3:.4g} kPa, max q={q.max() / 1e3:.4g} kPa")
    save_panel_fig(fig_a, "a")

    fig_b = build_panel(p_frac, q_frac, q_shear, p_tensile, p_compressive, yield_params,
                         show_intact=False, show_fractured=allow_fracture)
    if p_frac.size:
        print(f"Panel b (cracked points): max p={p_frac.max() / 1e3:.4g} kPa, max q={q_frac.max() / 1e3:.4g} kPa")
    else:
        print("Panel b (cracked points): no cracked points at this snapshot")
    save_panel_fig(fig_b, "b")


if __name__ == "__main__":
    main()
