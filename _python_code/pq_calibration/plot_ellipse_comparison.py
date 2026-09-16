#!/usr/bin/env python3
"""
Illustrates, for the supervisor comment on method.tex's "the portion of the
ellipse is practically a straight line" remark, where our simulated (p, q)
trial-stress cloud actually falls relative to a classical elliptical
fracture/failure envelope for ice -- and how flat that envelope's local
slope is over our simulation's working pressure range.

Ellipse: Derradji-Aouat (2003), "A unified failure envelope for isotropic
fresh water ice and iceberg ice", Cold Regions Science and Technology 36,
Eq. (1a) and Table 1 (freshwater ice and iceberg ice row):

    ((q - g) / q_max)^2 + ((P - k) / Pc)^2 = 1

with g=0, k=45 MPa, Pc=55 MPa (so P spans [-10, 100] MPa, matching the
range given in the supervisor discussion). q_max is not a single constant
in that paper -- it depends on strain rate/temperature via their Eq. (1b)
-- so we use the representative value given directly for this comparison,
q_max = 2 MPa.

Wolper et al. (2021) use a related but distinct Cohesive Cam-Clay ellipse,
y(p,q) = q^2(1+2*beta) + M^2*(p+beta*p0)*(p-p0); their Table 2 only tabulates
the PRODUCT beta*p0 (0.5-2 MPa across their glacier-calving cases, M~0.13-1.4),
never beta and p0 separately, so an exact curve can't be reconstructed from
that paper alone -- not plotted here, mentioned only for order-of-magnitude
context in comments.tex.

Compression is positive in both the ellipse (P) and our own convention (p,
eq:p_def in method.tex), so no sign flip is needed to overlay them.

Data source: same no-fracture snapshot as Fig 13 (setup.tex) -- genuine
unclipped trial (p, q) states, nothing projected back onto any envelope.

Two output images:
- ellipse_comparison_full: the whole ellipse (MPa scale), with our own
  working (p, q) range marked as a small box, showing the ~1000x scale gap.
- ellipse_comparison_zoom: zoomed to our own (p, q) hexbin range (kPa
  scale) -- the ellipse itself sits far above this frame (see the full
  plot / the printed local ellipse value), so this panel is annotated with
  the ellipse's local q value and slope at p=0 rather than showing the
  curve directly.

Configuration is via the plain variable assignments below (no command-line
arguments) -- edit these and rerun.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_pq_hexbin_single import (
    load_config_chain, resolve_snapshot_dir, load_pq_xi, intact_envelope,
)

# ----------------------------------------------------------------------
# Style -- Times New Roman throughout, matching plot_pq_hexbin_single.py.
# ----------------------------------------------------------------------
FONT_SIZE = 16
LEGEND_FONT_SIZE = 12

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# --- Configuration -- edit and rerun ---
SIMULATION_JSON = "/home/s2/Projects-CUDA/iceMPM_v5/_input_data/nares_5k_10_no_fracture/simulation.json"
SNAPSHOT_FILE = 'f00036.h5'   # same snapshot used for Fig 13 (setup.tex, fig:pq_hexbin)

OUT_DIR = Path(__file__).resolve().parent
FIGSIZE = (6.5, 5.5)
PNG_DPI = 250

# Derradji-Aouat (2003) ellipse, Table 1 (freshwater ice and iceberg ice), MPa.
ELLIPSE_G = 0.0     # q-center
ELLIPSE_K = 45.0    # P-center
ELLIPSE_PC = 55.0   # major axis (P direction) -> P in [-10, 100] MPa
ELLIPSE_QMAX = 2.0  # minor axis (q direction) -- representative value (see docstring)

# Our own working (p, q) hexbin range, kPa -- matches plot_pq_hexbin_single.py.
ZOOM_P_RANGE = (-150, 400)     # kPa
ZOOM_Q_RANGE = (0, 250)        # kPa
ZOOM_GRIDSIZE = 100

# Illustrative only: q_max rescaled (P-center/major-axis k, Pc unchanged) so
# the ellipse's real shape/curvature passes through (p=0, q=TOUCH_Q_AT_P0)
# instead of its real q~1.15 MPa there -- lets the local near-flatness be
# seen directly in the zoomed panel instead of only stated in an annotation.
# NOT the literal Derradji parameters -- see Fig 1 / the printed values for those.
TOUCH_Q_AT_P0 = 100  # kPa

HEXBIN_DISABLED = True  # exclude points that left the modeled domain -- see load_pq_xi


def ellipse_q(p_mpa, q_max=ELLIPSE_QMAX):
    """Derradji ellipse q(P), MPa, for the upper (q>=0) branch. q_max is
    exposed separately so the shape can be rescaled (see TOUCH_Q_AT_P0)."""
    arg = 1.0 - ((p_mpa - ELLIPSE_K) / ELLIPSE_PC) ** 2
    return ELLIPSE_G + q_max * np.sqrt(np.clip(arg, 0.0, None))


def main():
    cfg = load_config_chain(SIMULATION_JSON)
    print(f"Config: {SIMULATION_JSON}")
    snapshot_dir = resolve_snapshot_dir(cfg)
    path = str(Path(snapshot_dir) / SNAPSHOT_FILE)
    print(f"Snapshot: {path}")

    p, q, xi, utility, simulation_time_s = load_pq_xi(path)
    print(f"Loaded {p.size} points  (t={simulation_time_s / 3600.0:.3f} h)")

    xi_med = np.median(xi)
    k_s = cfg["IceStrengthFactor"]
    k_t = cfg["IceStrengthFactorTensile"]
    k_c = cfg["IceStrengthFactorCompressive"]
    q_shear = k_s * xi_med
    p_tensile = k_t * xi_med
    p_compressive = k_c * xi_med
    print(f"Our envelope (median xi, kPa): q_shear={q_shear/1e3:.4g}, "
          f"p_tensile={p_tensile/1e3:.4g}, p_compressive={p_compressive/1e3:.4g}")

    # Local ellipse value/slope at p=0, our working scale's reference point.
    q0 = ellipse_q(0.0)
    eps = 1e-3
    slope0 = (ellipse_q(eps) - ellipse_q(-eps)) / (2 * eps)
    print(f"Derradji ellipse at P=0: q={q0:.4g} MPa, local slope dq/dP={slope0:.4g} (dimensionless)")
    q_at_zoom_edges = ellipse_q(np.array(ZOOM_P_RANGE) / 1e3)
    print(f"Derradji ellipse across our zoom p-range {ZOOM_P_RANGE} kPa: "
          f"q = {q_at_zoom_edges[0]:.4g} to {q_at_zoom_edges[1]:.4g} MPa "
          f"(vs. our own q_shear = {q_shear/1e3:.4g} kPa)")

    # Illustrative rescaling: same shape (k, Pc unchanged), q_max chosen so
    # q(P=0) = TOUCH_Q_AT_P0.
    q_max_touch = (TOUCH_Q_AT_P0 / 1e3) / ellipse_q(0.0, q_max=1.0)   # MPa
    q_touch_edges = ellipse_q(np.array(ZOOM_P_RANGE) / 1e3, q_max=q_max_touch) * 1e3  # kPa
    print(f"Rescaled (illustrative) ellipse, q_max={q_max_touch*1e3:.4g} kPa "
          f"(vs. real {ELLIPSE_QMAX*1e3:.4g} kPa): "
          f"q = {q_touch_edges[0]:.4g} to {q_touch_edges[1]:.4g} kPa across the zoom p-range")

    # ------------------------------------------------------------------
    # Plot 1: full ellipse (MPa), our data cloud + working range marked.
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=FIGSIZE)

    p_curve = np.linspace(ELLIPSE_K - ELLIPSE_PC, ELLIPSE_K + ELLIPSE_PC, 600)
    q_curve = ellipse_q(p_curve)
    ax1.plot(p_curve, q_curve, color="red", lw=1.8, label="Derradji-Aouat (2003) ellipse")
    ax1.plot(p_curve, -q_curve, color="red", lw=1.8, linestyle=":", alpha=0.5,
              label="(mirror, $q$ is a magnitude)")

    # Our own working range, as a box -- essentially invisible at this scale, which is the point.
    box_p = np.array(ZOOM_P_RANGE) / 1e3   # kPa -> MPa
    box_q = np.array(ZOOM_Q_RANGE) / 1e3
    ax1.add_patch(Rectangle((box_p[0], box_q[0]), box_p[1] - box_p[0], box_q[1] - box_q[0],
                              facecolor="none", edgecolor="blue", lw=1.5, zorder=5,
                              label="our working range (Fig.~2)"))

    ax1.axhline(0, color="gray", lw=0.6)
    ax1.set_xlabel("Pressure, $p$ (MPa)")
    ax1.set_ylabel("Deviatoric stress, $q$ (MPa)")
    ax1.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")
    ax1.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, framealpha=0.9)
    fig1.tight_layout()
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": PNG_DPI})):
        out_path = OUT_DIR / f"ellipse_comparison_full.{ext}"
        fig1.savefig(out_path, bbox_inches="tight", **kwargs)
        print(f"Wrote {out_path}")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Plot 2: zoomed to our own working (p, q) range (kPa) -- hexbin cloud
    # + our piecewise-linear envelope + the ellipse's real SHAPE (k, Pc
    # unchanged), q_max rescaled so it passes through (p=0, q=TOUCH_Q_AT_P0)
    # instead of its real q~1.15 MPa there -- illustrative only, so the
    # local near-flatness is visible directly in this frame.
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=FIGSIZE)

    hb = ax2.hexbin(p / 1e3, q / 1e3, gridsize=ZOOM_GRIDSIZE,
                     extent=(*ZOOM_P_RANGE, *ZOOM_Q_RANGE),
                     bins="log", mincnt=1, cmap="viridis")
    hb.set_rasterized(True)
    fig2.colorbar(hb, ax=ax2, label=r"$\log_{10}$(count)")

    p_ellipse_touch = np.linspace(*ZOOM_P_RANGE, 400) / 1e3
    q_ellipse_touch = ellipse_q(p_ellipse_touch, q_max=q_max_touch) * 1e3
    ax2.plot(p_ellipse_touch * 1e3, q_ellipse_touch, color="black", lw=1.8, linestyle="--",
             label=f"Derradji ellipse shape\n(rescaled to touch $p$=0, $q\\approx${TOUCH_Q_AT_P0} kPa)")

    p_env = np.linspace(-p_tensile, p_compressive, 400) / 1e3
    q_env = intact_envelope(p_env * 1e3, q_shear, p_tensile, p_compressive) / 1e3
    ax2.plot(p_env, q_env, color="red", lw=1.8, label="our fracture envelope\n(median $\\bar\\sigma_f$)")
    ax2.plot([p_compressive / 1e3] * 2, [q_shear / 1e3, 0], color="red", lw=1.8)

    ax2.set_xlim(ZOOM_P_RANGE)
    ax2.set_ylim(ZOOM_Q_RANGE)
    ax2.set_xlabel("Pressure, $p$ (kPa)")
    ax2.set_ylabel("Deviatoric stress, $q$ (kPa)")
    ax2.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")
    ax2.text(0.03, 0.05,
              f"Real ellipse here (off-frame above):\n"
              f"$q\\approx${q0*1e3:.0f} kPa at $p$=0",
              transform=ax2.transAxes, va="bottom", ha="left", fontsize=LEGEND_FONT_SIZE,
              bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
    ax2.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, framealpha=0.9)
    fig2.tight_layout()
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": PNG_DPI})):
        out_path = OUT_DIR / f"ellipse_comparison_zoom.{ext}"
        fig2.savefig(out_path, bbox_inches="tight", **kwargs)
        print(f"Wrote {out_path}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
