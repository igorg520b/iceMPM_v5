#!/usr/bin/env python3
"""
Interactive version of plot_new_piecewise_curves.py -- same piecewise-linear
fracture (intact) and yield (fractured) curves from simulation/kernels.cu's
PlasticProjection/Q_From_Yield_Surface, but with sliders to tweak the shape
parameters live instead of editing constants and rerunning.

Requires a GUI-capable matplotlib backend (TkAgg, QtAgg, ...) and a display --
run this on a machine with a screen, not headless.

Models PlasticProjection's UseLiveIceStrength=true branch: q0/negT and both
compressive thresholds all scale off a single assumed-constant xi (the
per-point flexural strength, Pa) times a strength factor, matching kernels.cu
exactly. xi itself is a slider, standing in for whatever a real point's
idx_xi would be at some point in the simulation.

Intact fracture envelope:
    q0   = xi * IceStrengthFactor                (q at p=0)
    negT = -(xi * IceStrengthFactorTensile)       (tensile leg's zero-crossing)
    q(p) = (q0/-negT) * (p - negT)                for negT <= p <= 0
    q(p) = q0 + tan(IceFractureAngle) * p         for p >= 0

Intact material now also cracks outright once p exceeds a *separate*
compressive threshold, even under near-zero shear -- q_envelope keeps
growing with p (more confinement -> more shear capacity), so without this
explicit cap, high pressure alone would never crack the material:
    compressive_threshold_fracture = xi * IceStrengthFactorCompressive
Not scaled by Cf_now -- intact material has no plastic-strain history yet.

Fractured/yield surface (Q_From_Yield_Surface). The crossover point is where
the steep DP leg would otherwise cross the intact envelope's compression-shear
leg (p_intersect, q_intersect); Cf_now=1 puts the fractured surface's kink
exactly on the intact envelope, Cf_now<1 pulls it in below that:
    p_intersect = (q0 + DP_tan_phi*DP_threshold_p) / (DP_tan_phi - IceFractureAngle_tan)
    q_intersect = q0 + IceFractureAngle_tan * p_intersect
    q_crossover = q_intersect * Cf_now
    p_crossover = DP_threshold_p + q_crossover / tan(DP_phi)
    q_steep(p)  = (p - DP_threshold_p) * tan(DP_phi)
    q_2nd(p)    = q_crossover + tan(IceYieldFrictionAngle) * (p - p_crossover)
    q(p)        = min(q_steep(p), q_2nd(p))       for p >= DP_threshold_p

Cf_now softens with accumulated plastic shear strain gamma_p (see the
"Cf(gamma_p) softening" block in PlasticProjection, kernels.cu):
    Cf_now = Cf_residual + (Cf - Cf_residual) * exp(-gamma_p / GammaStar)
Cf here (c_f0) is gamma_p=0's value; drag the gamma_p slider to 0 to disable
the softening and see the constant-Cf curve.

Once cracked, PlasticProjection clamps p_new at a second, separate threshold
every step (the "no growing rubble-pile capacity" branch) -- so a fractured
point's reachable (p, q) region is bounded there, closing off the yield
surface at that pressure. This one *is* Cf_now-scaled -- heavily shear-worked
material holds pressure less well:
    compressive_threshold_yield = xi * IceStrengthFactorCompressiveYield * Cf_now
Since the two compressive thresholds use different factors (and only one is
Cf_now-scaled), they generally sit at different p -- each curve closes off
at its own threshold, not a shared one.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- Initial slider values -- match base_config.json / nares_10k_10/simulation.json ---
INIT = dict(
    xi=1.7e5,                              # assumed constant flexural strength, Pa
    ice_strength_factor=0.46,              # IceStrengthFactor
    ice_strength_factor_tensile=0.40,      # IceStrengthFactorTensile
    fracture_angle=10,                     # IceFractureAngle, deg
    cf0=0.9,                               # Cf (c_f0 -- Cf_now's value at gamma_p=0)
    cf_residual=0.1,                       # Cf_residual (c_res)
    gamma_star=0.3,                        # GammaStar
    gamma_p=0.0,                           # accumulated plastic shear strain, for exploring Cf(gamma_p)
    yield_friction_angle=10,               # IceYieldFrictionAngle, deg
    dp_phi=57,                             # DP_phi, deg
    dp_threshold_p=-1000,                  # DP_threshold_p
    ice_strength_factor_compressive=1.2,        # compressive_threshold_fracture = xi * this
    ice_strength_factor_compressive_yield=1.2,  # compressive_threshold_yield = xi * this * Cf_now
)


def intact_envelope(p, q0, negT, fracture_angle_deg):
    tan_fa = np.tan(np.radians(fracture_angle_deg))
    q_tensile = (q0 / -negT) * (p - negT)
    q_compression = q0 + tan_fa * p
    q = np.where(p <= 0, q_tensile, q_compression)
    return np.where(p < negT, np.nan, q)


def crossover_point(q0, fracture_angle_deg, dp_threshold_p, dp_phi_deg, cf_now):
    # Where the DP line would cross the intact envelope's compression-shear leg, scaled by cf_now.
    dp_tan_phi = np.tan(np.radians(dp_phi_deg))
    tan_fa = np.tan(np.radians(fracture_angle_deg))
    p_intersect = (q0 + dp_tan_phi * dp_threshold_p) / (dp_tan_phi - tan_fa)
    q_intersect = q0 + tan_fa * p_intersect
    return q_intersect * cf_now


def yield_surface(p, q_crossover, dp_threshold_p, dp_phi_deg, yield_friction_deg):
    dp_tan_phi = np.tan(np.radians(dp_phi_deg))
    tan_yf = np.tan(np.radians(yield_friction_deg))
    p_crossover = dp_threshold_p + q_crossover / dp_tan_phi
    q_steep = (p - dp_threshold_p) * dp_tan_phi
    q_2nd = q_crossover + tan_yf * (p - p_crossover)
    q = np.minimum(q_steep, q_2nd)
    return np.where(p < dp_threshold_p, np.nan, q), p_crossover


def main():
    fig = plt.figure(figsize=(10, 8))
    ax_zoom = fig.add_axes([0.08, 0.08, 0.62, 0.82])
    fig.suptitle("Piecewise-linear fracture/yield curves (drag sliders to explore)")

    # --- Sliders, stacked in a column on the right ---
    slider_specs = [
        ("xi",                              "xi (flexural strength, Pa)",   0.0,   5e5,  INIT["xi"],                              "%.0f"),
        ("ice_strength_factor",             "IceStrengthFactor",            0.0,   2.0,  INIT["ice_strength_factor"],             "%.3f"),
        ("ice_strength_factor_tensile",     "IceStrengthFactorTensile",     0.0,   2.0,  INIT["ice_strength_factor_tensile"],     "%.3f"),
        ("fracture_angle",                  "IceFractureAngle (deg)",       0.1,   30.0, INIT["fracture_angle"],                  "%.2f"),
        ("cf0",                             "Cf (c_f0)",                    0.05,  4.0,  INIT["cf0"],                             "%.3f"),
        ("cf_residual",                     "Cf_residual (c_res)",          0.0,   4.0,  INIT["cf_residual"],                     "%.3f"),
        ("gamma_star",                      "GammaStar",                    0.01,  2.0,  INIT["gamma_star"],                      "%.3f"),
        ("gamma_p",                         "gamma_p",                      0.0,   2.0,  INIT["gamma_p"],                         "%.3f"),
        ("yield_friction_angle",            "IceYieldFrictionAngle (deg)",  0.1,   60.0, INIT["yield_friction_angle"],            "%.2f"),
        ("dp_phi",                          "DP_phi (deg)",                 1.0,   89.0, INIT["dp_phi"],                          "%.1f"),
        ("dp_threshold_p",                  "DP_threshold_p",            -5000.0,  0.0,  INIT["dp_threshold_p"],                  "%.0f"),
        ("ice_strength_factor_compressive", "IceStrengthFactorCompressive", 0.0,   4.0,  INIT["ice_strength_factor_compressive"], "%.3f"),
        ("ice_strength_factor_compressive_yield", "...CompressiveYield",    0.0,   4.0,  INIT["ice_strength_factor_compressive_yield"], "%.3f"),
    ]

    sliders = {}
    n = len(slider_specs)
    top_y, bottom_y, height = 0.85, 0.08, 0.03
    step = (top_y - bottom_y) / n
    for i, (key, label, vmin, vmax, vinit, fmt) in enumerate(slider_specs):
        # width trimmed from 0.20 -- 6-digit values (e.g. xi) need more right-edge
        # margin for the value label than the default left it (was getting clipped).
        ax_s = fig.add_axes([0.76, top_y - i * step, 0.15, height])
        sliders[key] = Slider(ax_s, label, vmin, vmax, valinit=vinit, valfmt=fmt)
        sliders[key].label.set_fontsize(8)
        sliders[key].valtext.set_fontsize(8)

    ax_reset = fig.add_axes([0.76, bottom_y - 0.06, 0.09, 0.04])
    btn_reset = Button(ax_reset, "Reset")
    ax_save = fig.add_axes([0.87, bottom_y - 0.06, 0.09, 0.04])
    btn_save = Button(ax_save, "Save PNG")

    def draw_panel(ax, p_range, title, q0, negT, q_crossover, p_crossover, cf_now,
                    compressive_threshold_fracture, compressive_threshold_yield,
                    fracture_angle, dp_threshold_p, dp_phi, yield_friction_angle):
        ax.clear()
        p = np.linspace(*p_range, 2000)

        # Each curve closes off at its OWN threshold -- these are generally different p
        # now (only the yield one is Cf_now-scaled), unlike the old shared threshold.
        q_intact = intact_envelope(p, q0, negT, fracture_angle)
        q_intact = np.where(p <= compressive_threshold_fracture, q_intact, np.nan)
        ax.plot(p, q_intact, color="red", linewidth=1.5, label="intact fracture envelope")

        q_yield, _ = yield_surface(p, q_crossover, dp_threshold_p, dp_phi, yield_friction_angle)
        q_yield = np.where(p <= compressive_threshold_yield, q_yield, np.nan)
        ax.plot(p, q_yield, color="orange", linewidth=1.5, linestyle="--", label="fractured/yield surface")

        ax.axvline(0, color="gray", linewidth=0.6)
        ax.axhline(0, color="gray", linewidth=0.6)
        q_top = np.nanmax(np.concatenate([q_intact, q_yield, [1.0]])) * 1.08
        ax.set_ylim(0, q_top)

        ax.scatter([0], [q0], color="red", zorder=5, s=20)
        ax.annotate(f"q0={q0:.0f}", (0, q0), textcoords="offset points", xytext=(8, -10), fontsize=8, color="red")
        ax.scatter([negT], [0], color="red", zorder=5, s=20)
        ax.annotate(f"-T={negT:.0f}", (negT, 0), textcoords="offset points", xytext=(6, 10), fontsize=8, color="red")
        if p_range[0] <= p_crossover <= p_range[1]:
            ax.scatter([p_crossover], [q_crossover], color="orange", zorder=5, s=25)
            ax.annotate(f"crossover (Cf_now={cf_now:.3f})\n(p={p_crossover:.0f}, q={q_crossover:.0f})",
                        (p_crossover, q_crossover),
                        textcoords="offset points", xytext=(8, -28), fontsize=8, color="orange")

        if p_range[0] <= compressive_threshold_fracture <= p_range[1]:
            q_intact_ct = float(intact_envelope(np.array([compressive_threshold_fracture]), q0, negT,
                                                fracture_angle)[0])
            ax.plot([compressive_threshold_fracture, compressive_threshold_fracture], [q_intact_ct, 0],
                    color="red", linewidth=1.5)
            ax.text(compressive_threshold_fracture, q_top * 0.98,
                    f"compressive_threshold_fracture={compressive_threshold_fracture:.0f}",
                    rotation=90, va="top", ha="right", fontsize=7, color="darkred")

        if p_range[0] <= compressive_threshold_yield <= p_range[1]:
            q_yield_ct, _ = yield_surface(np.array([compressive_threshold_yield]), q_crossover, dp_threshold_p,
                                          dp_phi, yield_friction_angle)
            q_yield_ct = float(q_yield_ct[0])
            ax.plot([compressive_threshold_yield, compressive_threshold_yield], [q_yield_ct, 0],
                    color="orange", linewidth=1.5, linestyle="--")
            ax.text(compressive_threshold_yield, q_top * 0.90,
                    f"compressive_threshold_yield={compressive_threshold_yield:.0f}",
                    rotation=90, va="top", ha="right", fontsize=7, color="darkorange")

        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(p_range)
        ax.set_title(title)
        ax.set_xlabel("p")
        ax.set_ylabel("q")

    def update(_=None):
        xi = sliders["xi"].val
        ice_strength_factor = sliders["ice_strength_factor"].val
        ice_strength_factor_tensile = sliders["ice_strength_factor_tensile"].val
        fracture_angle = sliders["fracture_angle"].val
        cf0 = sliders["cf0"].val
        cf_residual = sliders["cf_residual"].val
        gamma_star = sliders["gamma_star"].val
        gamma_p = sliders["gamma_p"].val
        yield_friction_angle = sliders["yield_friction_angle"].val
        dp_phi = sliders["dp_phi"].val
        dp_threshold_p = sliders["dp_threshold_p"].val
        ice_strength_factor_compressive = sliders["ice_strength_factor_compressive"].val
        ice_strength_factor_compressive_yield = sliders["ice_strength_factor_compressive_yield"].val

        # Mirrors PlasticProjection's UseLiveIceStrength=true branch, kernels.cu.
        q0 = xi * ice_strength_factor
        negT = -(xi * ice_strength_factor_tensile)

        # Cf(gamma_p) softening -- mirrors the commentable block in PlasticProjection, kernels.cu.
        cf_now = cf_residual + (cf0 - cf_residual) * np.exp(-gamma_p / gamma_star)

        # Fracture-side cap: NOT Cf_now-scaled (intact material has no plastic-strain history).
        # Yield-side cap: Cf_now-scaled (heavily shear-worked material holds pressure less well).
        compressive_threshold_fracture = xi * ice_strength_factor_compressive
        compressive_threshold_yield = xi * ice_strength_factor_compressive_yield * cf_now

        q_crossover = crossover_point(q0, fracture_angle, dp_threshold_p, dp_phi, cf_now)
        dp_tan_phi = np.tan(np.radians(dp_phi))
        p_crossover = dp_threshold_p + q_crossover / dp_tan_phi

        zoom_p_hi = max(p_crossover * 1.8, compressive_threshold_fracture * 1.1,
                         compressive_threshold_yield * 1.1, -negT * 1.5)
        draw_panel(ax_zoom, (negT * 1.5, zoom_p_hi), "Near-origin detail", q0, negT, q_crossover, p_crossover,
                   cf_now, compressive_threshold_fracture, compressive_threshold_yield,
                   fracture_angle, dp_threshold_p, dp_phi, yield_friction_angle)

        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    def reset(_):
        for s in sliders.values():
            s.reset()
    btn_reset.on_clicked(reset)

    def save(_):
        out_path = "new_piecewise_curves_interactive.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    btn_save.on_clicked(save)

    update()
    plt.show()


if __name__ == "__main__":
    main()
