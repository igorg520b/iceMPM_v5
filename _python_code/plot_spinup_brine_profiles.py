#!/usr/bin/env python3
"""
For a sample of thermal spin-up points, plots the reconstructed vertical
temperature profile (formula (R), see heat_equation_spec.md), the brine
volume fraction profile derived from it, and the resulting flexural
strength profile (Timco & O'Brien 1994). Three side-by-side plots for one
point at a time -- use the '<'/'>' buttons to step through the sample.

Brine volume formula is switchable at runtime (radio buttons):
  - Frankenstein & Garner (1967): vb = S*(49.185/<-T> + 0.532) * 1e-3
    (<-T> is the Macaulay bracket max(-T, eps), not |T|)
  - Cox & Weeks (1983), via the 'Sea Ice' book closure (pure ice density +
    F1/F2/F3 polynomials, piecewise in T)
Salinity (ppt) is editable via a text box and applied to all points. The
flexural strength profile is depth-averaged over the ice thickness and the
resulting number is shown in its subplot title.

Input: paste the console line printed by DataPreparer::ThermalSpinUpSave
(preparer/data_preparer.cpp) into RAW_DATA_STRING below. Each entry is
{A1, A2, A3, Ts_old, thickness, lat, lon, pixel_x, pixel_y}.
"""
import ast

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, TextBox, Button
import numpy as np

# Workaround for a matplotlib 3.11.0 regression: TextBox._resize is wrapped by
# a decorator that unconditionally reads event.inaxes, but resize events don't
# have that attribute, so resizing the window crashes any figure with a
# TextBox. Patch it to the un-decorated behavior before any TextBox is built.
TextBox._resize = lambda self, event: self.stop_typing()

# --- Configuration -- edit and rerun ---
T_BOTTOM = -1.8  # deg C, SimParams::T_bottom
N_Z = 300

# Typical bulk salinity for first-year ice (roughly 4-8 ppt); lower it toward
# ~1-3 ppt for multi-year ice, or edit the text box at runtime.
DEFAULT_SALINITY = 6.0

# Paste the line printed by ThermalSpinUpSave here (Mathematica-style
# {{...}, {...}, ...}); each row is {A1, A2, A3, Ts_old, H, lat, lon, pixel_x, pixel_y}.
RAW_DATA_STRING = """
{{-2.59097,-1.04747,-0.42069,-1.95826,2.18202,79.874,-66.32,893,725},{-2.05925,-0.61694,-0.186185,-1.76826,2.10494,78.8066,-70.1308,739,965},{-1.72076,-0.262526,-0.0573515,-0.0345215,1.88187,80.1754,-69.7873,758,660},{-3.19073,-1.07486,-0.340787,-0.0138306,1.9008,79.896,-66.877,871,721},{-1.39847,-0.151934,-0.0190867,-0.026709,1.56997,80.5171,-68.9172,791,584},{-2.29938,-0.500935,-0.177231,-2.21723,2.94765,82.3768,-56.4341,1157,129},{-0.631146,-0.102851,-0.0162437,0.00200806,1.25368,79.3472,-73.4177,606,838},{-1.21935,-0.116262,0.0421967,-0.579138,1.58959,80.0866,-69.1031,784,680},{-0.0827288,-0.0342383,-0.0157869,-0.498846,0.8,78.467,-75.3122,507,1026},{-3.08194,-1.49753,-0.728068,-1.78681,2.0223,79.5373,-67.789,837,802}}
"""


def parse_data(raw: str):
    """Parses the Mathematica-braced RAW_DATA_STRING into a list of 9-tuples."""
    cleaned = raw.strip().replace("{", "[").replace("}", "]")
    rows = ast.literal_eval(cleaned)
    return [tuple(row) for row in rows]


def reconstruct_profile(z, Ts, A1, A2, A3, H, Tb):
    return (Ts + (Tb - Ts) * (z / H)
            + A1 * np.sin(np.pi * z / H)
            + A2 * np.sin(2 * np.pi * z / H)
            + A3 * np.sin(3 * np.pi * z / H))


# --- Brine volume formulas -----------------------------------------------

def brine_volume_frankenstein_garner(T, S):
    """Frankenstein & Garner (1967). T in degC, S in ppt. Returns Vb/V (fraction, clamped to [0,1]).

    Uses the Macaulay bracket <-T> = max(-T, eps), NOT |T| -- otherwise T>=0
    (melted/above-freezing, which the reconstructed profile can transiently
    hit near the surface) would be folded onto the same low vb as an
    equally-cold negative T, instead of correctly saturating toward vb=1.
    """
    T = np.asarray(T, dtype=float)
    Tabs = np.maximum(-T, 1e-6)
    vb = S * (49.185 / Tabs + 0.532) * 1e-3
    return np.clip(vb, 0.0, 1.0)


def _cw_coefficients(T):
    """Piecewise F1, F2 polynomial coefficients (Cox & Weeks 1983). Valid -30<=T<=0."""
    conditions = [
        (T > -2.0) & (T <= 0.0),
        (T <= -2.0) & (T >= -22.9),
        (T < -22.9) & (T >= -30.0),
    ]
    F1_sets = [(-0.041221, -18.407, 0.58402, 0.21454),
               (-4.732, -22.45, -0.6397, -0.01074),
               (9899.0, 1309.0, 55.27, 0.7160)]
    F2_sets = [(0.090312, -0.016111, 1.2291e-4, 1.3603e-4),
               (0.08903, -0.01763, -5.330e-4, -8.801e-6),
               (8.547, 1.089, 0.04518, 5.819e-4)]

    def select(coeff_sets):
        return [np.select(conditions, [c[i] for c in coeff_sets], default=np.nan) for i in range(4)]

    return select(F1_sets), select(F2_sets)


def brine_volume_cox_weeks(T, S, air_fraction=0.0):
    """Cox & Weeks (1983) via the 'Sea Ice' book closure (Eq. 1.9-1.11). Vb/V (fraction)."""
    T = np.asarray(T, dtype=float)
    (a1, b1, c1, d1), (a2, b2, c2, d2) = _cw_coefficients(T)
    F1 = a1 + b1 * T + c1 * T ** 2 + d1 * T ** 3
    F2 = a2 + b2 * T + c2 * T ** 2 + d2 * T ** 3

    rho_i = 917.0 - 0.1403 * T  # pure ice density, kg/m^3
    numerator = (1.0 - air_fraction) * (rho_i / 1000.0) * S
    denominator = F1 - (rho_i / 1000.0) * S * F2
    return numerator / denominator


FORMULAS = {
    "Frankenstein & Garner (1967)": brine_volume_frankenstein_garner,
    "Cox & Weeks (1983)": brine_volume_cox_weeks,
}


def flexural_strength_timco_obrien(vb):
    """Timco & O'Brien (1994). vb is Vb/V (fraction, 0-1). Returns sigma_f in MPa."""
    return 1.76 * np.exp(-5.88 * np.sqrt(vb))


def main():
    data = parse_data(RAW_DATA_STRING)
    n = len(data)
    print(f"Loaded {n} points from RAW_DATA_STRING")

    # Precompute the (fixed) depth axis and temperature profile per point --
    # only the brine volume curve depends on the formula/salinity controls.
    points = []
    for A1, A2, A3, Ts, H, lat, lon, px, py in data:
        z = np.linspace(0, H, N_Z)
        T = reconstruct_profile(z, Ts, A1, A2, A3, H, T_BOTTOM)
        points.append({"z": z, "T": T, "H": H, "lat": lat, "lon": lon})

    state = {"idx": 0}

    fig = plt.figure(figsize=(15, 6.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.09, wspace=0.35)
    ax_T = fig.add_subplot(1, 3, 1)
    ax_vb = fig.add_subplot(1, 3, 2)
    ax_sigma = fig.add_subplot(1, 3, 3)

    radio_ax = fig.add_axes((0.08, 0.86, 0.30, 0.11))
    radio_ax.set_title("Brine volume formula", fontsize=9)
    radio = RadioButtons(radio_ax, list(FORMULAS.keys()))

    text_ax = fig.add_axes((0.45, 0.90, 0.13, 0.045))
    text_box = TextBox(text_ax, "Salinity (ppt) ", initial=str(DEFAULT_SALINITY))

    prev_ax = fig.add_axes((0.75, 0.90, 0.05, 0.06))
    next_ax = fig.add_axes((0.92, 0.90, 0.05, 0.06))
    btn_prev = Button(prev_ax, "<")
    btn_next = Button(next_ax, ">")

    label_ax = fig.add_axes((0.81, 0.90, 0.10, 0.06))
    label_ax.set_axis_off()
    idx_text = label_ax.text(0.5, 0.5, "", ha="center", va="center", fontsize=10)

    def redraw(_event=None):
        idx = state["idx"]
        pt = points[idx]
        formula = FORMULAS[radio.value_selected]
        try:
            S = float(text_box.text)
        except ValueError:
            return
        vb_raw = formula(pt["T"], S)
        # Near T=0 the Cox & Weeks polynomial denominator crosses zero (a true
        # singularity of the piecewise fit, not a bug), sending the raw value
        # to +-1000%+ over a fraction of a mm. Mask those as gaps instead of
        # clamping through the pole, which would draw a false 0%->100% spike.
        vb = np.where(np.abs(vb_raw) > 2.0, np.nan, np.clip(vb_raw, 0.0, 1.0))

        sigma_f = flexural_strength_timco_obrien(vb)
        # z is uniformly spaced, so the plain mean over the (few) non-NaN
        # samples is the depth-average over the ice thickness.
        sigma_f_avg = np.nanmean(sigma_f)

        ax_T.clear()
        ax_T.plot(pt["T"], pt["z"], color="tab:blue", lw=2)
        ax_T.axvline(T_BOTTOM, color="navy", ls="--", lw=0.8, alpha=0.6)
        ax_T.invert_yaxis()
        ax_T.set_title("Temperature profile", fontsize=10)
        ax_T.set_xlabel("T (°C)")
        ax_T.set_ylabel("depth (m)")
        ax_T.grid(alpha=0.3)

        ax_vb.clear()
        ax_vb.plot(vb * 100.0, pt["z"], color="tab:red", lw=2)
        ax_vb.invert_yaxis()
        ax_vb.set_title("Brine volume fraction", fontsize=10)
        ax_vb.set_xlabel("brine volume (%)")
        ax_vb.set_ylabel("depth (m)")
        ax_vb.grid(alpha=0.3)

        ax_sigma.clear()
        ax_sigma.plot(sigma_f, pt["z"], color="tab:green", lw=2)
        ax_sigma.axvline(sigma_f_avg, color="black", ls="--", lw=1.0, alpha=0.7)
        ax_sigma.invert_yaxis()
        ax_sigma.set_title(f"Flexural strength (Timco & O'Brien)\nthickness-averaged <σf> = {sigma_f_avg:.3f} MPa",
                            fontsize=10)
        ax_sigma.set_xlabel("σf (MPa)")
        ax_sigma.set_ylabel("depth (m)")
        ax_sigma.grid(alpha=0.3)

        fig.suptitle(f"Point #{idx + 1}/{n}   ({pt['lat']:.3f}, {pt['lon']:.3f})   H={pt['H']:.2f} m",
                      fontsize=11)
        idx_text.set_text(f"{idx + 1} / {n}")
        fig.canvas.draw_idle()

    def go_prev(_event):
        state["idx"] = (state["idx"] - 1) % n
        redraw()

    def go_next(_event):
        state["idx"] = (state["idx"] + 1) % n
        redraw()

    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)
    radio.on_clicked(redraw)
    text_box.on_submit(redraw)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()
