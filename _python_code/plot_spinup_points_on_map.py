#!/usr/bin/env python3
"""
Places a sample of thermal spin-up points on the project's color image and
draws each point's reconstructed vertical temperature profile (formula (R),
see heat_equation_spec.md) alongside it.

Input: paste the console line printed by DataPreparer::ThermalSpinUpSave
(preparer/data_preparer.cpp) into RAW_DATA_STRING below. Each entry is
{A1, A2, A3, Ts_old, thickness, lat, lon, pixel_x, pixel_y} -- pixel_x/
pixel_y are already in the same top-down row convention as the original
ImageColor PNG (row 0 = top), so they can be plotted directly.

Configuration is via the plain variable assignments below (no command-line
arguments) -- edit these and rerun.
"""
import ast

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# --- Configuration -- edit and rerun ---
IMAGE_PATH = "../_input_data/nares_2k/color_2k.png"
OUTPUT_PATH = "spinup_points_map.png"
SHOW_PLOT = True

FIG_WIDTH_PX = 1920
FIG_HEIGHT_PX = 1080
DPI = 120

T_BOTTOM = -1.8  # deg C, SimParams::T_bottom

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


def main():
    data = parse_data(RAW_DATA_STRING)
    n = len(data)
    print(f"Loaded {n} points from RAW_DATA_STRING")

    img = mpimg.imread(IMAGE_PATH)

    fig = plt.figure(figsize=(FIG_WIDTH_PX / DPI, FIG_HEIGHT_PX / DPI), dpi=DPI)
    n_rows = int(np.ceil(n / 2))
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[2.2, 1, 1], wspace=0.35, hspace=0.6)

    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.imshow(img)  # default origin='upper': row 0 = top, matching pixel_x/pixel_y
    ax_map.set_title("Sampled spin-up points")
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for k, row in enumerate(data):
        A1, A2, A3, Ts, H, lat, lon, px, py = row
        color = colors[k % len(colors)]

        ax_map.plot(px, py, "o", color=color, markersize=10, markeredgecolor="black")
        ax_map.annotate(str(k + 1), (px, py), color="white", fontsize=9, fontweight="bold",
                         ha="center", va="center")

        ax_p = fig.add_subplot(gs[k // 2, 1 + k % 2])
        z = np.linspace(0, H, 200)
        T = reconstruct_profile(z, Ts, A1, A2, A3, H, T_BOTTOM)
        ax_p.plot(T, z, color=color, lw=2)
        ax_p.axvline(T_BOTTOM, color="navy", ls="--", lw=0.8, alpha=0.6)
        ax_p.invert_yaxis()
        ax_p.set_title(f"#{k + 1}  ({lat:.3f}, {lon:.3f})  H={H:.2f} m", fontsize=9)
        ax_p.set_xlabel("T (°C)", fontsize=8)
        ax_p.set_ylabel("depth (m)", fontsize=8)
        ax_p.tick_params(labelsize=7)
        ax_p.grid(alpha=0.3)
        for spine in ax_p.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    fig.savefig(OUTPUT_PATH, dpi=DPI)
    print(f"Saved {OUTPUT_PATH}")
    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
