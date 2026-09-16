#!/usr/bin/env python3
"""
Crops the three simulation/satellite comparison pairs for the paper's
Results section (June 8 13:00, June 10 22:00 -- the breakup, June 19 13:00
-- the final state).

Simulation source: _paper/fig_raw_results/*.jpg, rendered from the
visualizer's schematic-color option (grid_schematic) at 10000x6464 px --
the current production run is 10k resolution; a 20k rerun is expected
later, at which point SIM_OFFSET/SIM_CROP below stay the same (same crop
in *pixels*, same underlying ~50 m grid) but the source files swap in.

Satellite source: _python_code/image_download/nares_kane_basin_stereo/
<date>/final_stereographic_image.png, at 20000x12927 px -- exactly 2x the
simulation frames' resolution (same stereographic projection, same
geographic extent, just twice the pixel density), so the satellite crop
offset/size are simply the simulation ones scaled by SAT_SCALE=2, keeping
both crops covering the identical physical region before the final resize
to RESIZE_TO.

Usage:
    python3 crop_results_images.py
"""
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # these are legitimately large (20000x12927) source images

# --- Configuration -- edit and rerun ---
SIM_OFFSET = (1420, 1600)   # (x, y), top-left corner, in the 10k simulation frame
SIM_CROP = (4000, 4000)     # (width, height)

SAT_SCALE = 2                # satellite images are 2x the simulation frames' resolution
SAT_OFFSET = tuple(v * SAT_SCALE for v in SIM_OFFSET)
SAT_CROP = tuple(v * SAT_SCALE for v in SIM_CROP)

RESIZE_TO = (2000, 2000)    # final output size, both simulation and satellite

SIM_DIR = Path(__file__).resolve().parents[2] / "_paper" / "fig_raw_results"
SAT_DIR = Path(__file__).resolve().parents[1] / "image_download" / "nares_kane_basin_stereo"
OUT_DIR = Path(__file__).resolve().parent
PAPER_FIGURES_DIR = Path(__file__).resolve().parents[2] / "_paper" / "figures"

# (output basename, simulation source file, satellite source date directory)
PAIRS = [
    ("jun08", "j08-13h.jpg", "2024-06-08"),
    ("jun10", "j10-22h.jpg", "2024-06-10"),
    ("jun19", "j19-13h.jpg", "2024-06-19"),
]


def crop_and_resize(src_path, offset, crop_size, resize_to):
    with Image.open(src_path) as im:
        box = (offset[0], offset[1], offset[0] + crop_size[0], offset[1] + crop_size[1])
        cropped = im.crop(box)
        return cropped.resize(resize_to, Image.LANCZOS)


def main():
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for label, sim_file, sat_date in PAIRS:
        sim_src = SIM_DIR / sim_file
        sim_out = crop_and_resize(sim_src, SIM_OFFSET, SIM_CROP, RESIZE_TO)
        sim_out_path = PAPER_FIGURES_DIR / f"results_{label}_sim.jpg"
        sim_out.convert("RGB").save(sim_out_path, quality=92)
        print(f"Wrote {sim_out_path}  (from {sim_src.name}, offset={SIM_OFFSET}, crop={SIM_CROP})")

        sat_src = SAT_DIR / sat_date / "final_stereographic_image.png"
        sat_out = crop_and_resize(sat_src, SAT_OFFSET, SAT_CROP, RESIZE_TO)
        sat_out_path = PAPER_FIGURES_DIR / f"results_{label}_sat.jpg"
        sat_out.convert("RGB").save(sat_out_path, quality=92)
        print(f"Wrote {sat_out_path}  (from {sat_date}/final_stereographic_image.png, "
              f"offset={SAT_OFFSET}, crop={SAT_CROP})")


if __name__ == "__main__":
    main()
