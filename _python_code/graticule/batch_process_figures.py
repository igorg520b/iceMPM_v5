#!/usr/bin/env python3
"""
Runs add_graticule_frame.py's framing pipeline over a batch of input images,
then shrinks each result 5x (to 20%) with vips, preserving aspect ratio.

For each <stem>.<ext> in FILES:
  1. add_graticule_frame.make_figure() renders the framed figure to
     <stem>_.jpg (JPG regardless of the input's original format).
  2. vips resizes <stem>_.jpg down to 20% in place.

All the framing look (LAT_MIN, BORDER_WIDTH, colors, padding, etc.) is
controlled by the constants at the top of add_graticule_frame.py -- edit
those, not this file, to change how the figures look. This script only
loops over filenames and drives the resize step.

Usage:
    python3 batch_process_figures.py
"""
import subprocess
from pathlib import Path

import add_graticule_frame as g

FILES = [
    "fig08a.jpg", "fig08b.png",
    "fig09a.jpg", "fig09b.png",
    "fig10a.jpg", "fig10b.png",
    "fig11a.jpg", "fig11b.png",
    "fig12a.jpg", "fig12b.png",
]
RESIZE_SCALE = 0.2  # 5x smaller


def output_name(input_name):
    return f"{Path(input_name).stem}_.jpg"


def main():
    outputs = []
    for input_name in FILES:
        if not Path(input_name).exists():
            print(f"[skip] {input_name} not found")
            continue
        output_name_ = output_name(input_name)
        g.make_figure(input_name, output_name_)
        outputs.append(output_name_)

    print(f"\nResizing {len(outputs)} figures to {int(RESIZE_SCALE * 100)}% with vips...")
    for output_name_ in outputs:
        tmp_name = f"{Path(output_name_).stem}_resized_tmp.jpg"
        subprocess.run(["vips", "resize", output_name_, tmp_name, str(RESIZE_SCALE)], check=True)
        Path(tmp_name).replace(output_name_)
        print(f"[resized] {output_name_}")


if __name__ == "__main__":
    main()
