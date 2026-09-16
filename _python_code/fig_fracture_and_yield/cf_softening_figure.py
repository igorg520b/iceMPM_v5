"""
Plot of the cohesion-softening state c_f as a function of accumulated
plastic shear strain gamma_p (MPM_Paper.tex eq:cf_softening / eq. 23):

    c_f(gamma_p) = c_f^res + (c_f^0 - c_f^res) * exp(-gamma_p / gamma_star)

A version of this curve was shown in the slides; this is a from-scratch
redraw in the same Times New Roman / black-only / framed-box style as
Figures 5 and 6 (fracture_yield_figures_mpl.py), rather than a copy of the
slide version, so kept as its own script instead of importing that file
(which is a flat, unguarded script -- importing it would re-run its own
plotting/save calls as a side effect).

c_f^0 and c_f^res below must be kept in sync with cfFresh/cfResidual in
fracture_yield_figures_mpl.py by hand; there is no shared import between
the two for the reason above.

gamma_star: confirmed value in use is 0.16 (base_config.json's own
"GammaStar" key is stale at the time of writing -- 0.16 is the value
actually used for the production run, per direct confirmation).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# ----------------------------------------------------------------------
# Style -- Times New Roman throughout, matching make_thermal_figure.py /
# make_strength_figure.py / fracture_yield_figures_mpl.py.
# ----------------------------------------------------------------------
FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
TITLE_FONT_SIZE = 16

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['mathtext.cal'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# ----------------------------------------------------------------------
# Parameters -- edit and rerun. Must match cfFresh/cfResidual in
# fracture_yield_figures_mpl.py.
# ----------------------------------------------------------------------
CF_FRESH = 0.9       # c_f^0, fresh-crack value
CF_RESIDUAL = 0.3    # c_f^res, fully-softened residual value
GAMMA_STAR = 0.16     # characteristic strain scale -- confirmed value actually used

GAMMA_P_MAX = 0.6     # x-axis range, strain units -- tight enough to show the
                       # decay's shape given the current gamma_star
N_POINTS = 400

out_dir = Path(__file__).resolve().parent
paper_figures_dir = Path(__file__).resolve().parents[2] / '_paper' / 'figures'


def cf_softening(gamma_p, cf0, cf_res, gamma_star):
    return cf_res + (cf0 - cf_res) * np.exp(-gamma_p / gamma_star)


def main():
    gamma_p = np.linspace(0, GAMMA_P_MAX, N_POINTS)
    cf = cf_softening(gamma_p, CF_FRESH, CF_RESIDUAL, GAMMA_STAR)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(gamma_p, cf, color='black', lw=2.0, solid_capstyle='round')

    ax.axhline(CF_FRESH, color='black', lw=0.8, ls=(0, (6, 5)))
    ax.axhline(CF_RESIDUAL, color='black', lw=0.8, ls=(0, (6, 5)))
    ax.annotate(r'$c_f^{0}$', (GAMMA_P_MAX, CF_FRESH), textcoords='offset points',
                xytext=(-30, 9), fontsize=FONT_SIZE, ha='left')
    ax.annotate(r'$c_f^{\mathrm{res}}$', (GAMMA_P_MAX, CF_RESIDUAL), textcoords='offset points',
                xytext=(-38, 16), fontsize=FONT_SIZE, ha='left')

    ax.annotate(r'$\gamma^{*}$', (GAMMA_STAR, cf_softening(GAMMA_STAR, CF_FRESH, CF_RESIDUAL, GAMMA_STAR)),
                textcoords='offset points', xytext=(8, 10), fontsize=FONT_SIZE, ha='left')
    ax.plot(GAMMA_STAR, cf_softening(GAMMA_STAR, CF_FRESH, CF_RESIDUAL, GAMMA_STAR),
            'o', color='black', ms=6, zorder=5)

    ax.set_xlim(0, GAMMA_P_MAX)
    ax.set_ylim(0, CF_FRESH * 1.15)
    ax.set_xlabel(r'Accumulated plastic shear strain, $\gamma_{\mathrm{p}}$', fontsize=FONT_SIZE)
    ax.set_ylabel(r'Cohesion-softening state, $c_f$', fontsize=FONT_SIZE)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=FONT_SIZE * 0.85)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color('black')

    fig.tight_layout()

    fig.savefig(out_dir / 'cf_softening_mpl.svg', bbox_inches='tight')
    fig.savefig(out_dir / 'cf_softening_mpl.pdf', bbox_inches='tight')
    fig.savefig(out_dir / 'cf_softening_mpl.png', dpi=200, bbox_inches='tight')
    print(f"Wrote cf_softening_mpl.svg/.pdf/.png to {out_dir}")

    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    for name in ('cf_softening_mpl.svg', 'cf_softening_mpl.pdf'):
        shutil.copy2(out_dir / name, paper_figures_dir / name)
    print(f"Copied to {paper_figures_dir}")


if __name__ == "__main__":
    main()
