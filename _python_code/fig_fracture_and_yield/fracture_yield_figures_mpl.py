"""
Matplotlib re-creation of ice_fracture_yield_curves.nb's Figures 2 and 3
(fracture envelope / yield surface, piecewise-linear, in p-q space).

Same formulas, same base_config.json-derived parameter values, and the same
Times New Roman / black-only / framed-box style as the Mathematica notebook
next to this file -- kept here instead of touching the notebook itself so
the two can be compared side by side. Output is vector (SVG + PDF directly
from matplotlib, no intermediate rasterization, no inkscape round-trip).

Mirrors kernels.cu's PlasticProjection / Q_From_Yield_Surface piecewise-
linear envelope and yield surface -- see the notebook's Step 1-8 for the
same derivation spelled out cell-by-cell.
"""

import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Arc, FancyArrowPatch
from pathlib import Path

# ----------------------------------------------------------------------
# Step 1/2: base parameters (Pa -> kPa), from base_config.json
# ----------------------------------------------------------------------
q0Pa = 7.75e4
tensileTPa = 7.0e4
fractureAngleDeg = 0   # IceFractureAngle -- 0 in the final calibration: the intact
                        # envelope's compressive leg and the yield surface's second
                        # segment are both flat now, not sloped.
dpPhiDeg = 57
dpThresholdPPa = -1000
yieldFrictionAngleDeg = 0   # IceYieldFrictionAngle -- 0 in the final calibration
# Current run set uses Cf in [0.3, 0.8] (cfResidual corrected from the old
# 0.1 to the calibrated 0.3, method.tex Table 2) -- four states spread
# evenly across that range.
cfFresh = 0.8
cfMid1 = 0.63
cfMid2 = 0.47
cfResidual = 0.3
# kernels.cu's PlasticProjection now keeps two independent compressive caps
# (both xi*factor, base_config.json's IceStrengthFactorCompressive /
# IceStrengthFactorCompressiveYield -- currently both 1.2, hence the same
# formula here, but they're separately tunable so kept as two constants):
#  - compressive_threshold_fracture: caps intact pressure directly in the
#    elastic-domain check (Figure 2) -- NOT scaled by Cf, since intact
#    material has no plastic-strain history yet.
#  - compressive_threshold_yield: re-caps p every step post-crack (Figure 3)
#    -- IS scaled by Cf_now, since worked/softened material holds pressure
#    less well; each softening state therefore has its own, smaller cap.
compressiveThresholdFracturePa = q0Pa / 0.4 * 1.2
compressiveThresholdYieldBasePa = q0Pa / 0.4 * 1.2  # pre-Cf-scaling; actual cap is this * cf

q0 = q0Pa / 1000.0
tensileT = tensileTPa / 1000.0
dpThresholdP = dpThresholdPPa / 1000.0
compressiveThresholdFracture = compressiveThresholdFracturePa / 1000.0
compressiveThresholdYieldBase = compressiveThresholdYieldBasePa / 1000.0

def compressiveThresholdYield(cf):
    return compressiveThresholdYieldBase * cf

# ----------------------------------------------------------------------
# Step 3: derived slopes
# ----------------------------------------------------------------------
fractureAngleTan = np.tan(np.radians(fractureAngleDeg))
dpTanPhi = np.tan(np.radians(dpPhiDeg))
yieldFrictionTan = np.tan(np.radians(yieldFrictionAngleDeg))

# ----------------------------------------------------------------------
# Step 4/5: crossover point, Cf-scaled crossovers
# ----------------------------------------------------------------------
pIntersect = (q0 + dpTanPhi * dpThresholdP) / (dpTanPhi - fractureAngleTan)
qIntersect = q0 + fractureAngleTan * pIntersect

def qCrossover(cf):
    return qIntersect * cf

def pCrossover(cf):
    return dpThresholdP + qCrossover(cf) / dpTanPhi

pCrossoverFresh = pCrossover(cfFresh)
qCrossoverFresh = qCrossover(cfFresh)

# ----------------------------------------------------------------------
# Step 6/7: qIntact[p], qYield[p, cf]
# ----------------------------------------------------------------------
def qIntact(p):
    p = np.asarray(p, dtype=float)
    return np.where(p <= 0, (q0 / tensileT) * (p + tensileT), q0 + fractureAngleTan * p)

def qYield(p, cf):
    p = np.asarray(p, dtype=float)
    steep = (p - dpThresholdP) * dpTanPhi
    shallow = qIntersect * cf + yieldFrictionTan * (p - dpThresholdP - qCrossover(cf) / dpTanPhi)
    return np.where(p < dpThresholdP, 0.0, np.minimum(steep, shallow))

# ----------------------------------------------------------------------
# Step 8: plot ranges
# ----------------------------------------------------------------------
hPad = 20
pMin = -tensileT - hPad
pMax = max(compressiveThresholdFracture, compressiveThresholdYield(cfFresh)) + hPad
qTopEnvelope = float(qIntact(compressiveThresholdFracture))
qTopYieldFresh = float(qYield(compressiveThresholdYield(cfFresh), cfFresh))
qTopYieldResidual = float(qYield(compressiveThresholdYield(cfResidual), cfResidual))
qPlotMax = 1.2 * max(qTopEnvelope, qTopYieldFresh, qTopYieldResidual)

# ----------------------------------------------------------------------
# Style -- Times New Roman throughout, matching the notebook's
# BaseStyle -> {FontSize -> 24, FontFamily -> "Times New Roman"} at
# ImageSize -> 1200; here figwidth_pt * 0.02 keeps the same font-to-width
# ratio at a more matplotlib-typical figure size.
# ----------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['mathtext.cal'] = 'Times New Roman'  # unused (\mathcal), set anyway to silence the findfont fallback warning
# Type 42 (real embedded TrueType outlines) instead of matplotlib's default
# Type 3 -- Type 3 text renders fine visually but isn't real embedded text
# (not searchable/selectable/copyable in the final paper PDF).
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

FIGWIDTH = 12.0    # inches
ASPECT = 0.3       # matches Mathematica's AspectRatio -> 0.3
FIGHEIGHT = FIGWIDTH * ASPECT + 1.0   # + headroom for frame labels/padding
FONTSIZE = FIGWIDTH * 72 * 0.02       # same font-size/width ratio as the .nb (24/1200)
LW = 2.0
LW_THIN = 1.3
MS = 9

# ----------------------------------------------------------------------
# Label positions -- ALL per-annotation nudges live here so they're easy
# to find and tweak without hunting through the plotting code below.
# (dx, dy) entries are point offsets from the labeled data point (same
# convention as ax.annotate(..., textcoords='offset points')).
# ----------------------------------------------------------------------
LABEL_OFFSET = {
    'p_tensile':     (10, -22),   # Fig 2, at (-tensileT, 0)
    'q_shear':       (-46, -6),   # Fig 2, at (0, q0)
    'p_compressive': (0, -22),    # Fig 2, at (compressiveThresholdFracture, 0)
    'p_crush':       (0, -22),    # Fig 3, at (compressiveThresholdYield(cfFresh), 0)
    'p_threshold':   (17, -25),    # Fig 3, at (dpThresholdP, 0) -- own row below the "Pressure, p (kPa)" title; too wide (carries its value) to fit the tick-number row near either the "0" or "-50" tick
    'phi_label':     (-45, 19),      # Fig 3, extra nudge past the arc bisector point
    'dp_callout':    (0, 10),     # Fig 3, "Drucker-Prager" text above the arrow tail
}

# q_yield gets its own constant (not buried in LABEL_OFFSET) since it needed
# tuning separately -- (dx, dy) point offset from (pCrossoverFresh, qCrossoverFresh).
QYIELD_LABEL_OFFSET = (-45, -8)

# Cf = ... labels sit above their own curve, inset from that curve's own
# right end (compressive_threshold_yield(cf)) so they stay inside the frame.
# Inset is a FRACTION of each curve's own p-span (dpThresholdP to its own
# cap), not a fixed kPa distance -- the four caps now differ a lot in length
# (they scale with cf), so a fixed kPa inset pushes the label clean off the
# short, low-cf curves. One entry per curve, in the same order as
# [cfFresh, cfMid1, cfMid2, cfResidual] below -- tune per-curve if one label
# needs to sit differently than the others (e.g. the top/cfFresh curve runs
# close under the dashed envelope, so it gets a smaller CF_LABEL_DY).
#                     cfFresh  cfMid1  cfMid2  cfResidual
CF_LABEL_INSET_FRAC = [0.12,    0.12,   0.12,   0.05    ]  # fraction of this curve's own p-span --
                                                            # cfResidual's curve is so short that a
                                                            # 0.12 inset lands the label right where
                                                            # the phi arc/label already sit; 0.05
                                                            # keeps it close to its own cap instead.
CF_LABEL_DY         = [-25,       -20,      -18,      -15      ]  # points, above the curve -- cfResidual
                                                            # needs to clear the phi arc/label below it.
CF_LABEL_DX         = [93,       75,      75,      +45     ]  # points, extra horizontal nudge --
                                                            # requested: cfResidual's label
                                                            # moved one symbol space left, clear
                                                            # of the phi arc/label next to it.

def new_axes():
    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))
    ax.set_xlim(pMin, pMax)
    ax.set_ylim(0, qPlotMax)
    ax.set_xlabel('Pressure, p (kPa)', fontsize=FONTSIZE)
    ax.set_ylabel('Deviatoric stress, q (kPa)', fontsize=FONTSIZE)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=FONTSIZE * 0.85)
    for spine in ax.spines.values():
        spine.set_linewidth(LW_THIN)
        spine.set_color('black')
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.22)
    return fig, ax

out_dir = Path(__file__).resolve().parent
paper_figures_dir = Path(__file__).resolve().parents[2] / '_paper' / 'figures'

# ========================================================================
# Figure 2 -- fracture envelope
# ========================================================================
fig, ax = new_axes()

p_env = np.linspace(pMin, compressiveThresholdFracture, 400)
ax.plot(p_env, qIntact(p_env), color='black', lw=LW, solid_capstyle='round')
# close the envelope with a solid vertical line at the compressive threshold
ax.plot([compressiveThresholdFracture, compressiveThresholdFracture], [qTopEnvelope, 0], color='black', lw=LW)
# thin dashed guide from p=0 up to q_shear, showing that q_shear is q at p=0
ax.plot([0, 0], [0, q0], color='black', lw=LW_THIN, ls=(0, (6, 5)))

for x, y in [(-tensileT, 0), (0, q0), (compressiveThresholdFracture, 0)]:
    ax.plot(x, y, 'o', color='black', ms=MS, zorder=5)

ax.annotate(r'$p_{\mathrm{tensile}}$', (-tensileT, 0), textcoords='offset points',
            xytext=LABEL_OFFSET['p_tensile'], fontsize=FONTSIZE, ha='center')
ax.annotate(r'$q_{\mathrm{shear}}$', (0, q0), textcoords='offset points',
            xytext=LABEL_OFFSET['q_shear'], fontsize=FONTSIZE, ha='center')
ax.annotate(r'$p_{\mathrm{compressive}}$', (compressiveThresholdFracture, 0), textcoords='offset points',
            xytext=LABEL_OFFSET['p_compressive'], fontsize=FONTSIZE, ha='center')

ax.text((pMin + pMax) / 2, qPlotMax / 2, 'Intact (Elastic)',
        fontsize=FONTSIZE, ha='center', va='center', color='black')
ax.text(compressiveThresholdFracture * 0.35, qPlotMax * 0.91, 'Fractured',
        fontsize=FONTSIZE, ha='center', va='center', color='black')

fig.savefig(out_dir / 'fracture_envelope_mpl.svg', bbox_inches='tight')
fig.savefig(out_dir / 'fracture_envelope_mpl.pdf', bbox_inches='tight')
plt.close(fig)

# ========================================================================
# Figure 3 -- yield surface, four Cf states + Drucker-Prager angle callout
# ========================================================================
fig, ax = new_axes()
# extra bottom margin -- the p_threshold label below carries its numeric
# value (unlike the other point labels) and needs a row of its own,
# clear of both the tick numbers and the "Pressure, p (kPa)" title.
fig.subplots_adjust(bottom=0.30)

cfs = [cfFresh, cfMid1, cfMid2, cfResidual]
# Each softening state has its OWN compressive cap now (compressive_threshold_yield
# = xi*IceStrengthFactorCompressiveYield*Cf_now in kernels.cu) -- softer, more-worked
# states (lower cf) hold less pressure, so their curves terminate further left than
# the freshly-cracked (cfFresh) one, rather than all four sharing one cap.
for cf in cfs:
    p_cap = compressiveThresholdYield(cf)
    p_yield = np.linspace(dpThresholdP, p_cap, 400)
    ax.plot(p_yield, qYield(p_yield, cf), color='black', lw=LW, solid_capstyle='round')
    # closing line at this curve's own cap, down to 0
    ax.plot([p_cap, p_cap], [float(qYield(p_cap, cf)), 0], color='black', lw=LW)
    ax.plot(p_cap, 0, 'o', color='black', ms=MS, zorder=5)

# dashed reference: the intact fracture envelope from Figure 2 (its own,
# unscaled compressive_threshold_fracture cap -- independent of the Cf-scaled
# yield caps above), no labels/points
p_env = np.linspace(-tensileT, compressiveThresholdFracture, 300)
ax.plot(p_env, qIntact(p_env), color='black', lw=LW_THIN, ls=(0, (6, 5)))
ax.plot([compressiveThresholdFracture, compressiveThresholdFracture], [qTopEnvelope, 0],
        color='black', lw=LW_THIN, ls=(0, (6, 5)))

for x, y in [(dpThresholdP, 0), (pCrossoverFresh, qCrossoverFresh)]:
    ax.plot(x, y, 'o', color='black', ms=MS, zorder=5)

# Dashed extension of the (freshest curve's) Drucker-Prager segment beyond
# its own crossover, out to where it would cross the intact envelope's own
# compressive leg -- illustrates (p*, q*) from eq:crossover_base, the
# construction point the crossover is defined relative to.
p_dp_ext = np.linspace(pCrossoverFresh, pIntersect, 100)
ax.plot(p_dp_ext, (p_dp_ext - dpThresholdP) * dpTanPhi, color='black', lw=LW_THIN, ls=(0, (6, 5)))
ax.plot(pIntersect, qIntersect, 'o', color='black', ms=MS, zorder=5)
ax.annotate(r'$(p^{*}, q^{*})$', (pIntersect, qIntersect), textcoords='offset points',
            xytext=(10, 8), fontsize=FONTSIZE, ha='left')

ax.annotate(rf'$p_{{\mathrm{{th}}}}$ = {dpThresholdP:g} kPa', (dpThresholdP, 0),
            textcoords='offset points',
            xytext=LABEL_OFFSET['p_threshold'], fontsize=FONTSIZE, ha='center')
# labeled on the freshest (largest-cap) curve, the one most directly comparable
# to Figure 2's p_compressive point; the other three curves' own, smaller caps
# are visually self-evident from where each one terminates. This is the
# Cf-scaled yield-side cap, p_crush(cf) -- distinct from Figure 2's unscaled
# p_compressive, so it gets its own label.
ax.annotate(r'$p_{\mathrm{crush}}$', (compressiveThresholdYield(cfFresh), 0), textcoords='offset points',
            xytext=LABEL_OFFSET['p_crush'], fontsize=FONTSIZE, ha='center')
ax.annotate(r'$q_{\mathrm{yield}}$', (pCrossoverFresh, qCrossoverFresh), textcoords='offset points',
            xytext=QYIELD_LABEL_OFFSET, fontsize=FONTSIZE, ha='center')

# Cf = ... labels sit above their own curve, inset from that curve's own
# cap by a fraction of its own p-span (CF_LABEL_INSET_FRAC/CF_LABEL_DY/
# CF_LABEL_DX, set per-curve in the constants block above).
for cf, frac, dy, dx in zip(cfs, CF_LABEL_INSET_FRAC, CF_LABEL_DY, CF_LABEL_DX):
    p_cap = compressiveThresholdYield(cf)
    x_cf_label = p_cap - frac * (p_cap - dpThresholdP)
    y = float(qYield(x_cf_label, cf))
    ax.annotate(rf'$c_f$ = {cf:g}', (x_cf_label, y), textcoords='offset points',
                xytext=(dx, dy), fontsize=FONTSIZE, ha='center', va='bottom')

# Drucker-Prager friction angle phi -- arc from horizontal to the shared
# steep leg, centered at p_threshold.
arc_r = 0.3 * (pCrossoverFresh - dpThresholdP)
ax.add_patch(Arc((dpThresholdP, 0), 2 * arc_r, 2 * arc_r, angle=0,
                  theta1=0, theta2=dpPhiDeg, color='black', lw=LW_THIN))
phi_base_x = dpThresholdP + 0.42 * (pCrossoverFresh - dpThresholdP) * np.cos(np.radians(dpPhiDeg / 2))
phi_base_y = 0.42 * (pCrossoverFresh - dpThresholdP) * np.sin(np.radians(dpPhiDeg / 2))
ax.annotate('φ=57°', (phi_base_x, phi_base_y), textcoords='offset points',
            xytext=LABEL_OFFSET['phi_label'], fontsize=FONTSIZE, ha='center', va='center')

# "Drucker-Prager" callout arrow, pointing at the shared steep leg -- tail's
# q-position kept proportional to qPlotMax (not a fixed kPa value) so it
# stays inside the frame regardless of how tall the plot ends up being.
arrow_tail = (-50, 0.72 * qPlotMax)
arrow_head = (dpThresholdP + 0.55 * (pCrossoverFresh - dpThresholdP), 0.55 * qCrossoverFresh)
ax.add_patch(FancyArrowPatch(arrow_tail, arrow_head, arrowstyle='-|>',
                              mutation_scale=14, color='black', lw=LW_THIN))
ax.annotate('Drucker-Prager', arrow_tail, textcoords='offset points',
            xytext=(0, 10), fontsize=FONTSIZE, ha='center')

fig.savefig(out_dir / 'yield_surface_mpl.svg', bbox_inches='tight')
fig.savefig(out_dir / 'yield_surface_mpl.pdf', bbox_inches='tight')
plt.close(fig)

print('Wrote fracture_envelope_mpl.svg/.pdf and yield_surface_mpl.svg/.pdf to', out_dir)

# The paper's own figures/ directory needs these same files (MPM_Paper.tex's
# \graphicspath points there) -- copied rather than generated there directly,
# so this script's outputs live alongside the script itself like every other
# fig-prep script in _python_code/, with the paper copy kept in sync here.
for name in ('fracture_envelope_mpl.svg', 'fracture_envelope_mpl.pdf',
             'yield_surface_mpl.svg', 'yield_surface_mpl.pdf'):
    shutil.copy2(out_dir / name, paper_figures_dir / name)
print('Copied to', paper_figures_dir)
