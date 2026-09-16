#!/usr/bin/env python3
"""
Illustrates, for the supervisor comment on mpm_discretization.tex's contact
normal formula, what m_0, m_1, and grad(m_0 - m_1) actually are (eq. 43:
n_hat = -grad(m_0 - m_1) / ||grad(m_0 - m_1)||).

The comment ("does not make sense, m_0 is conserved, what does grad(m_0)
mean?") conflates two different things:
  - per-particle mass conservation: each material point's own carried mass
    never changes over time -- a statement about ONE particle, over TIME.
  - the nodal mass field m_0(x): the sum, at a fixed instant, of nearby
    intact-field particles' masses projected onto each background grid
    node via the usual P2G interpolation weights -- a smooth, well-defined
    function of POSITION, at one instant. m_1(x) is the same thing for the
    fractured field. Two separate material fields sharing one background
    grid naturally have two separate, spatially-varying nodal mass
    fields, and it is completely ordinary to take the spatial gradient of
    a difference of two fields -- no different from taking grad(p) or
    grad(T) elsewhere in the paper.

This script draws a simple 1D strip of grid nodes with a synthetic
(illustrative, not simulated) example: intact ice on the left, fractured
ice on the right, and a transition/contact zone in the middle where BOTH
fields have nonzero nodal mass at the same nodes (exactly the situation
eq. 43 is written for). It shows m_0(x), m_1(x), and (m_0 - m_1)(x), and
the resulting contact normal direction from eq. 43.

Output: contact_normal_diagram.pdf, saved to _paper/comments_figures/.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

FONT_SIZE = 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '_paper', 'comments_figures')

# Synthetic 1D row of 9 grid nodes -- illustrative only, not simulation output.
x = np.arange(9)
m0 = np.array([5, 5, 5, 4, 2, 1, 0, 0, 0], dtype=float)  # intact field, nodal mass
m1 = np.array([0, 0, 0, 1, 3, 4, 5, 5, 5], dtype=float)  # fractured field, nodal mass
diff = m0 - m1

# central-difference gradient of (m0 - m1) along x (spacing = 1)
grad = np.gradient(diff, x)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

ax1.plot(x, m0, 'o-', color='steelblue', label=r'$m_0(x)$ (intact field)', linewidth=2)
ax1.plot(x, m1, 's-', color='firebrick', label=r'$m_1(x)$ (fractured field)', linewidth=2)
ax1.axvspan(3, 5, color='gray', alpha=0.15)
ax1.text(4, 7.0, 'both fields present\nat the same nodes', ha='center', va='center', fontsize=10, style='italic')
ax1.set_ylabel('nodal mass')
ax1.legend(loc='upper right', frameon=False)
ax1.set_title('Two background grids, co-located node for node')
ax1.set_ylim(-0.5, 8.5)

ax2.plot(x, diff, 'D-', color='black', linewidth=2, label=r'$(m_0-m_1)(x)$')
ax2.axhline(0, color='gray', linewidth=0.8)
ax2.axvspan(3, 5, color='gray', alpha=0.15)

# annotate the contact normal at the middle of the transition zone (node 4)
i0 = 4
gx = grad[i0]
n_hat = -np.sign(gx)  # direction only, 1D
ax2.annotate('', xy=(i0 + 0.9 * n_hat, diff[i0] - 0.3), xytext=(i0, diff[i0] - 0.3),
             arrowprops=dict(arrowstyle='-|>', color='darkgreen', lw=2.5))
ax2.text(i0, diff[i0] - 2.2,
         r'$\hat n = -\nabla(m_0-m_1)\,/\,|\nabla(m_0-m_1)|$'
         '\npoints from intact into fractured field',
         color='darkgreen', fontsize=10, ha='center', va='top')

ax2.set_xlabel('grid node position (schematic)')
ax2.set_ylabel(r'$m_0-m_1$')
ax2.set_ylim(-7.5, 6.5)
ax2.legend(loc='upper right', frameon=False)
ax2.set_title(r'Spatial gradient of the mass difference gives the contact normal (eq. 43)')

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
outpath = os.path.join(OUT_DIR, 'contact_normal_diagram.pdf')
plt.savefig(outpath)
print(f"Saved {outpath}")

# ---------------------------------------------------------------------------
# Second figure: the central-difference stencil itself, on a small 2D patch
# of the background grid -- how d(m0-m1)/dx and d(m0-m1)/dy are actually
# computed from a node's four neighbors.
# ---------------------------------------------------------------------------

fig2, ax = plt.subplots(figsize=(5.5, 5.5))

h = 1.0  # grid spacing, schematic
cx, cy = 0.0, 0.0  # center node (i, j)

nodes = {
    'center': (cx, cy),
    'left':   (cx - h, cy),
    'right':  (cx + h, cy),
    'down':   (cx, cy - h),
    'up':     (cx, cy + h),
}

# illustrative (m0 - m1) values at each node
vals = {'center': 1.0, 'left': 3.0, 'right': -1.0, 'down': 0.5, 'up': 1.5}

# label offsets chosen so no label sits on either arrow
label_offsets = {
    'left':   (0.0, -0.24),
    'right':  (0.0, -0.24),
    'up':     (-0.22, 0.0),
    'down':   (-0.22, 0.0),
    'center': (0.4, -0.35),
}

for key, (px, py) in nodes.items():
    ax.plot(px, py, 'o', color='black', markersize=8, zorder=3)
    label = f"{vals[key]:+.1f}"
    dx_off, dy_off = label_offsets[key]
    ax.text(px + dx_off, py + dy_off, label, fontsize=11, ha='center', va='center')

# grid lines through the stencil, for context
ax.axhline(cy, color='lightgray', linewidth=1, zorder=0)
ax.axvline(cx, color='lightgray', linewidth=1, zorder=0)
ax.axhline(cy + h, color='lightgray', linewidth=0.6, zorder=0)
ax.axhline(cy - h, color='lightgray', linewidth=0.6, zorder=0)
ax.axvline(cx + h, color='lightgray', linewidth=0.6, zorder=0)
ax.axvline(cx - h, color='lightgray', linewidth=0.6, zorder=0)

# arrow for the x-derivative: left neighbor -> right neighbor
ax.annotate('', xy=(nodes['right'][0] - 0.15, cy), xytext=(nodes['left'][0] + 0.15, cy),
            arrowprops=dict(arrowstyle='-|>', color='steelblue', lw=2))

# arrow for the y-derivative: down neighbor -> up neighbor
ax.annotate('', xy=(cx, nodes['up'][1] - 0.15), xytext=(cx, nodes['down'][1] + 0.15),
            arrowprops=dict(arrowstyle='-|>', color='firebrick', lw=2))

ax.text(cx, 1.9,
         r'$\dfrac{\partial(m_0-m_1)}{\partial x}\approx\dfrac{(m_0-m_1)_{\mathrm{right}}-(m_0-m_1)_{\mathrm{left}}}{2h}$',
         color='steelblue', fontsize=11, ha='center', va='bottom')
ax.text(cx, -1.9,
         r'$\dfrac{\partial(m_0-m_1)}{\partial y}\approx\dfrac{(m_0-m_1)_{\mathrm{up}}-(m_0-m_1)_{\mathrm{down}}}{2h}$',
         color='firebrick', fontsize=11, ha='center', va='top')

ax.set_xlim(-1.9, 1.9)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Central-difference stencil for $\\nabla(m_0-m_1)$ at one node\n'
              '(numbers shown: illustrative $(m_0-m_1)$ values)', fontsize=12)

plt.tight_layout()
outpath2 = os.path.join(OUT_DIR, 'central_difference_stencil.pdf')
plt.savefig(outpath2)
print(f"Saved {outpath2}")
