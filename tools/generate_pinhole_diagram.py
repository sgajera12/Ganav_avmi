"""
generate_pinhole_diagram.py
Generates a clean black-and-white pinhole camera model diagram for the paper.
Output: results/pinhole_diagram.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import os

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('white')

for ax in axes:
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')

# ── shared style ──────────────────────────────────────────────────────────────
AX_KW  = dict(color='black', lw=1.4)
ARR_KW = dict(color='black', lw=1.2,
              arrowstyle='->', mutation_scale=10)
DOT_KW = dict(color='black', zorder=5)
TXT_KW = dict(fontsize=10, color='black', ha='center', va='center',
              fontfamily='serif')

def arrow(ax, x1, y1, x2, y2, **kw):
    d = dict(**ARR_KW); d.update(kw)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=d.pop('arrowstyle','-|>'),
                                color=d.pop('color','black'),
                                lw=d.pop('lw',1.2)))

def txt(ax, x, y, s, **kw):
    d = dict(**TXT_KW); d.update(kw)
    ax.text(x, y, s, **d)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT panel — 3D perspective view
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-1, 8); ax.set_ylim(-1.5, 5)

# --- camera centre C ---
C = np.array([0.0, 0.0])
ax.plot(*C, 'ko', ms=5)
txt(ax, -0.45, -0.2, 'C', fontsize=11, ha='right', fontfamily='serif',
    fontstyle='italic')
txt(ax, -0.45, -0.6, 'camera\ncentre', fontsize=8, ha='right')

# --- principal axis (Z) ---
arrow(ax, C[0], C[1], 7.2, 0.0)
txt(ax, 7.5, 0.0, '$Z$', fontsize=11, fontstyle='italic')

# --- image plane (vertical line at Z=2) ---
ax.plot([2, 2], [-1.1, 2.5], 'k-', lw=1.2)
txt(ax, 1.55, -1.3, 'image\nplane', fontsize=8)
# label f
ax.annotate('', xy=(2.0, -0.65), xytext=(0.0, -0.65),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
txt(ax, 1.0, -0.9, '$f$', fontsize=11, fontstyle='italic')

# --- 3D point X ---
X3 = np.array([5.5, 3.2])
ax.plot(*X3, 'ko', ms=5)
txt(ax, X3[0]+0.3, X3[1]+0.2, '$\\mathbf{X}$', fontsize=12, ha='left',
    fontweight='bold')

# --- Y axis (up) ---
arrow(ax, C[0], C[1], -0.6, 3.2)
txt(ax, -0.85, 3.4, '$Y$', fontsize=11, fontstyle='italic')

# --- X axis (into page — dashed diagonal) ---
ax.annotate('', xy=(1.5, 1.8), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2,
                            linestyle='dashed'))
txt(ax, 1.7, 2.0, '$X$', fontsize=11, fontstyle='italic')

# --- projection ray: C → X3, continues to image plane ---
# intersection with image plane x=2: t = 2/X3[0]
t_ip = 2.0 / X3[0]
p_ip = t_ip * X3   # projected point on image plane

ax.plot([C[0], X3[0]], [C[1], X3[1]], 'k--', lw=0.9)  # ray
ax.plot(*p_ip, 'ko', ms=4)
txt(ax, p_ip[0]-0.35, p_ip[1], '$p$', fontsize=11, fontstyle='italic',
    ha='right')

# small axes at p on image plane
ax.annotate('', xy=(p_ip[0]+0.0, p_ip[1]+0.9),
            xytext=p_ip,
            arrowprops=dict(arrowstyle='->', color='black', lw=0.9))
txt(ax, p_ip[0]+0.25, p_ip[1]+0.95, '$y$', fontsize=9, fontstyle='italic')

ax.annotate('', xy=(p_ip[0]+0.75, p_ip[1]+0.0),
            xytext=p_ip,
            arrowprops=dict(arrowstyle='->', color='black', lw=0.9))
txt(ax, p_ip[0]+0.85, p_ip[1]-0.2, '$x$', fontsize=9, fontstyle='italic')

# fY/Z annotation (vertical dashed from X3 to Z-axis)
ax.plot([X3[0], X3[0]], [0, X3[1]], 'k:', lw=0.8)
ax.plot([0, X3[0]], [0, 0], 'k:', lw=0.8)

# title
txt(ax, 3.5, 4.6, '(a) 3D Perspective', fontsize=10, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT panel — 2D side view
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_xlim(-1.5, 8); ax.set_ylim(-2.0, 5)

# --- camera centre ---
C2 = np.array([0.0, 0.0])
ax.plot(*C2, 'ko', ms=5)
txt(ax, -0.5, -0.3, '$C$', fontsize=11, fontstyle='italic', ha='right')

# --- Z axis ---
arrow(ax, -0.1, 0, 7.2, 0)
txt(ax, 7.5, 0.0, '$Z$', fontsize=11, fontstyle='italic')

# --- Y axis ---
arrow(ax, 0, -0.1, 0, 4.2)
txt(ax, 0.25, 4.4, '$Y$', fontsize=11, fontstyle='italic')

# --- image plane at Z=f=2 ---
ax.plot([2, 2], [-1.4, 3.2], 'k-', lw=1.2)

# --- f annotation ---
ax.annotate('', xy=(2.0, -1.1), xytext=(0.0, -1.1),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
txt(ax, 1.0, -1.4, '$f$', fontsize=11, fontstyle='italic')

# --- 3D point (Y, Z) ---
Y3, Z3 = 3.5, 5.5
ax.plot([Z3], [Y3], 'ko', ms=5)
txt(ax, Z3+0.25, Y3+0.2, '$Y$', fontsize=11, fontstyle='italic')

# dashed lines from Y3 and Z3 to axes
ax.plot([0, Z3], [Y3, Y3], 'k--', lw=0.8)
ax.plot([Z3, Z3], [0, Y3], 'k:', lw=0.8)

# --- projection ray ---
t_ip2 = 2.0 / Z3
p_y   = t_ip2 * Y3    # fY/Z
ax.plot([C2[0], Z3], [C2[1], Y3], 'k-', lw=0.9)
ax.plot([2.0], [p_y], 'ko', ms=4)

# --- fY/Z label with bracket ---
ax.annotate('', xy=(2.25, p_y), xytext=(2.25, 0),
            arrowprops=dict(arrowstyle='<->', color='black', lw=0.9))
txt(ax, 3.15, p_y/2, '$fY/Z$', fontsize=10, fontstyle='italic', ha='left')

# principal point p
ax.plot([2.0], [0.0], 'ko', ms=4)
txt(ax, 2.3, -0.3, '$p$', fontsize=11, fontstyle='italic')

# right-angle marker at p
sq = 0.15
ax.plot([2, 2+sq, 2+sq], [sq, sq, 0], 'k-', lw=0.8)

# title
txt(ax, 3.0, 4.6, '(b) Side View', fontsize=10, fontweight='bold')

# ── overall equation at top ────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         r'$(X, Y, Z)^\top \;\longrightarrow\; \left(\frac{fX}{Z},\; \frac{fY}{Z}\right)^\top$',
         ha='center', va='top', fontsize=14,
         fontfamily='serif')

fig.text(0.5, 0.90,
         r'$u = \frac{f_x \cdot X}{Z} + c_x \qquad v = \frac{f_y \cdot Y}{Z} + c_y$',
         ha='center', va='top', fontsize=12,
         fontfamily='serif')

plt.tight_layout(rect=[0, 0, 1, 0.88])
os.makedirs('results', exist_ok=True)
plt.savefig('results/pinhole_diagram.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print('Saved: results/pinhole_diagram.png')
