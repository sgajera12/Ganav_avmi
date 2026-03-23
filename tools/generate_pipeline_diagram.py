"""
generate_pipeline_diagram.py  — v2 clean grid layout
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(24, 14))
ax.set_xlim(0, 24); ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('#F0F4F8')
ax.set_facecolor('#F0F4F8')

# ── colour scheme ─────────────────────────────────────────────────────────────
IN_BG, IN_BD   = '#D6EAF8', '#2980B9'
SEG_BG,SEG_BD  = '#D5F5E3', '#1E8449'
FUS_BG,FUS_BD  = '#E8DAEF', '#7D3C98'
OUT_BG,OUT_BD  = '#FDECEA', '#C0392B'
HDR_BG         = '#2C3E50'

def box(x, y, w, h, fc='#fff', ec='#333', lw=1.5, alpha=1, r=0.2, z=3):
    p = FancyBboxPatch((x,y), w, h, boxstyle=f'round,pad={r}',
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(p); return p

def txt(x, y, s, fs=8.5, c='#1A1A2E', ha='center', va='center',
        bold=False, italic=False, z=5):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, zorder=z,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            multialignment='center')

def arr(x1,y1,x2,y2, c='#2C3E50', lw=1.8, z=4):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                connectionstyle='arc3,rad=0.0'), zorder=z)

def section_header(x, y, w, label, color):
    box(x, y, w, 0.55, fc=color, ec=color, lw=0, r=0.15, z=4)
    txt(x+w/2, y+0.275, label, fs=10, c='white', bold=True, z=5)

def divider(x, y1, y2, c='#BDC3C7'):
    ax.plot([x,x],[y1,y2], color=c, lw=1.5, ls='--', zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TITLE
# ══════════════════════════════════════════════════════════════════════════════
box(0.2, 13.0, 23.6, 0.85, fc=HDR_BG, ec=HDR_BG, lw=0, r=0.2)
txt(12.0, 13.42, 'Semantic Scene Understanding: GANav Segmentation + LiDAR–Camera Fusion Pipeline',
    fs=13, c='white', bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS
# ══════════════════════════════════════════════════════════════════════════════
box(0.2,  0.3, 4.2, 12.3, fc=IN_BG,  ec=IN_BD,  lw=2.0, alpha=0.5, r=0.3, z=1)
box(4.6,  0.3, 8.6, 12.3, fc=SEG_BG, ec=SEG_BD, lw=2.0, alpha=0.5, r=0.3, z=1)
box(13.4, 0.3, 5.4, 12.3, fc=FUS_BG, ec=FUS_BD, lw=2.0, alpha=0.5, r=0.3, z=1)
box(19.0, 0.3, 4.8, 12.3, fc=OUT_BG, ec=OUT_BD, lw=2.0, alpha=0.5, r=0.3, z=1)

# Section headers
section_header(0.2,  12.15, 4.2,  'INPUT',               IN_BD)
section_header(4.6,  12.15, 8.6,  'SEGMENTATION BRANCH', SEG_BD)
section_header(13.4, 12.15, 5.4,  'FUSION BRANCH',       FUS_BD)
section_header(19.0, 12.15, 4.8,  'OUTPUT',               OUT_BD)

# ══════════════════════════════════════════════════════════════════════════════
# COL 1 — INPUT
# ══════════════════════════════════════════════════════════════════════════════
# Camera
box(0.4, 8.8, 3.8, 3.0, fc='white', ec=IN_BD)
txt(2.3, 11.45, 'RGB Camera Frame', fs=9.5, c=IN_BD, bold=True)
txt(2.3, 11.0,  '640 x 480 px  |  bgr8 encoding', fs=8)
txt(2.3, 10.6,  'Topic: /camera/right/image_raw', fs=7.5, c='#666', italic=True)
txt(2.3, 10.15, 'Rate: ~25 Hz  |  ROS2 sensor_msgs/Image', fs=7.5, c='#555')
box(0.6, 8.95, 3.4, 0.95, fc='#EBF5FB', ec=IN_BD, lw=1, r=0.1)
txt(2.3,  9.42, 'Source: Unreal Engine 5 simulation', fs=8, c=IN_BD)

# LiDAR
box(0.4, 4.8, 3.8, 3.6, fc='white', ec=IN_BD)
txt(2.3,  8.05, 'LiDAR Point Cloud', fs=9.5, c=IN_BD, bold=True)
txt(2.3,  7.6,  'N x 4  float32:  [X, Y, Z, RGB_packed]', fs=8)
txt(2.3,  7.2,  '~4,700 points per frame  |  ~30 Hz', fs=7.5, c='#555')
txt(2.3,  6.8,  'Topic: /lidar/points2', fs=7.5, c='#666', italic=True)
txt(2.3,  6.4,  'sensor_msgs/PointCloud2  |  CDR encoded', fs=7.5, c='#555')
txt(2.3,  6.0,  'RGB packed as float32 bit-cast to uint32', fs=7.5, c='#555')
box(0.6, 4.95, 3.4, 0.7, fc='#EBF5FB', ec=IN_BD, lw=1, r=0.1)
txt(2.3,  5.30, 'Source: Velodyne VLP-16 (simulated)', fs=8, c=IN_BD)

# Time-sync
box(0.4, 2.2, 3.8, 2.25, fc='white', ec='#1565C0')
txt(2.3, 4.1,  'Timestamp Synchronisation', fs=9, c='#1565C0', bold=True)
txt(2.3, 3.65, 'ApproxTimeSynchronizer (ROS2)', fs=8)
txt(2.3, 3.25, 'slop = 200 ms', fs=8, c='#555')
txt(2.3, 2.85, 'Nearest LiDAR frame to each', fs=7.5, c='#555')
txt(2.3, 2.45, 'camera frame  (802 sync pairs)', fs=7.5, c='#555')

arr(2.3, 8.8, 2.3, 8.35)
arr(2.3, 4.8, 2.3, 4.45)

# ══════════════════════════════════════════════════════════════════════════════
# COL 2a — SEGMENTATION BRANCH (camera path, top)
# ══════════════════════════════════════════════════════════════════════════════
SEG_X = 5.0
SW = 7.7   # sub-box width inside seg column

# Pre-processing
box(SEG_X, 10.6, SW, 1.3, fc='white', ec=SEG_BD)
txt(SEG_X+SW/2, 11.6,  'Image Pre-processing', fs=9.5, c=SEG_BD, bold=True)
txt(SEG_X+SW/2, 11.15, 'Resize to 375 x 300 px (training crop size)  |  No padding — avoids mask shift', fs=8)
txt(SEG_X+SW/2, 10.78, 'Normalise: mean=[123.7, 116.3, 103.5]  std=[58.4, 57.1, 57.4]', fs=8, c='#555')

# Encoder
box(SEG_X, 8.35, SW, 2.0, fc='#EAF7EA', ec=SEG_BD)
txt(SEG_X+SW/2, 10.05, 'GANav Encoder — MixVisionTransformer-B0 (SegFormer backbone)', fs=9.5, c=SEG_BD, bold=True)
txt(SEG_X+SW/2, 9.60,  '4-stage hierarchical transformer with overlapping patch embedding', fs=8)
txt(SEG_X+SW/2, 9.20,  'Stage 1: 32x32 feat  |  Stage 2: 16x16  |  Stage 3: 8x8  |  Stage 4: 4x4', fs=8, c='#555')
txt(SEG_X+SW/2, 8.80,  'Efficient self-attention with reduction ratio  |  No positional encoding', fs=8, c='#555')
txt(SEG_X+SW/2, 8.48,  'Output: multi-scale feature maps  F1, F2, F3, F4', fs=8, c=SEG_BD, italic=True)

# Decoder
box(SEG_X, 5.75, SW, 2.35, fc='#EAF7EA', ec=SEG_BD)
txt(SEG_X+SW/2, 7.8,  'GANav Decoder — OursHeadClassAtt  (PSA + Class-Attention)', fs=9.5, c=SEG_BD, bold=True)
txt(SEG_X+SW/2, 7.35, 'Polarised Self-Attention (PSA): 97 x 97 spatial attention mask', fs=8)
txt(SEG_X+SW/2, 6.95, '384 channels  |  Class-specific attention weighting per query', fs=8, c='#555')
txt(SEG_X+SW/2, 6.55, 'Fuses F1-F4 through All-MLP head  |  Upsamples to H x W', fs=8, c='#555')
txt(SEG_X+SW/2, 6.15, 'Output: 6-class logit map (H x W x 6)  ->  argmax  ->  Semantic Map (H x W)', fs=8, c=SEG_BD, italic=True)
txt(SEG_X+SW/2, 5.88, 'Rescaled back to 640 x 480 via nearest-neighbour interpolation', fs=7.5, c='#555')

# Class legend
box(SEG_X, 4.05, SW, 1.45, fc='white', ec='#888', lw=1.2, r=0.15)
txt(SEG_X+SW/2, 5.25, '6 Semantic Classes (AVMI taxonomy)', fs=9, c='#333', bold=True)
classes = [
    ('Sky',      '#1866B2', SEG_X+0.7),
    ('Tree',     '#12B625', SEG_X+2.2),
    ('Bush',     '#B8C400', SEG_X+3.7),
    ('Ground',   '#5C1306', SEG_X+5.2),
    ('Obstacle', '#CC00CC', SEG_X+6.7),
    ('Rock',     '#CC0000', SEG_X+8.2),
]
for label, color, cx in classes:
    box(cx-0.05, 4.55, 1.1, 0.35, fc=color, ec=color, lw=0, r=0.08)
    txt(cx+0.5, 4.73, label, fs=8, c='white', bold=True)
    txt(cx+0.5, 4.25, f'class {classes.index((label,color,cx))}', fs=7.5, c='#555')

# LiDAR RGB unpack (bottom of seg col — for the lidar path)
box(SEG_X, 1.5, SW, 2.3, fc='white', ec=SEG_BD)
txt(SEG_X+SW/2, 3.5,  'LiDAR Pre-processing  &  RGB Unpack', fs=9.5, c=SEG_BD, bold=True)
txt(SEG_X+SW/2, 3.05, 'Parse CDR-serialised PointCloud2  |  Extract XYZ as float32  |  Filter NaN', fs=8)
txt(SEG_X+SW/2, 2.65, 'Unpack RGB: rgb_int = float32.view(uint32)', fs=8, c='#555')
txt(SEG_X+SW/2, 2.25, 'R = (rgb_int >> 16) & 0xFF     G = (rgb_int >> 8) & 0xFF     B = rgb_int & 0xFF', fs=8, c='#333', italic=True)
txt(SEG_X+SW/2, 1.82, 'Store as BGR for OpenCV  |  Build Nx3 XYZ and Nx3 RGB arrays', fs=8, c='#555')

# arrows in seg branch
arr(SEG_X+SW/2, 10.6,  SEG_X+SW/2, 10.35)
arr(SEG_X+SW/2, 8.35,  SEG_X+SW/2, 8.1)
arr(SEG_X+SW/2, 5.75,  SEG_X+SW/2, 5.5)

# ══════════════════════════════════════════════════════════════════════════════
# COL 3 — FUSION BRANCH
# ══════════════════════════════════════════════════════════════════════════════
FX = 13.6
FW = 5.0

box(FX, 10.35, FW, 1.55, fc='white', ec=FUS_BD)
txt(FX+FW/2, 11.6,  'Coordinate Transformation', fs=9.5, c=FUS_BD, bold=True)
txt(FX+FW/2, 11.15, 'P_c  =  R_L2C  .  P_L  +  t_L2C', fs=9, c='#5B2C6F', italic=True)
txt(FX+FW/2, 10.7,  'Extrinsics from Unreal Engine:  lidar@(-0.31, 0, 2.11)m  cam@(0.55, -0.15, 2.0)m', fs=7.5, c='#555')
txt(FX+FW/2, 10.45, 'R_base maps vehicle-ROS -> camera optical axes  |  R_L2C = R_base @ R_cam_in_veh', fs=7.5, c='#555')

box(FX, 8.45, FW, 1.65, fc='white', ec=FUS_BD)
txt(FX+FW/2, 9.8,   'Depth Filtering', fs=9.5, c=FUS_BD, bold=True)
txt(FX+FW/2, 9.35,  'Retain: Z_c  > 0.1 m  (remove behind-camera & too-close pts)', fs=8)
txt(FX+FW/2, 8.95,  'Typical: ~75% points pass  (~3,500 pts / frame)', fs=8, c='#555')
txt(FX+FW/2, 8.58,  'Removes sensor noise at very short range', fs=7.5, c='#777')

box(FX, 6.35, FW, 1.85, fc='white', ec=FUS_BD)
txt(FX+FW/2, 7.9,   'Perspective Projection  (Pinhole Model)', fs=9.5, c=FUS_BD, bold=True)
txt(FX+FW/2, 7.45,  '[u, v]^T  =  K . (P_c / Z_c)', fs=9, c='#5B2C6F', italic=True)
txt(FX+FW/2, 7.0,   'K:  fx=278.6  fy=334.3  cx=323.8  cy=229.2  (Unreal intrinsics)', fs=7.5, c='#555')
txt(FX+FW/2, 6.6,   'Bound check: 0 <= u < 640  and  0 <= v < 480', fs=7.5, c='#555')
txt(FX+FW/2, 6.45,  '~1,200 pts projected in-frame per frame', fs=7.5, c='#777', italic=True)

box(FX, 4.25, FW, 1.85, fc='white', ec=FUS_BD)
txt(FX+FW/2, 5.8,   'Class Lookup & Colour Assignment', fs=9.5, c=FUS_BD, bold=True)
txt(FX+FW/2, 5.35,  'class[i]  =  SemanticMap[ v[i], u[i] ]', fs=9, c='#5B2C6F', italic=True)
txt(FX+FW/2, 4.9,   'colour[i]  =  PALETTE[ class[i] ]', fs=9, c='#5B2C6F', italic=True)
txt(FX+FW/2, 4.5,   'Each 3D point tagged: (X,Y,Z) + class_id + BGR colour', fs=8, c='#555')
txt(FX+FW/2, 4.35,  'Enables semantic 3D map generation', fs=7.5, c='#777', italic=True)

box(FX, 2.5, FW, 1.5, fc='#EAF4FF', ec='#1565C0', lw=1.5)
txt(FX+FW/2, 3.7,   'Model: GANav (AVMI Scratch)', fs=9, c='#1565C0', bold=True)
txt(FX+FW/2, 3.3,   'Trained on 6-class AVMI UGV dataset', fs=8, c='#555')
txt(FX+FW/2, 2.9,   'Input pre-resized to 375x300 (no pad shift)', fs=8, c='#555')
txt(FX+FW/2, 2.6,   'mIoU: evaluated on held-out UGV test set', fs=7.5, c='#777', italic=True)

# fusion arrows
arr(FX+FW/2, 10.35, FX+FW/2, 10.1)
arr(FX+FW/2,  8.45, FX+FW/2,  8.2)
arr(FX+FW/2,  6.35, FX+FW/2,  6.1)
arr(FX+FW/2,  4.25, FX+FW/2,  4.0)

# ══════════════════════════════════════════════════════════════════════════════
# COL 4 — OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
OX = 19.2
OW = 4.4

box(OX, 8.8, OW, 2.9, fc='white', ec=OUT_BD)
txt(OX+OW/2, 11.35, 'Segmented Camera Image', fs=10, c=OUT_BD, bold=True)
txt(OX+OW/2, 10.85, 'Semantic mask blended over', fs=8.5, c='#333')
txt(OX+OW/2, 10.45, 'raw camera frame  (alpha=0.55)', fs=8, c='#555')
txt(OX+OW/2, 10.05, 'Resolution: 640 x 480 px', fs=8, c='#555')
txt(OX+OW/2,  9.6,  'Class colours overlaid at', fs=7.5, c='#555')
txt(OX+OW/2,  9.2,  'correct spatial positions', fs=7.5, c='#555')
txt(OX+OW/2,  8.95, '(shift-corrected pre-resize)', fs=7.5, c='#888', italic=True)

box(OX, 3.4, OW, 5.1, fc='white', ec=OUT_BD)
txt(OX+OW/2, 8.1,  'Segmented 3D Point Cloud', fs=10, c=OUT_BD, bold=True)
txt(OX+OW/2, 7.6,  'Each point: (X, Y, Z)', fs=8.5, c='#333')
txt(OX+OW/2, 7.2,  '+ semantic class label', fs=8.5, c='#333')
txt(OX+OW/2, 6.8,  '+ class BGR colour', fs=8.5, c='#333')
txt(OX+OW/2, 6.3,  '~1,200 pts/frame in-view', fs=8, c='#555')
txt(OX+OW/2, 5.9,  'Projected on black canvas', fs=8, c='#555')
# tiny colour legend in output
legend_items = [('Sky','#1866B2'),('Tree','#12B625'),('Bush','#B8C400'),
                ('Ground','#5C1306'),('Obstacle','#CC00CC'),('Rock','#CC0000')]
for i,(lbl,col) in enumerate(legend_items):
    rx = OX+0.35 + (i%2)*2.1
    ry = 4.95 - (i//2)*0.55
    box(rx, ry, 0.3, 0.28, fc=col, ec=col, lw=0, r=0.04)
    txt(rx+1.0, ry+0.14, lbl, fs=7.5, c='#333', ha='center')

box(OX, 1.5, OW, 1.6, fc='#FDECEA', ec=OUT_BD, lw=1.2)
txt(OX+OW/2, 2.9,  'Applications', fs=9, c=OUT_BD, bold=True)
txt(OX+OW/2, 2.5,  'Terrain classification for UGV nav', fs=7.5, c='#555')
txt(OX+OW/2, 2.1,  'Obstacle / rock detection in 3D', fs=7.5, c='#555')
txt(OX+OW/2, 1.7,  'Off-road path planning support', fs=7.5, c='#555')

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-COLUMN ARROWS
# ══════════════════════════════════════════════════════════════════════════════
# camera -> preprocess
arr(4.2,  9.8,  SEG_X, 11.2, c=IN_BD)
# lidar -> rgb unpack
arr(4.2,  5.5,  SEG_X, 2.65, c=IN_BD)
# sync -> fusion
arr(4.2,  3.2,  FX,    3.2,  c='#1565C0')

# seg map (decoder output) -> fusion coord transform
ax.annotate('', xy=(FX, 11.1), xytext=(SEG_X+SW, 6.5),
            arrowprops=dict(arrowstyle='->', color=SEG_BD, lw=2.0,
                            connectionstyle='arc3,rad=-0.15'), zorder=4)
txt(11.8, 9.1, 'Semantic\nMap', fs=7.5, c=SEG_BD, italic=True)

# lidar unpack -> coord transform
ax.annotate('', xy=(FX, 10.6), xytext=(SEG_X+SW, 2.65),
            arrowprops=dict(arrowstyle='->', color=SEG_BD, lw=2.0,
                            connectionstyle='arc3,rad=0.1'), zorder=4)
txt(11.8, 6.2, 'XYZ +\nRGB', fs=7.5, c=SEG_BD, italic=True)

# fusion -> outputs
arr(FX+FW, 7.0,  OX, 9.5,  c=FUS_BD)   # cam seg output
arr(FX+FW, 5.2,  OX, 6.0,  c=FUS_BD)   # 3d pc output

# ══════════════════════════════════════════════════════════════════════════════
os.makedirs('results', exist_ok=True)
plt.tight_layout(pad=0.2)
plt.savefig('results/pipeline_diagram.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print('Saved: results/pipeline_diagram.png')
