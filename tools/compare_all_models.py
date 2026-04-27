"""
compare_all_models.py
Runs all trained AVMI-class models on the same UGV images and produces a
comparison grid:  Original | AVMI Scratch | RUGD Mapped | RUGD Selective |
                  RUGD Weighted | RELLIS Fixed
Each row = one image.  Each column = one model.
"""

import os, sys, glob, random
import numpy as np
import cv2
import torch
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

torch.backends.cudnn.enabled = False

# ── class colour map (AVMI scheme) ──────────────────────────────────────
CLASSES = ['sky', 'tree', 'bush', 'ground', 'obstacle', 'rock']
PALETTE = np.array([
    [24,  102, 178],   # sky      - blue
    [18,  182,  37],   # tree     - green
    [239, 255,  15],   # bush     - yellow
    [92,   19,   6],   # ground   - dark brown
    [255,  63, 250],   # obstacle - pink
    [255,   0,   0],   # rock     - red
], dtype=np.uint8)

# ── model definitions ────────────────────────────────────────────────────
MODELS = [
    dict(
        label='AVMI\nScratch',
        cfg='work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py',
        ckpt='work_dirs/ganav_avmi_scratch/latest.pth',
    ),
    dict(
        label='RUGD\nMapped',
        cfg='work_dirs/ganav_rugd_avmi_mapped/ganav_rugd_avmi_mapped.py',
        ckpt='work_dirs/ganav_rugd_avmi_mapped/latest.pth',
    ),
    dict(
        label='RUGD\nSelective',
        cfg='work_dirs/ganav_rugd_avmi_selective/ganav_rugd_avmi_selective.py',
        ckpt='work_dirs/ganav_rugd_avmi_selective/latest.pth',
    ),
    dict(
        label='RUGD\nWeighted',
        cfg='work_dirs/ganav_rugd_avmi_weighted/ganav_rugd_avmi_weighted.py',
        ckpt='work_dirs/ganav_rugd_avmi_weighted/latest.pth',
    ),
    dict(
        label='RELLIS\nFixed',
        cfg='work_dirs/ganav_rellis_avmi_fixed/ganav_rellis_avmi_fixed.py',
        ckpt='work_dirs/ganav_rellis_avmi_fixed/latest.pth',
    ),
    dict(
        label='AVMI+RELLIS\nJoint',
        cfg='work_dirs/ganav_avmi_rellis_joint/ganav_avmi_rellis_joint.py',
        ckpt='work_dirs/ganav_avmi_rellis_joint/latest.pth',
    ),
    dict(
        label='AVMI+RUGD\nJoint',
        cfg='work_dirs/ganav_avmi_rugd_joint/ganav_avmi_rugd_joint.py',
        ckpt='work_dirs/ganav_avmi_rugd_joint/latest.pth',
    ),
]

UGV_DIR  = '/home/pinaka/dataset/AVMI/images_100'
OUT_DIR  = 'results/model_comparison'
N_IMAGES = 6   # how many UGV images to compare
THUMB_W  = 400
THUMB_H  = 300

os.makedirs(OUT_DIR, exist_ok=True)

# ── pick fixed images ────────────────────────────────────────────────────
random.seed(42)
all_imgs = sorted(glob.glob(os.path.join(UGV_DIR, '**', '*.png'), recursive=True) +
                  glob.glob(os.path.join(UGV_DIR, '**', '*.jpg'), recursive=True))
chosen = random.sample(all_imgs, min(N_IMAGES, len(all_imgs)))
print(f'Using {len(chosen)} UGV images.')


def seg_to_colour(seg):
    h, w = seg.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in enumerate(PALETTE):
        colour[seg == c] = col
    colour[seg == 255] = [80, 80, 80]   # ignore → grey
    return colour

def overlay(img, seg, alpha=0.55):
    col = seg_to_colour(seg)
    col_bgr = col[:, :, ::-1]
    return cv2.addWeighted(img, 1 - alpha, col_bgr, alpha, 0)

def thumb(img):
    return cv2.resize(img, (THUMB_W, THUMB_H))

def label_img(img, text, font_scale=0.55, bg=(30,30,30)):
    out = img.copy()
    lines = text.split('\n')
    y = 20
    for line in lines:
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255,255,255), 1, cv2.LINE_AA)
        y += 20
    return out

def legend_bar(width):
    """Colour legend strip."""
    bar_h = 28
    bar = np.full((bar_h, width, 3), 40, dtype=np.uint8)
    n = len(CLASSES)
    sw = width // n
    for i, (cls, col) in enumerate(zip(CLASSES, PALETTE)):
        x = i * sw
        bar[:, x:x+sw] = col[::-1]   # RGB→BGR
        cv2.putText(bar, cls, (x+4, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255,255,255), 1, cv2.LINE_AA)
    return bar

# ── load models ──────────────────────────────────────────────────────────
loaded = []
for m in MODELS:
    if not os.path.exists(m['cfg']) or not os.path.exists(m['ckpt']):
        print(f'  [skip] {m["label"].replace(chr(10)," ")} — files not found')
        continue
    print(f'  Loading {m["label"].replace(chr(10)," ")} ...', end=' ', flush=True)
    net = init_segmentor(m['cfg'], m['ckpt'], device='cuda:0')
    loaded.append(dict(label=m['label'], net=net))
    print('ready')

n_cols = 1 + len(loaded)   # original + one per model
n_rows = len(chosen)

# ── header row ───────────────────────────────────────────────────────────
col_labels = ['Original'] + [m['label'] for m in loaded]
header_h = 50
header = np.full((header_h, THUMB_W * n_cols, 3), 20, dtype=np.uint8)
for c, lbl in enumerate(col_labels):
    x = c * THUMB_W + THUMB_W // 2 - len(lbl.replace('\n','')) * 5
    for j, line in enumerate(lbl.split('\n')):
        cv2.putText(header, line,
                    (c * THUMB_W + 10, 20 + j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 1, cv2.LINE_AA)

rows_out = [header]

# ── per-image rows ───────────────────────────────────────────────────────
for idx, img_path in enumerate(chosen):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    name = os.path.splitext(os.path.basename(img_path))[0]
    row_cells = [thumb(img_bgr)]

    for m in loaded:
        result = inference_segmentor(m['net'], img_path)
        seg = result[0]
        if isinstance(seg, torch.Tensor):
            seg = seg.cpu().numpy()
        seg = seg.astype(np.int32)

        # align to image size
        seg_img = cv2.resize(seg.astype(np.uint8),
                             (img_bgr.shape[1], img_bgr.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        ov = overlay(img_bgr, seg_img)
        cell = thumb(ov)

        # per-class % overlay text
        y = THUMB_H - 10
        for c in range(len(CLASSES)):
            pct = (seg_img == c).sum() / seg_img.size * 100
            if pct > 2.0:
                txt = f'{CLASSES[c]}: {pct:.0f}%'
                col_bgr = tuple(int(x) for x in PALETTE[c][::-1])
                cv2.putText(cell, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, col_bgr, 1, cv2.LINE_AA)
                y -= 14

        row_cells.append(cell)

    row_img = np.hstack(row_cells)
    # row label on left edge
    cv2.putText(row_img, f'{idx+1}. {name}', (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1)
    rows_out.append(row_img)

# ── colour legend at bottom ──────────────────────────────────────────────
total_w = THUMB_W * n_cols
rows_out.append(legend_bar(total_w))

grid = np.vstack(rows_out)
out_path = os.path.join(OUT_DIR, 'ugv_all_models_comparison.png')
cv2.imwrite(out_path, grid)
print(f'\nSaved comparison grid → {out_path}')
print(f'Grid size: {grid.shape[1]}×{grid.shape[0]} px')
