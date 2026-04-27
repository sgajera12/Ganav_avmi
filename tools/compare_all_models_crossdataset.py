"""
compare_all_models_crossdataset.py
Same columns as ugv_all_models_comparison.png (all fine-tuned GANav B0 models)
but rows are RUGD and RELLIS images instead of AVMI UGV images.

Output:
    results/model_comparison/rugd_all_models_comparison.png
    results/model_comparison/rellis_all_models_comparison.png

Usage:
    conda run -n ganav python tools/compare_all_models_crossdataset.py
"""

import os, sys, random
import numpy as np
import cv2
import torch
import mmcv
from mmseg.apis import init_segmentor, inference_segmentor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

torch.backends.cudnn.enabled = False

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── class colour map (AVMI scheme, RGB for legend, BGR for cv2) ──────────
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
        cfg=os.path.join(_REPO_ROOT, 'work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py'),
        ckpt=os.path.join(_REPO_ROOT, 'work_dirs/ganav_avmi_scratch/latest.pth'),
    ),
    dict(
        label='RUGD\nMapped',
        cfg=os.path.join(_REPO_ROOT, 'work_dirs/ganav_rugd_avmi_mapped/ganav_rugd_avmi_mapped.py'),
        ckpt=os.path.join(_REPO_ROOT, 'work_dirs/ganav_rugd_avmi_mapped/latest.pth'),
    ),
    dict(
        label='RUGD\nSelective',
        cfg=os.path.join(_REPO_ROOT, 'work_dirs/ganav_rugd_avmi_selective/ganav_rugd_avmi_selective.py'),
        ckpt=os.path.join(_REPO_ROOT, 'work_dirs/ganav_rugd_avmi_selective/latest.pth'),
    ),
    dict(
        label='RUGD\nWeighted',
        cfg=os.path.join(_REPO_ROOT, 'work_dirs/ganav_rugd_avmi_weighted/ganav_rugd_avmi_weighted.py'),
        ckpt=os.path.join(_REPO_ROOT, 'work_dirs/ganav_rugd_avmi_weighted/latest.pth'),
    ),
    dict(
        label='RELLIS\nFixed',
        cfg=os.path.join(_REPO_ROOT, 'work_dirs/ganav_rellis_avmi_fixed/ganav_rellis_avmi_fixed.py'),
        ckpt=os.path.join(_REPO_ROOT, 'work_dirs/ganav_rellis_avmi_fixed/latest.pth'),
    ),
    dict(
        label='AVMI+RELLIS\nJoint',
        cfg=os.path.join(_REPO_ROOT, 'work_dirs/ganav_avmi_rellis_joint/ganav_avmi_rellis_joint.py'),
        ckpt=os.path.join(_REPO_ROOT, 'work_dirs/ganav_avmi_rellis_joint/latest.pth'),
    ),
    dict(
        label='AVMI+RUGD\nJoint',
        cfg=os.path.join(_REPO_ROOT, 'work_dirs/ganav_avmi_rugd_joint/ganav_avmi_rugd_joint.py'),
        ckpt=os.path.join(_REPO_ROOT, 'work_dirs/ganav_avmi_rugd_joint/latest.pth'),
    ),
]

OUT_DIR  = os.path.join(_REPO_ROOT, 'results', 'model_comparison')
N_IMAGES = 6
THUMB_W  = 400
THUMB_H  = 300

os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────
def seg_to_colour(seg):
    colour = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for c, col in enumerate(PALETTE):
        colour[seg == c] = col
    colour[seg == 255] = [80, 80, 80]
    return colour

def overlay(img, seg, alpha=0.55):
    col_bgr = seg_to_colour(seg)[:, :, ::-1]
    return cv2.addWeighted(img, 1 - alpha, col_bgr, alpha, 0)

def thumb(img):
    return cv2.resize(img, (THUMB_W, THUMB_H))

def legend_bar(width):
    bar_h = 28
    bar = np.full((bar_h, width, 3), 40, dtype=np.uint8)
    sw = width // len(CLASSES)
    for i, (cls, col) in enumerate(zip(CLASSES, PALETTE)):
        x = i * sw
        bar[:, x:x+sw] = col[::-1]
        cv2.putText(bar, cls, (x+4, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255,255,255), 1, cv2.LINE_AA)
    return bar

def is_natural(path):
    img = cv2.imread(path)
    if img is None: return False
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 40, 40),  (85, 255, 255))
    sky   = cv2.inRange(hsv, (90, 30, 100), (130, 255, 255))
    total = img.shape[0] * img.shape[1]
    return (green.sum() // 255 + sky.sum() // 255) / total > 0.15

# ── load models ──────────────────────────────────────────────────────────
print('Loading models...')
loaded = []
for m in MODELS:
    if not os.path.exists(m['cfg']) or not os.path.exists(m['ckpt']):
        print(f'  [skip] {m["label"].replace(chr(10)," ")} — files not found')
        continue
    print(f'  Loading {m["label"].replace(chr(10)," ")} ...', end=' ', flush=True)
    net = init_segmentor(m['cfg'], m['ckpt'], device='cuda:0')
    loaded.append(dict(label=m['label'], net=net))
    print('ready')

n_cols     = 1 + len(loaded)
col_labels = ['Original'] + [m['label'] for m in loaded]

def build_header():
    header = np.full((50, THUMB_W * n_cols, 3), 20, dtype=np.uint8)
    for c, lbl in enumerate(col_labels):
        for j, line in enumerate(lbl.split('\n')):
            cv2.putText(header, line,
                        (c * THUMB_W + 10, 20 + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
    return header

def build_grid(image_paths, out_path, dataset_label):
    print(f'\n=== {dataset_label} ({len(image_paths)} images) ===')
    rows_out = [build_header()]

    for idx, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f'  SKIP: {img_path}')
            continue
        name = os.path.basename(img_path)
        print(f'  [{idx+1}/{len(image_paths)}] {name}')

        row_cells = [thumb(img_bgr)]
        for m in loaded:
            result  = inference_segmentor(m['net'], img_path)
            seg     = result[0]
            if isinstance(seg, torch.Tensor):
                seg = seg.cpu().numpy()
            seg     = seg.astype(np.int32)
            seg_img = cv2.resize(seg.astype(np.uint8),
                                 (img_bgr.shape[1], img_bgr.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            ov   = overlay(img_bgr, seg_img)
            cell = thumb(ov)

            # per-class % text
            y = THUMB_H - 10
            for c in range(len(CLASSES)):
                pct = (seg_img == c).sum() / seg_img.size * 100
                if pct > 2.0:
                    col_bgr = tuple(int(x) for x in PALETTE[c][::-1])
                    cv2.putText(cell, f'{CLASSES[c]}: {pct:.0f}%',
                                (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.38, col_bgr, 1, cv2.LINE_AA)
                    y -= 14

            row_cells.append(cell)

        row_img = np.hstack(row_cells)
        cv2.putText(row_img, f'{idx+1}. {name}', (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        rows_out.append(row_img)

    total_w = THUMB_W * n_cols
    rows_out.append(legend_bar(total_w))
    grid = np.vstack(rows_out)
    cv2.imwrite(out_path, grid)
    print(f'  Saved: {out_path}  ({grid.shape[1]}×{grid.shape[0]} px)')

# ── RUGD trail images ─────────────────────────────────────────────────────
with open(os.path.join(_REPO_ROOT, 'data/rugd/val_ours.txt')) as f:
    rugd_ids = [l.strip() for l in f if l.strip()]

GOOD_SEQS = ['trail-4','trail-5','trail-6','trail-7','trail-9',
             'trail-10','trail-11','trail-12','trail-14','trail-15','trail']
good_paths = []
for rid in rugd_ids:
    seq = rid.split('/')[0]
    if seq in GOOD_SEQS:
        p = os.path.join(_REPO_ROOT, 'data/rugd/RUGD_frames-with-annotations', rid + '.png')
        if os.path.exists(p):
            good_paths.append(p)

natural = [p for p in good_paths if is_natural(p)]
random.seed(3)   # same seed as rugd_fixed_grid1
pool = natural.copy(); random.shuffle(pool)
step = max(1, len(pool) // 8)
rugd_sel = pool[::step][:N_IMAGES]

build_grid(rugd_sel,
           os.path.join(OUT_DIR, 'rugd_all_models_comparison.png'),
           'RUGD')

# ── RELLIS images ─────────────────────────────────────────────────────────
with open(os.path.join(_REPO_ROOT, 'data/rellis/val.txt')) as f:
    rellis_ids = [l.strip() for l in f if l.strip()]
rellis_all = [os.path.join(_REPO_ROOT, 'data/rellis/image', s + '.jpg')
              for s in rellis_ids if os.path.exists(
                  os.path.join(_REPO_ROOT, 'data/rellis/image', s + '.jpg'))]

random.seed(5)   # same seed as rellis_fixed_grid1
pool = rellis_all.copy(); random.shuffle(pool)
rellis_sel = pool[:N_IMAGES]

build_grid(rellis_sel,
           os.path.join(OUT_DIR, 'rellis_all_models_comparison.png'),
           'RELLIS')

print('\nAll done!')
