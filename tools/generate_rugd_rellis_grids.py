"""
generate_rugd_rellis_grids.py
Creates 4-col grids: [Original | Segmented | Original | Segmented]
4 rows x 2 pairs = 8 image+mask pairs per grid
3 grids each for RUGD and RELLIS (forest/trail/ground scenes)
"""
import os, sys, glob, random
import numpy as np
import cv2
import torch
import mmcv
sys.path.insert(0, os.path.abspath('.'))
torch.backends.cudnn.enabled = False

from mmseg.apis import init_segmentor, inference_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm

PALETTE = np.array([
    [178, 102,  24],
    [ 37, 182,  18],
    [ 15, 255, 239],
    [  6,  19,  92],
    [250,  63, 255],
    [  0,   0, 255],
], dtype=np.uint8)

GAP = 8
BG  = np.array([25, 25, 25], dtype=np.uint8)
W, H = 320, 240

print('Loading model on GPU...')
cfg  = mmcv.Config.fromfile('work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py')
model = init_segmentor(cfg, 'work_dirs/ganav_avmi_scratch/latest.pth', device='cuda:0')
model = revert_sync_batchnorm(model)
model.eval()
print('Model ready.')

def segment(img_bgr):
    seg = inference_segmentor(model, img_bgr)[0]
    return PALETTE[np.clip(seg, 0, 5)]

def make_grid(pairs, out_path):
    rows, cols = 4, 4
    cw = W*cols + GAP*(cols+1)
    ch = H*rows + GAP*(rows+1)
    canvas = np.full((ch, cw, 3), BG, dtype=np.uint8)
    for i, (orig, seg) in enumerate(pairs):
        row  = i // 2
        pair = i %  2
        col_orig = pair * 2
        col_seg  = pair * 2 + 1
        yo = GAP + row*(H+GAP)
        xo = GAP + col_orig*(W+GAP)
        xs = GAP + col_seg *(W+GAP)
        canvas[yo:yo+H, xo:xo+W] = cv2.resize(orig, (W,H))
        canvas[yo:yo+H, xs:xs+W] = cv2.resize(seg,  (W,H))
    cv2.imwrite(out_path, canvas)
    print(f'  Saved: {out_path}')

os.makedirs('results/ugv_comparison_grids', exist_ok=True)

# RUGD
print('\n=== RUGD ===')
with open('data/rugd/val_ours.txt') as f:
    rugd_ids = [l.strip() for l in f if l.strip()]
rugd_all = ['data/rugd/RUGD_frames-with-annotations/' + s + '.png' for s in rugd_ids]
rugd_all = [p for p in rugd_all if os.path.exists(p)]
trail = [p for p in rugd_all if any(k in p for k in ['trail','creek'])]
other = [p for p in rugd_all if p not in trail]

for g in range(3):
    random.seed(g*7)
    pool = trail.copy(); random.shuffle(pool)
    sel  = pool[g*8:(g+1)*8]
    if len(sel) < 8:
        extra = other.copy(); random.shuffle(extra)
        sel += extra[:8-len(sel)]
    pairs = []
    for j,p in enumerate(sel[:8]):
        img = cv2.imread(p)
        pairs.append((img, segment(img)))
        print(f'  grid{g+1} {j+1}/8: {os.path.basename(p)}')
    make_grid(pairs, f'results/ugv_comparison_grids/rugd_grid{g+1}.png')

# RELLIS
print('\n=== RELLIS ===')
with open('data/rellis/val.txt') as f:
    rellis_ids = [l.strip() for l in f if l.strip()]
rellis_all = ['data/rellis/image/' + s + '.jpg' for s in rellis_ids]
rellis_all = [p for p in rellis_all if os.path.exists(p)]

for g in range(3):
    random.seed(g*11)
    pool = rellis_all.copy(); random.shuffle(pool)
    step = max(1, len(pool)//24)
    sel  = pool[::step][g*8:(g+1)*8]
    pairs = []
    for j,p in enumerate(sel[:8]):
        img = cv2.imread(p)
        pairs.append((img, segment(img)))
        print(f'  grid{g+1} {j+1}/8: {os.path.basename(p)}')
    make_grid(pairs, f'results/ugv_comparison_grids/rellis_grid{g+1}.png')

print('\nAll done!')
