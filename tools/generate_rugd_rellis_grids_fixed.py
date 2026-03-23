"""
generate_rugd_rellis_grids_fixed.py
Same as before but with padding-aware rescaling for non-4:3 images (RELLIS=1920x1200)
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
    [178,102, 24],[37,182, 18],[15,255,239],
    [  6, 19, 92],[250, 63,255],[  0,  0,255],
], dtype=np.uint8)

GAP=8; BG=np.array([25,25,25],dtype=np.uint8); W,H=320,240
TARGET_W, TARGET_H = 375, 300   # model input size

print('Loading model on GPU...')
cfg   = mmcv.Config.fromfile('work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py')
model = init_segmentor(cfg, 'work_dirs/ganav_avmi_scratch/latest.pth', device='cuda:0')
model = revert_sync_batchnorm(model); model.eval()
print('Model ready.\n')

def segment(img_bgr):
    oh, ow = img_bgr.shape[:2]
    # compute what keep_ratio resize gives us
    scale    = min(TARGET_W / ow, TARGET_H / oh)
    rw       = int(ow * scale)
    rh       = int(oh * scale)
    # run inference (output is rescaled back to ori_shape but includes padding stretch)
    seg_full = inference_segmentor(model, img_bgr)[0]   # (oh, ow) but may be stretched
    # if padding was added (rh < TARGET_H), the valid content is only top fraction
    if rh < TARGET_H:
        valid_rows = int(oh * rh / TARGET_H)
        seg_crop   = seg_full[:valid_rows, :]
        seg_full   = cv2.resize(seg_crop.astype(np.uint8), (ow, oh),
                                interpolation=cv2.INTER_NEAREST)
    return PALETTE[np.clip(seg_full, 0, 5)]

def make_grid(pairs, out_path):
    cw = W*4+GAP*5; ch = H*4+GAP*5
    canvas = np.full((ch,cw,3), BG, dtype=np.uint8)
    for i,(orig,seg) in enumerate(pairs):
        row=i//2; pair=i%2
        yo=GAP+row*(H+GAP)
        xo=GAP+(pair*2)*(W+GAP)
        xs=GAP+(pair*2+1)*(W+GAP)
        canvas[yo:yo+H,xo:xo+W]=cv2.resize(orig,(W,H))
        canvas[yo:yo+H,xs:xs+W]=cv2.resize(seg,(W,H))
    cv2.imwrite(out_path,canvas)
    print(f'  Saved: {out_path}')

os.makedirs('results/ugv_comparison_grids', exist_ok=True)

# RELLIS (fixed)
print('=== RELLIS (shift-corrected) ===')
with open('data/rellis/val.txt') as f:
    rellis_ids = [l.strip() for l in f if l.strip()]
rellis_all = ['data/rellis/image/' + s + '.jpg' for s in rellis_ids]
rellis_all = [p for p in rellis_all if os.path.exists(p)]

for g in range(3):
    random.seed(g*13+5)
    pool = rellis_all.copy(); random.shuffle(pool)
    step = max(1, len(pool)//24)
    sel  = pool[g*8:(g+1)*8]
    if len(sel) < 8:
        sel += pool[:8-len(sel)]
    pairs=[]
    for j,p in enumerate(sel[:8]):
        img = cv2.imread(p)
        pairs.append((img, segment(img)))
        print(f'  grid{g+1} {j+1}/8: {os.path.basename(p)}')
    make_grid(pairs, f'results/ugv_comparison_grids/rellis_fixed_grid{g+1}.png')

# RUGD (no shift issue, but regenerate cleanly)
print('\n=== RUGD (trail only) ===')
with open('data/rugd/val_ours.txt') as f:
    rugd_ids = [l.strip() for l in f if l.strip()]
GOOD_SEQS = ['trail-4','trail-5','trail-6','trail-7','trail-9',
             'trail-10','trail-11','trail-12','trail-14','trail-15','trail']
good_paths = []
for rid in rugd_ids:
    seq = rid.split('/')[0]
    if seq in GOOD_SEQS:
        p = f'data/rugd/RUGD_frames-with-annotations/{rid}.png'
        if os.path.exists(p): good_paths.append(p)

def is_natural(path):
    img = cv2.imread(path)
    if img is None: return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv,(35,40,40),(85,255,255))
    sky   = cv2.inRange(hsv,(90,30,100),(130,255,255))
    total = img.shape[0]*img.shape[1]
    return (green.sum()//255+sky.sum()//255)/total > 0.15

natural = [p for p in good_paths if is_natural(p)]
print(f'Natural trail images: {len(natural)}')

for g in range(3):
    random.seed(g*19+3)
    pool = natural.copy(); random.shuffle(pool)
    step = max(1, len(pool)//8)
    sel  = pool[::step][:8]
    pairs=[]
    for j,p in enumerate(sel):
        img = cv2.imread(p)
        pairs.append((img, segment(img)))
        print(f'  grid{g+1} {j+1}/8: {os.path.basename(p)}')
    make_grid(pairs, f'results/ugv_comparison_grids/rugd_fixed_grid{g+1}.png')

print('\nAll done!')
