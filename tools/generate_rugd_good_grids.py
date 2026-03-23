"""
generate_rugd_good_grids.py - RUGD images with grass/trees/sky only (no roads/buildings)
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

print('Loading model on GPU...')
cfg   = mmcv.Config.fromfile('work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py')
model = init_segmentor(cfg, 'work_dirs/ganav_avmi_scratch/latest.pth', device='cuda:0')
model = revert_sync_batchnorm(model); model.eval()
print('Model ready.\n')

def segment(img_bgr):
    seg = inference_segmentor(model, img_bgr)[0]
    return PALETTE[np.clip(seg,0,5)]

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

# Only use trail sequences (no park, no village, no creek which has urban)
# Best sequences for UGV-like terrain: trail-4,5,6,7,9,10,11,12,14,15
GOOD_SEQS = ['trail-4','trail-5','trail-6','trail-7','trail-9',
             'trail-10','trail-11','trail-12','trail-14','trail-15','trail']

with open('data/rugd/val_ours.txt') as f:
    rugd_ids = [l.strip() for l in f if l.strip()]

# filter to good sequences only
good_paths = []
for rid in rugd_ids:
    seq = rid.split('/')[0]
    if seq in GOOD_SEQS:
        p = f'data/rugd/RUGD_frames-with-annotations/{rid}.png'
        if os.path.exists(p):
            good_paths.append(p)

print(f'Good trail images available: {len(good_paths)}')

# additionally filter by checking sky+green ratio in the image
# (skip images that are mostly grey/road)
def is_natural(path, min_green_ratio=0.15):
    img = cv2.imread(path)
    if img is None: return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # green pixels
    green = cv2.inRange(hsv, (35,40,40), (85,255,255))
    # sky-blue pixels
    sky   = cv2.inRange(hsv, (90,30,100), (130,255,255))
    total = img.shape[0]*img.shape[1]
    ratio = (green.sum()//255 + sky.sum()//255) / total
    return ratio > min_green_ratio

natural = [p for p in good_paths if is_natural(p)]
print(f'Natural/green images: {len(natural)}')

# make 3 grids
for g in range(3):
    random.seed(g*19+3)
    pool = natural.copy(); random.shuffle(pool)
    # take evenly spaced for variety
    step = max(1, len(pool)//8)
    sel  = pool[g*step*8::step*8][:8] if g > 0 else pool[::step][:8]
    if len(sel) < 8:
        extra = pool.copy(); random.shuffle(extra)
        sel += [x for x in extra if x not in sel][:8-len(sel)]
    sel = sel[:8]
    pairs=[]
    for j,p in enumerate(sel):
        img=cv2.imread(p)
        pairs.append((img,segment(img)))
        print(f'  grid{g+1} {j+1}/8: {os.path.basename(p)}')
    make_grid(pairs, f'results/ugv_comparison_grids/rugd_natural_grid{g+1}.png')

print('\nAll done!')
