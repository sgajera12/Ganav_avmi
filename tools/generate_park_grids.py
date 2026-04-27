"""
generate_park_grids.py
Generates Original | Segmented comparison grids for RUGD park-2 and park-8 sequences.
Layout: 4 columns (orig | seg | orig | seg) x 4 rows = 8 pairs per grid

Usage:
    conda run -n ganav python tools/generate_park_grids.py

Output: results/ugv_comparison_grids/
    rugd_park2_grid.png
    rugd_park8_grid.png
    rugd_park_mixed_grid.png
"""
import os, sys, numpy as np, cv2, torch, mmcv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.backends.cudnn.enabled = False
from mmseg.apis import init_segmentor, inference_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm

# ── Settings ──
MODEL_CFG  = 'work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py'
MODEL_CKPT = 'work_dirs/ganav_avmi_scratch/latest.pth'
OUT_DIR    = 'results/ugv_comparison_grids'

PARK2_RANGE = (81, 351)     # frame number range for park-2
PARK8_RANGE = (391, 801)    # frame number range for park-8

GAP = 3          # border thickness between images (pixels) — change here to adjust
W, H = 320, 240  # cell size per image
TW, TH = 375, 300  # model input size

PALETTE = np.array([
    [178, 102,  24],   # sky
    [ 37, 182,  18],   # tree
    [ 15, 255, 239],   # bush
    [  6,  19,  92],   # ground
    [250,  63, 255],   # obstacle
    [  0,   0, 255],   # rock
], dtype=np.uint8)

BG = np.array([25, 25, 25], dtype=np.uint8)  # background/border colour

# ── Load model 
print('Loading model...')
cfg   = mmcv.Config.fromfile(MODEL_CFG)
model = init_segmentor(cfg, MODEL_CKPT, device='cuda:0')
model = revert_sync_batchnorm(model)
model.eval()
print('Model ready.\n')

# ── Helpers ───
def segment(img_bgr):
    """Run segmentation with padding-aware rescaling for non-4:3 images."""
    oh, ow = img_bgr.shape[:2]
    scale  = min(TW / ow, TH / oh)
    rh     = int(oh * scale)
    seg    = inference_segmentor(model, img_bgr)[0]
    if rh < TH:  # image was padded — crop padding before rescaling back
        valid = int(oh * rh / TH)
        seg   = cv2.resize(seg[:valid, :].astype(np.uint8), (ow, oh),
                           interpolation=cv2.INTER_NEAREST)
    return PALETTE[np.clip(seg, 0, 5)]

def make_grid(pairs, out_path):
    """Build 4-col grid: orig|seg|orig|seg, 4 rows."""
    cw = W * 4 + GAP * 5
    ch = H * 4 + GAP * 5
    canvas = np.full((ch, cw, 3), BG, dtype=np.uint8)
    for i, (orig, seg) in enumerate(pairs):
        row  = i // 2
        pair = i %  2
        yo = GAP + row  * (H + GAP)
        xo = GAP + (pair * 2)     * (W + GAP)
        xs = GAP + (pair * 2 + 1) * (W + GAP)
        canvas[yo:yo+H, xo:xo+W] = cv2.resize(orig, (W, H))
        canvas[yo:yo+H, xs:xs+W] = cv2.resize(seg,  (W, H))
    cv2.imwrite(out_path, canvas)
    print(f'  Saved: {out_path}')

def load_sequence(base_dir, frame_range):
    lo, hi = frame_range
    files  = sorted([f for f in os.listdir(base_dir)
                     if f.endswith('.png') and
                     lo <= int(f.split('_')[1].split('.')[0]) <= hi])
    step   = max(1, len(files) // 8)
    return [os.path.join(base_dir, f) for f in files[::step][:8]]

# ── Main ──────
os.makedirs(OUT_DIR, exist_ok=True)

sel_p2 = load_sequence('data/rugd/RUGD_frames-with-annotations/park-2', PARK2_RANGE)
sel_p8 = load_sequence('data/rugd/RUGD_frames-with-annotations/park-8', PARK8_RANGE)

for label, sel, fname in [
    ('park-2',       sel_p2,              'rugd_park2_grid.png'),
    ('park-8',       sel_p8,              'rugd_park8_grid.png'),
    ('mixed (4+4)',  sel_p2[:4]+sel_p8[:4], 'rugd_park_mixed_grid.png'),
]:
    print(f'=== {label} ===')
    pairs = []
    for j, p in enumerate(sel):
        img = cv2.imread(p)
        pairs.append((img, segment(img)))
        print(f'  {j+1}/{len(sel)}: {os.path.basename(p)}')
    make_grid(pairs, os.path.join(OUT_DIR, fname))

print('\nAll done!')
