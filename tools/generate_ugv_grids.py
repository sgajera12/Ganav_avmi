"""
generate_ugv_grids.py
Generates two clean grid images (no text/labels):
  grid1_ugv_originals.png   — 3 rows × 6 cols of original UGV images
  grid2_ugv_segmented.png   — 3 rows × 6 cols of AVMI-scratch segmented images
                              (pure colour mask, no overlay)

Images are evenly sampled from the test set to cover diverse scenes.
"""
import os, sys, glob
import numpy as np
import cv2
import torch
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.backends.cudnn.enabled = False

# ── Settings ──────
ROWS, COLS   = 3, 6          # grid dimensions (18 images total)
THUMB_W      = 320           # width of each cell in the grid
THUMB_H      = 240           # height of each cell
OUT_DIR      = 'results/ugv_grids'

MODEL_CFG    = 'work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py'
MODEL_CKPT   = 'work_dirs/ganav_avmi_scratch/latest.pth'
UGV_GLOB     = 'data/avmi_ugv/images/test/*.png'

# AVMI 6-class colour palette (RGB)
PALETTE = np.array([
    [24,  102, 178],   # 0: sky      - blue
    [18,  182,  37],   # 1: tree     - green
    [239, 255,  15],   # 2: bush     - yellow
    [92,   19,   6],   # 3: ground   - dark brown
    [255,  63, 250],   # 4: obstacle - pink
    [255,   0,   0],   # 5: rock     - red
], dtype=np.uint8)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Pick 18 evenly-spaced images 
all_imgs = sorted(glob.glob(UGV_GLOB))
n_total  = ROWS * COLS   # 18
step     = max(1, len(all_imgs) // n_total)
chosen   = [all_imgs[i * step] for i in range(n_total)]
print(f'Using {len(chosen)} images (step={step} from {len(all_imgs)} total)')

# ── Load AVMI scratch model ─────
print('Loading AVMI scratch model …')
model = init_segmentor(MODEL_CFG, MODEL_CKPT, device='cuda:0')
model = revert_sync_batchnorm(model)
print('Ready.\n')

# ── Run inference on all chosen images 
orig_cells = []
seg_cells  = []

for i, img_path in enumerate(chosen):
    print(f'  {i+1}/{len(chosen)}: {os.path.basename(img_path)}')
    img_bgr = cv2.imread(img_path)

    # segmentation
    result = inference_segmentor(model, img_path)
    seg = result[0]
    if isinstance(seg, torch.Tensor):
        seg = seg.cpu().numpy()
    seg = seg.astype(np.uint8)

    H, W = img_bgr.shape[:2]
    if seg.shape != (H, W):
        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)

    # colour mask (pure, no overlay)
    seg_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c, col in enumerate(PALETTE):
        seg_rgb[seg == c] = col
    seg_rgb[seg == 255] = [80, 80, 80]   # ignored → grey

    # thumbnails
    orig_cells.append(cv2.resize(img_bgr, (THUMB_W, THUMB_H)))
    # convert seg from RGB to BGR for cv2
    seg_bgr = seg_rgb[:, :, ::-1].copy()
    seg_cells.append(cv2.resize(seg_bgr, (THUMB_W, THUMB_H),
                                interpolation=cv2.INTER_NEAREST))

# ── Build grids ───
GAP   = 6     # pixels between cells
BG    = 30    # background colour (dark grey)

def make_grid(cells, rows, cols):
    cell_h, cell_w = cells[0].shape[:2]
    total_w = cols * cell_w + (cols + 1) * GAP
    total_h = rows * cell_h + (rows + 1) * GAP
    canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        y = GAP + r * (cell_h + GAP)
        x = GAP + c * (cell_w + GAP)
        canvas[y:y + cell_h, x:x + cell_w] = cell
    return canvas

grid_orig = make_grid(orig_cells, ROWS, COLS)
grid_seg  = make_grid(seg_cells,  ROWS, COLS)

out1 = os.path.join(OUT_DIR, 'grid1_ugv_originals.png')
out2 = os.path.join(OUT_DIR, 'grid2_ugv_segmented.png')
cv2.imwrite(out1, grid_orig)
cv2.imwrite(out2, grid_seg)

print(f'\nSaved:')
print(f'  {out1}  ({grid_orig.shape[1]}×{grid_orig.shape[0]} px)')
print(f'  {out2}  ({grid_seg.shape[1]}×{grid_seg.shape[0]} px)')
