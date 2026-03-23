"""
test_joint_models.py
Tests the two joint-trained models (AVMI+RELLIS and AVMI+RUGD) on:
  - Source dataset images  (4 panels: Original | GT Mask | Prediction | Overlay)
  - UGV images             (3 panels: Original | Prediction | Overlay)

Usage:
    python tools/test_joint_models.py --model rellis_joint --n 6
    python tools/test_joint_models.py --model rugd_joint   --n 6
"""
import argparse, os, glob, random, sys
import numpy as np
import cv2
import torch
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.backends.cudnn.enabled = False

# ── AVMI 6-class colour map ────────────────────────────────────────────────────
CLASS_NAMES = ['sky', 'tree', 'bush', 'ground', 'obstacle', 'rock']
COLORS = np.array([
    [24,  102, 178],   # 0: sky      - blue
    [18,  182,  37],   # 1: tree     - green
    [239, 255,  15],   # 2: bush     - yellow
    [92,   19,   6],   # 3: ground   - dark brown
    [255,  63, 250],   # 4: obstacle - pink
    [255,   0,   0],   # 5: rock     - red
], dtype=np.uint8)
IGNORE_COLOR = np.array([80, 80, 80], dtype=np.uint8)   # grey for 255

# ── Label-to-AVMI lookup tables ────────────────────────────────────────────────
# RELLIS _orig.png: sequential 0-19
_RELLIS_LUT = np.full(256, 255, dtype=np.uint8)
_RELLIS_LUT[1]  = 3   # dirt      → ground
_RELLIS_LUT[2]  = 3   # grass     → ground
_RELLIS_LUT[3]  = 1   # tree      → tree
_RELLIS_LUT[4]  = 4   # pole      → obstacle
_RELLIS_LUT[5]  = 5   # water     → rock
_RELLIS_LUT[6]  = 0   # sky       → sky
_RELLIS_LUT[7]  = 4   # vehicle   → obstacle
_RELLIS_LUT[8]  = 4   # object    → obstacle
_RELLIS_LUT[9]  = 3   # asphalt   → ground
_RELLIS_LUT[10] = 4   # building  → obstacle
_RELLIS_LUT[11] = 4   # log       → obstacle
_RELLIS_LUT[12] = 4   # person    → obstacle
_RELLIS_LUT[13] = 4   # fence     → obstacle
_RELLIS_LUT[14] = 2   # bush      → bush
_RELLIS_LUT[15] = 3   # concrete  → ground
_RELLIS_LUT[16] = 4   # barrier   → obstacle
_RELLIS_LUT[17] = 5   # puddle    → rock
_RELLIS_LUT[18] = 3   # mud       → ground
_RELLIS_LUT[19] = 5   # rubble    → rock

# RUGD _orig.png: pixel values 0-24 (RUGD class IDs)
_RUGD_LUT = np.full(256, 255, dtype=np.uint8)
_RUGD_LUT[7]  = 0   # sky      → sky
_RUGD_LUT[4]  = 1   # tree     → tree
_RUGD_LUT[21] = 5   # rock     → rock
_RUGD_LUT[14] = 5   # rock-bed → rock
# all others remain 255 (ignore) — intentional for joint training

# ── Model / dataset configs ────────────────────────────────────────────────────
CONFIGS = {
    'rellis_joint': dict(
        label       = 'AVMI+RELLIS Joint',
        config      = 'work_dirs/ganav_avmi_rellis_joint/ganav_avmi_rellis_joint.py',
        checkpoint  = 'work_dirs/ganav_avmi_rellis_joint/latest.pth',
        split_file  = 'data/rellis/val.txt',   # stem paths, no extension
        img_root    = 'data/rellis/image',
        img_suffix  = '.jpg',
        ann_root    = 'data/rellis/annotation',
        ann_suffix  = '_orig.png',
        lut         = _RELLIS_LUT,
        out_src     = 'results/joint_rellis_test_rellis',
        out_ugv     = 'results/joint_rellis_test_ugv',
    ),
    'rugd_joint': dict(
        label       = 'AVMI+RUGD Joint',
        config      = 'work_dirs/ganav_avmi_rugd_joint/ganav_avmi_rugd_joint.py',
        checkpoint  = 'work_dirs/ganav_avmi_rugd_joint/latest.pth',
        split_file  = 'data/rugd/val_ours.txt',
        img_root    = 'data/rugd/RUGD_frames-with-annotations',
        img_suffix  = '.png',
        ann_root    = 'data/rugd/RUGD_annotations',
        ann_suffix  = '_orig.png',
        lut         = _RUGD_LUT,
        out_src     = 'results/joint_rugd_test_rugd',
        out_ugv     = 'results/joint_rugd_test_ugv',
    ),
}

UGV_GLOB = 'data/avmi_ugv/images/test/*.png'

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=list(CONFIGS.keys()), required=True)
parser.add_argument('--n', type=int, default=6)
args = parser.parse_args()
cfg = CONFIGS[args.model]

# ── Load model ─────────────────────────────────────────────────────────────────
print(f'Loading [{cfg["label"]}] …')
model = init_segmentor(cfg['config'], cfg['checkpoint'], device='cuda:0')
model = revert_sync_batchnorm(model)
print('Model ready.\n')


# ── Helpers ────────────────────────────────────────────────────────────────────
def seg_to_color(seg):
    """Convert segmentation index map (H,W) to RGB colour image."""
    out = np.full((*seg.shape, 3), IGNORE_COLOR, dtype=np.uint8)
    for c, col in enumerate(COLORS):
        out[seg == c] = col
    return out


def load_gt(img_path, img_root, ann_root, ann_suffix, lut):
    """Derive GT annotation path from image path and apply LUT."""
    rel = os.path.relpath(img_path, img_root)           # scene/frame.jpg
    base = os.path.splitext(rel)[0]                     # scene/frame
    ann_path = os.path.join(ann_root, base + ann_suffix) # .../scene/frame_orig.png
    if not os.path.exists(ann_path):
        return None
    raw = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    return lut[raw]


def run_inference(img_path):
    result = inference_segmentor(model, img_path)
    return result[0]


def add_legend(canvas, x_off, h):
    """Draw colour legend on the right panel."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (name, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        y = 55 + i * 26
        cv2.rectangle(canvas, (x_off + 8, y - 14), (x_off + 24, y + 2),
                      col[::-1].tolist(), -1)
        cv2.putText(canvas, name, (x_off + 30, y), font, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)
    # ignored
    y = 55 + len(CLASS_NAMES) * 26
    cv2.rectangle(canvas, (x_off + 8, y - 14), (x_off + 24, y + 2),
                  (80, 80, 80), -1)
    cv2.putText(canvas, 'ignored', (x_off + 30, y), font, 0.52,
                (200, 200, 200), 1, cv2.LINE_AA)


def make_canvas_src(img_path, gt_seg, pred_seg):
    """4-panel canvas: Original | GT Mask | Prediction | Overlay."""
    img_bgr = cv2.imread(img_path)
    H, W = img_bgr.shape[:2]

    # resize segs to match image
    gt_col  = cv2.resize(seg_to_color(gt_seg),   (W, H), interpolation=cv2.INTER_NEAREST)
    pred_col = cv2.resize(seg_to_color(pred_seg), (W, H), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(img_bgr, 0.45, pred_col[:, :, ::-1], 0.55, 0)

    canvas = np.zeros((H, W * 4, 3), dtype=np.uint8)
    canvas[:, :W]         = img_bgr
    canvas[:, W:W*2]      = gt_col[:, :, ::-1]    # RGB→BGR
    canvas[:, W*2:W*3]    = pred_col[:, :, ::-1]
    canvas[:, W*3:]       = overlay

    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, txt in [(8, 'Original'), (W+8, 'GT Mask'), (W*2+8, 'Prediction'), (W*3+8, 'Overlay')]:
        cv2.putText(canvas, txt, (x, 24), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    add_legend(canvas, W*3, H)

    # per-class stats on prediction panel
    font_s = cv2.FONT_HERSHEY_SIMPLEX
    sy = H - 10
    for c in range(len(CLASS_NAMES)):
        pct = (pred_seg == c).sum() / pred_seg.size * 100
        if pct < 1.0:
            continue
        col_bgr = tuple(int(v) for v in COLORS[c][::-1])
        cv2.putText(canvas, f'{CLASS_NAMES[c]}: {pct:.0f}%',
                    (W*2 + 6, sy), font_s, 0.38, col_bgr, 1, cv2.LINE_AA)
        sy -= 15

    for x in [W, W*2, W*3]:
        cv2.line(canvas, (x, 0), (x, H), (180, 180, 180), 2)
    return canvas


def make_canvas_ugv(img_path, pred_seg):
    """3-panel canvas: Original | Prediction | Overlay."""
    img_bgr = cv2.imread(img_path)
    H, W = img_bgr.shape[:2]

    pred_col = cv2.resize(seg_to_color(pred_seg), (W, H), interpolation=cv2.INTER_NEAREST)
    overlay  = cv2.addWeighted(img_bgr, 0.45, pred_col[:, :, ::-1], 0.55, 0)

    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, :W]      = img_bgr
    canvas[:, W:W*2]   = pred_col[:, :, ::-1]
    canvas[:, W*2:]    = overlay

    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, txt in [(8, 'Original'), (W+8, 'Prediction'), (W*2+8, 'Overlay')]:
        cv2.putText(canvas, txt, (x, 24), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    add_legend(canvas, W*2, H)

    sy = H - 10
    for c in range(len(CLASS_NAMES)):
        pct = (pred_seg == c).sum() / pred_seg.size * 100
        if pct < 1.0:
            continue
        col_bgr = tuple(int(v) for v in COLORS[c][::-1])
        cv2.putText(canvas, f'{CLASS_NAMES[c]}: {pct:.0f}%',
                    (W + 6, sy), font, 0.38, col_bgr, 1, cv2.LINE_AA)
        sy -= 15

    for x in [W, W*2]:
        cv2.line(canvas, (x, 0), (x, H), (180, 180, 180), 2)
    return canvas


# ── Test on source dataset ─────────────────────────────────────────────────────
os.makedirs(cfg['out_src'], exist_ok=True)
with open(cfg['split_file']) as f:
    stems = [l.strip() for l in f if l.strip()]
all_imgs = [os.path.join(cfg['img_root'], s + cfg['img_suffix']) for s in stems]
all_imgs = [p for p in all_imgs if os.path.exists(p)]
random.seed(42)
selected = random.sample(all_imgs, min(args.n, len(all_imgs)))
print(f'Source dataset test ({len(selected)} images) → {cfg["out_src"]}/')

for i, img_path in enumerate(selected):
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f'  {i+1}/{len(selected)}: {name}')

    gt_seg = load_gt(img_path, cfg['img_root'], cfg['ann_root'],
                     cfg['ann_suffix'], cfg['lut'])
    pred_seg = run_inference(img_path)
    if isinstance(pred_seg, torch.Tensor):
        pred_seg = pred_seg.cpu().numpy()

    if gt_seg is not None:
        canvas = make_canvas_src(img_path, gt_seg, pred_seg)
    else:
        print(f'    [no GT found, showing 3-panel]')
        canvas = make_canvas_ugv(img_path, pred_seg)

    out_path = os.path.join(cfg['out_src'], f'{i+1:02d}_{name}.png')
    cv2.imwrite(out_path, canvas)

print(f'  Done → {cfg["out_src"]}/\n')

# ── Test on UGV images ─────────────────────────────────────────────────────────
os.makedirs(cfg['out_ugv'], exist_ok=True)
ugv_imgs = sorted(glob.glob(UGV_GLOB))
if not ugv_imgs:
    print(f'No UGV images found at {UGV_GLOB}')
else:
    ugv_sel = random.sample(ugv_imgs, min(args.n, len(ugv_imgs)))
    print(f'UGV test ({len(ugv_sel)} images) → {cfg["out_ugv"]}/')
    for i, img_path in enumerate(ugv_sel):
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f'  {i+1}/{len(ugv_sel)}: {name}')
        pred_seg = run_inference(img_path)
        if isinstance(pred_seg, torch.Tensor):
            pred_seg = pred_seg.cpu().numpy()
        canvas = make_canvas_ugv(img_path, pred_seg)
        out_path = os.path.join(cfg['out_ugv'], f'{i+1:02d}_{name}.png')
        cv2.imwrite(out_path, canvas)
    print(f'  Done → {cfg["out_ugv"]}/')

print('\nAll done!')
