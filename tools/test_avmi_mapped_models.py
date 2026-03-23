"""
Test AVMI-mapped fine-tuned models (RUGD or RELLIS).
These models output AVMI 6 classes: sky, tree, bush, ground, obstacle, rock.
Tests both the source dataset images AND UGV images side by side.

Usage:
    python tools/test_avmi_mapped_models.py --dataset rugd   --n 5
    python tools/test_avmi_mapped_models.py --dataset rellis --n 5
"""
import argparse
import torch
import cv2
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm
import os
import glob
import random

torch.backends.cudnn.enabled = False  # RTX 5070 fix

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['rugd', 'rellis', 'rugd_sel', 'rellis_sel', 'rugd_w', 'rellis_fixed'], required=True)
parser.add_argument('--n', type=int, default=5,
                    help='Number of random images per source (default 5)')
args = parser.parse_args()

# ── Model paths ───────────────────────────────────────────────────────────────
CONFIGS = {
    'rugd': {
        'config':     'work_dirs/ganav_rugd_avmi_mapped/ganav_rugd_avmi_mapped.py',
        'checkpoint': 'work_dirs/ganav_rugd_avmi_mapped/latest.pth',
        'img_glob':   'data/rugd/RUGD_frames-with-annotations/**/*.png',
        'out_dir':    'results/avmi_mapped_test_rugd',
    },
    'rellis': {
        'config':     'work_dirs/ganav_rellis_avmi_mapped/ganav_rellis_avmi_mapped.py',
        'checkpoint': 'work_dirs/ganav_rellis_avmi_mapped/latest.pth',
        'img_glob':   'data/rellis/image/**/*.jpg',
        'out_dir':    'results/avmi_mapped_test_rellis',
    },
    'rugd_sel': {
        'config':     'work_dirs/ganav_rugd_avmi_selective/ganav_rugd_avmi_selective.py',
        'checkpoint': 'work_dirs/ganav_rugd_avmi_selective/latest.pth',
        'img_glob':   'data/rugd/RUGD_frames-with-annotations/**/*.png',
        'out_dir':    'results/avmi_selective_test_rugd',
    },
    'rellis_sel': {
        'config':     'work_dirs/ganav_rellis_avmi_selective/ganav_rellis_avmi_selective.py',
        'checkpoint': 'work_dirs/ganav_rellis_avmi_selective/latest.pth',
        'img_glob':   'data/rellis/image/**/*.jpg',
        'out_dir':    'results/avmi_selective_test_rellis',
    },
    'rugd_w': {
        'config':     'work_dirs/ganav_rugd_avmi_weighted/ganav_rugd_avmi_weighted.py',
        'checkpoint': 'work_dirs/ganav_rugd_avmi_weighted/latest.pth',
        'img_glob':   'data/rugd/RUGD_frames-with-annotations/**/*.png',
        'out_dir':    'results/avmi_weighted_test_rugd',
    },
    'rellis_fixed': {
        'config':     'work_dirs/ganav_rellis_avmi_fixed/ganav_rellis_avmi_fixed.py',
        'checkpoint': 'work_dirs/ganav_rellis_avmi_fixed/latest.pth',
        'img_glob':   'data/rellis/image/**/*.jpg',
        'out_dir':    'results/avmi_fixed_test_rellis',
    },
}

# ── AVMI 6 classes ────────────────────────────────────────────────────────────
CLASS_NAMES = ['sky', 'tree', 'bush', 'ground', 'obstacle', 'rock']
COLORS = np.array([
    [24,  102, 178],  # 0: sky    - blue
    [18,  182,  37],  # 1: tree   - green
    [239, 255,  15],  # 2: bush   - yellow
    [92,   19,   6],  # 3: ground - dark brown
    [255,  63, 250],  # 4: obstacle  - pink
    [255,   0,   0],  # 5: rock   - red
], dtype=np.uint8)

UGV_IMG_GLOB = 'data/avmi_ugv/images/test/*.png'

cfg = CONFIGS[args.dataset]
for f in (cfg['config'], cfg['checkpoint']):
    if not os.path.exists(f):
        print(f"ERROR: not found: {f}")
        exit(1)

print(f"Loading [{args.dataset} AVMI-mapped] model …")
model = init_segmentor(cfg['config'], cfg['checkpoint'], device='cuda:0')
model = revert_sync_batchnorm(model)
print("Model ready!\n")


def run_inference(img_path):
    result  = inference_segmentor(model, img_path)
    seg_map = result[0]
    orig_h, orig_w = seg_map.shape
    INFER_H, INFER_W = 300, 375
    scale       = min(INFER_H / orig_h, INFER_W / orig_w)
    resized_h   = int(round(orig_h * scale))
    valid_out_h = int(round(resized_h * orig_h / INFER_H))
    if valid_out_h < orig_h:
        seg_map = cv2.resize(seg_map[:valid_out_h, :], (orig_w, orig_h),
                             interpolation=cv2.INTER_NEAREST)
    return seg_map


def make_canvas(img_path, seg_map):
    colored_seg  = COLORS[seg_map]
    original     = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    H, W         = original_rgb.shape[:2]

    if colored_seg.shape[:2] != (H, W):
        colored_seg = cv2.resize(colored_seg, (W, H), interpolation=cv2.INTER_NEAREST)

    blended = cv2.addWeighted(original_rgb, 0.45,
                              colored_seg.astype(np.uint8), 0.55, 0)

    stats = [(cls, float(np.sum(seg_map == cls) / seg_map.size * 100))
             for cls in range(len(CLASS_NAMES))]

    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, :W]    = original_rgb
    canvas[:, W:W*2] = colored_seg
    canvas[:, W*2:]  = blended

    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, text in [(10, 'Original'), (W + 10, 'Segmentation'), (W*2 + 10, 'Overlay')]:
        cv2.putText(canvas, text, (x, 30), font, 0.8, (255, 255, 255), 2)

    for idx, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        y = 60 + idx * 28
        cv2.rectangle(canvas, (W + 10, y - 15), (W + 28, y + 3), color.tolist(), -1)
        cv2.putText(canvas, name, (W + 34, y), font, 0.6, (255, 255, 255), 1)

    sx, sy = W*2 + 8, 20
    for cls, pct in stats:
        if pct < 0.05:
            continue
        color = COLORS[cls].tolist()
        cv2.rectangle(canvas, (sx, sy - 11), (sx + 14, sy + 3), color, -1)
        cv2.putText(canvas, f"{CLASS_NAMES[cls]}: {pct:.1f}%",
                    (sx + 18, sy), font, 0.42, (255, 255, 255), 1)
        sy += 20

    cv2.line(canvas, (W,   0), (W,   H), (200, 200, 200), 2)
    cv2.line(canvas, (W*2, 0), (W*2, H), (200, 200, 200), 2)
    return canvas, stats


def test_batch(img_paths, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n── {label} ({len(img_paths)} images) ──")
    for i, img_path in enumerate(img_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  {i+1}/{len(img_paths)}: {name}")
        seg_map = run_inference(img_path)
        canvas, stats = make_canvas(img_path, seg_map)
        out = os.path.join(out_dir, f"{i+1:02d}_{name}.png")
        cv2.imwrite(out, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        for cls, pct in stats:
            if pct >= 1.0:
                print(f"    {CLASS_NAMES[cls]:8s}: {pct:.1f}%")
    print(f"  Saved → {out_dir}/")


# ── Test on source dataset images ─────────────────────────────────────────────
all_imgs = sorted(glob.glob(cfg['img_glob'], recursive=True))
if not all_imgs:
    print(f"No images found: {cfg['img_glob']}")
else:
    selected = random.sample(all_imgs, min(args.n, len(all_imgs)))
    test_batch(selected, cfg['out_dir'],
               f"{args.dataset.upper()} images → AVMI classes")

# ── Test on UGV images ────────────────────────────────────────────────────────
ugv_imgs = sorted(glob.glob(UGV_IMG_GLOB))
if ugv_imgs:
    ugv_selected = random.sample(ugv_imgs, min(args.n, len(ugv_imgs)))
    test_batch(ugv_selected, cfg['out_dir'] + '_ugv',
               f"UGV images with {args.dataset.upper()}-mapped model")

print("\nDone!")
