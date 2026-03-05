"""
Visual test for fine-tuned navigability models (RUGD / RELLIS).
Shows: Original | Segmentation | Overlay
Uses navigability colour coding (same for both RUGD and RELLIS).

Usage:
    # Test RUGD fine-tuned model
    python tools/test_nav_models.py --dataset rugd

    # Test RELLIS fine-tuned model
    python tools/test_nav_models.py --dataset rellis
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

# RTX 5070 (Blackwell) fix
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['rugd', 'rellis'], required=True,
                    help='Which fine-tuned model to test')
parser.add_argument('--n', type=int, default=5,
                    help='Number of random images per source to test (default 5)')
args = parser.parse_args()

# ── Model / data paths per dataset ───────────────────────────────────────────
CONFIGS = {
    'rugd': {
        'config':     'work_dirs/ganav_rugd_from_avmi/ganav_rugd_from_avmi.py',
        'checkpoint': 'work_dirs/ganav_rugd_from_avmi/latest.pth',
        'img_glob':   'data/rugd/RUGD_frames-with-annotations/**/*.png',
        'out_dir':    'results/nav_test_rugd',
    },
    'rellis': {
        'config':     'work_dirs/ganav_rellis_from_avmi/ganav_rellis_from_avmi.py',
        'checkpoint': 'work_dirs/ganav_rellis_from_avmi/latest.pth',
        'img_glob':   'data/rellis/image/**/*.jpg',
        'out_dir':    'results/nav_test_rellis',
    },
}

# UGV test images — tested alongside the main dataset to verify the model
# still recognises terrain from our own camera
UGV_IMG_GLOB = 'data/avmi_ugv/images/test/*.png'

# ── Navigability classes (same for both RUGD and RELLIS Group6) ───────────────
CLASS_NAMES = ['background', 'L1 Smooth', 'L2 Rough', 'L3 Bumpy', 'non-Nav', 'obstacle']
COLORS = np.array([
    [0,   0,   0],    # 0: background - black
    [0,   128, 0],    # 1: L1 Smooth  - dark green  (easy to drive)
    [255, 255, 0],    # 2: L2 Rough   - yellow      (driveable with care)
    [255, 128, 0],    # 3: L3 Bumpy   - orange      (challenging)
    [255, 0,   0],    # 4: non-Nav    - red          (avoid)
    [0,   0,   128],  # 5: obstacle   - dark blue    (blocked)
], dtype=np.uint8)

cfg = CONFIGS[args.dataset]

for f in (cfg['config'], cfg['checkpoint']):
    if not os.path.exists(f):
        print(f"ERROR: not found: {f}")
        print("Has training finished? Check work_dirs/ganav_{args.dataset}_from_avmi/")
        exit(1)

print(f"Loading [{args.dataset}] model …")
model = init_segmentor(cfg['config'], cfg['checkpoint'], device='cuda:0')
model = revert_sync_batchnorm(model)
print("Model ready!\n")

all_imgs = sorted(glob.glob(cfg['img_glob'], recursive=True))
if not all_imgs:
    print(f"No images found at: {cfg['img_glob']}")
    exit(1)

selected = random.sample(all_imgs, min(args.n, len(all_imgs)))
print(f"Testing {len(selected)} random images from {args.dataset.upper()} …\n")
os.makedirs(cfg['out_dir'], exist_ok=True)


def process(img_path):
    result  = inference_segmentor(model, img_path)
    seg_map = result[0]

    # Vertical alignment fix
    orig_h, orig_w = seg_map.shape
    INFER_H, INFER_W = 300, 375
    scale       = min(INFER_H / orig_h, INFER_W / orig_w)
    resized_h   = int(round(orig_h * scale))
    valid_out_h = int(round(resized_h * orig_h / INFER_H))
    if valid_out_h < orig_h:
        seg_map = cv2.resize(seg_map[:valid_out_h, :], (orig_w, orig_h),
                             interpolation=cv2.INTER_NEAREST)

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

    # 1×3 canvas: Original | Segmentation | Overlay
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, :W]    = original_rgb
    canvas[:, W:W*2] = colored_seg
    canvas[:, W*2:]  = blended

    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, text in [(10, 'Original'), (W + 10, 'Navigability Seg'), (W*2 + 10, 'Overlay')]:
        cv2.putText(canvas, text, (x, 30), font, 0.8, (255, 255, 255), 2)

    # Legend (middle panel)
    for idx, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        y = 60 + idx * 28
        cv2.rectangle(canvas, (W + 10, y - 15), (W + 28, y + 3), color.tolist(), -1)
        cv2.putText(canvas, name, (W + 34, y), font, 0.55, (255, 255, 255), 1)

    # Stats (overlay panel)
    sx, sy = W*2 + 8, 20
    for cls, pct in stats:
        if pct < 0.5:
            continue
        color = COLORS[cls].tolist()
        cv2.rectangle(canvas, (sx, sy - 11), (sx + 14, sy + 3), color, -1)
        cv2.putText(canvas, f"{CLASS_NAMES[cls]}: {pct:.1f}%",
                    (sx + 18, sy), font, 0.42, (255, 255, 255), 1)
        sy += 20

    cv2.line(canvas, (W,   0), (W,   H), (200, 200, 200), 2)
    cv2.line(canvas, (W*2, 0), (W*2, H), (200, 200, 200), 2)

    return canvas, stats


for i, img_path in enumerate(selected):
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"  {i+1}/{len(selected)}: {name}")
    canvas, stats = process(img_path)
    out = os.path.join(cfg['out_dir'], f"{i+1:02d}_{name}.png")
    cv2.imwrite(out, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    for cls, pct in stats:
        if pct >= 1.0:
            print(f"    {CLASS_NAMES[cls]:15s}: {pct:.1f}%")

# ── Also test on UGV images to verify terrain recognition is preserved ────────
ugv_imgs = sorted(glob.glob(UGV_IMG_GLOB))
if ugv_imgs:
    ugv_out_dir = cfg['out_dir'] + '_ugv'
    os.makedirs(ugv_out_dir, exist_ok=True)
    ugv_selected = random.sample(ugv_imgs, min(args.n, len(ugv_imgs)))
    print(f"\nTesting {len(ugv_selected)} UGV images with same model …")
    for i, img_path in enumerate(ugv_selected):
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  {i+1}/{len(ugv_selected)}: {name}")
        canvas, stats = process(img_path)
        out = os.path.join(ugv_out_dir, f"{i+1:02d}_{name}.png")
        cv2.imwrite(out, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        for cls, pct in stats:
            if pct >= 1.0:
                print(f"    {CLASS_NAMES[cls]:15s}: {pct:.1f}%")
    print(f"UGV results in {ugv_out_dir}/")

print(f"\nDone! {args.dataset.upper()} results in {cfg['out_dir']}/")
