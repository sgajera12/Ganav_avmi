"""
Test the AVMI scratch model on RUGD and RELLIS raw images.
No GT masks needed — just visual inspection of what the model predicts.
Saves a 1x3 canvas per image: Original | Segmentation | Overlay
"""
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

device = 'cuda:0'

config_file     = 'work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py'
checkpoint_file = 'work_dirs/ganav_avmi_scratch/latest.pth'

for f in (config_file, checkpoint_file):
    if not os.path.exists(f):
        print(f"ERROR: not found: {f}")
        exit(1)

print("Loading model …")
model = init_segmentor(config_file, checkpoint_file, device=device)
model = revert_sync_batchnorm(model)
print("Model ready!\n")

# ── AVMI class colours (RGB) 
CLASS_NAMES = ['sky', 'tree', 'bush', 'ground', 'obstacle', 'rock']
COLORS = np.array([
    [24,  102, 178],  # 0: sky    - blue
    [18,  182,  37],  # 1: tree   - green
    [239, 255,  15],  # 2: bush   - yellow
    [92,   19,   6],  # 3: ground - dark brown
    [255,  63, 250],  # 4: obstacle  - pink/magenta
    [255,   0,   0],  # 5: rock   - red
], dtype=np.uint8)

# ── Dataset image folders ───
DATASETS = {
    'rugd':   'data/rugd/RUGD_frames-with-annotations/**/*.png',
    'rellis': 'data/rellis/image/**/*.jpg',
}

N_PER_DATASET = 3   # how many random images to test per dataset

os.makedirs('results/cross_dataset_test', exist_ok=True)


def run_inference(img_path):
    """Run model on one image, return coloured seg map aligned to original size."""
    result  = inference_segmentor(model, img_path)
    seg_map = result[0]   # (H, W)

    # Fix vertical alignment (same as AVMI test script)
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
    """Build 1x3 canvas: Original | Segmentation | Overlay."""
    colored_seg  = COLORS[seg_map]   # (H, W, 3) RGB

    original     = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    H, W         = original_rgb.shape[:2]

    if colored_seg.shape[:2] != (H, W):
        colored_seg = cv2.resize(colored_seg, (W, H), interpolation=cv2.INTER_NEAREST)

    alpha   = 0.55
    blended = cv2.addWeighted(original_rgb, 1 - alpha,
                              colored_seg.astype(np.uint8), alpha, 0)

    # Class stats
    stats = [(cls, float(np.sum(seg_map == cls) / seg_map.size * 100))
             for cls in range(len(CLASS_NAMES))]

    # 1x3 canvas
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, :W]    = original_rgb
    canvas[:, W:W*2] = colored_seg
    canvas[:, W*2:]  = blended

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Panel labels
    for x, text in [(10, 'Original'), (W + 10, 'Segmentation'), (W*2 + 10, 'Overlay')]:
        cv2.putText(canvas, text, (x, 30), font, 0.8, (255, 255, 255), 2)

    # Legend on segmentation panel
    for idx, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        y = 60 + idx * 28
        cv2.rectangle(canvas, (W + 10, y - 15), (W + 28, y + 3), color.tolist(), -1)
        cv2.putText(canvas, name, (W + 34, y), font, 0.6, (255, 255, 255), 1)

    # Class % stats on overlay panel
    sx, sy = W*2 + 8, 20
    for cls, pct in stats:
        if pct < 0.05:
            continue
        color = COLORS[cls].tolist()
        cv2.rectangle(canvas, (sx, sy - 11), (sx + 14, sy + 3), color, -1)
        cv2.putText(canvas, f"{CLASS_NAMES[cls]}: {pct:.1f}%",
                    (sx + 18, sy), font, 0.42, (255, 255, 255), 1)
        sy += 20

    # Dividing lines
    cv2.line(canvas, (W,   0), (W,   H), (200, 200, 200), 2)
    cv2.line(canvas, (W*2, 0), (W*2, H), (200, 200, 200), 2)

    return canvas, stats


# ── Run across datasets ─────
for dataset_name, img_glob in DATASETS.items():
    all_imgs = sorted(glob.glob(img_glob, recursive=True))
    if not all_imgs:
        print(f"[{dataset_name}] No images found at: {img_glob}")
        continue

    selected = random.sample(all_imgs, min(N_PER_DATASET, len(all_imgs)))
    print(f"[{dataset_name}] Testing {len(selected)} images …")

    for i, img_path in enumerate(selected):
        print(f"  {i+1}/{len(selected)}: {os.path.basename(img_path)}")
        seg_map = run_inference(img_path)
        canvas, stats = make_canvas(img_path, seg_map)

        out_name = f"{dataset_name}_{i+1}_{os.path.splitext(os.path.basename(img_path))[0]}.png"
        out_path = os.path.join('results/cross_dataset_test', out_name)
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        for cls, pct in stats:
            if pct >= 0.5:
                print(f"    {CLASS_NAMES[cls]:8s}: {pct:.1f}%")
    print()

print("Done! Results in results/cross_dataset_test/")
