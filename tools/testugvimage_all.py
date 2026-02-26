import torch
import cv2
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm
import os
import glob
import random
from tqdm import tqdm

# RTX 5070 (Blackwell) fix
torch.backends.cudnn.enabled = False

device = 'cuda:0'
config_file     = 'work_dirs/ganav_avmi_finetune/ganav_avmi_finetune.py'
checkpoint_file = 'work_dirs/ganav_avmi_finetune/latest.pth'

print("Loading model …")
model = init_segmentor(config_file, checkpoint_file, device=device)
model = revert_sync_batchnorm(model)
print("Model ready!")

# ── AVMI class colours (RGB) ──────────────────────────────────────────────────
CLASS_NAMES = ['sky', 'tree', 'bush', 'ground', 'trunk', 'rock']
COLORS = np.array([
    [24,  102, 178],  # 0: sky    - blue
    [18,  182,  37],  # 1: tree   - green
    [239, 255,  15],  # 2: bush   - yellow
    [92,   19,   6],  # 3: ground - dark brown
    [255,  63, 250],  # 4: trunk  - pink/magenta
    [255,   0,   0],  # 5: rock   - red
], dtype=np.uint8)

# ── Padding artifact parameters ───────────────────────────────────────────────
INFER_H, INFER_W = 300, 375   # crop_size used at inference time

# ── Input / output ────────────────────────────────────────────────────────────
# ── AVMI test set (uncomment to use) ─────────────────────────────────────────
# input_folder  = 'data/avmi_ugv/images/test'
# ann_folder    = 'data/avmi_ugv/annotations/test'
# output_folder = 'results/avmi_test_batch/'
# os.makedirs(output_folder, exist_ok=True)
# all_paths   = sorted(glob.glob(os.path.join(input_folder, '*.png')))
# random.seed(42)
# image_paths = random.sample(all_paths, min(100, len(all_paths)))
# image_paths = sorted(image_paths)
# print(f"Randomly selected {len(image_paths)} / {len(all_paths)} images")

# ── RUGD sample images ────────────────────────────────────────────────────────
input_folder  = '/home/pinaka/Downloads/Temp/AVMI/RUGD_sample-data/images'
ann_folder    = '/home/pinaka/Downloads/Temp/AVMI/RUGD_sample-data/annotations'
output_folder = 'results/rugd_sample_test/'
os.makedirs(output_folder, exist_ok=True)
image_paths = sorted(glob.glob(os.path.join(input_folder, '*.png')))
print(f"Found {len(image_paths)} RUGD sample images")
print(f"Saving results to {output_folder}\n")

font = cv2.FONT_HERSHEY_SIMPLEX

for img_path in tqdm(image_paths):
    result  = inference_segmentor(model, img_path)
    seg_map = result[0]                        # (H, W)

    # Fix vertical alignment (same maths as testugvimage.py)
    orig_h, orig_w = seg_map.shape
    scale       = min(INFER_H / orig_h, INFER_W / orig_w)
    resized_h   = int(round(orig_h * scale))
    valid_out_h = int(round(resized_h * orig_h / INFER_H))
    if valid_out_h < orig_h:
        seg_map = cv2.resize(seg_map[:valid_out_h, :], (orig_w, orig_h),
                             interpolation=cv2.INTER_NEAREST)

    colored_seg  = COLORS[seg_map]             # (H, W, 3) RGB
    original     = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    H, W         = original_rgb.shape[:2]

    if colored_seg.shape[:2] != (H, W):
        colored_seg = cv2.resize(colored_seg, (W, H), interpolation=cv2.INTER_NEAREST)

    # Load GT RGB mask from annotations folder
    ann_path = os.path.join(ann_folder, os.path.basename(img_path))
    if os.path.exists(ann_path):
        gt_mask = cv2.cvtColor(cv2.imread(ann_path), cv2.COLOR_BGR2RGB)
        if gt_mask.shape[:2] != (H, W):
            gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        top_right = gt_mask
        tr_label  = 'GT mask (RGB)'
    else:
        top_right = colored_seg.copy()
        tr_label  = 'Seg (no GT)'

    alpha   = 0.55
    blended = cv2.addWeighted(original_rgb, 1 - alpha,
                              colored_seg.astype(np.uint8), alpha, 0)

    # Class % stats
    stats = [(cls, int(np.sum(seg_map == cls)) / seg_map.size * 100)
             for cls in range(len(CLASS_NAMES))]

    # 2×2 canvas:  Original | GT mask
    #              Overlay  | Seg + stats
    canvas = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
    canvas[:H,  :W]  = original_rgb
    canvas[:H,  W:]  = top_right
    canvas[H:,  :W]  = blended
    canvas[H:,  W:]  = colored_seg

    for (x, y), text in [
        ((10,     35),      'Original'),
        ((W + 10, 35),      tr_label),
        ((10,     H + 35),  'Overlay'),
        ((W + 10, H + 35),  'Segmentation'),
    ]:
        cv2.putText(canvas, text, (x, y), font, 0.9, (255, 255, 255), 2)

    # Legend on segmentation panel (bottom-right)
    for idx, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        y = H + 65 + idx * 28
        cv2.rectangle(canvas, (W + 10, y - 17), (W + 30, y + 3), color.tolist(), -1)
        cv2.putText(canvas, name, (W + 36, y), font, 0.65, (255, 255, 255), 1)

    # Class % on overlay panel (bottom-left) — small coloured text
    sx, sy0 = 8, H + 20
    for cls, pct in stats:
        if pct < 0.05:
            continue
        color = COLORS[cls].tolist()
        cv2.rectangle(canvas, (sx, sy0 - 11), (sx + 14, sy0 + 3), color, -1)
        cv2.putText(canvas, f"{CLASS_NAMES[cls]}: {pct:.1f}%",
                    (sx + 18, sy0), font, 0.42, (255, 255, 255), 1)
        sy0 += 20

    # Dividers
    cv2.line(canvas, (W, 0),    (W, H * 2),  (200, 200, 200), 2)
    cv2.line(canvas, (0, H),    (W * 2, H),  (200, 200, 200), 2)

    out_name = os.path.splitext(os.path.basename(img_path))[0] + '_seg.png'
    cv2.imwrite(os.path.join(output_folder, out_name),
                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

print(f"\nDone! {len(image_paths)} results saved to: {output_folder}")
