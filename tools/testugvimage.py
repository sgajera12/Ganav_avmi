import torch
import cv2
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm
import os

# RTX 5070 (Blackwell) fix
torch.backends.cudnn.enabled = False

device = 'cuda:0'
print(f"Using device: {device}")

config_file     = 'work_dirs/ganav_avmi_finetune/ganav_avmi_finetune.py'
checkpoint_file = 'work_dirs/ganav_avmi_finetune/latest.pth'

for f in (config_file, checkpoint_file):
    if not os.path.exists(f):
        print(f"ERROR: not found: {f}")
        exit(1)

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

# ── Test image — change this to any image you want to test ───────────────────
img_path = 'data/avmi_ugv/images/test/s1_00003.png'

if not os.path.exists(img_path):
    print(f"Error: image not found: {img_path}")
    exit(1)

print(f"Processing: {img_path}")
result  = inference_segmentor(model, img_path)
seg_map = result[0]                          # (H, W)
print(f"Segmentation shape: {seg_map.shape}")

# ── Fix vertical alignment ────────────────────────────────────────────────────
# inference_segmentor resizes the image to fit crop_size=(300,375), pads with
# zeros, then resizes the prediction back to original size using the FULL
# padded height.  This compresses valid content into only ~94% of the output
# height, shifting all boundaries upward by ~6-7%.
# Fix: identify the valid (non-padded) rows in the output and stretch them
# back to the full original height.
orig_h, orig_w = seg_map.shape
INFER_H, INFER_W = 300, 375                          # crop_size at inference
scale       = min(INFER_H / orig_h, INFER_W / orig_w)  # 0.586 for 480x640
resized_h   = int(round(orig_h * scale))             # rows of real content in inference input (281)
valid_out_h = int(round(resized_h * orig_h / INFER_H))  # rows of real content in output (450)
if valid_out_h < orig_h:
    seg_map = cv2.resize(seg_map[:valid_out_h, :], (orig_w, orig_h),
                         interpolation=cv2.INTER_NEAREST)
    print(f"Alignment corrected: stretched {valid_out_h} valid rows → {orig_h}")

# Coloured segmentation mask
colored_seg  = COLORS[seg_map]               # (H, W, 3) RGB

# Original image
original     = cv2.imread(img_path)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
H, W         = original_rgb.shape[:2]

if colored_seg.shape[:2] != (H, W):
    colored_seg = cv2.resize(colored_seg, (W, H), interpolation=cv2.INTER_NEAREST)

# Load ground-truth RGB mask if available (top-right panel)
mask_path = img_path.replace('images', 'annotations')
if os.path.exists(mask_path):
    gt_mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    if gt_mask.shape[:2] != (H, W):
        gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    top_right = gt_mask
    tr_label  = 'GT mask (RGB)'
else:
    top_right = colored_seg.copy()
    tr_label  = 'Seg (no GT)'

# Overlay blend
alpha   = 0.55
blended = cv2.addWeighted(
    original_rgb, 1 - alpha,
    colored_seg.astype(np.uint8), alpha, 0
)

# ── Class stats ───────────────────────────────────────────────────────────────
stats = []
for cls in range(len(CLASS_NAMES)):
    px  = int(np.sum(seg_map == cls))
    pct = px / seg_map.size * 100
    stats.append((cls, pct))

# ── 2×2 canvas ────────────────────────────────────────────────────────────────
#   Top-left: Original   |  Top-right: GT RGB mask
#   Bot-left: Seg        |  Bot-right: Overlay
canvas = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
canvas[:H,  :W]  = original_rgb
canvas[:H,  W:]  = top_right
canvas[H:,  :W]  = colored_seg
canvas[H:,  W:]  = blended

font = cv2.FONT_HERSHEY_SIMPLEX

# Panel labels
for (x, y), text in [
    ((10,     35),      'Original'),
    ((W + 10, 35),      tr_label),
    ((10,     H + 35),  'Segmentation'),
    ((W + 10, H + 35),  'Overlay'),
]:
    cv2.putText(canvas, text, (x, y), font, 0.9, (255, 255, 255), 2)

# Legend on segmentation panel (bottom-left)
for idx, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    y = H + 65 + idx * 28
    cv2.rectangle(canvas, (10, y - 17), (30, y + 3), color.tolist(), -1)
    cv2.putText(canvas, name, (36, y), font, 0.65, (255, 255, 255), 1)

# Class % stats on overlay panel (bottom-right) — small coloured text
sx, sy0 = W + 8, H + 20
for cls, pct in stats:
    if pct < 0.05:
        continue
    color = COLORS[cls].tolist()
    label = f"{CLASS_NAMES[cls]}: {pct:.1f}%"
    cv2.rectangle(canvas, (sx, sy0 - 11), (sx + 14, sy0 + 3), color, -1)
    cv2.putText(canvas, label, (sx + 18, sy0), font, 0.42, (255, 255, 255), 1)
    sy0 += 20

# Dividing lines
cv2.line(canvas, (W, 0),     (W, H * 2),  (200, 200, 200), 2)
cv2.line(canvas, (0, H),     (W * 2, H),  (200, 200, 200), 2)

output_path = 'avmi_test_result.png'
cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print(f"\nResult saved to: {output_path}")

print("\nDetected terrain types:")
for cls, pct in stats:
    if pct >= 0.05:
        print(f"  {CLASS_NAMES[cls]:8s}: {pct:5.1f}%")
