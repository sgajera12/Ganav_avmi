"""
test_rugd_v1_6class.py

Tests the OLD RUGD fine-tune model (segformer_b2_finetune_rugd):
  - 6-class: sky | tree | bush | ground | obstacle | rock
  - Trained with WRONG _orig.png mapping (off-by-one bug, water→rock, sky confused)
  - Kept for comparison against v2 (fixed mapping, 8-class)

Results saved to: results/test_rugd_v1_6class/
  rugd/     — RUGD test images with GT comparison
  avmi_ugv/ — AVMI UGV images (no GT, prediction only)

Run:
    conda activate segformer
    cd /home/pinaka/GANav-offroad
    python tools/test_rugd_v1_6class.py
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

# ── Config ────────────────────────────────────────────────────────────────────
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT        = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_finetune_rugd',
                           'iter_48000_miou0.6296.pth')
RUGD_ROOT   = os.path.join(_REPO_ROOT, 'data', 'rugd')
AVMI_ROOT   = os.path.join(_REPO_ROOT, 'data', 'avmi_ugv')
OUT_ROOT    = os.path.join(_REPO_ROOT, 'results', 'test_rugd_v1_6class')
PRETRAINED  = 'nvidia/mit-b2'
NUM_CLASSES = 6
NUM_SAMPLES = 30

CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')

PALETTE_RGB = np.array([
    [ 24, 102, 178],  # sky      - blue
    [ 18, 182,  37],  # tree     - green
    [239, 255,  15],  # bush     - yellow
    [ 92,  19,   6],  # ground   - dark brown
    [255,  63, 250],  # obstacle - magenta
    [255,   0,   0],  # rock     - red
], dtype=np.uint8)

# AVMI GT color → class
_AVMI_PALETTE = np.array([
    [ 24, 102, 178], [ 18, 182,  37], [239, 255,  15],
    [ 92,  19,   6], [255,  63, 250], [255,   0,   0],
], dtype=np.int32)

# ── OLD RUGD mapping (v1 — wrong off-by-one, kept for consistency with model) ─
# _orig.png stores colormap_id - 1, so sky=6, water=5, tree=3, grass=2
# v1 mapping was written assuming colormap IDs directly → sky was mapped as water
_RUGD_MAP_V1 = np.full(256, 255, dtype=np.uint8)
_RUGD_MAP_V1[1]  = 3  # (was "dirt"  but actual=sand)    → ground
_RUGD_MAP_V1[2]  = 3  # (was "sand"  but actual=grass)   → ground
_RUGD_MAP_V1[3]  = 3  # (was "grass" but actual=tree)    → ground  ← bug
_RUGD_MAP_V1[4]  = 1  # (was "tree"  but actual=pole)    → tree    ← bug
_RUGD_MAP_V1[5]  = 4  # (was "pole"  but actual=water)   → obstacle← bug
_RUGD_MAP_V1[6]  = 5  # (was "water" but actual=sky)     → rock    ← bug
_RUGD_MAP_V1[7]  = 0  # (was "sky"   but actual=vehicle) → sky     ← bug
_RUGD_MAP_V1[8]  = 4  # vehicle   → obstacle
_RUGD_MAP_V1[9]  = 4  # container → obstacle
_RUGD_MAP_V1[10] = 3  # asphalt   → ground
_RUGD_MAP_V1[11] = 3  # gravel    → ground
_RUGD_MAP_V1[12] = 4  # building  → obstacle
_RUGD_MAP_V1[13] = 3  # mulch     → ground
_RUGD_MAP_V1[14] = 5  # rock-bed  → rock
_RUGD_MAP_V1[15] = 4  # log       → obstacle
_RUGD_MAP_V1[16] = 4  # bicycle   → obstacle
_RUGD_MAP_V1[17] = 4  # person    → obstacle
_RUGD_MAP_V1[18] = 4  # fence     → obstacle
_RUGD_MAP_V1[19] = 2  # bush      → bush
_RUGD_MAP_V1[20] = 4  # sign      → obstacle
_RUGD_MAP_V1[21] = 5  # rock      → rock
_RUGD_MAP_V1[22] = 4  # bridge    → obstacle
_RUGD_MAP_V1[23] = 3  # concrete  → ground
_RUGD_MAP_V1[24] = 4  # picnic-table → obstacle

# ── Helpers ───────────────────────────────────────────────────────────────────
_PROCESSOR = SegformerImageProcessor(
    do_resize=True, size={'height': 480, 'width': 640},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)

def load_model(device):
    cfg = SegformerConfig.from_pretrained(PRETRAINED)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, config=cfg, ignore_mismatched_sizes=True)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    iter_n = ckpt.get('iter', '?')
    miou_n = ckpt.get('miou', float('nan'))
    print(f'  Loaded: iter={iter_n}  train_val_mIoU={miou_n:.4f}')
    return model.to(device).eval()

def infer(model, img, device):
    pv = _PROCESSOR(images=img, return_tensors='pt')['pixel_values'].to(device)
    with torch.no_grad():
        logits = model(pixel_values=pv).logits
        up = F.interpolate(logits, size=(img.height, img.width),
                           mode='bilinear', align_corners=False)
    return up.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

def avmi_rgb_to_index(rgb_np):
    flat = rgb_np.reshape(-1, 3).astype(np.int32)
    diff = flat[:, None, :] - _AVMI_PALETTE[None, :, :]
    return np.argmin((diff**2).sum(2), axis=1).reshape(rgb_np.shape[:2]).astype(np.uint8)

def draw_legend(img_pil):
    draw = ImageDraw.Draw(img_pil)
    W, H = img_pil.size
    cell = 22
    legend_h = NUM_CLASSES * cell + 6
    draw.rectangle([0, H - legend_h, 160, H], fill=(20, 20, 20))
    for i, name in enumerate(CLASSES):
        y = H - legend_h + 4 + i * cell
        draw.rectangle([4, y, 18, y + cell - 4], fill=tuple(int(c) for c in PALETTE_RGB[i]))
        draw.text((22, y), name, fill=(230, 230, 230))
    return img_pil

def save_result(out_dir, name, img, pred, gt=None):
    seg_rgb = PALETTE_RGB[pred]
    seg_pil = draw_legend(Image.fromarray(seg_rgb))
    W, H = img.size
    if gt is not None:
        void_mask = (gt == 255)
        gt_safe   = np.clip(gt, 0, NUM_CLASSES - 1)
        gt_rgb    = PALETTE_RGB[gt_safe.reshape(-1)].reshape(H, W, 3)
        gt_rgb[void_mask] = 0
        canvas = Image.new('RGB', (W * 3 + 8, H), (40, 40, 40))
        canvas.paste(img, (0, 0))
        canvas.paste(Image.fromarray(gt_rgb), (W + 4, 0))
        canvas.paste(seg_pil, (W * 2 + 8, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4),          'Original',     fill=(255, 255, 255))
        draw.text((W + 8, 4),      'Ground Truth', fill=(255, 255, 255))
        draw.text((W * 2 + 12, 4), 'RUGD v1',      fill=(255, 255, 255))
    else:
        canvas = Image.new('RGB', (W * 2 + 4, H), (40, 40, 40))
        canvas.paste(img, (0, 0))
        canvas.paste(seg_pil, (W + 4, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4),     'Original', fill=(255, 255, 255))
        draw.text((W + 8, 4), 'RUGD v1',  fill=(255, 255, 255))
    canvas.save(os.path.join(out_dir, f'{name}_result.png'))

def compute_miou(preds, gts):
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for p, g in zip(preds, gts):
        mask = g < NUM_CLASSES
        np.add.at(conf, (g[mask], p[mask]), 1)
    ious = []
    for i in range(NUM_CLASSES):
        tp = conf[i, i]; fp = conf[:, i].sum() - tp; fn = conf[i, :].sum() - tp
        d  = tp + fp + fn
        if d > 0: ious.append(tp / d)
    return ious, (sum(ious) / len(ious) if ious else 0.0)

# ── Tests ─────────────────────────────────────────────────────────────────────
def test_rugd(model, device):
    out_dir = os.path.join(OUT_ROOT, 'rugd')
    os.makedirs(out_dir, exist_ok=True)
    img_root = os.path.join(RUGD_ROOT, 'RUGD_frames-with-annotations')
    ann_root = os.path.join(RUGD_ROOT, 'RUGD_annotations')
    with open(os.path.join(RUGD_ROOT, 'test_ours.txt')) as f:
        ids = [l.strip() for l in f if l.strip()]
    print(f'  RUGD test: {len(ids)} images')
    all_preds, all_gts = [], []
    for i, rid in enumerate(ids):
        seq, name = rid.split('/')
        img = Image.open(os.path.join(img_root, seq, name + '.png')).convert('RGB')
        ann = np.array(Image.open(os.path.join(ann_root, seq, name + '_orig.png')))
        gt  = _RUGD_MAP_V1[ann]
        pred = infer(model, img, device)
        all_preds.append(pred); all_gts.append(gt)
        if i < NUM_SAMPLES:
            save_result(out_dir, name, img, pred, gt)
        if (i + 1) % 100 == 0:
            print(f'    {i+1}/{len(ids)}')
    ious, miou = compute_miou(all_preds, all_gts)
    print(f'  mIoU on RUGD test: {miou:.4f}')
    for cls, iou in zip(CLASSES, ious):
        print(f'    {cls:10s}: {iou:.4f}')

def test_avmi(model, device):
    out_dir = os.path.join(OUT_ROOT, 'avmi_ugv')
    os.makedirs(out_dir, exist_ok=True)
    img_dir = Path(AVMI_ROOT) / 'images' / 'test'
    ann_dir = Path(AVMI_ROOT) / 'annotations' / 'test'
    files   = sorted([f.stem for f in img_dir.glob('*.png')])
    print(f'  AVMI UGV test: {len(files)} images')
    all_preds, all_gts = [], []
    for i, name in enumerate(files):
        img     = Image.open(img_dir / f'{name}.png').convert('RGB')
        ann_rgb = np.array(Image.open(ann_dir / f'{name}.png').convert('RGB'))
        gt      = avmi_rgb_to_index(ann_rgb)
        pred    = infer(model, img, device)
        all_preds.append(pred); all_gts.append(gt)
        if i < NUM_SAMPLES:
            save_result(out_dir, name, img, pred, gt)
        if (i + 1) % 100 == 0:
            print(f'    {i+1}/{len(files)}')
    ious, miou = compute_miou(all_preds, all_gts)
    print(f'  mIoU on AVMI UGV: {miou:.4f}')
    for cls, iou in zip(CLASSES, ious):
        print(f'    {cls:10s}: {iou:.4f}')

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {CKPT}')
    model = load_model(device)

    print('\n[RUGD v1 → RUGD test set]')
    test_rugd(model, device)

    print('\n[RUGD v1 → AVMI UGV test]')
    test_avmi(model, device)

    print(f'\nAll results saved to: {OUT_ROOT}')
