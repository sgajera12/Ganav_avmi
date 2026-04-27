"""
Test trained HuggingFace SegFormer-B2 on AVMI, RUGD, and RELLIS images.
Saves side-by-side (original | segmentation) PNGs to results folders.

Run locally:
    conda activate segformer
    cd /home/pinaka/GANav-offroad
    python tools/test_segformer_hf.py
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

# ── Config ─────────────────────────────────────────────────────────────────────
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT   = os.path.join(_REPO_ROOT, 'data', 'avmi_ugv')
WORK_DIR    = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_hf')
OUT_DIR     = os.path.join(_REPO_ROOT, 'results', 'segformer_b2_test')
PRETRAINED  = 'nvidia/mit-b2'
NUM_CLASSES = 6
NUM_SAMPLES = 20   # how many test images to visualize per dataset

RUGD_ROOT  = os.path.join(_REPO_ROOT, 'data', 'rugd')
RELLIS_ROOT= os.path.join(_REPO_ROOT, 'data', 'rellis')

CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')

# RGB palette for visualization
PALETTE_RGB = np.array([
    [ 24, 102, 178],  # sky      - blue
    [ 18, 182,  37],  # tree     - green
    [239, 255,  15],  # bush     - yellow
    [ 92,  19,   6],  # ground   - dark brown
    [255,  63, 250],  # obstacle - magenta
    [255,   0,   0],  # rock     - red
], dtype=np.uint8)

# Annotation color → class index (same as training)
_PALETTE_COLORS = np.array([
    [24,  102, 178],
    [18,  182,  37],
    [239, 255,  15],
    [92,   19,   6],
    [255,  63, 250],
    [255,   0,   0],
], dtype=np.int32)

def rgb_mask_to_index(rgb_img):
    h, w = rgb_img.shape[:2]
    flat = rgb_img.reshape(-1, 3).astype(np.int32)
    diff = flat[:, None, :] - _PALETTE_COLORS[None, :, :]
    dist = (diff ** 2).sum(axis=2)
    nearest = np.argmin(dist, axis=1)
    return nearest.reshape(h, w).astype(np.uint8)


def draw_legend(img: Image.Image) -> Image.Image:
    """Draw class legend on bottom-left of PIL image."""
    draw = ImageDraw.Draw(img)
    W, H = img.size
    cell = 22
    legend_h = len(CLASSES) * cell + 6
    draw.rectangle([0, H - legend_h, 160, H], fill=(20, 20, 20))
    for i, name in enumerate(CLASSES):
        y = H - legend_h + 4 + i * cell
        col = tuple(int(c) for c in PALETTE_RGB[i])
        draw.rectangle([4, y, 18, y + cell - 4], fill=col)
        draw.text((22, y), name, fill=(230, 230, 230))
    return img


def compute_miou(preds, gts, num_classes):
    """Compute mIoU from list of (pred, gt) numpy arrays."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, g in zip(preds, gts):
        mask = g < num_classes
        np.add.at(confusion, (g[mask], p[mask]), 1)
    iou_list = []
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou_list.append(tp / denom)
    return iou_list, np.mean(iou_list) if iou_list else 0.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    ckpt_path = os.path.join(WORK_DIR, 'latest.pth')
    if not os.path.exists(ckpt_path):
        print(f'ERROR: No checkpoint found at {ckpt_path}')
        return

    print(f'Loading checkpoint: {ckpt_path}')
    cfg = SegformerConfig.from_pretrained(PRETRAINED)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, config=cfg, ignore_mismatched_sizes=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    print(f'Loaded from iter {ckpt["iter"]}, val mIoU={ckpt["miou"]:.4f}')

    processor = SegformerImageProcessor(
        do_resize=True,
        size={'height': 480, 'width': 640},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    # Test images
    test_img_dir = Path(DATA_ROOT) / 'images' / 'test'
    test_ann_dir = Path(DATA_ROOT) / 'annotations' / 'test'
    files = sorted([f.stem for f in test_img_dir.glob('*.png')])
    print(f'Test images: {len(files)} — running on {min(NUM_SAMPLES, len(files))}')

    all_preds, all_gts = [], []

    for i, name in enumerate(files[:NUM_SAMPLES]):
        img = Image.open(test_img_dir / f'{name}.png').convert('RGB')
        ann_rgb = np.array(Image.open(test_ann_dir / f'{name}.png').convert('RGB'))
        gt = rgb_mask_to_index(ann_rgb)

        # Inference
        encoded = processor(images=img, return_tensors='pt')
        pixels = encoded['pixel_values'].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixels)
            logits = F.interpolate(outputs.logits,
                                   size=(img.height, img.width),
                                   mode='bilinear', align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        all_preds.append(pred)
        all_gts.append(gt)

        # Colour mask
        seg_rgb = PALETTE_RGB[pred.reshape(-1)].reshape(pred.shape[0], pred.shape[1], 3)
        seg_pil = Image.fromarray(seg_rgb)
        seg_pil = draw_legend(seg_pil)

        # GT mask
        gt_rgb = PALETTE_RGB[gt.reshape(-1)].reshape(gt.shape[0], gt.shape[1], 3)
        gt_pil = Image.fromarray(gt_rgb)

        # Side-by-side: original | GT | prediction
        W, H = img.size
        out = Image.new('RGB', (W * 3 + 8, H), (40, 40, 40))
        out.paste(img,     (0,         0))
        out.paste(gt_pil,  (W + 4,     0))
        out.paste(seg_pil, (W * 2 + 8, 0))

        # Labels
        draw = ImageDraw.Draw(out)
        draw.text((4,  4), 'Original',   fill=(255, 255, 255))
        draw.text((W + 8,     4), 'Ground Truth', fill=(255, 255, 255))
        draw.text((W * 2 + 12, 4), 'SegFormer-B2', fill=(255, 255, 255))

        out.save(os.path.join(OUT_DIR, f'{name}_result.png'))
        print(f'  [{i+1}/{min(NUM_SAMPLES, len(files))}] {name}')

    # Final mIoU
    iou_list, miou = compute_miou(all_preds, all_gts, NUM_CLASSES)
    print(f'\nTest mIoU: {miou:.4f}')
    for i, (cls, iou) in enumerate(zip(CLASSES, iou_list)):
        print(f'  {cls:10s}: {iou:.4f}')
    print(f'\nResults saved to: {OUT_DIR}')
    return model, processor, device


def run_cross_dataset(model, processor, device, img_paths, dataset_name):
    """Run inference on a list of image paths, no GT needed."""
    out_dir = os.path.join(_REPO_ROOT, 'results', f'segformer_b2_{dataset_name}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n── {dataset_name} ({len(img_paths)} images) ──')

    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert('RGB')
        encoded = processor(images=img, return_tensors='pt')
        pixels = encoded['pixel_values'].to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixels)
            logits = F.interpolate(outputs.logits,
                                   size=(img.height, img.width),
                                   mode='bilinear', align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        seg_rgb = PALETTE_RGB[pred.reshape(-1)].reshape(pred.shape[0], pred.shape[1], 3)
        seg_pil = Image.fromarray(seg_rgb)
        seg_pil = draw_legend(seg_pil)

        W, H = img.size
        out = Image.new('RGB', (W * 2 + 4, H), (40, 40, 40))
        out.paste(img,     (0,       0))
        out.paste(seg_pil, (W + 4,   0))

        draw = ImageDraw.Draw(out)
        draw.text((4,  4), 'Original',     fill=(255, 255, 255))
        draw.text((W + 8, 4), 'SegFormer-B2', fill=(255, 255, 255))

        name = Path(img_path).stem
        out.save(os.path.join(out_dir, f'{name}_result.png'))
        print(f'  [{i+1}/{len(img_paths)}] {name}')

    print(f'Results saved to: {out_dir}')


if __name__ == '__main__':
    model, processor, device = main()

    # ── RUGD cross-dataset ────────────────────────────────────────────────────
    rugd_img_dir = Path(RUGD_ROOT) / 'RUGD_frames-with-annotations'
    rugd_files = []
    with open(os.path.join(RUGD_ROOT, 'test.txt')) as f:
        for line in f:
            p = rugd_img_dir / line.strip()
            if p.exists():
                rugd_files.append(str(p))
    run_cross_dataset(model, processor, device,
                      rugd_files[:NUM_SAMPLES], 'rugd')

    # ── RELLIS cross-dataset ──────────────────────────────────────────────────
    rellis_img_dir = Path(RELLIS_ROOT) / 'image'
    rellis_files = []
    with open(os.path.join(RELLIS_ROOT, 'test.txt')) as f:
        for line in f:
            p = rellis_img_dir / (line.strip() + '.jpg')
            if not p.exists():
                p = rellis_img_dir / (line.strip() + '.png')
            if p.exists():
                rellis_files.append(str(p))
    run_cross_dataset(model, processor, device,
                      rellis_files[:NUM_SAMPLES], 'rellis')
