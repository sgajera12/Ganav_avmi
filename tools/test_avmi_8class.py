"""
Test AVMI 8-class model on UGV images and GOD real-world dataset.
Use --dataset ugv or --dataset god

Run:
    conda activate segformer
    cd /home/pinaka/GANav-offroad
    python tools/test_avmi_8class.py --dataset god
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

# ── Config ────────────────────────────────────────────────────────────────────
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT        = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_8class', 'latest.pth')
DATA_ROOT   = os.path.join(_REPO_ROOT, 'data', 'avmi_ugv')
GOD_ROOT    = os.path.join(_REPO_ROOT, 'data', 'god')
NUM_CLASSES = 8
NUM_SAMPLES = 30

CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock', 'water', 'concrete')

PALETTE_RGB = np.array([
    [ 24, 102, 178],  # sky      - blue
    [ 18, 182,  37],  # tree     - green
    [239, 255,  15],  # bush     - yellow
    [ 92,  19,   6],  # ground   - dark brown
    [255,  63, 250],  # obstacle - magenta
    [255,   0,   0],  # rock     - red
    [  0, 200, 255],  # water    - cyan
    [180, 180, 180],  # concrete - grey
], dtype=np.uint8)

_PROCESSOR = SegformerImageProcessor(
    do_resize=True, size={'height': 480, 'width': 640},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)

def load_model(device):
    cfg = SegformerConfig.from_pretrained('nvidia/mit-b2')
    cfg.num_labels = NUM_CLASSES
    cfg.id2label = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b2', config=cfg, ignore_mismatched_sizes=True)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded: iter={ckpt["iter"]}  val_mIoU={ckpt["miou"]:.4f}')
    return model.to(device).eval()


def infer(model, img, device):
    pv = _PROCESSOR(images=img, return_tensors='pt')['pixel_values'].to(device)
    with torch.no_grad():
        logits = model(pixel_values=pv).logits
        up = F.interpolate(logits, size=(img.height, img.width),
                           mode='bilinear', align_corners=False)
    return up.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


def save_result(out_dir, name, img, pred):
    pred_rgb = PALETTE_RGB[pred]
    pred_img = Image.fromarray(pred_rgb)
    canvas   = Image.new('RGB', (img.width * 2, img.height))
    canvas.paste(img, (0, 0))
    canvas.paste(pred_img, (img.width, 0))
    canvas.save(os.path.join(out_dir, name))


def print_coverage(total_pixels):
    grand_total = total_pixels.sum()
    print('\n── Class coverage ──')
    for i, c in enumerate(CLASSES):
        pct = total_pixels[i] / grand_total * 100
        flag = '  ← spurious' if i in (6, 7) and pct > 2.0 else ''
        print(f'  {c:10s}: {pct:5.1f}%{flag}')


def test_ugv(model, device):
    out_dir = os.path.join(_REPO_ROOT, 'results', 'avmi_8class_ugv')
    os.makedirs(out_dir, exist_ok=True)
    files = sorted((Path(DATA_ROOT) / 'images' / 'val').glob('*.png'))
    files = files[::max(1, len(files) // NUM_SAMPLES)][:NUM_SAMPLES]
    total = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in files:
        img  = Image.open(f).convert('RGB')
        pred = infer(model, img, device)
        total += np.bincount(pred.ravel(), minlength=NUM_CLASSES)
        save_result(out_dir, f.name, img, pred)
        print(f'  {f.name}')
    print_coverage(total)
    print(f'Saved {len(files)} images to {out_dir}')


def test_god(model, device):
    out_dir = os.path.join(_REPO_ROOT, 'results', 'avmi_8class_god')
    os.makedirs(out_dir, exist_ok=True)

    img_root = Path(GOD_ROOT) / 'pylon_camera_node'
    all_files = []
    for seq in sorted(img_root.iterdir()):
        for f in sorted(seq.glob('*.png')):
            all_files.append(f)

    # Sample evenly
    step  = max(1, len(all_files) // NUM_SAMPLES)
    files = all_files[::step][:NUM_SAMPLES]

    total = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in files:
        img  = Image.open(f).convert('RGB')
        pred = infer(model, img, device)
        total += np.bincount(pred.ravel(), minlength=NUM_CLASSES)
        save_result(out_dir, f'{f.parent.name}_{f.name}', img, pred)
        print(f'  {f.parent.name}/{f.name}')
    print_coverage(total)
    print(f'Saved {len(files)} images to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ugv', 'god'], default='god')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = load_model(device)

    if args.dataset == 'ugv':
        test_ugv(model, device)
    else:
        test_god(model, device)
