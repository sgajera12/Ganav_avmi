"""
test_finetune_models.py

Tests both fine-tuned SegFormer-B2 models:
  1. RUGD fine-tune  → RUGD test set (with GT mIoU) + AVMI UGV test set
  2. RELLIS fine-tune → RELLIS test set (with GT mIoU) + AVMI UGV test set

Saves side-by-side (Original | GT | Prediction) PNGs for each run.

Run:
    conda activate segformer
    cd /home/pinaka/GANav-offroad
    python tools/test_finetune_models.py
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

# ── Config ─────────────────────────────────────────────────────────────────────
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED   = 'nvidia/mit-b2'
NUM_CLASSES  = 8
NUM_SAMPLES  = 30   # images to save as PNGs per dataset (mIoU uses all)
CLASSES      = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock', 'water', 'concrete')

RUGD_CKPT    = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_finetune_rugd_v2',   'iter_22000_miou0.7688.pth')
RELLIS_CKPT  = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_finetune_rellis_v2', 'iter_50000_miou0.7557.pth')
GOD_CKPT     = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_finetune_god',       'iter_28000_miou0.5442.pth')

RUGD_ROOT    = os.path.join(_REPO_ROOT, 'data', 'rugd')
RELLIS_ROOT  = os.path.join(_REPO_ROOT, 'data', 'rellis')
AVMI_ROOT    = os.path.join(_REPO_ROOT, 'data', 'avmi_ugv')
GOD_ROOT     = os.path.join(_REPO_ROOT, 'data', 'god')

# ── Palette ────────────────────────────────────────────────────────────────────
PALETTE_RGB = np.array([
    [ 24, 102, 178],  # sky
    [ 18, 182,  37],  # tree
    [239, 255,  15],  # bush
    [ 92,  19,   6],  # ground
    [255,  63, 250],  # obstacle
    [255,   0,   0],  # rock
    [  0, 200, 255],  # water    - cyan
    [180, 180, 180],  # concrete - grey
], dtype=np.uint8)

# For AVMI GT (RGB mask → class index)
_AVMI_PALETTE = np.array([
    [24,  102, 178],
    [18,  182,  37],
    [239, 255,  15],
    [92,   19,   6],
    [255,  63, 250],
    [255,   0,   0],
], dtype=np.int32)

# RUGD _orig.png stores colormap_id - 1 (void excluded, everything shifts down)
# 0=dirt,1=sand,2=grass,3=tree,4=pole,5=water,6=sky,7=vehicle,8=container,
# 9=asphalt,10=gravel,11=building,12=mulch,13=rock-bed,14=log,15=bicycle,
# 16=person,17=fence,18=bush,19=sign,20=rock,21=bridge,22=concrete,23=picnic-table
_RUGD_MAP = np.full(256, 255, dtype=np.uint8)
_RUGD_MAP[0]  = 3  # dirt         → ground
_RUGD_MAP[1]  = 3  # sand         → ground
_RUGD_MAP[2]  = 3  # grass        → ground
_RUGD_MAP[3]  = 1  # tree         → tree
_RUGD_MAP[4]  = 4  # pole         → obstacle
_RUGD_MAP[5]  = 6  # water        → water
_RUGD_MAP[6]  = 0  # sky          → sky
_RUGD_MAP[7]  = 4  # vehicle      → obstacle
_RUGD_MAP[8]  = 4  # container    → obstacle
_RUGD_MAP[9]  = 3  # asphalt      → ground
_RUGD_MAP[10] = 3  # gravel       → ground
_RUGD_MAP[11] = 4  # building     → obstacle
_RUGD_MAP[12] = 3  # mulch        → ground
_RUGD_MAP[13] = 5  # rock-bed     → rock
_RUGD_MAP[14] = 4  # log          → obstacle
_RUGD_MAP[15] = 4  # bicycle      → obstacle
_RUGD_MAP[16] = 4  # person       → obstacle
_RUGD_MAP[17] = 4  # fence        → obstacle
_RUGD_MAP[18] = 2  # bush         → bush
_RUGD_MAP[19] = 4  # sign         → obstacle
_RUGD_MAP[20] = 5  # rock         → rock
_RUGD_MAP[21] = 4  # bridge       → obstacle
_RUGD_MAP[22] = 7  # concrete     → concrete
_RUGD_MAP[23] = 4  # picnic-table → obstacle

# RELLIS ontology ID → 6-class
_RELLIS_MAP = np.full(256, 255, dtype=np.uint8)
_RELLIS_MAP[1]  = 3  # dirt      → ground
_RELLIS_MAP[3]  = 3  # grass     → ground
_RELLIS_MAP[4]  = 1  # tree      → tree
_RELLIS_MAP[5]  = 4  # pole      → obstacle
_RELLIS_MAP[6]  = 6  # water     → water
_RELLIS_MAP[7]  = 0  # sky       → sky
_RELLIS_MAP[8]  = 4  # vehicle   → obstacle
_RELLIS_MAP[9]  = 4  # object    → obstacle
_RELLIS_MAP[10] = 3  # asphalt   → ground
_RELLIS_MAP[12] = 4  # building  → obstacle
_RELLIS_MAP[15] = 4  # log       → obstacle
_RELLIS_MAP[17] = 4  # person    → obstacle
_RELLIS_MAP[18] = 4  # fence     → obstacle
_RELLIS_MAP[19] = 2  # bush      → bush
_RELLIS_MAP[23] = 7  # concrete  → concrete
_RELLIS_MAP[27] = 4  # barrier   → obstacle
_RELLIS_MAP[29] = 6  # puddle    → water
_RELLIS_MAP[31] = 255 # mud      → ignore
_RELLIS_MAP[33] = 5  # rubble    → rock

# GOD ontology IDs stored directly (verified against color annotations)
# 0=void,1=dirt,3=grass,4=trees,5=pole,6=water,7=sky,8=vehicle,9=object,
# 10=asphalt,12=building,15=log,17=person,18=fence,19=bush,23=concrete,
# 27=barrier,31=puddle,33=mud,34=rubble,35=mulch,36=gravel
_GOD_MAP = np.full(256, 255, dtype=np.uint8)
_GOD_MAP[1]  = 3  # dirt     → ground
_GOD_MAP[3]  = 3  # grass    → ground
_GOD_MAP[4]  = 1  # trees    → tree
_GOD_MAP[5]  = 4  # pole     → obstacle
_GOD_MAP[6]  = 6  # water    → water
_GOD_MAP[7]  = 0  # sky      → sky
_GOD_MAP[8]  = 4  # vehicle  → obstacle
_GOD_MAP[9]  = 4  # object   → obstacle
_GOD_MAP[10] = 3  # asphalt  → ground
_GOD_MAP[12] = 4  # building → obstacle
_GOD_MAP[15] = 4  # log      → obstacle
_GOD_MAP[17] = 4  # person   → obstacle
_GOD_MAP[18] = 4  # fence    → obstacle
_GOD_MAP[19] = 2  # bush     → bush
_GOD_MAP[23] = 7  # concrete → concrete
_GOD_MAP[27] = 4  # barrier  → obstacle
_GOD_MAP[31] = 6  # puddle   → water
_GOD_MAP[33] = 3  # mud      → ground
_GOD_MAP[34] = 5  # rubble   → rock
_GOD_MAP[35] = 3  # mulch    → ground
_GOD_MAP[36] = 3  # gravel   → ground

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    cfg = SegformerConfig.from_pretrained(PRETRAINED)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label   = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id   = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, config=cfg, ignore_mismatched_sizes=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # support both full checkpoint dicts and bare state dicts
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device).eval()
    iter_n = ckpt.get('iter', '?')
    miou_n = ckpt.get('miou', float('nan'))
    print(f'  Loaded: iter={iter_n}  train_val_mIoU={miou_n:.4f}')
    return model


def make_processor():
    return SegformerImageProcessor(
        do_resize=True, size={'height': 480, 'width': 640},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225])


def infer(model, processor, img_pil, device):
    pv = processor(images=img_pil, return_tensors='pt')['pixel_values'].to(device)
    with torch.no_grad():
        logits = F.interpolate(model(pixel_values=pv).logits,
                               size=(img_pil.height, img_pil.width),
                               mode='bilinear', align_corners=False)
    return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


def compute_miou(preds, gts):
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for p, g in zip(preds, gts):
        mask = g < NUM_CLASSES
        np.add.at(conf, (g[mask], p[mask]), 1)
    ious = []
    for i in range(NUM_CLASSES):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        d  = tp + fp + fn
        if d > 0:
            ious.append(tp / d)
    return ious, (sum(ious) / len(ious) if ious else 0.0)


def draw_legend(img_pil):
    draw = ImageDraw.Draw(img_pil)
    W, H = img_pil.size
    cell = 22
    legend_h = len(CLASSES) * cell + 6
    draw.rectangle([0, H - legend_h, 160, H], fill=(20, 20, 20))
    for i, name in enumerate(CLASSES):
        y   = H - legend_h + 4 + i * cell
        col = tuple(int(c) for c in PALETTE_RGB[i])
        draw.rectangle([4, y, 18, y + cell - 4], fill=col)
        draw.text((22, y), name, fill=(230, 230, 230))
    return img_pil


def save_result(out_dir, name, img_pil, pred, gt=None, model_label='SegFormer-B2'):
    seg_rgb = PALETTE_RGB[pred.reshape(-1)].reshape(pred.shape[0], pred.shape[1], 3)
    seg_pil = draw_legend(Image.fromarray(seg_rgb))

    W, H = img_pil.size
    if gt is not None:
        # Render GT: valid classes (0-5) → palette colour, void/255 → black
        void_mask = (gt == 255)
        gt_safe   = np.clip(gt, 0, NUM_CLASSES - 1)
        gt_rgb    = PALETTE_RGB[gt_safe.reshape(-1)].reshape(gt.shape[0], gt.shape[1], 3)
        gt_rgb[void_mask] = 0   # void pixels → black
        gt_pil = Image.fromarray(gt_rgb)
        canvas = Image.new('RGB', (W * 3 + 8, H), (40, 40, 40))
        canvas.paste(img_pil, (0, 0))
        canvas.paste(gt_pil,  (W + 4, 0))
        canvas.paste(seg_pil, (W * 2 + 8, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4),         'Original',    fill=(255, 255, 255))
        draw.text((W + 8, 4),     'Ground Truth', fill=(255, 255, 255))
        draw.text((W * 2 + 12, 4), model_label,  fill=(255, 255, 255))
    else:
        canvas = Image.new('RGB', (W * 2 + 4, H), (40, 40, 40))
        canvas.paste(img_pil, (0, 0))
        canvas.paste(seg_pil, (W + 4, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4),     'Original',   fill=(255, 255, 255))
        draw.text((W + 8, 4), model_label,  fill=(255, 255, 255))

    canvas.save(os.path.join(out_dir, f'{name}_result.png'))


def avmi_rgb_to_index(rgb_np):
    flat = rgb_np.reshape(-1, 3).astype(np.int32)
    diff = flat[:, None, :] - _AVMI_PALETTE[None, :, :]
    return np.argmin((diff ** 2).sum(2), axis=1).reshape(rgb_np.shape[:2]).astype(np.uint8)


# ── Test on RUGD ───────────────────────────────────────────────────────────────
def test_rugd(model, processor, device, out_dir):
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
        gt  = _RUGD_MAP[ann]

        pred = infer(model, processor, img, device)
        all_preds.append(pred)
        all_gts.append(gt)

        if i < NUM_SAMPLES:
            save_result(out_dir, name, img, pred, gt, 'RUGD Fine-tune')

        if (i + 1) % 100 == 0:
            print(f'    {i+1}/{len(ids)}')

    ious, miou = compute_miou(all_preds, all_gts)
    print(f'  mIoU on RUGD test: {miou:.4f}')
    for cls, iou in zip(CLASSES, ious):
        print(f'    {cls:10s}: {iou:.4f}')
    return miou


# ── Test on RELLIS ─────────────────────────────────────────────────────────────
def test_rellis(model, processor, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_root = os.path.join(RELLIS_ROOT, 'image')
    ann_root = os.path.join(RELLIS_ROOT, 'annotation')

    with open(os.path.join(RELLIS_ROOT, 'test.txt')) as f:
        ids = [l.strip() for l in f if l.strip()]

    # Group ids by sequence so we sample evenly across all folders
    from collections import defaultdict
    seq_ids = defaultdict(list)
    for rid in ids:
        seq_ids[rid.split('/')[0]].append(rid)
    samples_per_seq = max(1, NUM_SAMPLES // len(seq_ids))
    save_set = set()
    for seq_list in seq_ids.values():
        step = max(1, len(seq_list) // samples_per_seq)
        for rid in seq_list[::step][:samples_per_seq]:
            save_set.add(rid)

    print(f'  RELLIS test: {len(ids)} images across {len(seq_ids)} sequences')
    print(f'  Saving {len(save_set)} sample images ({samples_per_seq} per sequence)')
    all_preds, all_gts = [], []

    for i, rid in enumerate(ids):
        seq, name = rid.split('/')
        img_path = os.path.join(img_root, seq, name + '.jpg')
        ann_path = os.path.join(ann_root, seq, name + '.png')
        if not os.path.exists(img_path):
            img_path = img_path.replace('.jpg', '.png')

        img = Image.open(img_path).convert('RGB')
        ann = np.array(Image.open(ann_path))
        gt  = _RELLIS_MAP[ann]

        pred = infer(model, processor, img, device)
        all_preds.append(pred)
        all_gts.append(gt)

        if rid in save_set:
            save_result(out_dir, f'seq{seq}_{name}', img, pred, gt, 'RELLIS Fine-tune')

        if (i + 1) % 200 == 0:
            print(f'    {i+1}/{len(ids)}')

    ious, miou = compute_miou(all_preds, all_gts)
    print(f'  mIoU on RELLIS test: {miou:.4f}')
    for cls, iou in zip(CLASSES, ious):
        print(f'    {cls:10s}: {iou:.4f}')
    return miou


# ── Test on AVMI UGV ───────────────────────────────────────────────────────────
def test_avmi(model, processor, device, out_dir, model_label):
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

        pred = infer(model, processor, img, device)
        all_preds.append(pred)
        all_gts.append(gt)

        if i < NUM_SAMPLES:
            save_result(out_dir, name, img, pred, gt, model_label)

        if (i + 1) % 100 == 0:
            print(f'    {i+1}/{len(files)}')

    ious, miou = compute_miou(all_preds, all_gts)
    print(f'  mIoU on AVMI UGV: {miou:.4f}')
    for cls, iou in zip(CLASSES, ious):
        print(f'    {cls:10s}: {iou:.4f}')
    return miou


# ── Test on GOD ────────────────────────────────────────────────────────────────
def test_god(model, processor, device, out_dir, model_label):
    os.makedirs(out_dir, exist_ok=True)
    img_root = os.path.join(GOD_ROOT, 'pylon_camera_node')
    id_root  = os.path.join(GOD_ROOT, 'pylon_camera_node_label_id')

    # Collect all sequences
    seqs = sorted(Path(img_root).iterdir())
    all_ids = []
    for seq in seqs:
        for f in sorted(seq.glob('*.png')):
            all_ids.append((seq.name, f.stem))

    # Sample evenly across sequences
    from collections import defaultdict
    seq_map = defaultdict(list)
    for seq, name in all_ids:
        seq_map[seq].append(name)

    samples_per_seq = max(1, NUM_SAMPLES // len(seq_map))
    save_set = set()
    for seq, names in seq_map.items():
        step = max(1, len(names) // samples_per_seq)
        for n in names[::step][:samples_per_seq]:
            save_set.add((seq, n))

    print(f'  GOD: {len(all_ids)} images across {len(seq_map)} sequences')
    print(f'  Saving {len(save_set)} samples ({samples_per_seq} per sequence)')
    all_preds, all_gts = [], []

    for i, (seq, name) in enumerate(all_ids):
        img_path = os.path.join(img_root, seq, name + '.png')
        ann_path = os.path.join(id_root,  seq, name + '.png')
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            continue

        img = Image.open(img_path).convert('RGB')
        ann = np.array(Image.open(ann_path))
        gt  = _GOD_MAP[ann]

        pred = infer(model, processor, img, device)
        all_preds.append(pred)
        all_gts.append(gt)

        if (seq, name) in save_set:
            save_result(out_dir, f'{seq}_{name}', img, pred, gt, model_label)

        if (i + 1) % 200 == 0:
            print(f'    {i+1}/{len(all_ids)}')

    ious, miou = compute_miou(all_preds, all_gts)
    print(f'  mIoU on GOD: {miou:.4f}')
    for cls, iou in zip(CLASSES, ious):
        print(f'    {cls:10s}: {iou:.4f}')
    return miou

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['rugd', 'rellis', 'both', 'god', 'all'], default='both',
                        help='rugd / rellis / both / god / all')
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = make_processor()
    print(f'Device: {device}\n')

    results = {}

    # ── RUGD Fine-tune model ────────────────────────────────────────────────
    if args.dataset in ('rugd', 'both'):
        print('=' * 60)
        print('RUGD FINE-TUNE MODEL')
        print(f'  Checkpoint: {RUGD_CKPT}')
        model = load_model(RUGD_CKPT, device)

        rugd_out  = os.path.join(_REPO_ROOT, 'results', 'test_finetune_rugd', 'rugd')
        avmi_out1 = os.path.join(_REPO_ROOT, 'results', 'test_finetune_rugd', 'avmi_ugv')

        print('\n[RUGD fine-tune → RUGD test]')
        results['rugd_on_rugd'] = test_rugd(model, processor, device, rugd_out)

        print('\n[RUGD fine-tune → AVMI UGV test]')
        results['rugd_on_avmi'] = test_avmi(model, processor, device, avmi_out1, 'RUGD Fine-tune')

        del model
        torch.cuda.empty_cache()

    # ── RELLIS Fine-tune model ──────────────────────────────────────────────
    if args.dataset in ('rellis', 'both'):
        print('\n' + '=' * 60)
        print('RELLIS FINE-TUNE MODEL')
        print(f'  Checkpoint: {RELLIS_CKPT}')
        model = load_model(RELLIS_CKPT, device)

        rellis_out = os.path.join(_REPO_ROOT, 'results', 'test_finetune_rellis', 'rellis')
        avmi_out2  = os.path.join(_REPO_ROOT, 'results', 'test_finetune_rellis', 'avmi_ugv')

        print('\n[RELLIS fine-tune → RELLIS test]')
        results['rellis_on_rellis'] = test_rellis(model, processor, device, rellis_out)

        print('\n[RELLIS fine-tune → AVMI UGV test]')
        results['rellis_on_avmi'] = test_avmi(model, processor, device, avmi_out2, 'RELLIS Fine-tune')

        del model
        torch.cuda.empty_cache()

    # ── GOD (test both models on GOD) ──────────────────────────────────────
    if args.dataset in ('god', 'all'):
        print('\n' + '=' * 60)
        print('GOD FINE-TUNE MODEL → GOD test + AVMI UGV')
        model = load_model(GOD_CKPT, device)
        god_out  = os.path.join(_REPO_ROOT, 'results', 'test_finetune_god', 'god')
        avmi_out3 = os.path.join(_REPO_ROOT, 'results', 'test_finetune_god', 'avmi_ugv')
        results['god_on_god']  = test_god(model, processor, device, god_out, 'GOD Fine-tune')
        results['god_on_avmi'] = test_avmi(model, processor, device, avmi_out3, 'GOD Fine-tune')
        del model; torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SUMMARY')
    if 'rugd_on_rugd'      in results: print(f'  RUGD   model → RUGD  test:  {results["rugd_on_rugd"]:.4f}')
    if 'rugd_on_avmi'      in results: print(f'  RUGD   model → AVMI  test:  {results["rugd_on_avmi"]:.4f}')
    if 'rellis_on_rellis'  in results: print(f'  RELLIS model → RELLIS test: {results["rellis_on_rellis"]:.4f}')
    if 'rellis_on_avmi'    in results: print(f'  RELLIS model → AVMI  test:  {results["rellis_on_avmi"]:.4f}')
    if 'god_on_god'        in results: print(f'  GOD    model → GOD   test:  {results["god_on_god"]:.4f}')
    if 'god_on_avmi'       in results: print(f'  GOD    model → AVMI  test:  {results["god_on_avmi"]:.4f}')
    print('\nReference: AVMI B2 base model trained mIoU = 0.8051')


if __name__ == '__main__':
    main()
