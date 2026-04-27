"""
generate_rugd_rellis_grids_segformer.py
Same images as generate_rugd_rellis_grids_fixed.py but using SegFormer-B2.
Produces rellis_fixed_grid1-3 and rugd_fixed_grid1-3 with segformer suffix.

Usage:
    /home/pinaka/miniconda3/envs/segformer/bin/python \
        tools/generate_rugd_rellis_grids_segformer.py
"""
import os, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH  = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_hf', 'latest.pth')
PRETRAINED = 'nvidia/mit-b2'
OUT_DIR    = os.path.join(_REPO_ROOT, 'results', 'ugv_comparison_grids')
CLASSES    = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')

GAP = 8
BG  = np.array([25, 25, 25], dtype=np.uint8)
W, H = 320, 240

PALETTE_BGR = np.array([
    [178, 102,  24],
    [ 37, 182,  18],
    [ 15, 255, 239],
    [  6,  19,  92],
    [250,  63, 255],
    [  0,   0, 255],
], dtype=np.uint8)

# ── Load model ────────────────────────────────────────────────────────────────
print('Loading SegFormer-B2...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = SegformerConfig.from_pretrained(PRETRAINED)
cfg.num_labels = 6
cfg.id2label = {i: c for i, c in enumerate(CLASSES)}
cfg.label2id = {c: i for i, c in enumerate(CLASSES)}
model = SegformerForSemanticSegmentation.from_pretrained(
    PRETRAINED, config=cfg, ignore_mismatched_sizes=True)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt['model'])
model = model.to(device).eval()
print(f'Loaded iter={ckpt["iter"]}  val_mIoU={ckpt["miou"]:.4f}\n')

processor = SegformerImageProcessor(
    do_resize=True, size={'height': 480, 'width': 640},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def segment(img_bgr):
    pil_img = Image.fromarray(img_bgr[:, :, ::-1])
    pixels  = processor(images=pil_img, return_tensors='pt')['pixel_values'].to(device)
    oh, ow  = img_bgr.shape[:2]
    with torch.no_grad():
        logits = F.interpolate(model(pixel_values=pixels).logits,
                               size=(oh, ow), mode='bilinear', align_corners=False)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return PALETTE_BGR[pred.reshape(-1)].reshape(oh, ow, 3)

def make_grid(pairs, out_path):
    cw = W * 4 + GAP * 5
    ch = H * 4 + GAP * 5
    canvas = np.full((ch, cw, 3), BG, dtype=np.uint8)
    for i, (orig, seg) in enumerate(pairs):
        row  = i // 2
        pair = i %  2
        yo = GAP + row  * (H + GAP)
        xo = GAP + (pair * 2)     * (W + GAP)
        xs = GAP + (pair * 2 + 1) * (W + GAP)
        canvas[yo:yo+H, xo:xo+W] = cv2.resize(orig, (W, H))
        canvas[yo:yo+H, xs:xs+W] = cv2.resize(seg,  (W, H))
    cv2.imwrite(out_path, canvas)
    print(f'  Saved: {out_path}')

def is_natural(path):
    img = cv2.imread(path)
    if img is None: return False
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 40, 40),  (85, 255, 255))
    sky   = cv2.inRange(hsv, (90, 30, 100), (130, 255, 255))
    total = img.shape[0] * img.shape[1]
    return (green.sum() // 255 + sky.sum() // 255) / total > 0.15

os.makedirs(OUT_DIR, exist_ok=True)

# ── RELLIS ────────────────────────────────────────────────────────────────────
print('=== RELLIS ===')
with open(os.path.join(_REPO_ROOT, 'data/rellis/val.txt')) as f:
    rellis_ids = [l.strip() for l in f if l.strip()]
rellis_all = [os.path.join(_REPO_ROOT, 'data/rellis/image', s + '.jpg')
              for s in rellis_ids]
rellis_all = [p for p in rellis_all if os.path.exists(p)]

for g in range(3):
    random.seed(g * 13 + 5)
    pool = rellis_all.copy(); random.shuffle(pool)
    sel  = pool[g * 8:(g + 1) * 8]
    if len(sel) < 8:
        sel += pool[:8 - len(sel)]
    pairs = []
    for j, p in enumerate(sel[:8]):
        img = cv2.imread(p)
        pairs.append((img, segment(img)))
        print(f'  grid{g+1} {j+1}/8: {os.path.basename(p)}')
    make_grid(pairs, os.path.join(OUT_DIR, f'rellis_fixed_grid{g+1}_segformer.png'))

# ── RUGD ──────────────────────────────────────────────────────────────────────
print('\n=== RUGD (trail only) ===')
with open(os.path.join(_REPO_ROOT, 'data/rugd/val_ours.txt')) as f:
    rugd_ids = [l.strip() for l in f if l.strip()]

GOOD_SEQS = ['trail-4','trail-5','trail-6','trail-7','trail-9',
             'trail-10','trail-11','trail-12','trail-14','trail-15','trail']
good_paths = []
for rid in rugd_ids:
    seq = rid.split('/')[0]
    if seq in GOOD_SEQS:
        p = os.path.join(_REPO_ROOT,
                         'data/rugd/RUGD_frames-with-annotations', rid + '.png')
        if os.path.exists(p):
            good_paths.append(p)

natural = [p for p in good_paths if is_natural(p)]
print(f'Natural trail images: {len(natural)}')

for g in range(3):
    random.seed(g * 19 + 3)
    pool = natural.copy(); random.shuffle(pool)
    step = max(1, len(pool) // 8)
    sel  = pool[::step][:8]
    pairs = []
    for j, p in enumerate(sel):
        img = cv2.imread(p)
        pairs.append((img, segment(img)))
        print(f'  grid{g+1} {j+1}/8: {os.path.basename(p)}')
    make_grid(pairs, os.path.join(OUT_DIR, f'rugd_fixed_grid{g+1}_segformer.png'))

print('\nAll done!')
