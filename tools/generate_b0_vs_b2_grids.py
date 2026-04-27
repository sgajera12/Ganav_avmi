"""
generate_b0_vs_b2_grids.py
Side-by-side comparison: Original | GANav B0 (ONNX) | SegFormer B2 (HF)
on the same RUGD and RELLIS images used in the fixed grids (grid1 seed).

Output:
    results/ugv_comparison_grids/rugd_b0_vs_b2.png
    results/ugv_comparison_grids/rellis_b0_vs_b2.png

Usage:
    /home/pinaka/miniconda3/envs/segformer/bin/python \
        tools/generate_b0_vs_b2_grids.py
"""
import os, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor
import onnxruntime as ort

_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH   = os.path.join(_REPO_ROOT, 'work_dirs', 'ganav_avmi_scratch', 'model.onnx')
CKPT_PATH   = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_hf', 'latest.pth')
PRETRAINED  = 'nvidia/mit-b2'
OUT_DIR     = os.path.join(_REPO_ROOT, 'results', 'ugv_comparison_grids')
CLASSES     = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')

# GANav ONNX input size
GANAV_H, GANAV_W = 300, 375
MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
STD  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)

# Grid layout
N_IMAGES = 4   # images per dataset
CELL_W, CELL_H = 320, 240
GAP = 6
LABEL_H = 32   # height for column labels at top
BG = np.array([25, 25, 25], dtype=np.uint8)

PALETTE_BGR = np.array([
    [178, 102,  24],  # sky
    [ 37, 182,  18],  # tree
    [ 15, 255, 239],  # bush
    [  6,  19,  92],  # ground
    [250,  63, 255],  # obstacle
    [  0,   0, 255],  # rock
], dtype=np.uint8)

# ── Load GANav B0 (ONNX) ──────────────────────────────────────────────────────
print('Loading GANav B0 (ONNX)...')
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess = ort.InferenceSession(ONNX_PATH, providers=providers)
print(f'  Provider: {sess.get_providers()[0]}')

def segment_b0(img_bgr):
    oh, ow = img_bgr.shape[:2]
    inp = cv2.resize(img_bgr, (GANAV_W, GANAV_H))
    inp = inp[:, :, ::-1].astype(np.float32)
    inp = (inp - MEAN) / STD
    inp = inp.transpose(2, 0, 1)[np.newaxis]
    logits = sess.run(None, {'image': np.ascontiguousarray(inp)})[0]  # (1,6,H,W)
    seg = np.argmax(logits[0], axis=0).astype(np.uint8)
    seg = np.clip(seg, 0, 5)
    colour = PALETTE_BGR[seg.reshape(-1)].reshape(GANAV_H, GANAV_W, 3)
    return cv2.resize(colour, (ow, oh), interpolation=cv2.INTER_NEAREST)

# ── Load SegFormer B2 ─────────────────────────────────────────────────────────
print('Loading SegFormer B2...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = SegformerConfig.from_pretrained(PRETRAINED)
cfg.num_labels = 6
cfg.id2label = {i: c for i, c in enumerate(CLASSES)}
cfg.label2id = {c: i for i, c in enumerate(CLASSES)}
model = SegformerForSemanticSegmentation.from_pretrained(
    PRETRAINED, config=cfg, ignore_mismatched_sizes=True)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt['model'])
model = model.to(device).eval()
print(f'  Loaded iter={ckpt["iter"]}  val_mIoU={ckpt["miou"]:.4f}')

processor = SegformerImageProcessor(
    do_resize=True, size={'height': 480, 'width': 640},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)

def segment_b2(img_bgr):
    oh, ow = img_bgr.shape[:2]
    pil = Image.fromarray(img_bgr[:, :, ::-1])
    pixels = processor(images=pil, return_tensors='pt')['pixel_values'].to(device)
    with torch.no_grad():
        logits = F.interpolate(model(pixel_values=pixels).logits,
                               size=(oh, ow), mode='bilinear', align_corners=False)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return PALETTE_BGR[pred.reshape(-1)].reshape(oh, ow, 3)

# ── Grid builder ──────────────────────────────────────────────────────────────
COLS = ['Original', 'GANav B0', 'SegFormer B2']

def make_comparison_grid(image_paths, out_path):
    n = len(image_paths)
    total_w = len(COLS) * CELL_W + (len(COLS) + 1) * GAP
    total_h = LABEL_H + n * CELL_H + (n + 1) * GAP

    canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)

    # Column labels
    for ci, label in enumerate(COLS):
        x = GAP + ci * (CELL_W + GAP) + CELL_W // 2
        cv2.putText(canvas, label, (x - len(label) * 5, LABEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    for ri, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f'  SKIP (not found): {path}')
            continue
        b0  = segment_b0(img)
        b2  = segment_b2(img)
        print(f'  [{ri+1}/{n}] {os.path.basename(path)}')

        y = LABEL_H + GAP + ri * (CELL_H + GAP)
        for ci, panel in enumerate([img, b0, b2]):
            x = GAP + ci * (CELL_W + GAP)
            canvas[y:y+CELL_H, x:x+CELL_W] = cv2.resize(panel, (CELL_W, CELL_H),
                interpolation=cv2.INTER_NEAREST if ci > 0 else cv2.INTER_LINEAR)

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f'  Saved: {out_path}\n')

# ── RELLIS (grid1 seed: g=0, seed=5) ─────────────────────────────────────────
print('\n=== RELLIS ===')
with open(os.path.join(_REPO_ROOT, 'data/rellis/val.txt')) as f:
    rellis_ids = [l.strip() for l in f if l.strip()]
rellis_all = [os.path.join(_REPO_ROOT, 'data/rellis/image', s + '.jpg')
              for s in rellis_ids]
rellis_all = [p for p in rellis_all if os.path.exists(p)]

random.seed(5)   # same as rellis grid1 (g=0: seed = 0*13+5 = 5)
pool = rellis_all.copy(); random.shuffle(pool)
rellis_sel = pool[:8][:N_IMAGES]

make_comparison_grid(rellis_sel,
    os.path.join(OUT_DIR, 'rellis_b0_vs_b2.png'))

# ── RUGD trail (grid1 seed: g=0, seed=3) ─────────────────────────────────────
print('=== RUGD ===')

def is_natural(path):
    img = cv2.imread(path)
    if img is None: return False
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 40, 40),  (85, 255, 255))
    sky   = cv2.inRange(hsv, (90, 30, 100), (130, 255, 255))
    total = img.shape[0] * img.shape[1]
    return (green.sum() // 255 + sky.sum() // 255) / total > 0.15

with open(os.path.join(_REPO_ROOT, 'data/rugd/val_ours.txt')) as f:
    rugd_ids = [l.strip() for l in f if l.strip()]

GOOD_SEQS = ['trail-4','trail-5','trail-6','trail-7','trail-9',
             'trail-10','trail-11','trail-12','trail-14','trail-15','trail']
good_paths = []
for rid in rugd_ids:
    seq = rid.split('/')[0]
    if seq in GOOD_SEQS:
        p = os.path.join(_REPO_ROOT, 'data/rugd/RUGD_frames-with-annotations', rid + '.png')
        if os.path.exists(p):
            good_paths.append(p)

natural = [p for p in good_paths if is_natural(p)]
print(f'Natural trail images: {len(natural)}')

random.seed(3)   # same as rugd grid1 (g=0: seed = 0*19+3 = 3)
pool = natural.copy(); random.shuffle(pool)
step = max(1, len(pool) // 8)
rugd_sel = pool[::step][:N_IMAGES]

make_comparison_grid(rugd_sel,
    os.path.join(OUT_DIR, 'rugd_b0_vs_b2.png'))

print('All done!')
