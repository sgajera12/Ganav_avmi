"""
Custom selection grid using SegFormer-B2.
Row 1: park2 pairs 1, 3, 4
Row 2: park2 pair 8, park8 pairs 1, 3

Usage:
    /home/pinaka/miniconda3/envs/segformer/bin/python tools/generate_custom_grid_segformer.py
"""
import os, numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH  = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_hf', 'latest.pth')
PRETRAINED = 'nvidia/mit-b2'
OUT_DIR    = os.path.join(_REPO_ROOT, 'results', 'ugv_comparison_grids')
CLASSES    = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')

PARK2_RANGE = (81,  351)
PARK8_RANGE = (391, 801)

GAP = 3
W, H = 320, 240   # cell size per image

PALETTE_BGR = np.array([
    [178, 102,  24],
    [ 37, 182,  18],
    [ 15, 255, 239],
    [  6,  19,  92],
    [250,  63, 255],
    [  0,   0, 255],
], dtype=np.uint8)
BG = np.array([25, 25, 25], dtype=np.uint8)

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

def load_sequence(base_dir, frame_range):
    lo, hi = frame_range
    files  = sorted([f for f in os.listdir(base_dir)
                     if f.endswith('.png') and
                     lo <= int(f.split('_')[1].split('.')[0]) <= hi])
    step = max(1, len(files) // 8)
    return [os.path.join(base_dir, f) for f in files[::step][:8]]

def get_pair(path):
    img = cv2.imread(path)
    seg = segment(img)
    name = os.path.basename(path)
    print(f'  {name}')
    return img, seg

# ── Load sequences ────────────────────────────────────────────────────────────
park2_dir = os.path.join(_REPO_ROOT, 'data/rugd/RUGD_frames-with-annotations/park-2')
park8_dir = os.path.join(_REPO_ROOT, 'data/rugd/RUGD_frames-with-annotations/park-8')

p2 = load_sequence(park2_dir, PARK2_RANGE)  # 8 files
p8 = load_sequence(park8_dir, PARK8_RANGE)  # 8 files

# ── Select pairs (1-indexed → 0-indexed) ─────────────────────────────────────
# Row 1: park2[0], park2[2], park2[3]
# Row 2: park2[7], park8[0], park8[2]

print('=== Row 1 ===')
r1 = [get_pair(p2[0]), get_pair(p2[2]), get_pair(p2[3])]

print('=== Row 2 ===')
r2 = [get_pair(p2[7]), get_pair(p8[0]), get_pair(p8[2])]

# ── Build grid: 3 pairs wide x 2 rows ─────────────────────────────────────────
# Each pair = orig + seg side by side (W+GAP+W wide)
# 3 pairs per row with GAP between pairs
pair_w = W * 2 + GAP          # width of one orig|seg pair
total_w = pair_w * 3 + GAP * 4
total_h = H * 2 + GAP * 3

canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)

for row_idx, row in enumerate([r1, r2]):
    y = GAP + row_idx * (H + GAP)
    for col_idx, (orig, seg) in enumerate(row):
        x_orig = GAP + col_idx * (pair_w + GAP)
        x_seg  = x_orig + W + GAP
        canvas[y:y+H, x_orig:x_orig+W] = cv2.resize(orig, (W, H))
        canvas[y:y+H, x_seg:x_seg+W]   = cv2.resize(seg,  (W, H))

out_path = os.path.join(OUT_DIR, 'rugd_custom_grid_segformer.png')
os.makedirs(OUT_DIR, exist_ok=True)
cv2.imwrite(out_path, canvas)
print(f'\nSaved: {out_path}')
