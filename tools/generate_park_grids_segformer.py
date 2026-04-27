"""
generate_park_grids_segformer.py
Same grid as generate_park_grids.py but using SegFormer-B2 instead of GANav.
Layout: 4 columns (orig | seg | orig | seg) x 4 rows = 8 pairs per grid

Usage:
    /home/pinaka/miniconda3/envs/segformer/bin/python tools/generate_park_grids_segformer.py

Output: results/ugv_comparison_grids/
    rugd_park2_grid_segformer.png
    rugd_park8_grid_segformer.png
    rugd_park_mixed_grid_segformer.png
"""
import os, numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

# ── Settings ──────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH  = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_hf', 'latest.pth')
PRETRAINED = 'nvidia/mit-b2'
OUT_DIR    = os.path.join(_REPO_ROOT, 'results', 'ugv_comparison_grids')

PARK2_RANGE = (81,  351)
PARK8_RANGE = (391, 801)

GAP = 3
W, H = 320, 240

PALETTE_BGR = np.array([
    [178, 102,  24],  # sky
    [ 37, 182,  18],  # tree
    [ 15, 255, 239],  # bush
    [  6,  19,  92],  # ground
    [250,  63, 255],  # obstacle
    [  0,   0, 255],  # rock
], dtype=np.uint8)

BG = np.array([25, 25, 25], dtype=np.uint8)
CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')

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
    img_rgb = img_bgr[:, :, ::-1]
    pil_img = Image.fromarray(img_rgb)
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

def load_sequence(base_dir, frame_range):
    lo, hi = frame_range
    files  = sorted([f for f in os.listdir(base_dir)
                     if f.endswith('.png') and
                     lo <= int(f.split('_')[1].split('.')[0]) <= hi])
    step = max(1, len(files) // 8)
    return [os.path.join(base_dir, f) for f in files[::step][:8]]

# ── Main ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

park2_dir = os.path.join(_REPO_ROOT, 'data/rugd/RUGD_frames-with-annotations/park-2')
park8_dir = os.path.join(_REPO_ROOT, 'data/rugd/RUGD_frames-with-annotations/park-8')

sel_p2 = load_sequence(park2_dir, PARK2_RANGE)
sel_p8 = load_sequence(park8_dir, PARK8_RANGE)

for label, sel, fname in [
    ('park-2',      sel_p2,               'rugd_park2_grid_segformer.png'),
    ('park-8',      sel_p8,               'rugd_park8_grid_segformer.png'),
    ('mixed (4+4)', sel_p2[:4]+sel_p8[:4],'rugd_park_mixed_grid_segformer.png'),
]:
    print(f'=== {label} ===')
    pairs = []
    for j, p in enumerate(sel):
        img = cv2.imread(p)
        pairs.append((img, segment(img)))
        print(f'  {j+1}/{len(sel)}: {os.path.basename(p)}')
    make_grid(pairs, os.path.join(OUT_DIR, fname))

print('\nAll done!')
