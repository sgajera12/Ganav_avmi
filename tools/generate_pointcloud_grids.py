"""
generate_pointcloud_grids.py
============================
Reads camera + LiDAR frames directly from the ROS2 .db3 bag (no ROS2 required).
Produces two 3×6 grid images:
  results/pointcloud_grids/grid_raw.png       — camera + LiDAR dots colored by embedded RGB
  results/pointcloud_grids/grid_segmented.png — camera + LiDAR dots colored by GANav semantic class
"""

import os, sys, struct, sqlite3
import numpy as np
import cv2
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.backends.cudnn.enabled = False

# ── paths ─────────────────────────────────────────────────────────────────────
BAG_PATH  = Path('/home/pinaka/dataset/AVMI/dataset1/dataset1_0.db3')
MODEL_CFG  = 'work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py'
MODEL_CKPT = 'work_dirs/ganav_avmi_scratch/latest.pth'
OUT_DIR    = Path('results/pointcloud_grids')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── camera intrinsics (Unreal Engine calibration) ─────────────────────────────
FX, FY = 278.5899, 334.2900
CX, CY = 323.7999, 229.1999
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)

# ── lidar→camera extrinsics ───────────────────────────────────────────────────
def _ue_to_ros(pos_cm):
    return np.array([ pos_cm[0]/100., -pos_cm[1]/100., pos_cm[2]/100.], dtype=np.float32)

_p_C = _ue_to_ros(np.array([55.0, -15.0, 200.0]))   # camera pos (UE cm)
_p_L = _ue_to_ros(np.array([-31.0819, 0.0, 211.0])) # lidar pos  (UE cm)

_R_BASE = np.array([[0,-1,0],[0,0,-1],[1,0,0]], dtype=np.float32)
_r_ue   = R.from_euler('ZYX', [np.radians(0.), np.radians(-2.), np.radians(0.)])
R_L2C   = _R_BASE @ _r_ue.as_matrix().astype(np.float32)
t_L2C   = R_L2C @ (_p_C - _p_L)

# ── AVMI class colours (BGR for OpenCV) ───────────────────────────────────────
CLASSES = ['sky', 'tree', 'bush', 'ground', 'obstacle', 'rock']
PALETTE_BGR = np.array([
    [178, 102,  24],  # sky      blue  (RGB→BGR)
    [ 37, 182,  18],  # tree     green
    [ 15, 255, 239],  # bush     yellow
    [  6,  19,  92],  # ground   dark brown
    [250,  63, 255],  # obstacle pink
    [  0,   0, 255],  # rock     red
], dtype=np.uint8)

# ── CDR / PointCloud2 parser (pure Python, no ROS2) ───────────────────────────
def _read_string(data, offset):
    length = struct.unpack_from('<I', data, offset)[0]
    s = data[offset+4: offset+4+length-1].decode(errors='ignore')
    offset += 4 + length
    if offset % 4: offset += 4 - offset % 4
    return s, offset

def parse_pointcloud2(data: bytes):
    """Return Nx4 float32 array [x, y, z, rgb_packed_as_uint32_reinterpreted_as_float32]."""
    offset = 4  # CDR encapsulation header
    # stamp
    offset += 8
    # frame_id
    _, offset = _read_string(data, offset)
    # height, width
    height = struct.unpack_from('<I', data, offset)[0]; offset += 4
    width  = struct.unpack_from('<I', data, offset)[0]; offset += 4
    # fields
    nfields = struct.unpack_from('<I', data, offset)[0]; offset += 4
    for _ in range(nfields):
        _, offset = _read_string(data, offset)
        offset += 9   # f_offset(4) + datatype(1) + count(4)
        if offset % 4: offset += 4 - offset % 4
    # is_bigendian(1) → pad to 4
    offset += 1
    if offset % 4: offset += 4 - offset % 4
    # point_step, row_step
    point_step = struct.unpack_from('<I', data, offset)[0]; offset += 4
    _          = struct.unpack_from('<I', data, offset)[0]; offset += 4  # row_step
    # data array
    data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    raw = data[offset: offset + data_len]
    pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, point_step // 4)
    return pts  # columns: x, y, z, rgb_float

def decode_rgb(rgb_float_col: np.ndarray):
    """Packed float32 → uint8 BGR array (N, 3)."""
    rgb_int = rgb_float_col.view(np.uint32)
    r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
    g = ((rgb_int >> 8)  & 0xFF).astype(np.uint8)
    b = ( rgb_int        & 0xFF).astype(np.uint8)
    return np.stack([b, g, r], axis=1)  # BGR

def parse_image(data: bytes):
    """CDR Image → numpy BGR uint8."""
    offset = 4  # CDR header
    offset += 8  # stamp
    # frame_id string (uint32 len + bytes, then align)
    fid_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += fid_len
    if offset % 4: offset += 4 - offset % 4
    height = struct.unpack_from('<I', data, offset)[0]; offset += 4
    width  = struct.unpack_from('<I', data, offset)[0]; offset += 4
    # encoding string (uint32 len + bytes) — do NOT auto-align after, is_bigendian is next
    enc_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    enc = data[offset: offset + enc_len - 1].decode(errors='ignore')
    offset += enc_len
    # is_bigendian (uint8) — at unaligned position
    offset += 1
    # step (uint32) — align to 4 first
    if offset % 4: offset += 4 - offset % 4
    offset += 4  # step (skip)
    # data array: uint32 count + bytes
    data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    raw = np.frombuffer(data[offset: offset + data_len], dtype=np.uint8)
    if enc in ('rgb8', 'RGB8'):
        img = raw.reshape(height, width, 3)[..., ::-1].copy()  # RGB→BGR
    elif enc in ('bgr8', 'BGR8'):
        img = raw.reshape(height, width, 3).copy()
    else:
        img = raw.reshape(height, width, -1).copy()
    return img

# ── project LiDAR onto image ──────────────────────────────────────────────────
def project_lidar(pts_xyz, img_h, img_w):
    """Returns (u, v, Z, valid_mask)."""
    Pc = (R_L2C @ pts_xyz.T + t_L2C.reshape(3, 1)).T
    Z = Pc[:, 2]
    valid = Z > 0.1
    Pc, Z = Pc[valid], Z[valid]
    uv = (K @ (Pc.T / Z)).T
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u[inside].astype(np.int32), v[inside].astype(np.int32), Z[inside], valid, inside

def overlay_dots(img, u, v, colors_bgr, radius=2):
    out = img.copy()
    for x, y, c in zip(u, v, colors_bgr):
        cv2.circle(out, (int(x), int(y)), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out

# ── load model ────────────────────────────────────────────────────────────────
from mmseg.apis import init_segmentor, inference_segmentor
import mmcv

def convert_syncbn_to_bn(module):
    """Replace SyncBatchNorm with BatchNorm2d so the model runs on CPU."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.SyncBatchNorm):
            bn = torch.nn.BatchNorm2d(
                child.num_features, child.eps, child.momentum,
                child.affine, child.track_running_stats
            )
            if child.affine:
                bn.weight.data = child.weight.data.clone()
                bn.bias.data   = child.bias.data.clone()
            if child.track_running_stats:
                bn.running_mean.data = child.running_mean.data.clone()
                bn.running_var.data  = child.running_var.data.clone()
                bn.num_batches_tracked.data = child.num_batches_tracked.data.clone()
            setattr(module, name, bn)
        else:
            convert_syncbn_to_bn(child)

print('Loading GANav model …')
model = init_segmentor(
    mmcv.Config.fromfile(MODEL_CFG),
    MODEL_CKPT,
    device='cpu'
)
convert_syncbn_to_bn(model)
model.eval()
print('Model ready.')

# ── read bag and build sync pairs ─────────────────────────────────────────────
print(f'Reading {BAG_PATH} …')
conn = sqlite3.connect(str(BAG_PATH))

# get topic ids
rows = conn.execute("SELECT id, name FROM topics").fetchall()
tid = {name: tid for tid, name in rows}

CAM_TOPIC  = '/camera/right/image_raw'
LID_TOPIC  = '/lidar/points2'

# load all messages with timestamps
cam_msgs = conn.execute(
    "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
    (tid[CAM_TOPIC],)
).fetchall()

lid_msgs = conn.execute(
    "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
    (tid[LID_TOPIC],)
).fetchall()

conn.close()
print(f'Camera frames: {len(cam_msgs)},  LiDAR frames: {len(lid_msgs)}')

# build synchronized pairs: for each camera frame, find nearest LiDAR frame
lid_ts  = np.array([r[0] for r in lid_msgs], dtype=np.int64)
lid_data = [r[1] for r in lid_msgs]

pairs = []
for cam_ts, cam_data in cam_msgs:
    idx = int(np.argmin(np.abs(lid_ts - cam_ts)))
    dt_ms = abs(int(lid_ts[idx]) - int(cam_ts)) / 1e6
    if dt_ms < 200:  # within 200 ms
        pairs.append((cam_ts, bytes(cam_data), bytes(lid_data[idx])))

print(f'Synchronized pairs: {len(pairs)}')

# pick 18 evenly-spaced pairs
N = 18
step = max(1, len(pairs) // N)
selected = [pairs[i * step] for i in range(N)][:N]

# ── semantic segmentation colour overlay helper ───────────────────────────────
def seg_to_color(seg_map):
    """Convert (H,W) class index map → (H,W,3) BGR colour image."""
    out = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(PALETTE_BGR):
        out[seg_map == cls_id] = color
    return out

def camera_seg_overlay(img, seg_map, alpha=0.55):
    """Blend semantic colour mask over the raw camera image."""
    color_mask = seg_to_color(seg_map)
    return cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

# ── generate all four sets of images ─────────────────────────────────────────
cam_imgs       = []   # grid 1: raw camera
cam_seg_imgs   = []   # grid 2: segmented camera
pc_raw_imgs    = []   # grid 3: white dots on black (raw point cloud)
pc_seg_imgs    = []   # grid 4: semantic-coloured dots on black
pc_raw_bg_imgs = []   # grid 5: white dots on camera image background
pc_raw_bev_imgs= []   # grid 6: white dots on black, front-view large dots

for idx, (ts, cam_raw, lid_raw) in enumerate(selected, 1):
    print(f'  {idx}/{N} …', end=' ', flush=True)

    # ── camera image ────────────────────────────────────────────────────────
    img = parse_image(cam_raw)
    h, w = img.shape[:2]

    # ── LiDAR parse & project ───────────────────────────────────────────────
    pts = parse_pointcloud2(lid_raw)
    xyz = pts[:, :3].copy()
    valid_xyz = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[valid_xyz]

    u, v, Z, valid_front, inside = project_lidar(xyz, h, w)

    # ── GRID 1: raw camera ──────────────────────────────────────────────────
    cam_imgs.append(img.copy())

    # ── GRID 2: segmented camera (GANav mask blended over image) ───────────
    # Pre-resize to training crop size (300×375 h×w) so no padding is added
    # during inference — this prevents the ~30px upward shift caused by
    # rescaling from padded output back to original size.
    CROP_H, CROP_W = 300, 375
    img_resized = cv2.resize(img, (CROP_W, CROP_H))
    seg_small = inference_segmentor(model, img_resized)[0]  # (300, 375)
    seg_map = cv2.resize(seg_small.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST)   # back to (480, 640)
    cam_seg_imgs.append(camera_seg_overlay(img, seg_map))

    # ── GRID 3: raw point cloud — white dots on black ───────────────────────
    black = np.zeros((h, w, 3), dtype=np.uint8)
    white_colors = np.full((len(u), 3), 230, dtype=np.uint8)
    pc_raw = overlay_dots(black, u, v, white_colors, radius=2)
    pc_raw_imgs.append(pc_raw)

    # ── GRID 4: segmented point cloud — semantic colours on black ───────────
    class_at_point = seg_map[v, u]
    colors_seg = PALETTE_BGR[class_at_point]
    pc_seg = overlay_dots(black, u, v, colors_seg, radius=2)
    pc_seg_imgs.append(pc_seg)

    # ── GRID 5: white dots overlaid on raw camera image ─────────────────────
    pc_on_img = overlay_dots(img.copy(), u, v, white_colors, radius=3)
    pc_raw_bg_imgs.append(pc_on_img)

    # ── GRID 6: larger white dots on pure black — front-view clean style ────
    black2 = np.zeros((h, w, 3), dtype=np.uint8)
    pc_clean = overlay_dots(black2, u, v, white_colors, radius=3)
    pc_raw_bev_imgs.append(pc_clean)

    print(f'pts={len(xyz)} proj={len(u)}')

# ── assemble 3×6 grids ────────────────────────────────────────────────────────
GAP = 6   # px gap between images
BG  = (30, 30, 30)  # dark grey border

def make_grid(imgs, rows=3, cols=6, gap=GAP):
    h, w = imgs[0].shape[:2]
    canvas_h = rows * h + (rows + 1) * gap
    canvas_w = cols * w + (cols + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y0 = gap + r * (h + gap)
        x0 = gap + c * (w + gap)
        canvas[y0:y0+h, x0:x0+w] = im
    return canvas

grid1 = make_grid(cam_imgs)
grid2 = make_grid(cam_seg_imgs)
grid3 = make_grid(pc_raw_imgs)
grid4 = make_grid(pc_seg_imgs)
grid5 = make_grid(pc_raw_bg_imgs)
grid6 = make_grid(pc_raw_bev_imgs)

paths = [
    (OUT_DIR / 'grid1_camera_raw.png',              grid1),
    (OUT_DIR / 'grid2_camera_segmented.png',        grid2),
    (OUT_DIR / 'grid3_pointcloud_raw.png',          grid3),
    (OUT_DIR / 'grid4_pointcloud_segmented.png',    grid4),
    (OUT_DIR / 'grid5_pointcloud_on_camera.png',    grid5),
    (OUT_DIR / 'grid6_pointcloud_clean_frontview.png', grid6),
]

print(f'\nSaved:')
for path, grid in paths:
    cv2.imwrite(str(path), grid)
    print(f'  {path}  ({grid.shape[1]}×{grid.shape[0]} px)')
