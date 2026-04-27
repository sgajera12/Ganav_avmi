"""
generate_lidar_grids.py

Extracts N frames from the ROS2 bag at even intervals and saves
a 4-panel grid image for each:

  Original  | Segmented
  Raw 3D PC | Segmented 3D PC
             (camera-visible points = class colour, rest = white)

3D point clouds are rendered from an angled top-down perspective
(elevation 35°, not flat top-down).

Usage:
    /home/pinaka/miniconda3/envs/segformer/bin/python \
        tools/generate_lidar_grids.py
"""

import os, sys, struct, sqlite3
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (SegformerForSemanticSegmentation,
                          SegformerConfig, SegformerImageProcessor)


# ── Config ────────────────────────────────────────────────────────────────────
BAG_PATH   = '/home/pinaka/dataset/AVMI/dataset1/dataset1_0.db3'
CAM_TOPIC  = '/camera/right/image_raw'
LID_TOPIC  = '/lidar/points2'

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH  = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_hf', 'latest.pth')
PRETRAINED = 'nvidia/mit-b2'
OUT_DIR    = os.path.join(_REPO_ROOT, 'results', 'lidar_grids')
N_FRAMES   = None      # None = all frames
PANEL_W    = 640       # width of each panel
PANEL_H    = 480       # height of each panel
PC_RANGE   = 40.0      # max LiDAR range to show (metres)
NUM_CLASSES= 6
CLASSES    = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')

# ── Camera intrinsics / extrinsics ────────────────────────────────────────────
_K = np.array([[278.5899, 0, 323.7999],
               [0, 334.2900, 229.1999],
               [0, 0, 1]], dtype=np.float32)
_R_L2C = np.array([[ 0.0,      -1.0,      0.0     ],
                   [-0.034899,  0.0,      -0.999391],
                   [ 0.999391,  0.0,      -0.034899]], dtype=np.float32)
_t_L2C = np.array([-0.15, 0.07989, 0.86414], dtype=np.float32)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE_BGR = np.array([
    [178, 102,  24],  # sky
    [ 37, 182,  18],  # tree
    [ 15, 255, 239],  # bush
    [  6,  19,  92],  # ground
    [250,  63, 255],  # obstacle
    [  0,   0, 255],  # rock
], dtype=np.uint8)

# ── CDR parsers ───────────────────────────────────────────────────────────────
def _read_str(data, offset):
    n = struct.unpack_from('<I', data, offset)[0]; offset += 4
    s = data[offset: offset + n - 1].decode(errors='ignore'); offset += n
    if offset % 4: offset += 4 - offset % 4
    return s, offset

def parse_pointcloud2(data: bytes):
    offset = 4 + 8
    _, offset = _read_str(data, offset)
    offset += 8
    nfields = struct.unpack_from('<I', data, offset)[0]; offset += 4
    for _ in range(nfields):
        _, offset = _read_str(data, offset)
        offset += 9
        if offset % 4: offset += 4 - offset % 4
    offset += 1
    if offset % 4: offset += 4 - offset % 4
    point_step = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += 4
    data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    pts = np.frombuffer(data[offset: offset + data_len],
                        dtype=np.float32).reshape(-1, point_step // 4)
    return pts[:, :3]

def parse_image(data: bytes):
    offset = 4 + 8
    fid_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += fid_len
    if offset % 4: offset += 4 - offset % 4
    height = struct.unpack_from('<I', data, offset)[0]; offset += 4
    width  = struct.unpack_from('<I', data, offset)[0]; offset += 4
    enc_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    enc = data[offset: offset + enc_len - 1].decode(errors='ignore')
    offset += enc_len + 1
    if offset % 4: offset += 4 - offset % 4
    offset += 4
    data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    raw = np.frombuffer(data[offset: offset + data_len], dtype=np.uint8)
    if enc in ('rgb8', 'RGB8'):
        return raw.reshape(height, width, 3)[..., ::-1].copy()
    return raw.reshape(height, width, 3).copy()

# ── Point cloud projection ─────────────────────────────────────────────────────
def project_lidar(xyz, img_h, img_w):
    """Returns (u, v, indices_into_xyz) for points that land inside the image."""
    Pc   = (_R_L2C @ xyz.T + _t_L2C.reshape(3, 1)).T
    Z    = Pc[:, 2]
    fwd  = Z > 0.1
    idx_fwd = np.where(fwd)[0]
    Pc, Z = Pc[fwd], Z[fwd]
    uv = (_K @ (Pc.T / Z)).T
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return (u[inside].astype(np.int32),
            v[inside].astype(np.int32),
            idx_fwd[inside])

# ── 3D point cloud renderer (manual perspective projection) ───────────────────
def render_3d(xyz, point_colors_bgr=None):
    """
    Virtual camera floats above and slightly behind the LiDAR,
    looking forward — same direction as the camera image.

    xyz:              Nx3 LiDAR points (x=forward, y=left, z=up)
    point_colors_bgr: Nx3 uint8 BGR, or None → all white
    Returns:          (PANEL_H, PANEL_W, 3) BGR uint8
    """
    # ── Virtual camera pose ───────────────────────────────────────────────────
    cam_pos = np.array([-30.0,  0.0, 22.0], dtype=np.float64)  # behind+above
    look_at = np.array([20.0,   0.0,  0.0], dtype=np.float64)  # look forward

    fwd = look_at - cam_pos;  fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 0, 1.0])); right /= np.linalg.norm(right)
    up_v  = np.cross(right, fwd)

    # ── Filter points ─────────────────────────────────────────────────────────
    r    = np.linalg.norm(xyz[:, :2], axis=1)
    mask = (r < PC_RANGE) & np.all(np.isfinite(xyz), axis=1)
    pts  = xyz[mask]
    cols = point_colors_bgr[mask] if point_colors_bgr is not None else None

    # ── Transform to camera space ─────────────────────────────────────────────
    d     = pts - cam_pos                   # (N,3)
    xc    =  d @ right                      # right
    yc    =  d @ up_v                       # up
    zc    =  d @ fwd                        # depth

    valid = zc > 0.1
    xc, yc, zc = xc[valid], yc[valid], zc[valid]
    if cols is not None:
        cols = cols[valid]

    # ── Perspective divide ────────────────────────────────────────────────────
    fov_rad = np.radians(70)
    f       = 1.0 / np.tan(fov_rad / 2)
    px      = f * xc / zc
    py      = f * yc / zc

    # ── Map to pixel coords ───────────────────────────────────────────────────
    W, H = PANEL_W, PANEL_H
    u = ((px + 1.0) * W / 2).astype(np.int32)
    v = ((1.0 - py) * H / 2).astype(np.int32)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v   = u[inside], v[inside]

    # ── Draw onto canvas (depth-sorted so near points on top) ────────────────
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    depth  = zc[inside]
    order  = np.argsort(-depth)             # far → near
    u, v   = u[order], v[order]

    if cols is not None:
        c = cols[inside][order].astype(np.uint8)
    else:
        c = np.full((len(u), 3), 210, dtype=np.uint8)

    # Vectorised pixel write (2×2 dot per point)
    for dy in range(2):
        for dx in range(2):
            uu = np.clip(u + dx, 0, W - 1)
            vv = np.clip(v + dy, 0, H - 1)
            canvas[vv, uu] = c

    return canvas

# ── Legend bar ────────────────────────────────────────────────────────────────
def draw_legend(img):
    H, W = img.shape[:2]
    cell_h = 22
    legend_h = NUM_CLASSES * cell_h + 6
    overlay  = img.copy()
    cv2.rectangle(overlay, (0, H - legend_h), (160, H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, name in enumerate(CLASSES):
        y   = H - legend_h + 4 + i * cell_h
        col = tuple(int(c) for c in PALETTE_BGR[i])
        cv2.rectangle(img, (4, y), (18, y + cell_h - 4), col, -1)
        cv2.putText(img, name, (22, y + cell_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return img

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    print('Loading SegFormer-B2...')
    cfg = SegformerConfig.from_pretrained(PRETRAINED)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, config=cfg, ignore_mismatched_sizes=True)
    ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    print(f'  iter={ckpt["iter"]}  mIoU={ckpt["miou"]:.4f}')

    processor = SegformerImageProcessor(
        do_resize=True, size={'height': 480, 'width': 640},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225])

    # Open bag
    conn    = sqlite3.connect(BAG_PATH)
    tmap    = {n: tid for tid, n in conn.execute("SELECT id, name FROM topics").fetchall()}
    frames  = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tmap[CAM_TOPIC],)).fetchall()
    lid_rows = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tmap[LID_TOPIC],)).fetchall()
    lid_ts   = np.array([r[0] for r in lid_rows], dtype=np.int64)
    lid_data = [bytes(r[1]) for r in lid_rows]
    print(f'Camera frames: {len(frames)} | LiDAR frames: {len(lid_rows)}')

    # Pick N evenly spaced frames (or all frames if N_FRAMES is None)
    if N_FRAMES is None:
        indices = np.arange(len(frames))
    else:
        indices = np.linspace(0, len(frames) - 1, N_FRAMES, dtype=int)
    n_out = len(indices)
    print(f'Saving {n_out} grid images to {OUT_DIR}\n')

    GAP  = 6
    BG   = np.full((GAP, PANEL_W * 2 + GAP, 3), 30, dtype=np.uint8)
    VDIV = np.full((PANEL_H, GAP, 3), 30, dtype=np.uint8)

    for out_i, fi in enumerate(indices):
        ts, raw = frames[fi]
        print(f'  [{out_i+1}/{n_out}] frame {fi} ...')

        img_bgr = parse_image(bytes(raw))
        oh, ow  = img_bgr.shape[:2]

        # Segmentation
        pil = Image.fromarray(img_bgr[:, :, ::-1])
        pv  = processor(images=pil, return_tensors='pt')['pixel_values'].to(device)
        with torch.no_grad():
            logits = F.interpolate(
                model(pixel_values=pv).logits,
                size=(oh, ow), mode='bilinear', align_corners=False)
        seg_map   = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        colour_mask = PALETTE_BGR[seg_map.reshape(-1)].reshape(oh, ow, 3)
        seg_bgr   = draw_legend(colour_mask.copy())

        # LiDAR
        li  = int(np.argmin(np.abs(lid_ts - ts)))
        xyz = parse_pointcloud2(lid_data[li])
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]

        # Project to camera → get class per visible point
        u, v, vis_idx = project_lidar(xyz, oh, ow)

        # ── Panel 3: Raw 3D point cloud (all white) ──────────────────────────
        pc3d_raw = render_3d(xyz, point_colors_bgr=None)

        # ── Panel 4: Segmented 3D — visible=class colour, rest=dim gray ──────
        seg_colors = np.full((len(xyz), 3), 200, dtype=np.uint8)  # rest = dim gray
        if len(vis_idx) > 0:
            cls_ids = np.clip(seg_map[v, u], 0, NUM_CLASSES - 1)
            seg_colors[vis_idx] = PALETTE_BGR[cls_ids]
        pc3d_seg = render_3d(xyz, point_colors_bgr=seg_colors)

        # ── Resize panels ─────────────────────────────────────────────────────
        orig_r = cv2.resize(img_bgr,  (PANEL_W, PANEL_H))
        seg_r  = cv2.resize(seg_bgr,  (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)

        # Labels
        for panel, lbl in [(orig_r,'Original'), (seg_r,'Segmented'),
                           (pc3d_raw,'Raw 3D Point Cloud'),
                           (pc3d_seg,'Segmented 3D Point Cloud')]:
            cv2.putText(panel, lbl, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # ── Assemble 2×2 grid ─────────────────────────────────────────────────
        top = np.hstack([orig_r,  VDIV, seg_r])
        bot = np.hstack([pc3d_raw, VDIV, pc3d_seg])
        grid = np.vstack([top, BG, bot])

        out_path = os.path.join(OUT_DIR, f'grid_{out_i+1:02d}_frame{fi:05d}.png')
        cv2.imwrite(out_path, grid)
        print(f'     Saved: {out_path}')

    conn.close()
    print('\nAll done!')

if __name__ == '__main__':
    main()
