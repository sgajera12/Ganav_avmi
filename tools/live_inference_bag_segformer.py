"""
Live inference from ROS2 bag using SegFormer-B2 (HuggingFace, GPU).
Uses the AVMI-trained SegFormer-B2 model instead of GANav ONNX.

Run in segformer env:
    cd ~ && /home/pinaka/miniconda3/envs/segformer/bin/python \
        /home/pinaka/GANav-offroad/tools/live_inference_bag_segformer.py

Controls:
  SPACE  — pause / resume
  D / →  — step one frame forward
  A / ←  — step one frame back
  + / =  — speed up (2x, 4x, …)
  -      — slow down (0.5x, 0.25x, …)
  1      — reset to real-time (1x)
  Q/ESC  — quit
"""

import os, sys, struct, sqlite3, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

# ── Config ───
BAG_PATH  = '/home/pinaka/dataset/AVMI/dataset1/dataset1_0.db3'
CAM_TOPIC = '/camera/right/image_raw'
LID_TOPIC = '/lidar/points2'

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH  = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_hf', 'latest.pth')
PRETRAINED = 'nvidia/mit-b2'
NUM_CLASSES= 6
DISPLAY_W  = 640
SKIP       = 1
SAVE_VIDEO = True   # set False to disable recording
VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'segformer_b2_live.mp4')
VIDEO_FPS  =24

CLASSES = ('sky', 'tree', 'bush', 'ground', 'stump', 'rock')

# ── Camera intrinsics / extrinsics (same as ONNX script) ─────
_K = np.array([[278.5899, 0, 323.7999],
               [0, 334.2900, 229.1999],
               [0, 0, 1]], dtype=np.float32)
_R_L2C = np.array([[ 0.0,      -1.0,      0.0     ],
                   [-0.034899,  0.0,      -0.999391],
                   [ 0.999391,  0.0,      -0.034899]], dtype=np.float32)
_t_L2C = np.array([-0.15, 0.07989, 0.86414], dtype=np.float32)

# ── Class palette (BGR for OpenCV) 
PALETTE_BGR = np.array([
    [178, 102,  24],  # 0 sky
    [ 37, 182,  18],  # 1 tree
    [ 15, 255, 239],  # 2 bush
    [  6,  19,  92],  # 3 ground
    [250,  63, 255],  # 4 stump
    [  0,   0, 255],  # 5 rock
], dtype=np.uint8)

# ── CDR parsers (identical to ONNX script) 
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

def project_lidar(xyz, img_h, img_w):
    Pc = (_R_L2C @ xyz.T + _t_L2C.reshape(3, 1)).T
    Z  = Pc[:, 2]
    mask = Z > 0.1
    Pc, Z = Pc[mask], Z[mask]
    uv = (_K @ (Pc.T / Z)).T
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u[inside].astype(np.int32), v[inside].astype(np.int32)

def draw_dots(canvas, u, v, colors):
    for x, y, c in zip(u, v, colors):
        cv2.circle(canvas, (int(x), int(y)), 2,
                   (int(c[0]), int(c[1]), int(c[2])), -1)

def draw_legend(img):
    H, W = img.shape[:2]
    cell_h = 22
    legend_h = len(CLASSES) * cell_h + 6
    overlay = img.copy()
    cv2.rectangle(overlay, (0, H - legend_h), (160, H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, name in enumerate(CLASSES):
        y = H - legend_h + 4 + i * cell_h
        col = tuple(int(c) for c in PALETTE_BGR[i])
        cv2.rectangle(img, (4, y), (18, y + cell_h - 4), col, -1)
        cv2.putText(img, name, (22, y + cell_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return img

# ── Main ─────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load SegFormer-B2
    if not os.path.exists(CKPT_PATH):
        print(f'ERROR: checkpoint not found at {CKPT_PATH}')
        sys.exit(1)

    print('Loading SegFormer-B2...')
    cfg = SegformerConfig.from_pretrained(PRETRAINED)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, config=cfg, ignore_mismatched_sizes=True)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    print(f'Loaded iter={ckpt["iter"]}  val_mIoU={ckpt["miou"]:.4f}')

    processor = SegformerImageProcessor(
        do_resize=True,
        size={'height': 480, 'width': 640},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    # Open bag
    conn = sqlite3.connect(BAG_PATH)
    topic_map = {name: tid for tid, name in conn.execute("SELECT id, name FROM topics").fetchall()}

    if CAM_TOPIC not in topic_map:
        print(f'ERROR: {CAM_TOPIC} not in bag. Available: {list(topic_map.keys())}')
        sys.exit(1)

    frames = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (topic_map[CAM_TOPIC],)
    ).fetchall()[::SKIP]
    total = len(frames)

    has_lidar = LID_TOPIC in topic_map
    if has_lidar:
        lid_rows = conn.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (topic_map[LID_TOPIC],)
        ).fetchall()
        lid_ts   = np.array([r[0] for r in lid_rows], dtype=np.int64)
        lid_data = [bytes(r[1]) for r in lid_rows]
        print(f'LiDAR frames: {len(lid_rows)}')
    else:
        print(f'WARNING: {LID_TOPIC} not in bag — point-cloud panels will be blank')

    print(f'{total} camera frames. Starting...\n')

    win = 'SegFormer-B2 GPU | SPACE=pause  D=fwd  A=back  +/-=speed  1=realtime  Q=quit'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Video writer — initialised on first frame when we know the display size
    writer = None

    paused = False
    speed  = 1.0
    idx    = 0

    while 0 <= idx < total:
        t_start = time.monotonic()
        ts, raw = frames[idx]

        try:
            img_bgr = parse_image(bytes(raw))
        except Exception as e:
            print(f'Frame {idx}: parse error — {e}')
            idx += 1
            continue

        orig_h, orig_w = img_bgr.shape[:2]

        # ── Inference 
        img_rgb = img_bgr[:, :, ::-1]   # BGR → RGB for PIL
        pil_img = Image.fromarray(img_rgb)
        pixels  = processor(images=pil_img, return_tensors='pt')['pixel_values'].to(device)

        with torch.no_grad():
            with torch.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                logits = model(pixel_values=pixels).logits  # (1,6,H/4,W/4)
        logits_up = F.interpolate(logits, size=(orig_h, orig_w),
                                  mode='bilinear', align_corners=False)
        seg_map = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        colour_mask = PALETTE_BGR[seg_map.reshape(-1)].reshape(orig_h, orig_w, 3)
        seg_bgr = draw_legend(colour_mask.copy())

        # ── Point cloud panels 
        scale = DISPLAY_W / orig_w
        dH    = int(orig_h * scale)

        if has_lidar:
            li = int(np.argmin(np.abs(lid_ts - ts)))
            try:
                xyz = parse_pointcloud2(lid_data[li])
                xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
                u, v = project_lidar(xyz, orig_h, orig_w)
                pc_raw = img_bgr.copy()
                draw_dots(pc_raw, u, v, np.full((len(u), 3), 220, dtype=np.uint8))
                pc_seg = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                draw_dots(pc_seg, u, v, PALETTE_BGR[np.clip(seg_map[v, u], 0, 5)])
            except Exception as e:
                print(f'Frame {idx}: lidar error — {e}')
                pc_raw = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                pc_seg = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        else:
            pc_raw = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            pc_seg = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        # ── Build 2×2 display 
        vdiv = np.full((dH, 4, 3), 60, dtype=np.uint8)
        hdiv = np.full((4, DISPLAY_W * 2 + 4, 3), 60, dtype=np.uint8)

        tl = cv2.resize(img_bgr, (DISPLAY_W, dH))
        tr = cv2.resize(seg_bgr, (DISPLAY_W, dH), interpolation=cv2.INTER_NEAREST)
        bl = cv2.resize(pc_raw,  (DISPLAY_W, dH), interpolation=cv2.INTER_NEAREST)
        br = cv2.resize(pc_seg,  (DISPLAY_W, dH), interpolation=cv2.INTER_NEAREST)

        cv2.putText(tl, 'Original',      (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(tr, 'Segmented',  (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(bl, 'Raw Points',    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(br, 'Seg. Points',   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # info = f'Frame {idx+1}/{total}  ts={ts/1e9:.2f}s  {speed:.2g}x  {"[PAUSED]" if paused else ""}'
        # cv2.putText(tl, info, (10, dH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        frame_out = np.vstack([np.hstack([tl, vdiv, tr]),
                               hdiv,
                               np.hstack([bl, vdiv, br])])
        cv2.imshow(win, frame_out)

        # Init video writer on first frame
        if SAVE_VIDEO and writer is None:
            os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
            fh, fw = frame_out.shape[:2]
            writer = cv2.VideoWriter(VIDEO_PATH,
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     VIDEO_FPS, (fw, fh))
            print(f'Recording to: {VIDEO_PATH}')
        if SAVE_VIDEO and writer is not None and not paused:
            writer.write(frame_out)

        # Speed-controlled delay
        if not paused and idx + 1 < total:
            gap_s   = (frames[idx + 1][0] - ts) / 1e9
            wait_ms = max(1, int((gap_s / speed - (time.monotonic() - t_start)) * 1000))
        else:
            wait_ms = 0

        key = cv2.waitKey(wait_ms if not paused else 0) & 0xFF
        if key in (ord('q'), 27):     break
        elif key == ord(' '):         paused = not paused
        elif key in (83, ord('d')):   idx = min(idx + 1, total - 1); continue
        elif key in (81, ord('a')):   idx = max(0, idx - 1); paused = True; continue
        elif key in (ord('+'), ord('=')): speed = min(speed * 2, 32.0)
        elif key == ord('-'):         speed = max(speed / 2, 0.125)
        elif key == ord('1'):         speed = 1.0

        if not paused:
            idx += 1

    if writer is not None:
        writer.release()
        print(f'Video saved: {VIDEO_PATH}')
    cv2.destroyAllWindows()
    conn.close()
    print('Done.')

if __name__ == '__main__':
    main()
