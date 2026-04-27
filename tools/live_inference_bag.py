"""
Live inference from ROS2 bag file using ganav_avmi_scratch model.
Reads camera frames sequentially from the .db3 bag, runs segmentation,
and displays original | segmented side by side in a window.

Controls:
  SPACE  — pause / resume
  →      — step one frame (while paused)
  q/ESC  — quit

Run with ganav env:
  conda run -n ganav python tools/live_inference_bag.py
"""

import os, sys, struct, sqlite3
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmseg.apis import init_segmentor, inference_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm

# ── Config ────────────────────────────────────────────────────────────────────
BAG_PATH   = '/home/pinaka/dataset/AVMI/dataset1/dataset1_0.db3'
CAM_TOPIC  = '/camera/right/image_raw'
MODEL_CFG  = 'work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py'
MODEL_CKPT = 'work_dirs/ganav_avmi_scratch/latest.pth'
DEVICE     = 'cpu'        # works via PTX JIT compilation on RTX 5070
FPS        = 10              # playback speed (frames per second)
DISPLAY_W  = 640             # width of each panel in the display window
SKIP       = 1               # process every Nth frame (1=all, 2=half, 3=third...)

# ── Class palette (BGR for OpenCV) ────────────────────────────────────────────
PALETTE_BGR = np.array([
    [178, 102,  24],  # 0 sky      - blue
    [ 37, 182,  18],  # 1 tree     - green
    [ 15, 255, 239],  # 2 bush     - yellow
    [  6,  19,  92],  # 3 ground   - dark brown
    [250,  63, 255],  # 4 obstacle - pink
    [  0,   0, 255],  # 5 rock     - red
], dtype=np.uint8)

CLASS_NAMES = ['sky', 'tree', 'bush', 'ground', 'obstacle', 'rock']

# ── CDR image parser (no ROS2 required) ───────────────────────────────────────
def parse_image(data: bytes):
    """CDR serialized sensor_msgs/Image → numpy BGR uint8."""
    offset = 4   # CDR encapsulation header
    offset += 8  # stamp (sec + nanosec)
    # frame_id string
    fid_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += fid_len
    if offset % 4: offset += 4 - offset % 4
    height = struct.unpack_from('<I', data, offset)[0]; offset += 4
    width  = struct.unpack_from('<I', data, offset)[0]; offset += 4
    # encoding string (do NOT align after — is_bigendian byte is next)
    enc_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    enc = data[offset: offset + enc_len - 1].decode(errors='ignore')
    offset += enc_len
    offset += 1  # is_bigendian (uint8, unaligned)
    if offset % 4: offset += 4 - offset % 4
    offset += 4  # step
    data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    raw = np.frombuffer(data[offset: offset + data_len], dtype=np.uint8)
    if enc in ('rgb8', 'RGB8'):
        img = raw.reshape(height, width, 3)[..., ::-1].copy()
    elif enc in ('bgr8', 'BGR8'):
        img = raw.reshape(height, width, 3).copy()
    else:
        img = raw.reshape(height, width, -1).copy()
    return img

# ── Segmentation colour overlay ────────────────────────────────────────────────
def seg_to_color(seg, orig_shape):
    """Convert class index map to coloured BGR image, resized to orig_shape."""
    seg = np.clip(seg, 0, 5)
    h, w = seg.shape
    colour = PALETTE_BGR[seg.reshape(-1)].reshape(h, w, 3)
    return cv2.resize(colour, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

def draw_legend(img, cell_h=22):
    """Draw a small class legend on the bottom of the image."""
    H, W = img.shape[:2]
    n = len(CLASS_NAMES)
    legend_h = n * cell_h + 6
    overlay = img.copy()
    cv2.rectangle(overlay, (0, H - legend_h), (160, H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, name in enumerate(CLASS_NAMES):
        y = H - legend_h + 4 + i * cell_h
        col = tuple(int(c) for c in PALETTE_BGR[i])
        cv2.rectangle(img, (4, y), (18, y + cell_h - 4), col, -1)
        cv2.putText(img, name, (22, y + cell_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return img

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load model
    print('Loading model...')
    model = init_segmentor(MODEL_CFG, MODEL_CKPT, device=DEVICE)
    model = revert_sync_batchnorm(model)
    model.eval()
    print('Model ready.\n')

    # Open bag
    if not os.path.exists(BAG_PATH):
        print(f'ERROR: bag not found at {BAG_PATH}')
        sys.exit(1)

    conn = sqlite3.connect(BAG_PATH)
    rows = conn.execute("SELECT id, name FROM topics").fetchall()
    topic_map = {name: tid for tid, name in rows}

    if CAM_TOPIC not in topic_map:
        print(f'ERROR: topic "{CAM_TOPIC}" not found in bag.')
        print('Available topics:', list(topic_map.keys()))
        sys.exit(1)

    cam_id = topic_map[CAM_TOPIC]
    rows = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (cam_id,)
    ).fetchall()
    # Apply frame skip
    rows = rows[::SKIP]
    total = len(rows)
    print(f'Found {total} frames to process (skip={SKIP}).')

    # ── Live frame-by-frame inference ────────────────────────────────────────
    win = 'GANav Live | SPACE=pause  D/→=step fwd  A/←=step back  Q=quit'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    paused   = False
    idx      = 0

    while 0 <= idx < total:
        ts, raw = rows[idx]

        try:
            img_bgr = parse_image(bytes(raw))
        except Exception as e:
            print(f'Frame {idx}: parse error — {e}')
            idx += 1
            continue

        H, W = img_bgr.shape[:2]

        result = inference_segmentor(model, img_bgr)
        seg = result[0]
        if isinstance(seg, torch.Tensor):
            seg = seg.cpu().numpy()

        seg_bgr = seg_to_color(seg.astype(np.uint8), (H, W))
        seg_bgr = draw_legend(seg_bgr)

        scale = DISPLAY_W / W
        dH    = int(H * scale)
        left  = cv2.resize(img_bgr, (DISPLAY_W, dH))
        right = cv2.resize(seg_bgr, (DISPLAY_W, dH), interpolation=cv2.INTER_NEAREST)

        cv2.putText(left,  'Original',     (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(right, 'Segmentation', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        info = f'Frame {idx+1}/{total}  ts={ts/1e9:.2f}s  {"[PAUSED]" if paused else ""}'
        cv2.putText(left, info, (10, dH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        divider = np.full((dH, 4, 3), 60, dtype=np.uint8)
        canvas  = np.hstack([left, divider, right])
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (83, ord('d')):
            idx = min(idx + 1, total - 1)
            continue
        elif key in (81, ord('a')):
            idx = max(0, idx - 1)
            paused = True
            continue

        if not paused:
            idx += 1

    cv2.destroyAllWindows()
    conn.close()
    print('Done.')

if __name__ == '__main__':
    main()
