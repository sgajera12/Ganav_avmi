"""
Check RGB colours in a segmentation mask PNG.
Shows the actual dominant colours and how they map to classes
using nearest-colour matching (same logic as avmi_dataset.py).
"""

import numpy as np
from PIL import Image
import cv2

MASK_PATH = "/home/pinaka/offroad_slam/Segmentation_OffRoad/scripts/img_seg2/seg_right/1770.png"
IMG_PATH  = "/home/pinaka/offroad_slam/Segmentation_OffRoad/scripts/img_seg2/right/1770.png"

COLOR_TO_CLASS = {
    (24,  102, 178): (0, "sky    (blue)"),
    (18,  182,  37): (1, "tree   (green)"),
    (239, 255,  15): (2, "bush   (yellow)"),
    (92,   19,   6): (3, "ground (dark brown)"),
    (255,  63, 250): (4, "trunk  (pink)"),
    (255,   0,   0): (5, "rock   (red)"),
}

PALETTE_COLORS = np.array([k for k in COLOR_TO_CLASS], dtype=np.int32)
PALETTE_LABELS = np.array([v[0] for v in COLOR_TO_CLASS.values()], dtype=np.uint8)
PALETTE_NAMES  = [v[1] for v in COLOR_TO_CLASS.values()]


def nearest_class(rgb_pixel):
    diff = np.array(rgb_pixel, dtype=np.int32) - PALETTE_COLORS
    dist = (diff ** 2).sum(axis=1)
    idx  = np.argmin(dist)
    return PALETTE_LABELS[idx], PALETTE_NAMES[idx], dist[idx]


def check_mask(mask_path):
    mask  = np.array(Image.open(mask_path).convert("RGB"))
    h, w  = mask.shape[:2]
    total = h * w

    flat  = mask.reshape(-1, 3)
    unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)

    # Aggregate by nearest class
    class_pixels = {i: 0 for i in range(len(PALETTE_COLORS))}
    for rgb, cnt in zip(unique_colors, counts):
        cls, _, _ = nearest_class(tuple(rgb))
        class_pixels[cls] += cnt

    print(f"\n=== Mask: {mask_path} ===")
    print(f"Size: {w}x{h}  ({total} pixels)\n")

    print("--- Top 15 actual colours in mask ---")
    print(f"{'RGB':>20}  {'Pixels':>8}  {'%':>6}  Nearest class")
    print("-" * 65)
    for i in order[:15]:
        rgb   = tuple(unique_colors[i])
        count = counts[i]
        pct   = count / total * 100
        cls, name, dist = nearest_class(rgb)
        exact = " (exact)" if dist == 0 else f" (dist={int(dist**0.5)})"
        print(f"  {str(rgb):>20}  {count:>8}  {pct:>5.2f}%  {name}{exact}")

    print(f"\n--- Class coverage after nearest-colour assignment ---")
    print(f"{'Class':>4}  {'Name':>20}  {'Pixels':>8}  {'%':>6}")
    print("-" * 48)
    for i, name in enumerate(PALETTE_NAMES):
        px  = class_pixels[i]
        pct = px / total * 100
        print(f"  {i:>4}  {name:>20}  {px:>8}  {pct:>5.2f}%")


def save_side_by_side(img_path, mask_path, out_path="color_check_output.png"):
    orig = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if orig is None:
        print(f"[WARN] Could not load original: {img_path}")
        return
    if mask is None:
        print(f"[WARN] Could not load mask: {mask_path}")
        return

    h = min(orig.shape[0], mask.shape[0])
    orig_r = cv2.resize(orig, (int(orig.shape[1] * h / orig.shape[0]), h))
    mask_r = cv2.resize(mask, (int(mask.shape[1] * h / mask.shape[0]), h),
                        interpolation=cv2.INTER_NEAREST)

    canvas = np.hstack([orig_r, mask_r])
    cv2.imwrite(out_path, canvas)
    print(f"\n[Saved side-by-side → {out_path}]")


if __name__ == "__main__":
    check_mask(MASK_PATH)
    save_side_by_side(IMG_PATH, MASK_PATH)
