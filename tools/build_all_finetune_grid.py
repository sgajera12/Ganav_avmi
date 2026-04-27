"""
build_all_finetune_grid.py
Single grid showing one example orig|seg pair per fine-tuned model.

Rows: RUGD Mapped | RUGD Selective | RUGD Weighted | AVMI+RUGD Joint | AVMI+RELLIS Joint
Each row: row label | orig | seg

Usage:
    /home/pinaka/miniconda3/envs/segformer/bin/python \
        tools/build_all_finetune_grid.py
"""
import os
import numpy as np
import cv2

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES     = os.path.join(_REPO_ROOT, 'results')
OUT_DIR = os.path.join(RES, 'model_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

CELL_W, CELL_H = 320, 240
GAP      = 6
LABEL_W  = 150
N_PAIRS  = 3      # orig|seg pairs per row
BG       = np.array([20, 20, 20], dtype=np.uint8)

# (row label, folder)
ROWS = [
    ('RUGD\nMapped',       'avmi_mapped_test_rugd'),
    ('RUGD\nSelective',    'avmi_selective_test_rugd'),
    ('RUGD\nWeighted',     'avmi_weighted_test_rugd'),
    ('AVMI+RUGD\nJoint',   'joint_rugd_test_rugd'),
    ('AVMI+RELLIS\nJoint', 'joint_rellis_test_rellis'),
]

def get_panels(folder, n):
    folder = os.path.join(RES, folder)
    paths  = sorted([os.path.join(folder, f)
                     for f in os.listdir(folder) if f.endswith('.png')])[:n]
    pairs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        w3 = img.shape[1] // 3
        pairs.append((img[:, :w3], img[:, w3:w3*2]))
    return pairs

# ── Build canvas ──────────────────────────────────────────────────────────────
pair_w  = CELL_W * 2 + GAP
total_w = GAP + LABEL_W + GAP + N_PAIRS * pair_w + (N_PAIRS - 1) * GAP + GAP
title_h = 40
row_h   = CELL_H + GAP * 2
total_h = title_h + len(ROWS) * row_h + GAP

canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)

# Title
cv2.putText(canvas, 'Fine-tuned GANav B0 Models — Sample Test Results',
            (GAP + LABEL_W + GAP, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)

# Column headers above first row
header_y = title_h - 4
for pi in range(N_PAIRS):
    xo = GAP + LABEL_W + GAP + pi * (pair_w + GAP)
    xs = xo + CELL_W + GAP
    cv2.putText(canvas, 'Original',    (xo + 4, header_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Segmented', (xs + 4, header_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

for ri, (label, folder) in enumerate(ROWS):
    pairs = get_panels(folder, N_PAIRS)
    y0    = title_h + ri * row_h + GAP

    # Label box
    cv2.rectangle(canvas, (GAP, y0), (GAP + LABEL_W, y0 + CELL_H), (35, 35, 35), -1)
    lines = label.split('\n')
    start_y = y0 + CELL_H // 2 - len(lines) * 10 + 10
    for li, line in enumerate(lines):
        tw = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(canvas, line,
                    (GAP + (LABEL_W - tw) // 2, start_y + li * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)

    # Image pairs
    for pi, (orig, seg) in enumerate(pairs):
        xo = GAP + LABEL_W + GAP + pi * (pair_w + GAP)
        xs = xo + CELL_W + GAP
        canvas[y0:y0+CELL_H, xo:xo+CELL_W] = cv2.resize(orig, (CELL_W, CELL_H))
        canvas[y0:y0+CELL_H, xs:xs+CELL_W] = cv2.resize(
            seg, (CELL_W, CELL_H), interpolation=cv2.INTER_NEAREST)

    # Separator
    if ri < len(ROWS) - 1:
        sep_y = y0 + CELL_H + GAP
        cv2.line(canvas, (0, sep_y), (total_w, sep_y), (50, 50, 50), 1)

out_path = os.path.join(OUT_DIR, 'all_finetune_models_grid.png')
cv2.imwrite(out_path, canvas)
print(f'Saved: {out_path}  ({canvas.shape[1]}x{canvas.shape[0]})')
