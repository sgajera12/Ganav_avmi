"""
build_finetune_comparison_grid.py
Takes existing 3-panel test output images (orig | seg | overlay) and
assembles them into a multi-row comparison grid.

Each row = one model, showing N orig|seg pairs.
Row label on the left.

Output:
    results/model_comparison/rugd_finetune_grid.png
    results/model_comparison/rellis_finetune_grid.png

Usage:
    /home/pinaka/miniconda3/envs/segformer/bin/python \
        tools/build_finetune_comparison_grid.py
"""
import os
import numpy as np
import cv2

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_REPO_ROOT, 'results')
OUT_DIR = os.path.join(RES, 'model_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

# Cell size per image (orig or seg panel)
CELL_W, CELL_H = 300, 200
GAP      = 4
LABEL_W  = 130   # left column for row label
BG       = np.array([20, 20, 20], dtype=np.uint8)
N_PAIRS  = 4     # how many orig|seg pairs to show per model row

def load_panels(folder, n):
    """Load first n 3-panel images from folder, return list of (orig, seg)."""
    paths = sorted([os.path.join(folder, f)
                    for f in os.listdir(folder) if f.endswith('.png')])[:n]
    pairs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        w3 = img.shape[1] // 3
        orig = img[:, :w3]
        seg  = img[:, w3:w3*2]
        pairs.append((orig, seg))
    return pairs

def draw_label(canvas, text, x, y, w, h, font_scale=0.45):
    """Draw multi-word label centred vertically in a box."""
    words = text.split()
    lines, line = [], ''
    for word in words:
        test = (line + ' ' + word).strip()
        tw = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        if tw > w - 6 and line:
            lines.append(line)
            line = word
        else:
            line = test
    lines.append(line)
    total_th = len(lines) * 18
    start_y  = y + h // 2 - total_th // 2 + 14
    for i, l in enumerate(lines):
        tw = cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        cv2.putText(canvas, l, (x + (w - tw) // 2, start_y + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 220, 255), 1, cv2.LINE_AA)

def build_grid(rows_data, out_path, title):
    """
    rows_data: list of (label, list_of_(orig,seg)_pairs)
    """
    n_rows  = len(rows_data)
    n_pairs = max(len(r[1]) for r in rows_data)
    pair_w  = CELL_W * 2 + GAP
    total_w = LABEL_W + GAP + n_pairs * pair_w + (n_pairs - 1) * GAP + GAP
    title_h = 36
    row_h   = CELL_H + GAP * 2
    total_h = title_h + n_rows * row_h + GAP

    canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)

    # Title
    cv2.putText(canvas, title, (LABEL_W + GAP + 4, title_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)

    for ri, (label, pairs) in enumerate(rows_data):
        y0 = title_h + ri * row_h + GAP

        # Row label box
        cv2.rectangle(canvas, (GAP, y0), (LABEL_W, y0 + CELL_H), (35, 35, 35), -1)
        draw_label(canvas, label, GAP, y0, LABEL_W - GAP, CELL_H)

        for pi, (orig, seg) in enumerate(pairs[:n_pairs]):
            xo = LABEL_W + GAP + pi * (pair_w + GAP)
            xs = xo + CELL_W + GAP
            canvas[y0:y0+CELL_H, xo:xo+CELL_W] = cv2.resize(orig, (CELL_W, CELL_H))
            canvas[y0:y0+CELL_H, xs:xs+CELL_W] = cv2.resize(
                seg, (CELL_W, CELL_H), interpolation=cv2.INTER_NEAREST)

        # Thin separator line between rows
        if ri < n_rows - 1:
            sep_y = y0 + CELL_H + GAP
            cv2.line(canvas, (0, sep_y), (total_w, sep_y), (50, 50, 50), 1)

    cv2.imwrite(out_path, canvas)
    print(f'Saved: {out_path}  ({canvas.shape[1]}x{canvas.shape[0]})')

# ── RUGD grid ─────────────────────────────────────────────────────────────────
rugd_rows = [
    ('RUGD Mapped',        load_panels(os.path.join(RES, 'avmi_mapped_test_rugd'),    N_PAIRS)),
    ('RUGD Selective',     load_panels(os.path.join(RES, 'avmi_selective_test_rugd'), N_PAIRS)),
    ('RUGD Weighted',      load_panels(os.path.join(RES, 'avmi_weighted_test_rugd'),  N_PAIRS)),
    ('AVMI+RUGD Joint',    load_panels(os.path.join(RES, 'joint_rugd_test_rugd'),     N_PAIRS)),
]

build_grid(rugd_rows,
           os.path.join(OUT_DIR, 'rugd_finetune_grid.png'),
           'Fine-tuned Models — RUGD Test Images')

# ── RELLIS grid ───────────────────────────────────────────────────────────────
rellis_rows = [
    ('RELLIS Fixed\n(AVMI scratch\non RELLIS)',  load_panels(os.path.join(RES, 'avmi_fixed_test_rellis'),    N_PAIRS)),
    ('AVMI+RELLIS\nJoint',                       load_panels(os.path.join(RES, 'joint_rellis_test_rellis'),  N_PAIRS)),
]

build_grid(rellis_rows,
           os.path.join(OUT_DIR, 'rellis_finetune_grid.png'),
           'Fine-tuned Models — RELLIS Test Images')

print('Done.')
