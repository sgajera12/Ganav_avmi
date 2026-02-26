"""
Copy img_seg1 and img_seg2 into data/avmi_ugv, rename files to avoid
collisions, and create train/val/test splits.

Usage:
    python3 tools/setup_avmi_dataset.py           # dry-run (shows what will happen)
    python3 tools/setup_avmi_dataset.py --run      # actually copies files
"""

import os
import shutil
import random
import argparse
from pathlib import Path

# ── Source folders ────────────────────────────────────────────────────────────
SRC_ROOT = Path("/home/pinaka/offroad_slam/Segmentation_OffRoad/scripts")
SOURCES = [
    (SRC_ROOT / "img_seg1" / "right",     SRC_ROOT / "img_seg1" / "seg_right",  "s1"),
    (SRC_ROOT / "img_seg2" / "right",     SRC_ROOT / "img_seg2" / "seg_right",  "s2"),
]

# ── Destination ───────────────────────────────────────────────────────────────
DST_ROOT  = Path("data/avmi_ugv")
IMG_DIRS  = {split: DST_ROOT / "images"      / split for split in ("train", "val", "test")}
ANN_DIRS  = {split: DST_ROOT / "annotations" / split for split in ("train", "val", "test")}
SPLIT_TXT = {split: DST_ROOT / f"{split}.txt"        for split in ("train", "val", "test")}

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = remainder (0.10)

RANDOM_SEED = 42


def collect_pairs():
    """Return list of (img_path, ann_path, new_stem) for every valid pair."""
    pairs = []
    for img_dir, ann_dir, prefix in SOURCES:
        ann_stems = {p.stem for p in ann_dir.glob("*.png")}
        for img_path in sorted(img_dir.glob("*.png")):
            if img_path.stem in ann_stems:
                ann_path = ann_dir / img_path.name          # same filename
                new_stem = f"{prefix}_{int(img_path.stem):05d}"
                pairs.append((img_path, ann_path, new_stem))
            # silently skip images with no matching mask
    return pairs


def make_dirs(dry_run):
    for d in list(IMG_DIRS.values()) + list(ANN_DIRS.values()):
        if not dry_run:
            d.mkdir(parents=True, exist_ok=True)
        else:
            print(f"  mkdir -p {d}")


def clear_old_data(dry_run):
    """Remove existing images/annotations so we start clean."""
    for split in ("train", "val", "test"):
        for d in (IMG_DIRS[split], ANN_DIRS[split]):
            if d.exists():
                existing = list(d.iterdir())
                if existing:
                    print(f"  Clearing {len(existing)} old files from {d}")
                    if not dry_run:
                        shutil.rmtree(d)
                        d.mkdir(parents=True, exist_ok=True)
    for txt in SPLIT_TXT.values():
        if txt.exists():
            print(f"  Removing old {txt}")
            if not dry_run:
                txt.unlink()


def split_pairs(pairs):
    rng = random.Random(RANDOM_SEED)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }


def copy_and_write(splits, dry_run):
    for split, pairs in splits.items():
        stems = []
        for img_src, ann_src, stem in pairs:
            img_dst = IMG_DIRS[split] / f"{stem}.png"
            ann_dst = ANN_DIRS[split] / f"{stem}.png"
            if not dry_run:
                shutil.copy2(img_src, img_dst)
                shutil.copy2(ann_src, ann_dst)
            stems.append(stem)

        if not dry_run:
            with open(SPLIT_TXT[split], "w") as f:
                f.write("\n".join(stems) + "\n")

        print(f"  {split:5s}: {len(pairs):5d} pairs  →  {IMG_DIRS[split]}")

    if dry_run:
        print("\n  (dry-run: no files written)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="Actually copy files (default is dry-run)")
    args = parser.parse_args()
    dry_run = not args.run

    print("Scanning source folders for valid image-mask pairs …")
    pairs = collect_pairs()
    print(f"  Found {len(pairs)} paired files\n")

    # break down by source
    for _, _, prefix in SOURCES:
        n = sum(1 for *_, s in pairs if s.startswith(prefix))
        print(f"  {prefix}: {n} pairs")

    splits = split_pairs(pairs)
    print(f"\nSplit plan (seed={RANDOM_SEED}):")
    for split, p in splits.items():
        print(f"  {split:5s}: {len(p)}")

    if dry_run:
        print("\n*** DRY-RUN — pass --run to actually copy files ***\n")
    else:
        print("\n*** RUNNING — files will be copied ***\n")

    make_dirs(dry_run)
    clear_old_data(dry_run)
    copy_and_write(splits, dry_run)

    if not dry_run:
        print("\nDone! Update avmi_dataset.py if img_suffix is still .jpg")


if __name__ == "__main__":
    main()
