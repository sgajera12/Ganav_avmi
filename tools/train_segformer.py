"""
Training script for SegFormer models using mmsegmentation 1.x.
Uses the 'segformer' conda environment (NOT 'ganav').

Usage:
    /home/pinaka/miniconda3/envs/segformer/bin/python tools/train_segformer.py \
        configs/segformer_b2_avmi.py \
        --work-dir work_dirs/segformer_b2_avmi
"""
import argparse
import os
import os.path as osp
import sys

# NOTE: Do NOT add GANav project root to sys.path — it contains a local mmseg/
# folder that conflicts with the installed mmsegmentation 1.x package.
# Run this script from /tmp or home directory using absolute paths.

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor (mmseg 1.x)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='directory to save logs and models')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='enable mixed precision training (FP16) - default ON')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.resume:
        cfg.resume = True

    # Enable FP16 mixed precision (essential for 12GB VRAM)
    if args.amp:
        cfg.optim_wrapper.setdefault('type', 'AmpOptimWrapper')
        cfg.optim_wrapper['type'] = 'AmpOptimWrapper'

    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
