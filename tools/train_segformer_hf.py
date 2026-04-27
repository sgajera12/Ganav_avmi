"""
SegFormer-B2 training on AVMI 6-class dataset using HuggingFace Transformers.
No mmcv required. Uses RTX 5070 with CUDA 12.8 via PyTorch 2.10.

Run from home directory:
    cd ~ && /home/pinaka/miniconda3/envs/segformer/bin/python \
        /home/pinaka/GANav-offroad/tools/train_segformer_hf.py
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ── Annotation color → class index mapping (from avmi_dataset.py) ────────────
_PALETTE_COLORS = np.array([
    [24,  102, 178],  # 0: sky
    [18,  182,  37],  # 1: tree
    [239, 255,  15],  # 2: bush
    [92,   19,   6],  # 3: ground
    [255,  63, 250],  # 4: obstacle
    [255,   0,   0],  # 5: rock
], dtype=np.int32)
_PALETTE_LABELS = np.arange(6, dtype=np.uint8)

def rgb_mask_to_index(rgb_img):
    """Convert RGB color-coded annotation mask to class index mask."""
    h, w = rgb_img.shape[:2]
    flat = rgb_img.reshape(-1, 3).astype(np.int32)
    diff = flat[:, None, :] - _PALETTE_COLORS[None, :, :]
    dist = (diff ** 2).sum(axis=2)
    nearest = np.argmin(dist, axis=1)
    return _PALETTE_LABELS[nearest].reshape(h, w)

# ── Config ────────────────────────────────────────────────────────────────────
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT   = os.path.join(_REPO_ROOT, 'data', 'avmi_ugv')
WORK_DIR    = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_hf')
PRETRAINED  = 'nvidia/mit-b2'          # downloads MiT-B2 ImageNet weights
NUM_CLASSES = 6
CROP_SIZE   = 512
BATCH_SIZE  = 8
MAX_ITERS   = 40000
VAL_INTERVAL= 4000
LR_ENCODER  = 3e-5
LR_DECODER  = 3e-4
WEIGHT_DECAY= 0.01
WARMUP_ITERS= 1500

CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')
# Inverse frequency weights (sky/ground are majority, rock/obstacle are rare)
CLASS_WEIGHTS = torch.tensor([0.5, 1.0, 1.5, 0.8, 2.5, 3.0])

# ── Dataset ───────────────────────────────────────────────────────────────────
class AVMIDataset(Dataset):
    def __init__(self, data_root, split='train', crop_size=512):
        self.img_dir = Path(data_root) / 'images' / split
        self.ann_dir = Path(data_root) / 'annotations' / split
        self.files   = sorted([f.stem for f in self.img_dir.glob('*.png')])
        self.split   = split
        self.crop_size = crop_size
        self.processor = SegformerImageProcessor(
            do_resize=True,
            size={'height': 480, 'width': 640},
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(self.img_dir / f'{name}.png').convert('RGB')
        ann_rgb = np.array(Image.open(self.ann_dir / f'{name}.png').convert('RGB'))
        ann = rgb_mask_to_index(ann_rgb).astype(np.int64)

        if self.split == 'train':
            img, ann = self._random_crop_flip(img, ann)

        encoded = self.processor(images=img, return_tensors='pt')
        pixel_values = encoded['pixel_values'].squeeze(0)  # (3, H, W)

        # Resize annotation to match processor output
        ann_img = Image.fromarray(ann.astype(np.uint8))
        ann_img = ann_img.resize(
            (pixel_values.shape[2], pixel_values.shape[1]),
            Image.NEAREST)
        labels = torch.from_numpy(np.array(ann_img)).long()

        return {'pixel_values': pixel_values, 'labels': labels}

    def _random_crop_flip(self, img, ann):
        """Random resize, crop, and horizontal flip."""
        import random
        # Random resize 0.5–2.0x (clamp so image is never smaller than crop)
        scale = random.uniform(0.5, 2.0)
        new_w = max(self.crop_size, int(640 * scale))
        new_h = max(self.crop_size, int(480 * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        ann = np.array(
            Image.fromarray(ann.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
        ).astype(np.int64)

        # Random crop (no padding needed since new_h/new_w >= crop_size)
        top  = random.randint(0, new_h - self.crop_size)
        left = random.randint(0, new_w - self.crop_size)
        img  = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        ann  = ann[top:top+self.crop_size, left:left+self.crop_size]

        # Random flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ann = ann[:, ::-1].copy()

        return img, ann


# ── Loss ──────────────────────────────────────────────────────────────────────
def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    targets_oh = torch.zeros_like(probs)
    targets_oh.scatter_(1, targets.unsqueeze(1).clamp(0, num_classes-1), 1)
    intersection = (probs * targets_oh).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


# ── Training ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} ({torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"})')

    # Model
    from transformers import SegformerConfig
    cfg = SegformerConfig.from_pretrained(PRETRAINED)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, config=cfg, ignore_mismatched_sizes=True)
    model = model.to(device)
    print(f'Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    # Optimizer: separate LR for encoder vs decoder
    encoder_params = [p for n, p in model.named_parameters() if 'segformer' in n]
    decoder_params = [p for n, p in model.named_parameters() if 'decode_head' in n]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': decoder_params, 'lr': LR_DECODER},
    ], weight_decay=WEIGHT_DECAY)

    # LR scheduler: linear warmup + poly decay
    def lr_lambda(step):
        if step < WARMUP_ITERS:
            return step / WARMUP_ITERS
        pct = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
        return max(0.0, 1.0 - pct)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Data
    train_ds = AVMIDataset(DATA_ROOT, 'train', CROP_SIZE)
    val_ds   = AVMIDataset(DATA_ROOT, 'val',   CROP_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    # Loss weights
    ce_weight = CLASS_WEIGHTS.to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=255)
    scaler = GradScaler()

    # Resume
    start_iter = 0
    ckpt_path = os.path.join(WORK_DIR, 'latest.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter']
        print(f'Resumed from iter {start_iter}')

    # Train loop
    model.train()
    data_iter = iter(train_loader)
    log_file  = open(os.path.join(WORK_DIR, 'train_log.txt'), 'a')

    for itr in range(start_iter, MAX_ITERS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        pixels = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with torch.autocast('cuda'):
            outputs = model(pixel_values=pixels)
            logits  = outputs.logits  # (B, C, H/4, W/4)
            # Upsample logits to label size
            logits_up = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            # Compute losses in float32 to avoid FP16 NaN
            logits_f32 = logits_up.float()
            loss_ce   = ce_loss_fn(logits_f32, labels)
            loss_dice = dice_loss(logits_f32, labels, NUM_CLASSES)
            loss      = loss_ce + 3.0 * loss_dice

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # tighter clip
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if itr % 50 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            msg = f'[{itr:6d}/{MAX_ITERS}] loss={loss.item():.4f} ce={loss_ce.item():.4f} dice={loss_dice.item():.4f} lr={lr_now:.2e}'
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()

        # Validation
        if (itr + 1) % VAL_INTERVAL == 0:
            miou = validate(model, val_loader, device, NUM_CLASSES)
            msg = f'[Val @ {itr+1}] mIoU = {miou:.4f}'
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()

            # Save checkpoint
            torch.save({
                'iter': itr + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'miou': miou,
            }, ckpt_path)
            torch.save(model.state_dict(),
                       os.path.join(WORK_DIR, f'iter_{itr+1}_miou{miou:.4f}.pth'))
            model.train()

    log_file.close()
    print('Training complete.')


def validate(model, loader, device, num_classes):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            pixels = batch['pixel_values'].to(device)
            labels = batch['labels']
            with torch.autocast('cuda'):
                outputs = model(pixel_values=pixels)
                logits  = outputs.logits
                logits_up = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            preds = logits_up.argmax(dim=1).cpu()
            for p, g in zip(preds.view(-1), labels.view(-1)):
                if 0 <= g < num_classes:
                    confusion[g, p] += 1

    iou_list = []
    for i in range(num_classes):
        tp = confusion[i, i].item()
        fp = confusion[:, i].sum().item() - tp
        fn = confusion[i, :].sum().item() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou_list.append(tp / denom)
    miou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    print(f'  Per-class IoU: {[f"{CLASSES[i]}={iou_list[i]:.3f}" for i in range(len(iou_list))]}')
    return miou


if __name__ == '__main__':
    main()
