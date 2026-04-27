"""
finetune_segformer_rugd_v2.py

Fine-tunes AVMI 8-class SegFormer-B2 on RUGD offroad sequences only.
  sky | tree | bush | ground | obstacle | rock | water | concrete
  - water(6)    → water(6)     [was incorrectly rock — caused red-sky bug]
  - concrete(23)→ concrete(7)
  - all sequences used (village included — mappings handle class diversity)

Starts from:  work_dirs/segformer_b2_avmi_8class/latest.pth
Saves to:     work_dirs/segformer_b2_finetune_rugd_v2/

Run on Turing HPC via train_finetune_rugd_v2.sh
"""

import os, random
from datetime import datetime
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

# ── Config ────────────────────────────────────────────────────────────────────
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVMI_CKPT    = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_8class', 'latest.pth')
WORK_DIR     = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_finetune_rugd_v2')
PRETRAINED   = 'nvidia/mit-b2'
NUM_CLASSES  = 8
BATCH_SIZE   = 8
MAX_ITERS    = 30000
VAL_INTERVAL = 2000
LR_ENCODER   = 6e-6
LR_DECODER   = 6e-5
WEIGHT_DECAY = 0.01
WARMUP_ITERS = 500
CROP_SIZE    = 512

CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock', 'water', 'concrete')
# Higher weight for water/concrete — new classes absent from AVMI training
CLASS_WEIGHTS = torch.tensor([0.5, 1.0, 1.5, 0.8, 2.5, 3.0, 2.5, 2.5])

# ── Class mappings ────────────────────────────────────────────────────────────
# RUGD _orig.png stores colormap_id - 1 (void excluded, everything shifts down):
# 0=dirt, 1=sand, 2=grass, 3=tree, 4=pole, 5=water, 6=sky, 7=vehicle,
# 8=container, 9=asphalt, 10=gravel, 11=building, 12=mulch, 13=rock-bed,
# 14=log, 15=bicycle, 16=person, 17=fence, 18=bush, 19=sign, 20=rock,
# 21=bridge, 22=concrete, 23=picnic-table
_RUGD_MAP = np.full(256, 255, dtype=np.uint8)
_RUGD_MAP[0]  = 3  # dirt         → ground
_RUGD_MAP[1]  = 3  # sand         → ground
_RUGD_MAP[2]  = 3  # grass        → ground
_RUGD_MAP[3]  = 1  # tree         → tree
_RUGD_MAP[4]  = 4  # pole         → obstacle
_RUGD_MAP[5]  = 6  # water        → water
_RUGD_MAP[6]  = 0  # sky          → sky
_RUGD_MAP[7]  = 4  # vehicle      → obstacle
_RUGD_MAP[8]  = 4  # container    → obstacle
_RUGD_MAP[9]  = 3  # asphalt      → ground
_RUGD_MAP[10] = 3  # gravel       → ground
_RUGD_MAP[11] = 4  # building     → obstacle
_RUGD_MAP[12] = 3  # mulch        → ground
_RUGD_MAP[13] = 5  # rock-bed     → rock
_RUGD_MAP[14] = 4  # log          → obstacle
_RUGD_MAP[15] = 4  # bicycle      → obstacle
_RUGD_MAP[16] = 4  # person       → obstacle
_RUGD_MAP[17] = 4  # fence        → obstacle
_RUGD_MAP[18] = 2  # bush         → bush
_RUGD_MAP[19] = 4  # sign         → obstacle
_RUGD_MAP[20] = 5  # rock         → rock
_RUGD_MAP[21] = 4  # bridge       → obstacle
_RUGD_MAP[22] = 7  # concrete     → concrete
_RUGD_MAP[23] = 4  # picnic-table → obstacle

# ── Shared processor ──────────────────────────────────────────────────────────
_PROCESSOR = SegformerImageProcessor(
    do_resize=True,
    size={'height': 480, 'width': 640},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)

def augment(img, ann):
    scale  = random.uniform(0.5, 2.0)
    orig_w, orig_h = img.size
    new_w  = max(CROP_SIZE, int(orig_w * scale))
    new_h  = max(CROP_SIZE, int(orig_h * scale))
    img    = img.resize((new_w, new_h), Image.BILINEAR)
    ann    = np.array(Image.fromarray(ann).resize((new_w, new_h), Image.NEAREST))
    top    = random.randint(0, new_h - CROP_SIZE)
    left   = random.randint(0, new_w - CROP_SIZE)
    img    = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    ann    = ann[top:top+CROP_SIZE, left:left+CROP_SIZE]
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        ann = ann[:, ::-1].copy()
    return img, ann

def encode(img, ann):
    pv  = _PROCESSOR(images=img, return_tensors='pt')['pixel_values'].squeeze(0)
    ann = Image.fromarray(ann)
    ann = ann.resize((pv.shape[2], pv.shape[1]), Image.NEAREST)
    lbl = torch.from_numpy(np.array(ann)).long()
    return pv, lbl

# ── RUGD Dataset (offroad sequences only) ─────────────────────────────────────
class RUGDDataset(Dataset):
    def __init__(self, split='train', train_aug=True):
        txt = 'train_ours.txt' if split == 'train' else 'val_ours.txt'
        with open(os.path.join(_REPO_ROOT, 'data', 'rugd', txt)) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        self.img_root = os.path.join(_REPO_ROOT, 'data', 'rugd', 'RUGD_frames-with-annotations')
        self.ann_root = os.path.join(_REPO_ROOT, 'data', 'rugd', 'RUGD_annotations')
        self.train_aug = train_aug

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        rid = self.ids[idx]
        seq, name = rid.split('/')
        img = Image.open(os.path.join(self.img_root, seq, name + '.png')).convert('RGB')
        ann = np.array(Image.open(
            os.path.join(self.ann_root, seq, name + '_orig.png')))
        ann = _RUGD_MAP[ann]
        if self.train_aug:
            img, ann = augment(img, ann)
        pv, lbl = encode(img, ann)
        return {'pixel_values': pv, 'labels': lbl}

# ── Loss ──────────────────────────────────────────────────────────────────────
def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    oh    = torch.zeros_like(probs)
    oh.scatter_(1, targets.unsqueeze(1).clamp(0, num_classes - 1), 1)
    inter = (probs * oh).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()

# ── Validation ────────────────────────────────────────────────────────────────
def validate(model, loader, device):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            pv  = batch['pixel_values'].to(device)
            lbl = batch['labels']
            with torch.autocast('cuda'):
                out = model(pixel_values=pv)
                up  = nn.functional.interpolate(out.logits, size=lbl.shape[-2:],
                                                mode='bilinear', align_corners=False)
            pred = up.argmax(1).cpu()
            for p, g in zip(pred.view(-1), lbl.view(-1)):
                if 0 <= g < NUM_CLASSES:
                    conf[g, p] += 1
    ious = []
    for i in range(NUM_CLASSES):
        tp = conf[i, i].item()
        fp = conf[:, i].sum().item() - tp
        fn = conf[i, :].sum().item() - tp
        d  = tp + fp + fn
        if d > 0: ious.append(tp / d)
    miou = sum(ious) / len(ious) if ious else 0.0
    print(f'  Per-class IoU: {[f"{CLASSES[i]}={ious[i]:.3f}" for i in range(len(ious))]}')
    return miou

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  ({torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"})')

    cfg = SegformerConfig.from_pretrained(PRETRAINED)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label   = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id   = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, config=cfg, ignore_mismatched_sizes=True)

    print(f'Loading AVMI 8-class checkpoint: {AVMI_CKPT}')
    ckpt = torch.load(AVMI_CKPT, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'])
    print(f'  Loaded iter={ckpt["iter"]}  val_mIoU={ckpt["miou"]:.4f}')
    model = model.to(device)

    enc_params = [p for n, p in model.named_parameters() if 'segformer' in n]
    dec_params = [p for n, p in model.named_parameters() if 'decode_head' in n]
    optimizer  = torch.optim.AdamW([
        {'params': enc_params, 'lr': LR_ENCODER},
        {'params': dec_params, 'lr': LR_DECODER},
    ], weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < WARMUP_ITERS: return step / WARMUP_ITERS
        pct = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
        return max(0.0, 1.0 - pct)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_ds = RUGDDataset('train', train_aug=True)
    val_ds   = RUGDDataset('val',   train_aug=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)
    print(f'Train: {len(train_ds)} RUGD images')
    print(f'Val:   {len(val_ds)}   RUGD images')

    ce_loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device), ignore_index=255)
    scaler = GradScaler()

    start_iter = 0
    ckpt_path  = os.path.join(WORK_DIR, 'latest.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter']
        print(f'Resumed fine-tune from iter {start_iter}')

    model.train()
    data_iter = iter(train_loader)
    log_file  = open(os.path.join(WORK_DIR, 'train_log.txt'), 'a')

    for itr in range(start_iter, MAX_ITERS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        pv  = batch['pixel_values'].to(device)
        lbl = batch['labels'].to(device)

        optimizer.zero_grad()
        with torch.autocast('cuda'):
            out      = model(pixel_values=pv)
            logits   = out.logits
            logits_up = nn.functional.interpolate(
                logits, size=lbl.shape[-2:], mode='bilinear', align_corners=False)
            lf     = logits_up.float()
            l_ce   = ce_loss_fn(lf, lbl)
            l_dice = dice_loss(lf, lbl, NUM_CLASSES)
            loss   = l_ce + 3.0 * l_dice

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if itr % 50 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            ts  = datetime.now().strftime('%H:%M:%S')
            msg = (f'[{ts}] [{itr:6d}/{MAX_ITERS}] loss={loss.item():.4f} '
                   f'ce={l_ce.item():.4f} dice={l_dice.item():.4f} lr={lr_now:.2e}')
            print(msg, flush=True)
            log_file.write(msg + '\n'); log_file.flush()

        if (itr + 1) % VAL_INTERVAL == 0:
            miou = validate(model, val_loader, device)
            msg  = f'[Val @ {itr+1}] mIoU = {miou:.4f}'
            print(msg, flush=True)
            log_file.write(msg + '\n'); log_file.flush()
            torch.save({'iter': itr+1, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'miou': miou}, ckpt_path)
            torch.save(model.state_dict(),
                       os.path.join(WORK_DIR, f'iter_{itr+1}_miou{miou:.4f}.pth'))
            model.train()

    log_file.close()
    print('Fine-tuning complete.')

if __name__ == '__main__':
    main()
