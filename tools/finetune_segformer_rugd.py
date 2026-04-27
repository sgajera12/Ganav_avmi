"""
finetune_segformer_rugd.py

Fine-tunes our AVMI-trained SegFormer-B2 (80.51% mIoU) on RUGD only
with 6-class mapping:  sky | tree | bush | ground | obstacle | rock

Starts from:  work_dirs/segformer_b2_avmi_hf/latest.pth
Saves to:     work_dirs/segformer_b2_finetune_rugd/

Run on Turing HPC via train_finetune_rugd.sh
"""

import os, random
from datetime import datetime
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor

# ── Config ────────────────────────────────────────────────────────────────────
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVMI_CKPT    = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_avmi_hf', 'latest.pth')
WORK_DIR     = os.path.join(_REPO_ROOT, 'work_dirs', 'segformer_b2_finetune_rugd')
PRETRAINED   = 'nvidia/mit-b2'
NUM_CLASSES  = 6
BATCH_SIZE   = 8
MAX_ITERS    = 30000
VAL_INTERVAL = 2000
LR_ENCODER   = 6e-6    # 5× lower than scratch (fine-tuning)
LR_DECODER   = 6e-5
WEIGHT_DECAY = 0.01
WARMUP_ITERS = 500
CROP_SIZE    = 512

CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')
CLASS_WEIGHTS = torch.tensor([0.5, 1.0, 1.5, 0.8, 2.5, 3.0])

# ── Class mappings ────────────────────────────────────────────────────────────
# RUGD _orig.png: indices 0-24
# 0=void,1=dirt,2=sand,3=grass,4=tree,5=pole,6=water,7=sky,8=vehicle,
# 9=container,10=asphalt,11=gravel,12=building,13=mulch,14=rock-bed,
# 15=log,16=bicycle,17=person,18=fence,19=bush,20=sign,21=rock,
# 22=bridge,23=concrete,24=picnic-table
_RUGD_MAP = np.full(256, 255, dtype=np.uint8)
_RUGD_MAP[1]  = 3  # dirt      → ground
_RUGD_MAP[2]  = 3  # sand      → ground
_RUGD_MAP[3]  = 3  # grass     → ground
_RUGD_MAP[4]  = 1  # tree      → tree
_RUGD_MAP[5]  = 4  # pole      → obstacle
_RUGD_MAP[6]  = 5  # water     → rock (non-navigable)
_RUGD_MAP[7]  = 0  # sky       → sky
_RUGD_MAP[8]  = 4  # vehicle   → obstacle
_RUGD_MAP[9]  = 4  # container → obstacle
_RUGD_MAP[10] = 3  # asphalt   → ground
_RUGD_MAP[11] = 3  # gravel    → ground
_RUGD_MAP[12] = 4  # building  → obstacle
_RUGD_MAP[13] = 3  # mulch     → ground
_RUGD_MAP[14] = 5  # rock-bed  → rock
_RUGD_MAP[15] = 4  # log       → obstacle
_RUGD_MAP[16] = 4  # bicycle   → obstacle
_RUGD_MAP[17] = 4  # person    → obstacle
_RUGD_MAP[18] = 4  # fence     → obstacle
_RUGD_MAP[19] = 2  # bush      → bush
_RUGD_MAP[20] = 4  # sign      → obstacle
_RUGD_MAP[21] = 5  # rock      → rock
_RUGD_MAP[22] = 4  # bridge    → obstacle
_RUGD_MAP[23] = 3  # concrete  → ground
_RUGD_MAP[24] = 4  # picnic-table → obstacle

# RELLIS uses ontology IDs (not sequential):
# 0=void,1=dirt,3=grass,4=tree,5=pole,6=water,7=sky,8=vehicle,9=object,
# 10=asphalt,12=building,15=log,17=person,18=fence,19=bush,23=concrete,
# 27=barrier,29=puddle,31=mud,33=rubble
_RELLIS_MAP = np.full(256, 255, dtype=np.uint8)
_RELLIS_MAP[1]  = 3  # dirt      → ground
_RELLIS_MAP[3]  = 3  # grass     → ground
_RELLIS_MAP[4]  = 1  # tree      → tree
_RELLIS_MAP[5]  = 4  # pole      → obstacle
_RELLIS_MAP[6]  = 5  # water     → rock
_RELLIS_MAP[7]  = 0  # sky       → sky
_RELLIS_MAP[8]  = 4  # vehicle   → obstacle
_RELLIS_MAP[9]  = 4  # object    → obstacle
_RELLIS_MAP[10] = 3  # asphalt   → ground
_RELLIS_MAP[12] = 4  # building  → obstacle
_RELLIS_MAP[15] = 4  # log       → obstacle
_RELLIS_MAP[17] = 4  # person    → obstacle
_RELLIS_MAP[18] = 4  # fence     → obstacle
_RELLIS_MAP[19] = 2  # bush      → bush
_RELLIS_MAP[23] = 3  # concrete  → ground
_RELLIS_MAP[27] = 4  # barrier   → obstacle
_RELLIS_MAP[29] = 5  # puddle    → rock
_RELLIS_MAP[31] = 3  # mud       → ground
_RELLIS_MAP[33] = 5  # rubble    → rock

# ── Shared processor ──────────────────────────────────────────────────────────
_PROCESSOR = SegformerImageProcessor(
    do_resize=True,
    size={'height': 480, 'width': 640},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)

def augment(img, ann):
    """Random scale crop + horizontal flip."""
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

# ── RUGD Dataset ──────────────────────────────────────────────────────────────
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

# ── RELLIS Dataset ────────────────────────────────────────────────────────────
class RELLISDataset(Dataset):
    def __init__(self, split='train', train_aug=True):
        txt = 'train.txt' if split == 'train' else 'val.txt'
        with open(os.path.join(_REPO_ROOT, 'data', 'rellis', txt)) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        self.img_root = os.path.join(_REPO_ROOT, 'data', 'rellis', 'image')
        self.ann_root = os.path.join(_REPO_ROOT, 'data', 'rellis', 'annotation')
        self.train_aug = train_aug

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        rid = self.ids[idx]
        seq, name = rid.split('/')
        img_path = os.path.join(self.img_root, seq, name + '.jpg')
        ann_path = os.path.join(self.ann_root, seq, name + '.png')
        img = Image.open(img_path).convert('RGB')
        ann = np.array(Image.open(ann_path))
        ann = _RELLIS_MAP[ann]
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

    # ── Model: load AVMI checkpoint ──────────────────────────────────────────
    cfg = SegformerConfig.from_pretrained(PRETRAINED)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label   = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id   = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, config=cfg, ignore_mismatched_sizes=True)

    print(f'Loading AVMI checkpoint: {AVMI_CKPT}')
    ckpt = torch.load(AVMI_CKPT, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'])
    print(f'  Loaded iter={ckpt["iter"]}  val_mIoU={ckpt["miou"]:.4f}')
    model = model.to(device)

    # ── Optimizer (lower LR for fine-tuning) ────────────────────────────────
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

    # ── Datasets: RUGD + RELLIS joint ───────────────────────────────────────
    train_ds = RUGDDataset('train', train_aug=True)
    val_ds   = RUGDDataset('val',   train_aug=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)
    print(f'Train: {len(train_ds)} RUGD images')
    print(f'Val:   {len(val_ds)}   RUGD images')

    # ── Loss ────────────────────────────────────────────────────────────────
    ce_loss_fn = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS.to(device), ignore_index=255)
    scaler = GradScaler()

    # ── Resume fine-tune if interrupted ─────────────────────────────────────
    start_iter = 0
    ckpt_path  = os.path.join(WORK_DIR, 'latest.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter']
        print(f'Resumed fine-tune from iter {start_iter}')

    # ── Train loop ───────────────────────────────────────────────────────────
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
            lf  = logits_up.float()
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
