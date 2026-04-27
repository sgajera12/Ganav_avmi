"""
train_joint_8class.py

Trains SegFormer-B2 from scratch on AVMI + RELLIS + RUGD jointly with 8 classes:
  0=sky  1=tree  2=bush  3=ground  4=obstacle  5=rock  6=water  7=concrete

--mode opt1  AVMI batches get 2× loss scale  (prioritise real-world domain)
--mode opt2  per-dataset class weights        (AVMI: rock/obstacle high;
                                               RELLIS/RUGD: water/concrete high)

Key ideas:
  - AVMI has no water/concrete → those classes are never the GT label for AVMI
    pixels, so the model naturally learns not to predict them on AVMI imagery.
  - RELLIS / RUGD supply water (class 6) and concrete (class 7) supervision.
  - Datasets alternate each iteration (1/3 AVMI, 1/3 RELLIS, 1/3 RUGD).

Run:
    python tools/train_joint_8class.py --mode opt1
    python tools/train_joint_8class.py --mode opt2
"""

import os, random, argparse
from datetime import datetime
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from transformers import (SegformerForSemanticSegmentation,
                          SegformerConfig, SegformerImageProcessor)

# ── Config ────────────────────────────────────────────────────────────────────
_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED   = 'nvidia/mit-b2'
NUM_CLASSES  = 8
CROP_SIZE    = 512
BATCH_SIZE   = 8
MAX_ITERS    = 60000
VAL_INTERVAL = 4000
LR_ENCODER   = 3e-5
LR_DECODER   = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_ITERS = 1500

CLASSES = ('sky','tree','bush','ground','obstacle','rock','water','concrete')

AVMI_ROOT   = os.path.join(_REPO_ROOT, 'data', 'avmi_ugv')
RELLIS_ROOT = os.path.join(_REPO_ROOT, 'data', 'rellis')
RUGD_ROOT   = os.path.join(_REPO_ROOT, 'data', 'rugd')

# ── Per-dataset class weights ──────────────────────────────────────────────────
# opt1 uses the same weights for all datasets; only the batch-level scale differs
_W_SHARED = torch.tensor([0.5, 1.0, 1.5, 0.8, 2.5, 3.0, 2.0, 1.5])

# opt2: AVMI emphasises the 6 existing classes (water/concrete irrelevant but set low)
#       RELLIS/RUGD emphasise water and concrete (the new additions)
_W_AVMI   = torch.tensor([0.5, 1.0, 1.5, 0.8, 2.5, 3.0, 0.5, 0.5])
_W_RELLIS = torch.tensor([0.5, 1.0, 1.5, 0.8, 2.0, 2.5, 2.5, 2.0])
_W_RUGD   = torch.tensor([0.5, 1.0, 1.5, 0.8, 2.5, 3.0, 2.0, 2.0])

# ── AVMI palette → 6-class index ──────────────────────────────────────────────
_AVMI_PALETTE = np.array([
    [24,  102, 178], [18,  182,  37], [239, 255,  15],
    [92,   19,   6], [255,  63, 250], [255,   0,   0],
], dtype=np.int32)

def avmi_rgb_to_index(rgb):
    flat = rgb.reshape(-1, 3).astype(np.int32)
    diff = flat[:, None, :] - _AVMI_PALETTE[None, :, :]
    return np.argmin((diff**2).sum(2), axis=1).reshape(rgb.shape[:2]).astype(np.uint8)

# ── RUGD 8-class mapping ──────────────────────────────────────────────────────
_RUGD_MAP = np.full(256, 255, dtype=np.uint8)
_RUGD_MAP[1]=3; _RUGD_MAP[2]=3;  _RUGD_MAP[3]=3;  _RUGD_MAP[4]=1
_RUGD_MAP[5]=4; _RUGD_MAP[6]=6;  _RUGD_MAP[7]=0;  _RUGD_MAP[8]=4   # water→6
_RUGD_MAP[9]=4; _RUGD_MAP[10]=3; _RUGD_MAP[11]=3; _RUGD_MAP[12]=4
_RUGD_MAP[13]=3; _RUGD_MAP[14]=5; _RUGD_MAP[15]=4; _RUGD_MAP[16]=4
_RUGD_MAP[17]=4; _RUGD_MAP[18]=4; _RUGD_MAP[19]=2; _RUGD_MAP[20]=4
_RUGD_MAP[21]=5; _RUGD_MAP[22]=4; _RUGD_MAP[23]=7; _RUGD_MAP[24]=4  # concrete→7

# ── RELLIS 8-class mapping ────────────────────────────────────────────────────
_RELLIS_MAP = np.full(256, 255, dtype=np.uint8)
_RELLIS_MAP[1]=3;  _RELLIS_MAP[3]=3;  _RELLIS_MAP[4]=1;  _RELLIS_MAP[5]=4
_RELLIS_MAP[6]=6;  _RELLIS_MAP[7]=0;  _RELLIS_MAP[8]=4;  _RELLIS_MAP[9]=4  # water→6
_RELLIS_MAP[10]=3; _RELLIS_MAP[12]=4; _RELLIS_MAP[15]=4; _RELLIS_MAP[17]=4
_RELLIS_MAP[18]=4; _RELLIS_MAP[19]=2; _RELLIS_MAP[23]=7; _RELLIS_MAP[27]=4  # concrete→7
_RELLIS_MAP[29]=6; _RELLIS_MAP[31]=3; _RELLIS_MAP[33]=5                     # puddle→6

# ── Shared processor ──────────────────────────────────────────────────────────
_PROCESSOR = SegformerImageProcessor(
    do_resize=True, size={'height': 480, 'width': 640},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225])

def augment(img, ann):
    scale  = random.uniform(0.5, 2.0)
    new_w  = max(CROP_SIZE, int(img.width  * scale))
    new_h  = max(CROP_SIZE, int(img.height * scale))
    img    = img.resize((new_w, new_h), Image.BILINEAR)
    ann    = np.array(Image.fromarray(ann).resize((new_w, new_h), Image.NEAREST))
    top    = random.randint(0, new_h - CROP_SIZE)
    left   = random.randint(0, new_w - CROP_SIZE)
    img    = img.crop((left, top, left+CROP_SIZE, top+CROP_SIZE))
    ann    = ann[top:top+CROP_SIZE, left:left+CROP_SIZE]
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        ann = ann[:, ::-1].copy()
    return img, ann

def encode(img, ann):
    pv  = _PROCESSOR(images=img, return_tensors='pt')['pixel_values'].squeeze(0)
    ann = Image.fromarray(ann).resize((pv.shape[2], pv.shape[1]), Image.NEAREST)
    return pv, torch.from_numpy(np.array(ann)).long()

# ── Datasets ──────────────────────────────────────────────────────────────────
class AVMIDataset(Dataset):
    def __init__(self, split='train'):
        self.img_dir = Path(AVMI_ROOT) / 'images' / split
        self.ann_dir = Path(AVMI_ROOT) / 'annotations' / split
        self.files   = sorted([f.stem for f in self.img_dir.glob('*.png')])
        self.train   = (split == 'train')

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img  = Image.open(self.img_dir / f'{name}.png').convert('RGB')
        ann  = avmi_rgb_to_index(np.array(
               Image.open(self.ann_dir / f'{name}.png').convert('RGB')))
        if self.train:
            img, ann = augment(img, ann)
        pv, lbl = encode(img, ann)
        return {'pixel_values': pv, 'labels': lbl}


class RUGDDataset(Dataset):
    def __init__(self, split='train'):
        txt = 'train_ours.txt' if split == 'train' else 'val_ours.txt'
        with open(os.path.join(RUGD_ROOT, txt)) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        self.img_root = os.path.join(RUGD_ROOT, 'RUGD_frames-with-annotations')
        self.ann_root = os.path.join(RUGD_ROOT, 'RUGD_annotations')
        self.train    = (split == 'train')

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        seq, name = self.ids[idx].split('/')
        img = Image.open(os.path.join(self.img_root, seq, name+'.png')).convert('RGB')
        ann = _RUGD_MAP[np.array(Image.open(
              os.path.join(self.ann_root, seq, name+'_orig.png')))]
        if self.train:
            img, ann = augment(img, ann)
        pv, lbl = encode(img, ann)
        return {'pixel_values': pv, 'labels': lbl}


class RELLISDataset(Dataset):
    def __init__(self, split='train'):
        txt = 'train.txt' if split == 'train' else 'val.txt'
        with open(os.path.join(RELLIS_ROOT, txt)) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        self.img_root = os.path.join(RELLIS_ROOT, 'image')
        self.ann_root = os.path.join(RELLIS_ROOT, 'annotation')
        self.train    = (split == 'train')

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        seq, name = self.ids[idx].split('/')
        img_path = os.path.join(self.img_root, seq, name+'.jpg')
        if not os.path.exists(img_path):
            img_path = img_path.replace('.jpg', '.png')
        img = Image.open(img_path).convert('RGB')
        ann = _RELLIS_MAP[np.array(Image.open(
              os.path.join(self.ann_root, seq, name+'.png')))]
        if self.train:
            img, ann = augment(img, ann)
        pv, lbl = encode(img, ann)
        return {'pixel_values': pv, 'labels': lbl}

# ── Loss ──────────────────────────────────────────────────────────────────────
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    oh    = torch.zeros_like(probs)
    oh.scatter_(1, targets.unsqueeze(1).clamp(0, NUM_CLASSES-1), 1)
    inter = (probs * oh).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
    return (1 - (2*inter+eps)/(union+eps)).mean()

# ── Validation ────────────────────────────────────────────────────────────────
def validate(model, loader, device, label):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            pv  = batch['pixel_values'].to(device)
            lbl = batch['labels']
            with torch.autocast('cuda'):
                up = nn.functional.interpolate(
                    model(pixel_values=pv).logits,
                    size=lbl.shape[-2:], mode='bilinear', align_corners=False)
            pred = up.argmax(1).cpu()
            for p, g in zip(pred.view(-1), lbl.view(-1)):
                if 0 <= g < NUM_CLASSES:
                    conf[g, p] += 1
    ious = []
    for i in range(NUM_CLASSES):
        tp = conf[i,i].item(); fp = conf[:,i].sum().item()-tp; fn = conf[i,:].sum().item()-tp
        d  = tp+fp+fn
        if d > 0: ious.append(tp/d)
    miou = sum(ious)/len(ious) if ious else 0.0
    print(f'  [{label}] Per-class: {[f"{CLASSES[i]}={ious[i]:.3f}" for i in range(len(ious))]}')
    print(f'  [{label}] mIoU = {miou:.4f}')
    return miou

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['opt1','opt2'], required=True,
                        help='opt1=dataset-level loss scaling | opt2=per-dataset class weights')
    args = parser.parse_args()

    WORK_DIR = os.path.join(_REPO_ROOT, 'work_dirs', f'segformer_b2_joint_8class_{args.mode}')
    os.makedirs(WORK_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  Mode: {args.mode}')

    # Model (8 classes, from ImageNet pretrained MiT-B2)
    cfg = SegformerConfig.from_pretrained(PRETRAINED)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label   = {i: c for i, c in enumerate(CLASSES)}
    cfg.label2id   = {c: i for i, c in enumerate(CLASSES)}
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, config=cfg, ignore_mismatched_sizes=True).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    # Optimizer
    enc_params = [p for n, p in model.named_parameters() if 'segformer' in n]
    dec_params = [p for n, p in model.named_parameters() if 'decode_head' in n]
    optimizer  = torch.optim.AdamW(
        [{'params': enc_params, 'lr': LR_ENCODER},
         {'params': dec_params, 'lr': LR_DECODER}], weight_decay=WEIGHT_DECAY)

    def lr_lambda(s):
        if s < WARMUP_ITERS: return s / WARMUP_ITERS
        return max(0.0, 1.0 - (s-WARMUP_ITERS)/(MAX_ITERS-WARMUP_ITERS))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Datasets — alternating loaders (equal representation from all three)
    def make_loader(ds, shuffle=True):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=4, pin_memory=True, drop_last=True)

    avmi_loader   = make_loader(AVMIDataset('train'))
    rellis_loader = make_loader(RELLISDataset('train'))
    rugd_loader   = make_loader(RUGDDataset('train'))

    avmi_val   = make_loader(AVMIDataset('val'),   shuffle=False)
    rellis_val = make_loader(RELLISDataset('val'), shuffle=False)
    rugd_val   = make_loader(RUGDDataset('val'),   shuffle=False)

    print(f'AVMI: {len(avmi_loader.dataset)} | RELLIS: {len(rellis_loader.dataset)} | RUGD: {len(rugd_loader.dataset)}')

    # Loss functions
    if args.mode == 'opt1':
        # Single shared class weights; batch-level scale handles AVMI priority
        ce_fn = {s: nn.CrossEntropyLoss(weight=_W_SHARED.to(device), ignore_index=255)
                 for s in ('avmi','rellis','rugd')}
        AVMI_SCALE = 2.0   # AVMI loss ×2
    else:  # opt2
        # Per-dataset class weights
        ce_fn = {
            'avmi':   nn.CrossEntropyLoss(weight=_W_AVMI.to(device),   ignore_index=255),
            'rellis': nn.CrossEntropyLoss(weight=_W_RELLIS.to(device), ignore_index=255),
            'rugd':   nn.CrossEntropyLoss(weight=_W_RUGD.to(device),   ignore_index=255),
        }
        AVMI_SCALE = 1.0   # no extra scaling; class weights handle prioritisation

    scaler = GradScaler()

    # Resume
    start_iter = 0
    ckpt_path  = os.path.join(WORK_DIR, 'latest.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter']
        print(f'Resumed from iter {start_iter}')

    # Infinite iterators
    def inf(loader):
        while True:
            yield from loader

    avmi_it   = inf(avmi_loader)
    rellis_it = inf(rellis_loader)
    rugd_it   = inf(rugd_loader)
    sources   = ['avmi', 'rellis', 'rugd']

    model.train()
    log_file = open(os.path.join(WORK_DIR, 'train_log.txt'), 'a')

    for itr in range(start_iter, MAX_ITERS):
        src   = sources[itr % 3]          # cycle: AVMI → RELLIS → RUGD → …
        batch = next({'avmi': avmi_it, 'rellis': rellis_it, 'rugd': rugd_it}[src])

        pv  = batch['pixel_values'].to(device)
        lbl = batch['labels'].to(device)

        optimizer.zero_grad()
        with torch.autocast('cuda'):
            logits = nn.functional.interpolate(
                model(pixel_values=pv).logits,
                size=lbl.shape[-2:], mode='bilinear', align_corners=False)
            lf      = logits.float()
            l_ce    = ce_fn[src](lf, lbl)
            l_dice  = dice_loss(lf, lbl)
            scale   = AVMI_SCALE if src == 'avmi' else 1.0
            loss    = scale * (l_ce + 3.0 * l_dice)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if itr % 50 == 0:
            ts  = datetime.now().strftime('%H:%M:%S')
            lr  = optimizer.param_groups[0]['lr']
            msg = (f'[{ts}] [{itr:6d}/{MAX_ITERS}] [{src:6s}] '
                   f'loss={loss.item():.4f} ce={l_ce.item():.4f} '
                   f'dice={l_dice.item():.4f} lr={lr:.2e}')
            print(msg, flush=True)
            log_file.write(msg+'\n'); log_file.flush()

        if (itr+1) % VAL_INTERVAL == 0:
            avmi_miou   = validate(model, avmi_val,   device, 'AVMI')
            rellis_miou = validate(model, rellis_val, device, 'RELLIS')
            rugd_miou   = validate(model, rugd_val,   device, 'RUGD')
            mean_miou   = (avmi_miou + rellis_miou + rugd_miou) / 3
            msg = (f'[Val @ {itr+1}] AVMI={avmi_miou:.4f} '
                   f'RELLIS={rellis_miou:.4f} RUGD={rugd_miou:.4f} '
                   f'mean={mean_miou:.4f}')
            print(msg, flush=True)
            log_file.write(msg+'\n'); log_file.flush()
            torch.save({'iter': itr+1, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'miou_avmi': avmi_miou, 'miou': mean_miou},
                       ckpt_path)
            torch.save(model.state_dict(),
                       os.path.join(WORK_DIR, f'iter_{itr+1}_avmi{avmi_miou:.4f}.pth'))
            model.train()

    log_file.close()
    print('Training complete.')

if __name__ == '__main__':
    main()
