# Joint training: AVMI (synthetic, all 6 classes) + RELLIS (real, fixed index)
#
# Why joint instead of fine-tune:
#   Fine-tuning on RELLIS forgets AVMI rock/stump knowledge (RELLIS has no rock/stump).
#   Joint training keeps AVMI samples in every epoch so rock/stump are never forgotten.
#
# Contribution by dataset:
#   AVMI  → rock, stump, ground (bright green UGV terrain), bush, sky, tree
#   RELLIS → sky, tree, bush, ground (real-world forest appearance)
#
# ConcatDataset samples uniformly from both — AVMI has fewer images so we
# repeat it to roughly balance the number of samples per epoch.

_base_ = './ganav_avmi_scratch.py'

custom_imports = dict(
    imports=[
        'mmseg.datasets.avmi_dataset',
        'mmseg.datasets.rellis_avmi_mapped',
    ],
    allow_failed_imports=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (300, 375)

# ── pipelines ────────────────────────────────────────────────────────────
avmi_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRGBAnnotations'),
    dict(type='Resize', img_scale=(688, 550), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

rellis_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRELLISMappedAnnotations'),
    dict(type='Resize', img_scale=(1920, 1200), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# ── datasets ─────────────────────────────────────────────────────────────
# Repeat AVMI 4x to balance sample counts against RELLIS (3302 train images)
avmi_train = dict(
    type='RepeatDataset',
    times=4,
    dataset=dict(
        type='AVMIDataset',
        data_root='data/avmi_ugv/',
        img_dir='images/train',
        ann_dir='annotations/train',
        split='train.txt',
        pipeline=avmi_train_pipeline))

rellis_train = dict(
    type='RELLISDataset_AVMIMapped',
    data_root='data/rellis/',
    img_dir='image',
    ann_dir='annotation',
    split='train.txt',
    pipeline=rellis_train_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[avmi_train, rellis_train],
    # Validate on RELLIS (real-world performance is what we care about)
    val=dict(
        type='RELLISDataset_AVMIMapped',
        data_root='data/rellis/',
        img_dir='image',
        ann_dir='annotation',
        split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type='RELLISDataset_AVMIMapped',
        data_root='data/rellis/',
        img_dir='image',
        ann_dir='annotation',
        split='test.txt',
        pipeline=test_pipeline))

# Fine-tune LR (not scratch LR — we're starting from trained weights)
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=4e-5)
lr_config = dict(
    policy='poly', power=0.9, min_lr=1e-5,
    warmup='linear', warmup_iters=500, warmup_ratio=1e-6, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=100000)
total_iters = 100000
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)

load_from = 'work_dirs/ganav_avmi_scratch/latest.pth'
work_dir  = './work_dirs/ganav_avmi_rellis_joint'
