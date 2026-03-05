# Fine-tune on RELLIS (navigability classes) starting from AVMI scratch model.
# The base RELLIS config uses crop_size=(375,600) which breaks PSA mask_size=(97,97).
# We override to crop_size=(300,375) to match AVMI scratch architecture exactly,
# ensuring 100% weight transfer from backbone + decode head.

_base_ = [
    './_base_/models/ours_class_att.py',
    './_base_/datasets/rellis_group6.py',
    './_base_/default_runtime.py'
]

# ── Override RELLIS crop_size to match AVMI scratch (300,375) ─────────────────
# Default rellis_group6.py uses (375,600) which changes PSA feature map dims.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (300, 375)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1200), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size),
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# ── Optimizer — lower LR than scratch ────────────────────────────────────────
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,
    by_epoch=False)

# ── Training schedule ─────────────────────────────────────────────────────────
runner = dict(type='IterBasedRunner', max_iters=100000)
total_iters = 100000
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)

# ── Init from AVMI scratch ────────────────────────────────────────────────────
load_from = 'work_dirs/ganav_avmi_scratch/latest.pth'

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])

cudnn_benchmark = False
work_dir = './work_dirs/ganav_rellis_from_avmi'
gpu_ids = [0]
