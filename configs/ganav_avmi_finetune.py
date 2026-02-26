_base_ = './ours/ganav_group6_rugd.py'

# Custom imports
custom_imports = dict(
    imports=['mmseg.datasets.avmi_dataset'],
    allow_failed_imports=False
)

# Dataset settings
dataset_type = 'AVMIDataset'
data_root = 'data/avmi_ugv/'

# Number of classes from your _classes.csv
num_classes = 6

# Update model for 5 classes
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='OursHeadClassAtt',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=384,
        mask_size=(97, 97),
        psa_type='bi-direction',
        compact=False,
        shrink_factor=2,
        normalization_factor=1.0,
        psa_softmax=True,
        dropout_ratio=0.1,
        num_classes=num_classes,
        input_transform='multiple_select',
        norm_cfg=norm_cfg,
        align_corners=False,
        attn_split=1,
        strides=(2, 1),
        size_index=1,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            static_weight=False)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=160,
            channels=32,
            num_convs=1,
            num_classes=num_classes,
            in_index=-2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=num_classes,
            in_index=-3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Image settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = (688, 550)
crop_size = (300, 375)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRGBAnnotations'),
    dict(type='Resize', img_scale=img_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
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
            dict(type='Collect', keys=['img'])
        ])
]

# Data
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test',
        split='test.txt',
        pipeline=test_pipeline))

# Optimizer - Lower LR for fine-tuning
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict()

# Long fine-tuning run (resume from 25k checkpoint)
runner = dict(type='IterBasedRunner', max_iters=300000)
total_iters = 300000

# Save checkpoint every 25k iters (don't flood disk)
checkpoint_config = dict(by_epoch=False, interval=25000)

# Evaluate every 25k iters
evaluation = dict(interval=25000, metric='mIoU', pre_eval=True)

# Learning rate schedule — lower min_lr for longer run
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-05,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-06,
    by_epoch=False)

# Weights loaded via --resume-from; keep load_from for reference
load_from = 'trained_models/rugd_group6/ganav_rugd_6.pth'

# Logging
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
cudnn_benchmark = False

# Output directory
work_dir = './work_dirs/ganav_avmi_finetune'
gpu_ids = [0]
auto_resume = False