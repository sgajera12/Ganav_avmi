# Fine-tune on RUGD with full class mapping + class-weighted loss.
# Uses same full mapping as ganav_rugd_avmi_mapped.py (all pixels labeled, no
# coverage gaps), but adds class weights to counteract ground dominance (68%).
#
# Pixel distribution in RUGD mapped: sky=2%, tree=0.2%, bush~0%,
#   ground=68%, obstacle=18%, rock=12%
# Class weights (inverse-sqrt frequency, capped):
#   sky=6, tree=15, bush=20, ground=0.3, obstacle=1.0, rock=1.5

_base_ = './ganav_avmi_scratch.py'

custom_imports = dict(
    imports=['mmseg.datasets.rugd_avmi_mapped'],
    allow_failed_imports=False)

dataset_type = 'RUGDDataset_AVMIMapped'
data_root    = 'data/rugd/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (300, 375)
img_size  = (688, 550)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRUGDMappedAnnotations'),
    dict(type='Resize', img_scale=img_size, ratio_range=(0.5, 2.0)),
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

# Class weights: [sky, tree, bush, ground, obstacle, rock]
# High weights for rare classes, low weights for dominant ground
_class_weight = [6.0, 15.0, 20.0, 0.3, 1.0, 1.5]

norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 6

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=_class_weight)),
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
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=0.4)),
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
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=0.4)),
    ])

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='RUGD_frames-with-annotations',
        ann_dir='RUGD_annotations',
        split='train_ours.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='RUGD_frames-with-annotations',
        ann_dir='RUGD_annotations',
        split='val_ours.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='RUGD_frames-with-annotations',
        ann_dir='RUGD_annotations',
        split='test_ours.txt',
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=4e-5)
lr_config = dict(
    policy='poly', power=0.9, min_lr=1e-5,
    warmup='linear', warmup_iters=500, warmup_ratio=1e-6, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=100000)
total_iters = 100000
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)

load_from = 'work_dirs/ganav_avmi_scratch/latest.pth'
work_dir  = './work_dirs/ganav_rugd_avmi_weighted'
