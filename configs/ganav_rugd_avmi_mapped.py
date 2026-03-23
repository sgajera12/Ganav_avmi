# Fine-tune on RUGD with classes REMAPPED to AVMI 6 visual classes.
# RUGD 24 classes → sky, tree, bush, ground, obstacle, rock
# Base: AVMI scratch model (same 6-class output head → all weights transfer).

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

# Fine-tune LR (lower than scratch 0.06)
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=4e-5)
lr_config = dict(
    policy='poly', power=0.9, min_lr=1e-5,
    warmup='linear', warmup_iters=500, warmup_ratio=1e-6, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=100000)
total_iters = 100000
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)

load_from = 'work_dirs/ganav_avmi_scratch/latest.pth'
work_dir  = './work_dirs/ganav_rugd_avmi_mapped'
