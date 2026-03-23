# Fine-tune on RELLIS with CORRECTED class mapping (sequential indices).
# The previous RELLIS mapped model was completely wrong because it used
# non-sequential original IDs, but the annotations use sequential 0-19 indices.
# This config uses the fixed rellis_avmi_mapped.py.
#
# Corrected pixel distribution: sky=31%, tree=10%, bush=17%, ground=40%,
#   obstacle=0.4%, rock=0.8%  — naturally balanced, no class weighting needed.

_base_ = './ganav_avmi_scratch.py'

custom_imports = dict(
    imports=['mmseg.datasets.rellis_avmi_mapped'],
    allow_failed_imports=False)

dataset_type = 'RELLISDataset_AVMIMapped'
data_root    = 'data/rellis/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (300, 375)

train_pipeline = [
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='annotation',
        split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='annotation',
        split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='annotation',
        split='test.txt',
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
work_dir  = './work_dirs/ganav_rellis_avmi_fixed'
