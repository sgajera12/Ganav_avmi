# Joint training: AVMI (synthetic, all 6 classes) + RUGD (sky+tree+rock only)
#
# Why only sky/tree/rock from RUGD:
#   RUGD grass (green) conflicts with UGV ground (also green) — mapping grass→ground
#   teaches the model "green = ground" which contradicts AVMI training.
#   RUGD dark dirt does not match UGV bright green grass — also confusing.
#   So we ONLY take from RUGD what is visually consistent:
#     sky  → same blue sky everywhere
#     tree → tree canopy recognizable across domains
#     rock → rocky surfaces look similar in RUGD and UGV
#
# Contribution by dataset:
#   AVMI → ground (bright green UGV grass), bush, stump, rock, sky, tree
#   RUGD → sky, tree, rock (real-world appearance for these 3 classes only)

_base_ = './ganav_avmi_scratch.py'

custom_imports = dict(
    imports=[
        'mmseg.datasets.avmi_dataset',
        'mmseg.datasets.rugd_avmi_skytreerock',
    ],
    allow_failed_imports=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (300, 375)

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

rugd_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadRUGDSkyTreeRockAnnotations'),
    dict(type='Resize', img_scale=(688, 550), ratio_range=(0.5, 2.0)),
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

# Repeat AVMI 4x to balance against RUGD (4748 train images)
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

rugd_train = dict(
    type='RUGDDataset_SkyTreeRock',
    data_root='data/rugd/',
    img_dir='RUGD_frames-with-annotations',
    ann_dir='RUGD_annotations',
    split='train_ours.txt',
    pipeline=rugd_train_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[avmi_train, rugd_train],
    # Validate on RUGD (rock/tree/sky classes only)
    val=dict(
        type='RUGDDataset_SkyTreeRock',
        data_root='data/rugd/',
        img_dir='RUGD_frames-with-annotations',
        ann_dir='RUGD_annotations',
        split='val_ours.txt',
        pipeline=test_pipeline),
    test=dict(
        type='RUGDDataset_SkyTreeRock',
        data_root='data/rugd/',
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
work_dir  = './work_dirs/ganav_avmi_rugd_joint'
