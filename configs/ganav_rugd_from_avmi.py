# Fine-tune on RUGD (navigability classes) starting from AVMI scratch model.
# The architecture is identical (6 classes, mask_size=(97,97), crop_size=(300,375))
# so ALL weights transfer — backbone + decode head.
# The head re-learns navigability semantics instead of AVMI visual classes.

_base_ = [
    './_base_/models/ours_class_att.py',
    './_base_/datasets/rugd_group6.py',
    './_base_/default_runtime.py'
]

#  Optimizer — lower LR than scratch since backbone is already trained 
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

#  Training schedule 
runner = dict(type='IterBasedRunner', max_iters=100000)
total_iters = 100000
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)

#  Data 
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

#  Init from AVMI scratch 
# All weights load (same 6-class architecture). Head adapts to RUGD classes.
load_from = 'work_dirs/ganav_avmi_scratch/latest.pth'

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])

cudnn_benchmark = False
work_dir = './work_dirs/ganav_rugd_from_avmi'
gpu_ids = [0]
