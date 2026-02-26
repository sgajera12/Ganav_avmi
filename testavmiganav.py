import os
from mmseg.datasets import build_dataset
from mmcv import Config

print("="*60)
print("VERIFYING DATASET SETUP")
print("="*60)

# Load config
cfg = Config.fromfile('configs/ganav_avmi_finetune.py')

# Check train dataset
print("\n1. Loading training dataset...")
try:
    train_dataset = build_dataset(cfg.data.train)
    print(f"   ✓ Train samples: {len(train_dataset)}")
    print(f"   ✓ Classes: {train_dataset.CLASSES}")
    print(f"   ✓ Num classes: {len(train_dataset.CLASSES)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Check val dataset
print("\n2. Loading validation dataset...")
try:
    val_dataset = build_dataset(cfg.data.val)
    print(f"   ✓ Val samples: {len(val_dataset)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Check test dataset
print("\n3. Loading test dataset...")
try:
    test_dataset = build_dataset(cfg.data.test)
    print(f"   ✓ Test samples: {len(test_dataset)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Check class mapping
print("\n4. Checking class mapping...")
print(f"   Pixel Value -> Class Name")
for idx, cls_name in enumerate(train_dataset.CLASSES):
    print(f"   {idx} -> {cls_name}")

# Check a sample
print("\n5. Loading a sample...")
try:
    sample = train_dataset[0]
    print(f"   ✓ Image shape: {sample['img'].data.shape}")
    print(f"   ✓ Mask shape: {sample['gt_semantic_seg'].data.shape}")
    print(f"   ✓ Mask values: {sample['gt_semantic_seg'].data.unique()}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ ALL CHECKS PASSED - READY TO TRAIN!")
print("="*60)