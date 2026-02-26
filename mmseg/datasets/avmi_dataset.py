import os.path as osp
import numpy as np
import mmcv
from mmseg.datasets.builder import DATASETS, PIPELINES
from mmseg.datasets.custom import CustomDataset


# Actual RGB colors used in the annotation masks.
# These are the true dominant colors found in the mask PNGs.
# Nearest-colour matching is used at load time, so JPEG-compressed
# boundary pixels are still assigned to the correct class.
COLOR_TO_CLASS = {
    (24,  102, 178): 0,  # sky    - blue
    (18,  182,  37): 1,  # tree   - green
    (239, 255,  15): 2,  # bush   - yellow
    (92,   19,   6): 3,  # ground - dark brown
    (255,  63, 250): 4,  # trunk  - pink/magenta
    (255,   0,   0): 5,  # rock   - red
}

# Pre-build lookup arrays once for fast nearest-colour matching
_PALETTE_COLORS = np.array(list(COLOR_TO_CLASS.keys()), dtype=np.int32)   # (N,3)
_PALETTE_LABELS = np.array(list(COLOR_TO_CLASS.values()), dtype=np.uint8)  # (N,)


def rgb_mask_to_index(rgb_img):
    """Convert an RGB colour-coded mask to a class index mask (uint8).

    Uses nearest-colour (L2) matching so JPEG-compressed boundary pixels
    are still assigned to the correct class rather than ignored.
    """
    h, w = rgb_img.shape[:2]
    flat = rgb_img.reshape(-1, 3).astype(np.int32)          # (H*W, 3)
    diff = flat[:, None, :] - _PALETTE_COLORS[None, :, :]   # (H*W, N, 3)
    dist = (diff ** 2).sum(axis=2)                           # (H*W, N)
    nearest = np.argmin(dist, axis=1)                        # (H*W,)
    seg = _PALETTE_LABELS[nearest].reshape(h, w)
    return seg


@PIPELINES.register_module()
class LoadRGBAnnotations:
    """Pipeline transform: load a colour-coded PNG mask and return class indices."""

    def __call__(self, results):
        filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        img = mmcv.imread(filename, channel_order='rgb')
        seg = rgb_mask_to_index(img)
        results['gt_semantic_seg'] = seg
        results['seg_fields'].append('gt_semantic_seg')
        return results


@DATASETS.register_module()
class AVMIDataset(CustomDataset):
    """AVMI UGV Dataset for terrain segmentation."""

    CLASSES = ('sky', 'tree', 'bush', 'ground', 'trunk', 'rock')

    # Visualization palette — matches the actual mask colours above
    PALETTE = [
        [24,  102, 178],  # 0: sky    - blue
        [18,  182,  37],  # 1: tree   - green
        [239, 255,  15],  # 2: bush   - yellow
        [92,   19,   6],  # 3: ground - dark brown
        [255,  63, 250],  # 4: trunk  - pink/magenta
        [255,   0,   0],  # 5: rock   - red
    ]

    def __init__(self, **kwargs):
        super(AVMIDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs
        )
        # Replace the default grayscale loader with our RGB loader so that
        # evaluation (get_gt_seg_map_by_idx) also goes through colour conversion.
        self.gt_seg_map_loader = LoadRGBAnnotations()
