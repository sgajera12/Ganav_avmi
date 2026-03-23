import os.path as osp
import numpy as np
import mmcv
from .builder import DATASETS, PIPELINES
from .custom import CustomDataset

# ── RUGD original pixel value → AVMI class index ─────────────────────────────
# RUGD _orig.png pixel values: 0=void, 1=dirt, 2=sand, 3=grass, 4=tree,
# 5=pole, 6=water, 7=sky, 8=vehicle, 9=container, 10=asphalt, 11=gravel,
# 12=building, 13=mulch, 14=rock-bed, 15=log, 16=bicycle, 17=person,
# 18=fence, 19=bush, 20=sign, 21=rock, 22=bridge, 23=concrete, 24=picnic-table
#
# AVMI 6 classes: 0=sky, 1=tree, 2=bush, 3=ground, 4=obstacle, 5=rock

# Build a 256-element lookup table (covers all possible uint8 values)
_RUGD_TO_AVMI = np.full(256, 255, dtype=np.uint8)   # default = ignore
_RUGD_TO_AVMI[0]  = 255   # void          → ignore
_RUGD_TO_AVMI[1]  = 3     # dirt          → ground
_RUGD_TO_AVMI[2]  = 3     # sand          → ground
_RUGD_TO_AVMI[3]  = 3     # grass         → ground
_RUGD_TO_AVMI[4]  = 1     # tree          → tree
_RUGD_TO_AVMI[5]  = 4     # pole          → obstacle
_RUGD_TO_AVMI[6]  = 5     # water         → rock (non-navigable)
_RUGD_TO_AVMI[7]  = 0     # sky           → sky
_RUGD_TO_AVMI[8]  = 4     # vehicle       → obstacle
_RUGD_TO_AVMI[9]  = 4     # container     → obstacle
_RUGD_TO_AVMI[10] = 3     # asphalt       → ground
_RUGD_TO_AVMI[11] = 3     # gravel        → ground
_RUGD_TO_AVMI[12] = 4     # building      → obstacle
_RUGD_TO_AVMI[13] = 3     # mulch         → ground
_RUGD_TO_AVMI[14] = 5     # rock-bed      → rock
_RUGD_TO_AVMI[15] = 4     # log           → obstacle
_RUGD_TO_AVMI[16] = 4     # bicycle       → obstacle
_RUGD_TO_AVMI[17] = 4     # person        → obstacle
_RUGD_TO_AVMI[18] = 4     # fence         → obstacle
_RUGD_TO_AVMI[19] = 2     # bush          → bush
_RUGD_TO_AVMI[20] = 4     # sign          → obstacle
_RUGD_TO_AVMI[21] = 5     # rock          → rock
_RUGD_TO_AVMI[22] = 4     # bridge        → obstacle
_RUGD_TO_AVMI[23] = 3     # concrete      → ground
_RUGD_TO_AVMI[24] = 4     # picnic-table  → obstacle


@PIPELINES.register_module()
class LoadRUGDMappedAnnotations:
    """Load RUGD _orig.png annotation and remap to AVMI 6 classes."""

    def __call__(self, results):
        filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        img = mmcv.imread(filename, flag='grayscale')
        seg = _RUGD_TO_AVMI[img]
        results['gt_semantic_seg'] = seg
        results['seg_fields'].append('gt_semantic_seg')
        return results


@DATASETS.register_module()
class RUGDDataset_AVMIMapped(CustomDataset):
    """RUGD dataset with classes remapped to AVMI 6-class scheme.

    Original RUGD 24 classes → AVMI: sky, tree, bush, ground, obstacle, rock
    Uses _orig.png annotations (original class indices).
    """

    CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')
    PALETTE = [
        [24,  102, 178],  # 0: sky    - blue
        [18,  182,  37],  # 1: tree   - green
        [239, 255,  15],  # 2: bush   - yellow
        [92,   19,   6],  # 3: ground - dark brown
        [255,  63, 250],  # 4: obstacle  - pink
        [255,   0,   0],  # 5: rock   - red
    ]

    def __init__(self, **kwargs):
        super(RUGDDataset_AVMIMapped, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_orig.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadRUGDMappedAnnotations()
