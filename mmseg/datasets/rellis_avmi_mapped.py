import os.path as osp
import numpy as np
import mmcv
from .builder import DATASETS, PIPELINES
from .custom import CustomDataset

# ── RELLIS pixel value → AVMI class index ────────────────────────────────────
# RELLIS _orig.png uses SEQUENTIAL indices 0-19 matching the CLASSES list:
# 0=void, 1=dirt, 2=grass, 3=tree, 4=pole, 5=water, 6=sky,
# 7=vehicle, 8=object, 9=asphalt, 10=building, 11=log,
# 12=person, 13=fence, 14=bush, 15=concrete, 16=barrier,
# 17=puddle, 18=mud, 19=rubble
#
# AVMI 6 classes: 0=sky, 1=tree, 2=bush, 3=ground, 4=obstacle, 5=rock

# Build a 256-element lookup table
_RELLIS_TO_AVMI = np.full(256, 255, dtype=np.uint8)  # default = ignore
_RELLIS_TO_AVMI[0]  = 255   # void      → ignore
_RELLIS_TO_AVMI[1]  = 3     # dirt      → ground
_RELLIS_TO_AVMI[2]  = 3     # grass     → ground
_RELLIS_TO_AVMI[3]  = 1     # tree      → tree
_RELLIS_TO_AVMI[4]  = 4     # pole      → obstacle
_RELLIS_TO_AVMI[5]  = 5     # water     → rock (non-navigable)
_RELLIS_TO_AVMI[6]  = 0     # sky       → sky
_RELLIS_TO_AVMI[7]  = 4     # vehicle   → obstacle
_RELLIS_TO_AVMI[8]  = 4     # object    → obstacle
_RELLIS_TO_AVMI[9]  = 3     # asphalt   → ground
_RELLIS_TO_AVMI[10] = 4     # building  → obstacle
_RELLIS_TO_AVMI[11] = 4     # log       → obstacle
_RELLIS_TO_AVMI[12] = 4     # person    → obstacle
_RELLIS_TO_AVMI[13] = 4     # fence     → obstacle
_RELLIS_TO_AVMI[14] = 2     # bush      → bush
_RELLIS_TO_AVMI[15] = 3     # concrete  → ground
_RELLIS_TO_AVMI[16] = 4     # barrier   → obstacle
_RELLIS_TO_AVMI[17] = 5     # puddle    → rock
_RELLIS_TO_AVMI[18] = 3     # mud       → ground
_RELLIS_TO_AVMI[19] = 5     # rubble    → rock


@PIPELINES.register_module()
class LoadRELLISMappedAnnotations:
    """Load RELLIS _orig.png annotation and remap to AVMI 6 classes."""

    def __call__(self, results):
        filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        img = mmcv.imread(filename, flag='grayscale')
        seg = _RELLIS_TO_AVMI[img]
        results['gt_semantic_seg'] = seg
        results['seg_fields'].append('gt_semantic_seg')
        return results


@DATASETS.register_module()
class RELLISDataset_AVMIMapped(CustomDataset):
    """RELLIS dataset with classes remapped to AVMI 6-class scheme.

    Original RELLIS 20 classes (non-sequential IDs) → AVMI: sky, tree, bush, ground, obstacle, rock
    Uses _orig.png annotations (original class indices).
    """

    CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')
    PALETTE = [
        [24,  102, 178], # 0: sky - blue
        [18,  182,  37], # 1: tree - green
        [239, 255,  15], # 2: bush - yellow
        [92,   19,   6], # 3: ground- dark brown
        [255,  63, 250], # 4: obstacle - pink
        [255,   0,   0], # 5: rock-red
    ]

    def __init__(self, **kwargs):
        super(RELLISDataset_AVMIMapped, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_orig.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadRELLISMappedAnnotations()
