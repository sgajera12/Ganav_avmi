import os.path as osp
import numpy as np
import mmcv
from .builder import DATASETS, PIPELINES
from .custom import CustomDataset

# ── RUGD ultra-selective: sky + tree + rock only ──────────────────────────
# Only map classes whose visual appearance is consistent with UGV images.
# Ground/grass are IGNORED to avoid the green-confusion problem where RUGD
# green grass conflicts with AVMI green ground/trees.
#
# RUGD pixel values: 0=void, 1=dirt, 2=sand, 3=grass, 4=tree, 5=pole,
# 6=water, 7=sky, 8=vehicle, 9=container, 10=asphalt, 11=gravel,
# 12=building, 13=mulch, 14=rock-bed, 15=log, 16=bicycle, 17=person,
# 18=fence, 19=bush, 20=sign, 21=rock, 22=bridge, 23=concrete, 24=picnic-table
#
# Mapped (3 classes only):
#   sky(7)       → sky(0)   — blue sky looks the same everywhere
#   tree(4)      → tree(1)  — canopy recognizable across domains
#   rock(21)     → rock(5)  — rocky surfaces visually consistent
#   rock-bed(14) → rock(5)  — rocky terrain, consistent
#
# Everything else → 255 (ignore)
#   Ground/grass are ignored intentionally: RUGD dark dirt/green grass does
#   NOT match UGV bright green grass — mapping these causes the model to
#   associate green pixels with ground, conflicting with AVMI training.

_RUGD_TO_AVMI_STR = np.full(256, 255, dtype=np.uint8)
_RUGD_TO_AVMI_STR[7]  = 0   # sky      → sky
_RUGD_TO_AVMI_STR[4]  = 1   # tree     → tree
_RUGD_TO_AVMI_STR[21] = 5   # rock     → rock
_RUGD_TO_AVMI_STR[14] = 5   # rock-bed → rock


@PIPELINES.register_module()
class LoadRUGDSkyTreeRockAnnotations:
    """Load RUGD _orig.png and map only sky/tree/rock to AVMI classes.
    All other classes (including grass and dirt) are set to 255 (ignored).
    """
    def __call__(self, results):
        filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        img = mmcv.imread(filename, flag='grayscale')
        seg = _RUGD_TO_AVMI_STR[img]
        results['gt_semantic_seg'] = seg
        results['seg_fields'].append('gt_semantic_seg')
        return results


@DATASETS.register_module()
class RUGDDataset_SkyTreeRock(CustomDataset):
    """RUGD with only sky/tree/rock mapped to AVMI classes.
    Used for joint training with AVMI to contribute real-world rock/tree/sky
    appearance without polluting ground/bush/obstacle learning.
    """
    CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')
    PALETTE = [
        [24,  102, 178],
        [18,  182,  37],
        [239, 255,  15],
        [92,   19,   6],
        [255,  63, 250],
        [255,   0,   0],
    ]

    def __init__(self, **kwargs):
        super(RUGDDataset_SkyTreeRock, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_orig.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadRUGDSkyTreeRockAnnotations()
