import os.path as osp
import numpy as np
import mmcv
from .builder import DATASETS, PIPELINES
from .custom import CustomDataset

# ── RUGD selective mapping → AVMI 6 visual classes ───────────────────────────
# Strategy: only map classes that VISUALLY match AVMI terrain classes.
# Everything else → 255 (ignore) so it doesn't pollute the loss.
#
# RUGD pixel values: 0=void, 1=dirt, 2=sand, 3=grass, 4=tree,
# 5=pole, 6=water, 7=sky, 8=vehicle, 9=container, 10=asphalt, 11=gravel,
# 12=building, 13=mulch, 14=rock-bed, 15=log, 16=bicycle, 17=person,
# 18=fence, 19=bush, 20=sign, 21=rock, 22=bridge, 23=concrete, 24=picnic-table
#
# AVMI 6 classes: 0=sky, 1=tree, 2=bush, 3=ground, 4=obstacle, 5=rock
#
# Kept classes (direct visual matches only):
#   sky(7)      → sky(0)
#   tree(4)     → tree(1)
#   bush(19)    → bush(2)
#   dirt(1)     → ground(3)   ← brown dirt == AVMI ground
#   gravel(11)  → ground(3)   ← loose gravel similar to UGV terrain
#   rock(21)    → rock(5)
#   rock-bed(14)→ rock(5)
#
# Ignored (→ 255):
#   sand, grass, asphalt, concrete, mulch  — differ too much from AVMI ground
#   pole, water, vehicle, container, building, log, bicycle,
#   person, fence, sign, bridge, picnic-table — not terrain classes

_RUGD_TO_AVMI_SEL = np.full(256, 255, dtype=np.uint8)
_RUGD_TO_AVMI_SEL[7]  = 0   # sky           → sky
_RUGD_TO_AVMI_SEL[4]  = 1   # tree          → tree
_RUGD_TO_AVMI_SEL[19] = 2   # bush          → bush
_RUGD_TO_AVMI_SEL[1]  = 3   # dirt          → ground
_RUGD_TO_AVMI_SEL[11] = 3   # gravel        → ground
_RUGD_TO_AVMI_SEL[21] = 5   # rock          → rock
_RUGD_TO_AVMI_SEL[14] = 5   # rock-bed      → rock


@PIPELINES.register_module()
class LoadRUGDSelectiveAnnotations:
    """Load RUGD _orig.png annotation and selectively remap to AVMI 6 classes.
    Non-matching classes are set to 255 (ignored in loss).
    """

    def __call__(self, results):
        filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        img = mmcv.imread(filename, flag='grayscale')
        seg = _RUGD_TO_AVMI_SEL[img]
        results['gt_semantic_seg'] = seg
        results['seg_fields'].append('gt_semantic_seg')
        return results


@DATASETS.register_module()
class RUGDDataset_AVMISelective(CustomDataset):
    """RUGD dataset with SELECTIVE class mapping to AVMI 6-class scheme.

    Only visually matching classes are kept (sky, tree, bush, dirt→ground,
    gravel→ground, rock, rock-bed→rock). All other classes are ignored (255)
    so they don't bias training toward ground.
    """

    CLASSES = ('sky', 'tree', 'bush', 'ground', 'obstacle', 'rock')
    PALETTE = [
        [24,  102, 178],  # 0: sky      - blue
        [18,  182,  37],  # 1: tree     - green
        [239, 255,  15],  # 2: bush     - yellow
        [92,   19,   6],  # 3: ground   - dark brown
        [255,  63, 250],  # 4: obstacle - pink
        [255,   0,   0],  # 5: rock     - red
    ]

    def __init__(self, **kwargs):
        super(RUGDDataset_AVMISelective, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_orig.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadRUGDSelectiveAnnotations()
