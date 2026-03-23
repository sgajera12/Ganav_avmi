import os.path as osp
import numpy as np
import mmcv
from .builder import DATASETS, PIPELINES
from .custom import CustomDataset

# ── RELLIS selective mapping → AVMI 6 visual classes ─────────────────────────
# Strategy: only map classes that VISUALLY match AVMI terrain classes.
# Everything else → 255 (ignore) so it doesn't pollute the loss.
#
# RELLIS _orig.png uses SEQUENTIAL indices 0-19:
# 0=void, 1=dirt, 2=grass, 3=tree, 4=pole, 5=water, 6=sky,
# 7=vehicle, 8=object, 9=asphalt, 10=building, 11=log,
# 12=person, 13=fence, 14=bush, 15=concrete, 16=barrier,
# 17=puddle, 18=mud, 19=rubble
#
# AVMI 6 classes: 0=sky, 1=tree, 2=bush, 3=ground, 4=obstacle, 5=rock
#
# Kept classes (direct visual matches only):
#   sky(6)     → sky(0)
#   tree(3)    → tree(1)
#   bush(14)   → bush(2)
#   dirt(1)    → ground(3)   ← brown dirt == AVMI ground
#   mud(18)    → ground(3)   ← wet dirt, same visual class
#   rubble(19) → rock(5)     ← broken rocks, similar to AVMI rock
#
# Ignored (→ 255):
#   grass, asphalt, concrete  — differ from AVMI brown dirt ground
#   pole, water, vehicle, object, building, log, person,
#   fence, barrier, puddle    — not terrain classes or too ambiguous

_RELLIS_TO_AVMI_SEL = np.full(256, 255, dtype=np.uint8)
_RELLIS_TO_AVMI_SEL[6]  = 0   # sky     → sky
_RELLIS_TO_AVMI_SEL[3]  = 1   # tree    → tree
_RELLIS_TO_AVMI_SEL[14] = 2   # bush    → bush
_RELLIS_TO_AVMI_SEL[1]  = 3   # dirt    → ground
_RELLIS_TO_AVMI_SEL[18] = 3   # mud     → ground
_RELLIS_TO_AVMI_SEL[19] = 5   # rubble  → rock


@PIPELINES.register_module()
class LoadRELLISSelectiveAnnotations:
    """Load RELLIS _orig.png annotation and selectively remap to AVMI 6 classes.
    Non-matching classes are set to 255 (ignored in loss).
    """

    def __call__(self, results):
        filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        img = mmcv.imread(filename, flag='grayscale')
        seg = _RELLIS_TO_AVMI_SEL[img]
        results['gt_semantic_seg'] = seg
        results['seg_fields'].append('gt_semantic_seg')
        return results


@DATASETS.register_module()
class RELLISDataset_AVMISelective(CustomDataset):
    """RELLIS dataset with SELECTIVE class mapping to AVMI 6-class scheme.

    Only visually matching classes are kept (sky, tree, bush, dirt→ground,
    mud→ground, rubble→rock). All other classes are ignored (255) so they
    don't bias training toward ground.
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
        super(RELLISDataset_AVMISelective, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_orig.png',
            reduce_zero_label=False,
            **kwargs)
        self.gt_seg_map_loader = LoadRELLISSelectiveAnnotations()
