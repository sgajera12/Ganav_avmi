"""
Export ganav_avmi_scratch model to ONNX format.
Run once in ganav env (CPU only — no cuDNN needed for export):

    conda run -n ganav python tools/export_onnx.py

Output: work_dirs/ganav_avmi_scratch/model.onnx
"""

import os, sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmseg.apis import init_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm

MODEL_CFG  = 'work_dirs/ganav_avmi_scratch/ganav_avmi_scratch.py'
MODEL_CKPT = 'work_dirs/ganav_avmi_scratch/latest.pth'
OUT_PATH   = 'work_dirs/ganav_avmi_scratch/model.onnx'

# ── Load model ────────────────────────────────────────────────────────────────
print('Loading model on CPU...')
model = init_segmentor(MODEL_CFG, MODEL_CKPT, device='cpu')
model = revert_sync_batchnorm(model)
model.eval()
print('Model loaded.')

# ── Find exact input tensor shape via hook on extract_feat ───────────────────
# PSA head has a fixed spatial size baked in — must use the exact size
# that the test pipeline produces, not the raw image size.
captured = {}
def _feat_hook(module, inp, out):
    if 'shape' not in captured:
        captured['shape'] = inp[0].shape  # (1, 3, H, W)

hook = model.backbone.register_forward_hook(_feat_hook)
from mmseg.apis import inference_segmentor
sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
_ = inference_segmentor(model, sample_img)
hook.remove()

IMG_H, IMG_W = captured['shape'][2], captured['shape'][3]
print(f'Detected model input size: {IMG_H} x {IMG_W}')

# ── Wrapper: image tensor → seg logits ───────────────────────────────────────
class SegWrapper(torch.nn.Module):
    def __init__(self, model, h, w):
        super().__init__()
        self.model = model
        self.h = h
        self.w = w

    def forward(self, x):
        # x: (1, 3, H, W) — preprocessed, normalized
        feats = self.model.extract_feat(x)
        out = self.model.decode_head.forward(feats)
        # decode_head.forward returns (main_logits, aux_logits) tuple
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = torch.nn.functional.interpolate(
            out, size=(self.h, self.w),
            mode='bilinear', align_corners=False)
        return out  # (1, 6, H, W)

wrapper = SegWrapper(model, IMG_H, IMG_W)
wrapper.eval()

# ── Dummy input ───────────────────────────────────────────────────────────────
dummy = torch.zeros(1, 3, IMG_H, IMG_W)

# ── Export ────────────────────────────────────────────────────────────────────
print(f'Exporting to {OUT_PATH} ...')
torch.onnx.export(
    wrapper,
    dummy,
    OUT_PATH,
    opset_version=12,
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={
        'image':  {0: 'batch'},
        'logits': {0: 'batch'},
    },
    do_constant_folding=True,
)
print(f'Exported successfully → {OUT_PATH}')

# ── Quick verify ──────────────────────────────────────────────────────────────
import onnx
model_onnx = onnx.load(OUT_PATH)
onnx.checker.check_model(model_onnx)
print('ONNX model verified OK.')
print(f'Input:  {[d.dim_value for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]}')
print(f'Output: {[d.dim_value for d in model_onnx.graph.output[0].type.tensor_type.shape.dim]}')
