"""
Classical (non-neural) prediction baseline for QFormer-RevMark.

Implements the canonical "rhombus" / Cross-Dot 4-neighbour prediction used in
Thodi & Rodriguez 2007 PEE and subsequent classical RDH literature (Sachnev
2009, Hosny 2024 quaternion-moment models -- all rest on this kind of local
linear prediction).

API mirrors QFormerPredictor:  takes RGB (B, 3, H, W) and returns predicted RGB.

The prediction at position (i, j) is the average of the four orthogonal
neighbours.  At borders, missing neighbours are dropped.

Used as a baseline to isolate the contribution of the deep quaternion
transformer predictor in our pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalRhombusPredictor(nn.Module):
    """4-neighbour rhombus predictor (no learnable parameters)."""

    def __init__(self, image_size: int = 256, patch_size: int = 16):
        super().__init__()
        # patch_size kept for compatibility with QFormerPredictor signature
        self.image_size = image_size
        self.patch_size = patch_size
        # Fixed 4-neighbour averaging kernel.
        kernel = torch.tensor([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]], dtype=torch.float32) / 4.0
        # (3, 1, 3, 3): one filter per channel, grouped conv
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, 3, 3).contiguous())

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.dim() != 4 or rgb.size(1) != 3:
            raise ValueError(f"expected (B,3,H,W), got {tuple(rgb.shape)}")
        # Border handling: replicate padding so border averages drop to a sane value
        padded = F.pad(rgb, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, self.kernel, groups=3)

    def count_parameters(self) -> dict:
        return {"total": 0, "trainable": 0}


# ============================================================
# Drop-in wrapper that uses ClassicalRhombusPredictor instead of QFormerPredictor
# but reuses Adaptive Q-PEE and the embed/extract pipeline.
# ============================================================
from ..pee.adaptive_pee import AdaptiveQuaternionPEE  # noqa: E402
from .qformer_revmark import QFormerRevMark  # noqa: E402
from .qformer_noshift_revmark import QFormerNoShiftRevMark  # noqa: E402


class ClassicalRevMark(QFormerRevMark):
    """QFormerRevMark with neural predictor replaced by 4-neighbour average."""

    def __init__(self, image_size: int = 256, patch_size: int = 16, init_T: float = 5.0):
        nn.Module.__init__(self)
        self.predictor = ClassicalRhombusPredictor(image_size=image_size, patch_size=patch_size)
        self.apee = AdaptiveQuaternionPEE(init_T=init_T)
        self.image_size = image_size


class ClassicalNoShiftRevMark(QFormerNoShiftRevMark):
    """QFormerNoShiftRevMark with neural predictor replaced by 4-neighbour average."""

    def __init__(self, image_size: int = 256, patch_size: int = 16, init_T: float = 5.0):
        nn.Module.__init__(self)
        self.predictor = ClassicalRhombusPredictor(image_size=image_size, patch_size=patch_size)
        self.apee = AdaptiveQuaternionPEE(init_T=init_T)
        self.image_size = image_size


def _self_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Standard pipeline
    H = W = 64
    img = torch.randint(0, 256, (1, 3, H, W), dtype=torch.int64, device=device)
    bits = torch.randint(0, 2, img.shape, dtype=torch.int64, device=device)

    model_std = ClassicalRevMark(image_size=H, patch_size=8, init_T=5.0).to(device).eval()
    er = model_std.embed(img, bits)
    xr = model_std.extract(er.watermarked_uint8)
    be_std = torch.equal(xr.recovered_uint8, img)
    print(f"  Classical standard PEE: bit-exact={be_std}, capacity={er.capacity_bpp:.4f} bpp")

    # No-shift pipeline
    payload = torch.randint(0, 2, (200,), dtype=torch.int64, device=device)
    model_ns = ClassicalNoShiftRevMark(image_size=H, patch_size=8, init_T=5.0).to(device).eval()
    er = model_ns.embed_noshift(img, payload, T_select=0)
    xr = model_ns.extract_noshift(er.watermarked_uint8, er.side_data)
    be_ns = torch.equal(xr.recovered_uint8, img)
    print(f"  Classical no-shift PEE: bit-exact={be_ns}, embedded={er.n_payload_bits}, PSNR={er.psnr_quick:.2f}")

    print("classical_predictor self-test: ALL PASS" if (be_std and be_ns) else "FAIL")


if __name__ == "__main__":
    _self_test()
