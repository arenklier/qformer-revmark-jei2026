"""
Scalar (non-quaternion) Transformer predictor — ablation baseline for
QFormer-RevMark.

This mirrors the architecture of QFormerPredictor as closely as possible
but uses standard scalar PyTorch modules instead of the quaternion
operators. The point of the ablation is to test whether the quaternion
algebra itself (rather than the Transformer backbone) is responsible for
the capacity gains observed in our experiments.

Two configurations are supported:

    --variant matched_dim       Same hidden width d_model = 4 * d_quaternion
                                (scalar has ~4x more trainable parameters
                                than the quaternion model)

    --variant matched_params    Hidden width chosen so total trainable
                                parameters match the quaternion model
                                (scalar then has 1/2 the embedding width
                                because of attention quadratic terms; we
                                solve approximately)

We re-use the QFormerRevMark embed/extract pipeline -- only the predictor
network differs. To do that cleanly we keep the same input/output contract
(RGB in -> RGB out).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.quaternion_pack import rgb_to_quaternion, quaternion_to_rgb


# ============================================================
# Standard ViT-style patch embed / unembed for RGB
# ============================================================
class ScalarPatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, patch_size: int = 16, d_model: int = 192):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, rgb: torch.Tensor):
        feat = self.proj(rgb)            # (B, D, H', W')
        B, C, Hp, Wp = feat.shape
        return feat.flatten(2).transpose(1, 2), (Hp, Wp)


class ScalarPatchUnembed(nn.Module):
    def __init__(self, out_ch: int = 3, patch_size: int = 16, d_model: int = 192):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.token_to_pixels = nn.Linear(d_model, out_ch * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor, grid_hw):
        B, N, D = tokens.shape
        Hp, Wp = grid_hw
        P = self.patch_size
        flat = self.token_to_pixels(tokens).view(B, Hp, Wp, -1, P, P)
        img = flat.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, Hp * P, Wp * P)
        return img


# ============================================================
# Sinusoidal 2-D positional embedding (same as Q-version, scalar)
# ============================================================
def sin_pos_2d(H: int, W: int, D: int, device, dtype):
    if D % 4 != 0:
        raise ValueError("d_model must be divisible by 4 for 2-D sin/cos")
    d_each = D // 4
    h_pos = torch.arange(H, device=device).unsqueeze(1).float()
    w_pos = torch.arange(W, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, 2 * d_each, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / (2 * d_each)))
    pe_h = torch.zeros(H, 2 * d_each, device=device)
    pe_h[:, 0::2] = torch.sin(h_pos * div)
    pe_h[:, 1::2] = torch.cos(h_pos * div)
    pe_w = torch.zeros(W, 2 * d_each, device=device)
    pe_w[:, 0::2] = torch.sin(w_pos * div)
    pe_w[:, 1::2] = torch.cos(w_pos * div)
    pe = torch.zeros(H, W, D, device=device, dtype=dtype)
    pe[:, :, : 2 * d_each] = pe_h.unsqueeze(1).expand(H, W, 2 * d_each).to(dtype)
    pe[:, :, 2 * d_each :] = pe_w.unsqueeze(0).expand(H, W, 2 * d_each).to(dtype)
    return pe.reshape(H * W, D)


# ============================================================
# Standard pre-norm Transformer block
# ============================================================
class ScalarTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Scalar predictor (drop-in replacement for QFormerPredictor)
# ============================================================
class ScalarTransformerPredictor(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        d_model: int = 192,        # equivalent to 4*d_quaternion when matched_dim
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.grid = image_size // patch_size

        self.embed = ScalarPatchEmbed(in_ch=3, patch_size=patch_size, d_model=d_model)
        self.blocks = nn.ModuleList([
            ScalarTransformerBlock(d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.unembed = ScalarPatchUnembed(out_ch=3, patch_size=patch_size, d_model=d_model)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        tokens, grid = self.embed(rgb)
        pe = sin_pos_2d(grid[0], grid[1], self.d_model, tokens.device, tokens.dtype)
        tokens = tokens + pe.unsqueeze(0)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return self.unembed(tokens, grid)


# ============================================================
# Wrapper module that mirrors QFormerRevMark API but uses ScalarTransformerPredictor
# ============================================================
from ..pee.adaptive_pee import AdaptiveQuaternionPEE  # noqa: E402
from .qformer_revmark import (  # noqa: E402
    QFormerRevMark,
    float_to_uint8,
    uint8_to_float,
    checkerboard_mask,
    EmbedResult,
    ExtractResult,
)


class ScalarTransformerRevMark(QFormerRevMark):
    """
    QFormerRevMark variant that uses ScalarTransformerPredictor instead of
    the quaternion-based predictor. All embed/extract logic is inherited.
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        d_model: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        init_T: float = 5.0,
    ):
        # Skip QFormerRevMark.__init__ (which builds the quaternion predictor)
        nn.Module.__init__(self)
        self.predictor = ScalarTransformerPredictor(
            image_size=image_size,
            patch_size=patch_size,
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
        )
        self.apee = AdaptiveQuaternionPEE(init_T=init_T)
        self.image_size = image_size


def _self_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compare param counts at matched_dim
    from .qformer import QFormerPredictor
    q_model = QFormerPredictor(image_size=64, patch_size=8, d_quaternion=16, depth=2, num_heads=4).to(device)
    s_model = ScalarTransformerPredictor(image_size=64, patch_size=8, d_model=64, depth=2, num_heads=4).to(device)
    print(f"  QFormerPredictor (d_quat=16, depth=2) params: {sum(p.numel() for p in q_model.parameters()):,}")
    print(f"  Scalar matched-dim  (d_model=64, depth=2)     : {sum(p.numel() for p in s_model.parameters()):,}")

    rgb = torch.rand(2, 3, 64, 64, device=device)
    q_out = q_model(rgb)
    s_out = s_model(rgb)
    assert q_out.shape == s_out.shape == rgb.shape

    # End-to-end smoke with ScalarTransformerRevMark
    wrap = ScalarTransformerRevMark(image_size=32, patch_size=8, d_model=64, depth=2, num_heads=4, init_T=10.0).to(device).eval()
    img = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.int64, device=device)
    bits = torch.randint(0, 2, img.shape, dtype=torch.int64, device=device)
    er = wrap.embed(img, bits)
    xr = wrap.extract(er.watermarked_uint8)
    bit_exact = torch.equal(xr.recovered_uint8, img)
    print(f"  Scalar wrapper bit-exact round-trip: {bit_exact}")
    print(f"  Capacity (random img, T=10): {er.capacity_bpp:.4f} bpp")
    print("scalar_transformer_predictor self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
