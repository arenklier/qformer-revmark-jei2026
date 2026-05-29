"""
QFormer predictor: a quaternion transformer that maps RGB images to per-pixel
quaternion predictions. Used as the deep prediction network inside the
reversible-PEE watermarking pipeline.

Pipeline
--------
RGB (B, 3, H, W)
  -> rgb_to_quaternion -> (B, 4, H, W)
  -> QuaternionPatchEmbed -> (B, N, 4*d_quat) + pos
  -> N x QuaternionTransformerBlock
  -> QuaternionPatchUnembed -> (B, 4, H, W)
  -> quaternion_to_rgb -> (B, 3, H, W)  predicted RGB image
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..data.quaternion_pack import rgb_to_quaternion, quaternion_to_rgb
from .patch_embed import QuaternionPatchEmbed, QuaternionPatchUnembed, SinusoidalPosEmbed2D
from .quaternion_attention import QuaternionTransformerBlock
from .quaternion_ops import QuaternionLayerNorm


class QFormerPredictor(nn.Module):
    """
    Encoder-style QFormer that outputs a same-size RGB prediction.

    Args
    ----
    image_size   : input H = W
    patch_size   : square patch side (must divide image_size)
    d_quaternion : quaternion-feature dim per token (output is 4 * d)
    depth        : number of QuaternionTransformerBlock
    num_heads    : attention heads
    mlp_ratio    : FFN expansion factor
    dropout      : attn + FFN dropout
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        d_quaternion: int = 48,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_quaternion = d_quaternion
        self.grid_size = image_size // patch_size

        self.embed = QuaternionPatchEmbed(in_channels=1, patch_size=patch_size, d_quaternion=d_quaternion)
        self.pos = SinusoidalPosEmbed2D(d_model=4 * d_quaternion, max_h=self.grid_size, max_w=self.grid_size)
        self.blocks = nn.ModuleList([
            QuaternionTransformerBlock(
                d_quaternion=d_quaternion,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = QuaternionLayerNorm(d_quaternion)
        self.unembed = QuaternionPatchUnembed(out_channels=1, patch_size=patch_size, d_quaternion=d_quaternion)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb : (B, 3, H, W) tensor in [0, 1]
        returns predicted RGB image, same shape, in approximately [0, 1].
        """
        # Image size check relaxed: architecture supports arbitrary multiples of patch_size.
        h, w = rgb.size(-2), rgb.size(-1)
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"input H={h}, W={w} must be multiples of patch_size={self.patch_size}")
        q = rgb_to_quaternion(rgb)                       # (B, 4, H, W)
        tokens, grid = self.embed(q)                     # (B, N, 4*d)
        tokens = self.pos(tokens, grid)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        q_out = self.unembed(tokens, grid)               # (B, 4, H, W)
        return quaternion_to_rgb(q_out)                  # (B, 3, H, W)

    def count_parameters(self) -> dict:
        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": n_total, "trainable": n_trainable}


def _self_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = QFormerPredictor(
        image_size=64,         # small for quick test
        patch_size=8,
        d_quaternion=16,
        depth=2,
        num_heads=4,
    ).to(device)

    rgb = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
    pred = model(rgb)
    assert pred.shape == rgb.shape
    loss = ((pred - rgb) ** 2).mean()
    loss.backward()

    info = model.count_parameters()
    print("qformer self-test: ALL PASS")
    print(f"  Device: {device}")
    print(f"  Input/Output: {tuple(rgb.shape)}")
    print(f"  Initial MSE loss: {loss.item():.6f}")
    print(f"  Total params: {info['total']:,}  (trainable: {info['trainable']:,})")


if __name__ == "__main__":
    _self_test()
