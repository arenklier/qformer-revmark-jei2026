"""
Quaternion patch embedding and inverse for QFormer-RevMark.

Pipeline
--------
RGB image (B, 3, H, W)
    -> rgb_to_quaternion: (B, 4, H, W)              [r=0, i=R, j=G, k=B]
    -> QuaternionPatchEmbed: (B, N, 4*d_quat)       N = (H/P)*(W/P) tokens
        ...QFormer blocks...
    -> QuaternionPatchUnembed: (B, 4, H, W)
    -> quaternion_to_rgb: (B, 3, H, W)              predicted RGB

A quaternion patch embedding is a 2-D convolution with stride=patch_size whose
weights form a Hamilton-product transform on the 4-component channel axis.

We implement this as four convolutions (one per quaternion component) combined
according to the Hamilton product formula -- analogous to QuaternionLinear but
in conv form.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Quaternion 2-D convolution
# ============================================================
class QuaternionConv2d(nn.Module):
    """
    Quaternion 2-D convolution.

    Input:  (B, 4 * in_channels, H, W)
    Output: (B, 4 * out_channels, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Four real conv weights
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight_r = nn.Parameter(torch.empty(*shape))
        self.weight_i = nn.Parameter(torch.empty(*shape))
        self.weight_j = nn.Parameter(torch.empty(*shape))
        self.weight_k = nn.Parameter(torch.empty(*shape))

        if bias:
            self.bias = nn.Parameter(torch.zeros(4 * out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot-style init scaled by 1/sqrt(2*fan_in) so quaternion gain matches.
        fan_in = self.in_channels * (self.kernel_size ** 2)
        std = 1.0 / math.sqrt(2.0 * fan_in)
        for w in (self.weight_r, self.weight_i, self.weight_j, self.weight_k):
            nn.init.normal_(w, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 4 * self.in_channels:
            raise ValueError(
                f"expected channel dim {4 * self.in_channels}, got {x.size(1)}"
            )
        r, i, j, k = torch.chunk(x, 4, dim=1)

        out_r = (F.conv2d(r, self.weight_r, stride=self.stride, padding=self.padding)
               - F.conv2d(i, self.weight_i, stride=self.stride, padding=self.padding)
               - F.conv2d(j, self.weight_j, stride=self.stride, padding=self.padding)
               - F.conv2d(k, self.weight_k, stride=self.stride, padding=self.padding))
        out_i = (F.conv2d(r, self.weight_i, stride=self.stride, padding=self.padding)
               + F.conv2d(i, self.weight_r, stride=self.stride, padding=self.padding)
               + F.conv2d(j, self.weight_k, stride=self.stride, padding=self.padding)
               - F.conv2d(k, self.weight_j, stride=self.stride, padding=self.padding))
        out_j = (F.conv2d(r, self.weight_j, stride=self.stride, padding=self.padding)
               - F.conv2d(i, self.weight_k, stride=self.stride, padding=self.padding)
               + F.conv2d(j, self.weight_r, stride=self.stride, padding=self.padding)
               + F.conv2d(k, self.weight_i, stride=self.stride, padding=self.padding))
        out_k = (F.conv2d(r, self.weight_k, stride=self.stride, padding=self.padding)
               + F.conv2d(i, self.weight_j, stride=self.stride, padding=self.padding)
               - F.conv2d(j, self.weight_i, stride=self.stride, padding=self.padding)
               + F.conv2d(k, self.weight_r, stride=self.stride, padding=self.padding))

        out = torch.cat([out_r, out_i, out_j, out_k], dim=1)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# ============================================================
# Patch embedding (image -> token sequence)
# ============================================================
class QuaternionPatchEmbed(nn.Module):
    """
    Tokenise a quaternion image via a single Hamilton-product convolution.

    Input:  (B, 4, H, W)        a "pure" quaternion image (real part zero)
    Output: (B, N, 4 * d_quat)  N = (H / patch) * (W / patch)
    """

    def __init__(self, in_channels: int = 1, patch_size: int = 16, d_quaternion: int = 32):
        super().__init__()
        if in_channels != 1:
            # in_channels is the number of quaternion-tuples in input (1 for pure quat image)
            raise ValueError("for RGB-as-pure-quaternion use in_channels=1")
        self.patch_size = patch_size
        self.d_quaternion = d_quaternion
        self.proj = QuaternionConv2d(
            in_channels=in_channels,
            out_channels=d_quaternion,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        x: (B, 4, H, W)
        returns:
            tokens (B, N, 4*d_quat)
            grid_hw (H', W') so the inverse can reshape correctly
        """
        if x.size(1) != 4:
            raise ValueError(f"expected (B,4,H,W), got {tuple(x.shape)}")
        feat = self.proj(x)                       # (B, 4*d_quat, H', W')
        B, C, Hp, Wp = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, 4*d_quat)
        return tokens, (Hp, Wp)


# ============================================================
# Patch un-embedding (token sequence -> image)
# ============================================================
class QuaternionPatchUnembed(nn.Module):
    """
    Inverse of QuaternionPatchEmbed via Hamilton-product transpose convolution.

    Input:  (B, N, 4 * d_quat),  grid_hw = (H', W')
    Output: (B, 4, H, W)
    """

    def __init__(self, out_channels: int = 1, patch_size: int = 16, d_quaternion: int = 32):
        super().__init__()
        if out_channels != 1:
            raise ValueError("for quaternion image output use out_channels=1")
        self.patch_size = patch_size
        self.d_quaternion = d_quaternion
        # Use a small head: Q-conv1x1 to expand channels, then PixelShuffle-like reshape.
        # Simpler & exact: a Q-Linear that maps each token to (4 * patch^2) pixels.
        self.token_to_pixels = nn.Linear(4 * d_quaternion, 4 * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        B, N, C = tokens.shape
        Hp, Wp = grid_hw
        assert N == Hp * Wp, f"token count {N} does not match grid {grid_hw}"
        P = self.patch_size
        # (B, N, 4*P*P) where the 4 quaternion components live in stride-of-PxP blocks.
        flat = self.token_to_pixels(tokens)                 # (B, N, 4*P*P)
        flat = flat.view(B, Hp, Wp, 4, P, P)                # (B, Hp, Wp, 4, P, P)
        # rearrange to (B, 4, Hp*P, Wp*P)
        img = flat.permute(0, 3, 1, 4, 2, 5).contiguous()   # (B, 4, Hp, P, Wp, P)
        img = img.view(B, 4, Hp * P, Wp * P)
        return img


# ============================================================
# Sinusoidal positional embedding for 2-D tokens
# ============================================================
class SinusoidalPosEmbed2D(nn.Module):
    """Fixed sinusoidal positional embedding for 2-D token grids."""

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(max_h, max_w, d_model)
        self.register_buffer("pe", pe, persistent=False)
        self.max_h = max_h
        self.max_w = max_w

    @staticmethod
    def _build_pe(H: int, W: int, D: int) -> torch.Tensor:
        if D % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2-D sin/cos")
        d_each = D // 4  # half for h sin/cos, half for w sin/cos
        h_pos = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        w_pos = torch.arange(W, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, 2 * d_each, 2, dtype=torch.float32) * (-math.log(10000.0) / (2 * d_each)))

        pe_h = torch.zeros(H, 2 * d_each)
        pe_h[:, 0::2] = torch.sin(h_pos * div)
        pe_h[:, 1::2] = torch.cos(h_pos * div)
        pe_w = torch.zeros(W, 2 * d_each)
        pe_w[:, 0::2] = torch.sin(w_pos * div)
        pe_w[:, 1::2] = torch.cos(w_pos * div)

        pe = torch.zeros(H, W, D)
        pe[:, :, : 2 * d_each] = pe_h.unsqueeze(1).expand(H, W, 2 * d_each)
        pe[:, :, 2 * d_each :] = pe_w.unsqueeze(0).expand(H, W, 2 * d_each)
        return pe.reshape(H * W, D)

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        Hp, Wp = grid_hw
        if Hp > self.max_h or Wp > self.max_w:
            # Rebuild on the fly for larger grids
            pe = self._build_pe(Hp, Wp, self.d_model).to(tokens.device).to(tokens.dtype)
        else:
            pe = self.pe[: Hp * Wp].view(self.max_h, self.max_w, self.d_model)[:Hp, :Wp].reshape(Hp * Wp, self.d_model)
            pe = pe.to(tokens.dtype)
        return tokens + pe.unsqueeze(0)


def _self_test():
    torch.manual_seed(0)

    B, H, W = 2, 64, 64
    patch = 8
    d_quat = 16

    img_rgb = torch.rand(B, 3, H, W)
    img_quat = torch.cat([torch.zeros_like(img_rgb[:, :1]), img_rgb], dim=1)  # (B,4,H,W)
    assert img_quat.shape == (B, 4, H, W)

    embed = QuaternionPatchEmbed(in_channels=1, patch_size=patch, d_quaternion=d_quat)
    unembed = QuaternionPatchUnembed(out_channels=1, patch_size=patch, d_quaternion=d_quat)
    pos = SinusoidalPosEmbed2D(d_model=4 * d_quat, max_h=H // patch, max_w=W // patch)

    tokens, grid = embed(img_quat)
    assert tokens.shape == (B, (H // patch) * (W // patch), 4 * d_quat)
    tokens_pe = pos(tokens, grid)
    assert tokens_pe.shape == tokens.shape

    img_back = unembed(tokens_pe, grid)
    assert img_back.shape == img_quat.shape

    # Backward sanity
    loss = img_back.sum()
    loss.backward()
    print("patch_embed self-test: ALL PASS")
    print(f"  Input img:       {tuple(img_quat.shape)}")
    print(f"  Token sequence:  {tuple(tokens.shape)}  (grid {grid})")
    print(f"  Recon img:       {tuple(img_back.shape)}")
    print(f"  Patch-embed params: {sum(p.numel() for p in embed.parameters()):,}")
    print(f"  Patch-unembed params: {sum(p.numel() for p in unembed.parameters()):,}")


if __name__ == "__main__":
    _self_test()
