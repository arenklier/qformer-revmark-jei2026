"""
Quaternion self-attention modules for QFormer-RevMark.

Design notes
------------
We use Hamilton-product-based Q/K/V projections (each a QuaternionLinear).
The attention score is the real (scalar) part of <Q_i, conj(K_j)> -- this is
the natural inner product on the quaternion algebra:

    <q, p> = Re(q * conj(p))

This is symmetric, real-valued, and reduces to ordinary dot-product attention
when imaginary parts vanish.  Softmax over the j-axis weights are applied to
quaternion-valued V tokens via scalar multiplication.

For multi-head, we split the quaternion feature dimension across heads (each
head keeps the 4-tuple intact). Recombination via output projection follows
standard ViT design.

References
----------
Tay et al., "Lightweight and Efficient Neural Transformer Architectures via
  Quaternion Algebra" (ACL 2019).
Parcollet et al., "Quaternion Recurrent Neural Networks" (ICLR 2019).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quaternion_ops import (
    QuaternionLinear,
    QuaternionLayerNorm,
    hamilton_product,
    quaternion_conjugate,
    split_quaternion,
    merge_quaternion,
)


def _real_inner(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Real part of  q * conj(k)  elementwise. Both tensors have last dim 4*d.

    For q = a + bi + cj + dk and k = e + fi + gj + hk:
        Re(q * conj(k)) = ae + bf + cg + dh   (the Euclidean dot product of
        the 4-vectors).

    Therefore <q, k>_q == dot4(q, k). We expose this as a separate name to
    make the geometric interpretation explicit.
    """
    return (q * k).sum(dim=-1)


class QuaternionMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention over quaternion-valued tokens.

    Input:  (B, N, 4 * d_quat)
    Output: (B, N, 4 * d_quat)
    """

    def __init__(self, d_quaternion: int, num_heads: int = 4, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if d_quaternion % num_heads != 0:
            raise ValueError(
                f"d_quaternion={d_quaternion} must be divisible by num_heads={num_heads}"
            )
        self.d_quaternion = d_quaternion
        self.num_heads = num_heads
        self.d_head = d_quaternion // num_heads  # quaternion-dim per head
        self.scale = 1.0 / math.sqrt(4 * self.d_head)

        self.q_proj = QuaternionLinear(d_quaternion, d_quaternion, bias=bias)
        self.k_proj = QuaternionLinear(d_quaternion, d_quaternion, bias=bias)
        self.v_proj = QuaternionLinear(d_quaternion, d_quaternion, bias=bias)
        self.out_proj = QuaternionLinear(d_quaternion, d_quaternion, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, 4*d_quat) -> (B, H, N, 4*d_head)
        We interleave the 4 components per-head: each head receives one slice of
        each of (r, i, j, k).
        """
        B, N, _ = x.shape
        r, i, j, k = split_quaternion(x, dim=-1)  # each (B, N, d_quat)
        r = r.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        i = i.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        j = j.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        return merge_quaternion(r, i, j, k, dim=-1)  # (B, H, N, 4*d_head)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, N, 4*d_head) -> (B, N, 4*d_quat)"""
        B, H, N, _ = x.shape
        r, i, j, k = split_quaternion(x, dim=-1)  # each (B, H, N, d_head)
        r = r.transpose(1, 2).contiguous().view(B, N, H * self.d_head)
        i = i.transpose(1, 2).contiguous().view(B, N, H * self.d_head)
        j = j.transpose(1, 2).contiguous().view(B, N, H * self.d_head)
        k = k.transpose(1, 2).contiguous().view(B, N, H * self.d_head)
        return merge_quaternion(r, i, j, k, dim=-1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))  # (B, H, N, 4*d_head)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Real-valued attention scores: <q_i, k_j>_q = sum over component pairs.
        # Implementing as q @ k^T after packing is equivalent to dot4.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of quaternion-valued V tokens via scalar weighting
        # (B,H,N,N) x (B,H,N,4*d_head) -> (B,H,N,4*d_head)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return self.out_proj(out)


class QuaternionFeedForward(nn.Module):
    """Quaternion two-layer MLP with GELU activation."""

    def __init__(self, d_quaternion: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_quaternion * mlp_ratio)
        self.fc1 = QuaternionLinear(d_quaternion, hidden)
        self.fc2 = QuaternionLinear(hidden, d_quaternion)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class QuaternionTransformerBlock(nn.Module):
    """Pre-norm quaternion transformer block."""

    def __init__(
        self,
        d_quaternion: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = QuaternionLayerNorm(d_quaternion)
        self.attn = QuaternionMultiHeadAttention(d_quaternion, num_heads, dropout=dropout)
        self.norm2 = QuaternionLayerNorm(d_quaternion)
        self.ffn = QuaternionFeedForward(d_quaternion, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def _self_test():
    torch.manual_seed(0)

    d_quat = 32
    num_heads = 4
    block = QuaternionTransformerBlock(d_quat, num_heads=num_heads, mlp_ratio=2.0)

    # Token sequence
    B, N = 2, 16
    x = torch.randn(B, N, 4 * d_quat, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"

    # Backward
    loss = y.sum()
    loss.backward()
    assert x.grad is not None

    # Parameter count
    n_params = sum(p.numel() for p in block.parameters())
    # Equivalent scalar transformer would have 4*4 = 16x more attention matmul params
    print(f"quaternion_attention self-test: ALL PASS")
    print(f"  Block params (d_quat={d_quat}, heads={num_heads}, mlp_ratio=2): {n_params:,}")
    print(f"  Output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    _self_test()
