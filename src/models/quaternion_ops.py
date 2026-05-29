"""
Quaternion neural-network primitives for QFormer-RevMark.

Conventions
-----------
A quaternion tensor has its 4 components packed along the channel axis:
    q.shape = (..., 4 * d)  with components ordered (real, i, j, k),
where d is the quaternion-feature dimension.

For 2D image feature maps we use (B, 4*d, H, W); for token sequences (B, N, 4*d).

Hamilton product (q1 * q2) is implemented via four real-valued matmuls.
See Parcollet et al. (2019), "Quaternion Recurrent Neural Networks".
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Component split / merge helpers
# ============================================================
def split_quaternion(q: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed quaternion tensor into (r, i, j, k) parts along `dim`."""
    if q.size(dim) % 4 != 0:
        raise ValueError("quaternion tensor channel size must be divisible by 4")
    return torch.chunk(q, 4, dim=dim)


def merge_quaternion(r: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.cat([r, i, j, k], dim=dim)


def quaternion_conjugate(q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Conjugate q = a + bi + cj + dk  ->  a - bi - cj - dk."""
    r, i, j, k = split_quaternion(q, dim=dim)
    return merge_quaternion(r, -i, -j, -k, dim=dim)


def quaternion_norm_sq(q: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    """Squared modulus per quaternion: |q|^2 = a^2 + b^2 + c^2 + d^2."""
    r, i, j, k = split_quaternion(q, dim=dim)
    return (r * r + i * i + j * j + k * k).sum(dim=dim, keepdim=keepdim) if False else (r ** 2 + i ** 2 + j ** 2 + k ** 2)


# ============================================================
# Hamilton product
# ============================================================
def hamilton_product(q1: torch.Tensor, q2: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Hamilton product q1 * q2 elementwise.

    For q1 = a + bi + cj + dk, q2 = e + fi + gj + hk:
        r = ae - bf - cg - dh
        i = af + be + ch - dg
        j = ag - bh + ce + df
        k = ah + bg - cf + de
    """
    a, b, c, d = split_quaternion(q1, dim=dim)
    e, f, g, h = split_quaternion(q2, dim=dim)
    r = a * e - b * f - c * g - d * h
    i = a * f + b * e + c * h - d * g
    j = a * g - b * h + c * e + d * f
    k = a * h + b * g - c * f + d * e
    return merge_quaternion(r, i, j, k, dim=dim)


# ============================================================
# Quaternion Linear layer
# ============================================================
class QuaternionLinear(nn.Module):
    """
    Linear layer whose weight matrix realises a Hamilton-product transform.

    Input:  (..., 4 * in_features)   packed as (r, i, j, k) chunks
    Output: (..., 4 * out_features)

    Reduces real-valued parameters by 4x compared with a scalar Linear of the
    same effective dimension. See Parcollet et al. (2019).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Four real-valued weight matrices: r, i, j, k
        self.weight_r = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_i = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_j = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_k = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(4 * out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Quaternion-aware init: scale by 1/sqrt(2*in_features) to keep variance.
        # Each weight component is initialised from a normal distribution.
        std = 1.0 / math.sqrt(2.0 * self.in_features)
        for w in (self.weight_r, self.weight_i, self.weight_j, self.weight_k):
            nn.init.normal_(w, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (..., 4 * in_features) packed (r, i, j, k).
        Returns:  (..., 4 * out_features).
        """
        if x.size(-1) != 4 * self.in_features:
            raise ValueError(
                f"expected last dim {4 * self.in_features}, got {x.size(-1)}"
            )
        r, i, j, k = split_quaternion(x, dim=-1)

        # Hamilton-product weight matrix acting on (r, i, j, k)^T.
        # See Parcollet (2019) Eq. (6).
        out_r = F.linear(r, self.weight_r) - F.linear(i, self.weight_i) \
              - F.linear(j, self.weight_j) - F.linear(k, self.weight_k)
        out_i = F.linear(r, self.weight_i) + F.linear(i, self.weight_r) \
              + F.linear(j, self.weight_k) - F.linear(k, self.weight_j)
        out_j = F.linear(r, self.weight_j) - F.linear(i, self.weight_k) \
              + F.linear(j, self.weight_r) + F.linear(k, self.weight_i)
        out_k = F.linear(r, self.weight_k) + F.linear(i, self.weight_j) \
              - F.linear(j, self.weight_i) + F.linear(k, self.weight_r)

        out = merge_quaternion(out_r, out_i, out_j, out_k, dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# ============================================================
# Quaternion LayerNorm
# ============================================================
class QuaternionLayerNorm(nn.Module):
    """
    Per-component LayerNorm: normalise each of (r, i, j, k) independently.

    A truly geometry-aware normaliser would use the quaternion modulus, but
    per-component normalisation has been shown to train more stably in
    practice (Tay et al., 2019).
    """

    def __init__(self, d_quaternion: int, eps: float = 1e-5):
        super().__init__()
        self.d_quaternion = d_quaternion
        self.eps = eps
        # Separate gain/bias per component to preserve quaternion symmetry
        self.weight = nn.Parameter(torch.ones(4 * d_quaternion))
        self.bias = nn.Parameter(torch.zeros(4 * d_quaternion))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r, i, j, k = split_quaternion(x, dim=-1)
        normed = []
        for part in (r, i, j, k):
            mean = part.mean(dim=-1, keepdim=True)
            var = part.var(dim=-1, unbiased=False, keepdim=True)
            normed.append((part - mean) / torch.sqrt(var + self.eps))
        out = merge_quaternion(*normed, dim=-1)
        return out * self.weight + self.bias


# ============================================================
# Quick self-tests (run as: python -m src.models.quaternion_ops)
# ============================================================
def _self_test():
    torch.manual_seed(0)

    # 1) Conjugate involution: (q*)* == q
    q = torch.randn(2, 8)  # 8 = 4*2
    assert torch.allclose(quaternion_conjugate(quaternion_conjugate(q)), q)

    # 2) Hamilton product non-commutative
    q1 = torch.randn(2, 8)
    q2 = torch.randn(2, 8)
    p1 = hamilton_product(q1, q2)
    p2 = hamilton_product(q2, q1)
    assert not torch.allclose(p1, p2), "Hamilton product is non-commutative"

    # 3) q * q.conj() yields real-valued quaternion with norm-squared in real part
    qq = hamilton_product(q1, quaternion_conjugate(q1))
    r, i, j, k = split_quaternion(qq, dim=-1)
    assert torch.allclose(i, torch.zeros_like(i), atol=1e-5)
    assert torch.allclose(j, torch.zeros_like(j), atol=1e-5)
    assert torch.allclose(k, torch.zeros_like(k), atol=1e-5)
    expected_norm = (q1 ** 2).reshape(2, 4, 2).sum(dim=1)
    assert torch.allclose(r, expected_norm, atol=1e-5)

    # 4) QuaternionLinear forward / backward
    layer = QuaternionLinear(in_features=8, out_features=16)
    x = torch.randn(3, 32, requires_grad=True)  # 32 = 4*8
    y = layer(x)
    assert y.shape == (3, 64)  # 64 = 4*16
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    for p in layer.parameters():
        assert p.grad is not None

    # 5) Parameter count is 4x lower than equivalent real Linear
    real_layer = nn.Linear(4 * 8, 4 * 16, bias=False)
    quat_params = sum(p.numel() for p in layer.parameters() if "bias" not in str(p))
    quat_params_no_bias = layer.weight_r.numel() + layer.weight_i.numel() + layer.weight_j.numel() + layer.weight_k.numel()
    real_params = real_layer.weight.numel()
    assert quat_params_no_bias * 4 == real_params, f"quat={quat_params_no_bias}, real={real_params}"

    # 6) QuaternionLayerNorm
    ln = QuaternionLayerNorm(d_quaternion=8)
    out = ln(torch.randn(3, 32))
    assert out.shape == (3, 32)

    print("quaternion_ops self-test: ALL PASS")
    print(f"  Q-Linear (in=8, out=16) params (excl bias): {quat_params_no_bias}  vs scalar equivalent: {real_params}  (ratio = {real_params / quat_params_no_bias:.1f}x)")


if __name__ == "__main__":
    _self_test()
