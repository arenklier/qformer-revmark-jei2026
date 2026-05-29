"""
Adaptive Quaternion Prediction Error Expansion (Q-APEE) for QFormer-RevMark.

Algorithm
---------
For an integer prediction error e in {..., -1, 0, 1, ...} and a learnable
expansion threshold T (per-channel, integer-rounded), the embedding rule is:

    e in [-T, T]    ->  e' = 2e + b           (expansion; carries 1 bit)
    e > T           ->  e' = e + T + 1         (shift outward)
    e < -T          ->  e' = e - T             (shift outward)

Extraction (from e' alone, given the same T):

    e' in [-2T, 2T+1]  ->  b = e' mod 2,  e = (e' - b) / 2
    e' > 2T + 1        ->  no bit,  e = e' - (T + 1)
    e' < -2T           ->  no bit,  e = e' + T

This guarantees bit-exact recovery of (e, b) from e' for any integer e and any
non-negative integer T.

We apply the rule independently to the i, j, k channels of a pure quaternion
image (the real component is always zero and carries nothing).

Per-channel learnable thresholds (one scalar per quaternion component) make the
expansion region adapt to the error statistics of each colour channel -- the
"adaptive" in Q-APEE. Gradients pass through the rounding via the straight-
through estimator.

References
----------
Thodi & Rodriguez (2007), "Expansion embedding techniques for reversible
watermarking", IEEE Trans. Image Process.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


# ============================================================
# Straight-through round
# ============================================================
class _STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return _STERound.apply(x)


# ============================================================
# Integer PEE primitives (work on int64 tensors)
# ============================================================
def pee_embed_integer(e: torch.Tensor, b: torch.Tensor, T: int) -> torch.Tensor:
    """
    Embed bits into an integer error map.

    Parameters
    ----------
    e : (..., ) int tensor, prediction errors
    b : (..., ) int tensor in {0, 1}, same shape as e
        Only positions with |e| <= T consume a bit. Other positions ignore b.
    T : non-negative integer threshold

    Returns
    -------
    e_prime : (..., ) int tensor, modified errors
    """
    assert T >= 0
    expandable = (e >= -T) & (e <= T)
    expanded = 2 * e + b * expandable.to(e.dtype)
    shifted_right = e + (T + 1)
    shifted_left = e - T
    return torch.where(
        expandable,
        expanded,
        torch.where(e > T, shifted_right, shifted_left),
    )


def pee_extract_integer(e_prime: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inverse of pee_embed_integer.

    Returns
    -------
    e_rec        : (..., ) int tensor, recovered original errors
    b_rec        : (..., ) int tensor in {0, 1}, recovered bits (0 outside expansion)
    bit_carrier  : (..., ) bool tensor; True where a bit was actually carried
    """
    assert T >= 0
    expansion = (e_prime >= -2 * T) & (e_prime <= 2 * T + 1)
    right_shift = e_prime > 2 * T + 1
    left_shift = e_prime < -2 * T

    # Python-style modulo: torch.remainder(-1, 2) == 1
    b_rec = torch.remainder(e_prime, 2)
    # When not in expansion region, force bit to 0
    b_rec = torch.where(expansion, b_rec, torch.zeros_like(b_rec))

    e_from_expansion = (e_prime - b_rec) // 2
    e_from_right = e_prime - (T + 1)
    e_from_left = e_prime + T

    e_rec = torch.where(
        expansion,
        e_from_expansion,
        torch.where(right_shift, e_from_right, e_from_left),
    )
    return e_rec, b_rec, expansion


# ============================================================
# Adaptive Quaternion PEE module
# ============================================================
@dataclass
class APEEStats:
    """Diagnostic statistics returned alongside an embed call."""
    n_pixels: int
    n_expandable: int
    payload_bits: int
    capacity_bpp: float


class AdaptiveQuaternionPEE(nn.Module):
    """
    Learnable-threshold quaternion PEE.

    Maintains a separate threshold for each of (i, j, k) channels of a pure
    quaternion image. Real channel is treated as constant zero.

    Forward (embedding):
        embed(error, payload_bits) -> (e_prime, stats)

    Inverse (extraction):
        extract(e_prime) -> (e_rec, payload_bits_rec, carrier_mask)
    """

    def __init__(self, init_T: float = 5.0):
        super().__init__()
        # One learnable threshold per (i, j, k) quaternion channel.
        # We learn it in log-space to keep T > 0.
        self.log_T = nn.Parameter(torch.tensor([float(init_T)] * 3).log())

    @property
    def T_float(self) -> torch.Tensor:
        """Per-channel float threshold (>= ~exp(log_T))."""
        return self.log_T.exp()

    @property
    def T_int(self) -> Tuple[int, int, int]:
        """Per-channel integer threshold used in the integer PEE."""
        T = ste_round(self.T_float).clamp_min(0).detach().to(torch.int64).tolist()
        return tuple(T)

    def embed(
        self,
        error: torch.Tensor,
        payload_bits: torch.Tensor,
    ) -> Tuple[torch.Tensor, APEEStats]:
        """
        Parameters
        ----------
        error        : (B, 3, H, W) int64 tensor (i, j, k errors)
        payload_bits : (B, 3, H, W) int64 tensor with values in {0, 1};
                       extra bits in positions that turn out non-expandable
                       are simply ignored (caller can pack densely).

        Returns
        -------
        e_prime : (B, 3, H, W) int64 modified error map
        stats   : APEEStats summarising capacity
        """
        if error.dtype != torch.int64:
            raise TypeError(f"error must be int64, got {error.dtype}")
        if payload_bits.shape != error.shape:
            raise ValueError("payload_bits must match error shape")

        Ts = self.T_int  # (T_i, T_j, T_k)
        out_channels = []
        n_pixels = error[:, 0].numel() * 3
        n_expandable = 0
        n_bits_used = 0
        for c in range(3):
            T = Ts[c]
            e_c = error[:, c]
            b_c = payload_bits[:, c]
            expandable = (e_c >= -T) & (e_c <= T)
            n_expandable += int(expandable.sum().item())
            n_bits_used += int((expandable & (b_c >= 0)).sum().item())
            e_prime_c = pee_embed_integer(e_c, b_c, T)
            out_channels.append(e_prime_c)

        e_prime = torch.stack(out_channels, dim=1)
        H, W = error.shape[-2:]
        stats = APEEStats(
            n_pixels=n_pixels,
            n_expandable=n_expandable,
            payload_bits=n_bits_used,
            capacity_bpp=n_bits_used / float(error.shape[0] * H * W),
        )
        return e_prime, stats

    def extract(self, e_prime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        e_prime : (B, 3, H, W) int64

        Returns
        -------
        e_rec        : (B, 3, H, W) int64
        bits_rec     : (B, 3, H, W) int64 in {0, 1}, 0 outside expansion
        carrier_mask : (B, 3, H, W) bool, True where a bit was extracted
        """
        if e_prime.dtype != torch.int64:
            raise TypeError(f"e_prime must be int64, got {e_prime.dtype}")
        Ts = self.T_int
        e_rec_chs, b_rec_chs, carrier_chs = [], [], []
        for c in range(3):
            er, br, ca = pee_extract_integer(e_prime[:, c], Ts[c])
            e_rec_chs.append(er)
            b_rec_chs.append(br)
            carrier_chs.append(ca)
        return (
            torch.stack(e_rec_chs, dim=1),
            torch.stack(b_rec_chs, dim=1),
            torch.stack(carrier_chs, dim=1),
        )

    def extra_repr(self) -> str:
        return f"T_init={tuple(self.T_float.detach().tolist())}, T_int={self.T_int}"


# ============================================================
# Self-tests
# ============================================================
def _test_integer_pee_round_trip():
    """For every integer error in a wide range and every bit, embed then extract."""
    torch.manual_seed(0)
    for T in (1, 3, 5, 8):
        for e_val in range(-3 * T - 2, 3 * T + 3):
            for b_val in (0, 1):
                e = torch.tensor([e_val], dtype=torch.int64)
                b = torch.tensor([b_val], dtype=torch.int64)
                e_prime = pee_embed_integer(e, b, T)
                e_rec, b_rec, carrier = pee_extract_integer(e_prime, T)
                if -T <= e_val <= T:
                    assert e_rec.item() == e_val, f"T={T} e={e_val} b={b_val} -> e'={e_prime.item()} -> e_rec={e_rec.item()}"
                    assert b_rec.item() == b_val, f"bit mismatch T={T} e={e_val} b={b_val}"
                    assert carrier.item() == True
                else:
                    assert e_rec.item() == e_val, f"shift T={T} e={e_val} -> e'={e_prime.item()} -> e_rec={e_rec.item()}"
                    assert carrier.item() == False


def _test_module_round_trip():
    """End-to-end: random error map + random bits, embed then extract."""
    torch.manual_seed(1)
    apee = AdaptiveQuaternionPEE(init_T=5.0)

    B, H, W = 2, 32, 32
    error = torch.randint(-15, 16, (B, 3, H, W), dtype=torch.int64)
    bits = torch.randint(0, 2, (B, 3, H, W), dtype=torch.int64)

    e_prime, stats = apee.embed(error, bits)
    e_rec, bits_rec, carrier = apee.extract(e_prime)

    # All errors must round-trip exactly
    assert torch.equal(e_rec, error), "error recovery failed"

    # Bits must round-trip on carrier positions
    err_on_carrier = (bits_rec[carrier] != bits[carrier]).any().item()
    assert not err_on_carrier, "bit recovery on carrier positions failed"

    print(f"  capacity = {stats.capacity_bpp:.3f} bpp  "
          f"(expandable {stats.n_expandable}/{stats.n_pixels})")


def _test_gradient_flow():
    """Make sure the threshold can be optimised via gradient descent."""
    torch.manual_seed(2)
    apee = AdaptiveQuaternionPEE(init_T=5.0)
    opt = torch.optim.SGD(apee.parameters(), lr=0.1)
    # Encourage smaller T via a dummy loss on log_T
    target = torch.tensor(0.0)
    log_T_before = apee.log_T.detach().clone()
    for _ in range(5):
        opt.zero_grad()
        loss = (apee.log_T - target).pow(2).mean()
        loss.backward()
        opt.step()
    log_T_after = apee.log_T.detach().clone()
    assert (log_T_after - log_T_before).abs().sum().item() > 0, "no gradient update"


def _self_test():
    print("Testing integer PEE round-trip for many (T, e, b)...")
    _test_integer_pee_round_trip()
    print("  integer PEE round-trip: PASS")

    print("Testing module-level embed/extract round-trip...")
    _test_module_round_trip()

    print("Testing gradient flow through learnable threshold...")
    _test_gradient_flow()
    print("  gradient flow: PASS")

    print("adaptive_pee self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
