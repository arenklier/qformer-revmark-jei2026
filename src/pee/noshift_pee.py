"""
No-shift PEE for QFormer-RevMark — high-PSNR / low-payload regime.

Standard PEE shifts every pixel with |e| > T by +-(T+1). The shift cost
dominates PSNR at low payload sizes because the number of shifted pixels
is independent of payload.

This module skips shifting entirely. Bits are embedded only in pixels with
e == 0 (when T_select == 0), or e in [-T, T] (when T_select > 0). All other
pixels remain unmodified.

To resolve the extraction ambiguity between "embedded bit" and "unmodified
non-zero e", a side-channel **location map** is produced at embedding time
and consumed at extraction time. The location map is a compact bitmap of
which candidate pixels actually carry a bit, plus the payload size header.

API mirrors AdaptiveQuaternionPEE but is wrapped in a side-channel
producer/consumer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch


# ============================================================
# Helpers
# ============================================================
def _flat_indices_of_candidates(e: torch.Tensor, T_select: int) -> torch.Tensor:
    """Return flat indices (1-D) of positions where |e| <= T_select."""
    mask = (e.abs() <= T_select)
    return mask.flatten().nonzero(as_tuple=False).flatten()


def _pack_bits_to_uint8(bits: List[int]) -> bytes:
    out = bytearray()
    acc = 0
    n = 0
    for b in bits:
        acc = (acc << 1) | (b & 1)
        n += 1
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n > 0:
        out.append(acc << (8 - n))
    return bytes(out)


def _unpack_bits_from_uint8(data: bytes, nbits: int) -> List[int]:
    out = []
    for byte in data:
        for i in range(7, -1, -1):
            if len(out) == nbits:
                return out
            out.append((byte >> i) & 1)
    return out


# ============================================================
# Side-channel payload
# ============================================================
@dataclass
class SideChannel:
    """Auxiliary data needed for extraction (NOT carried inside the image)."""
    n_payload_bits: int                          # how many bits encoded
    n_candidate_per_pass: List[int]              # # candidates considered per (pass, channel)
    location_bitmap: bytes                       # packed 1 bit per candidate position
    T_select: int


@dataclass
class NoShiftEmbedResult:
    watermarked_uint8: torch.Tensor              # (B, 3, H, W) int64 - usually B=1
    side: SideChannel
    n_modified: int                              # number of pixels actually flipped
    psnr_estimate: float                         # quick PSNR vs original


@dataclass
class NoShiftExtractResult:
    recovered_uint8: torch.Tensor                # (B, 3, H, W) int64
    payload_bits: torch.Tensor                   # (n_payload_bits,) int64 in {0,1}


# ============================================================
# No-shift embedding (single-batch, single-pass over Dot OR Cross)
# ============================================================
def embed_one_pass_noshift(
    error_int: torch.Tensor,             # (3, H, W) int64; error at the positions of this pass
    pass_mask: torch.Tensor,             # (H, W) bool - True where this pass embeds
    payload_bits: torch.Tensor,          # (M,) int64 in {0,1}; remaining bits to embed
    T_select: int = 0,
    pred_int: torch.Tensor = None,       # (3, H, W) int64; predictor output -- needed for overflow check
):
    """
    Returns
    -------
    delta : (3, H, W) int64        the modification map to ADD to the predicted image
    n_consumed : int                how many payload bits were consumed
    location_bits : List[int]       per-candidate flag list (1 = embedded, 0 = not)
    n_candidates : List[int]        # candidates per channel
    """
    C, H, W = error_int.shape
    delta = torch.zeros_like(error_int)
    location_bits = []
    n_candidates = []
    bit_cursor = 0
    payload_list = payload_bits.tolist()

    # IMPORTANT: candidate set must match what the extractor will see.
    # Embedded positions have |e| <= T_select; after +1 flip, |e_prime| <= T_select+1.
    # Non-embedded positions with |e| = T_select+1 collide with the above. So we
    # must also include them in the location map (with bit = 0).
    for c in range(C):
        e_c = error_int[c]
        mask_c = pass_mask & (e_c.abs() <= T_select + 1)  # include boundary
        flat_positions = mask_c.flatten().nonzero(as_tuple=False).flatten().tolist()
        n_candidates.append(len(flat_positions))
        for pos in flat_positions:
            yi = pos // W
            xi = pos % W
            e_val = int(e_c[yi, xi].item())
            embeddable = abs(e_val) <= T_select
            # Overflow guard: pixel + 1 must fit in [0, 255]. Pixel value = pred + e.
            overflow_safe = True
            if pred_int is not None:
                pixel_val = int(pred_int[c, yi, xi].item()) + e_val
                if pixel_val >= 255:
                    overflow_safe = False
            if embeddable and overflow_safe and bit_cursor < len(payload_list):
                b = payload_list[bit_cursor]
                location_bits.append(1)
                bit_cursor += 1
                if b == 1:
                    delta[c, yi, xi] = 1
            else:
                location_bits.append(0)
                # no modification

    return delta, bit_cursor, location_bits, n_candidates


def extract_one_pass_noshift(
    e_prime_int: torch.Tensor,           # (3, H, W) int64 prediction error of WATERMARKED image
    pass_mask: torch.Tensor,             # (H, W) bool
    location_bits: List[int],            # length = total candidates over channels
    n_candidates_per_channel: List[int],
    T_select: int = 0,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Returns (recovered_e, recovered_payload_bits).
        recovered_e : (3,H,W) int64    original prediction error
        recovered_payload_bits : list of {0,1}
    """
    C, H, W = e_prime_int.shape
    recovered_e = e_prime_int.clone()
    payload = []

    offset = 0
    for c in range(C):
        e_c = e_prime_int[c]
        # candidate positions are: pass_mask & |e_prime| <= T_select+1 (because embed could
        # have shifted from 0 to 1, but |e_prime|=1 already happens for unembedded |e|=1).
        # We use |e_prime| in {0,1} as candidate set (matches embedding criterion exactly
        # because e_original in [-T_select, T_select] -> e_prime in [-T_select, T_select+1]).
        mask_c = pass_mask & (e_c.abs() <= T_select + 1)
        flat_positions = mask_c.flatten().nonzero(as_tuple=False).flatten().tolist()
        # Sanity check: # candidates seen at extract MUST equal what was at embed.
        # Because we only flipped by +1 (which can take a |e|=0 to |e|=1 but never
        # widens the |e|<=T_select+1 set), this holds.
        expected = n_candidates_per_channel[c]
        if len(flat_positions) != expected:
            raise RuntimeError(
                f"candidate set drift on channel {c}: embed={expected}, extract={len(flat_positions)}"
            )

        for pos in flat_positions:
            flag = location_bits[offset]
            offset += 1
            yi = pos // W
            xi = pos % W
            if flag == 1:
                b = int(e_c[yi, xi].item())
                payload.append(b)
                # recover original (subtract bit)
                recovered_e[c, yi, xi] = 0
            # else: leave recovered_e as-is (unmodified original)

    return recovered_e, payload


# ============================================================
# Self-test
# ============================================================
def _self_test():
    torch.manual_seed(0)
    H, W = 32, 32
    C = 3

    # Synthetic prediction errors: mostly 0 with a few non-zero values
    error = torch.zeros(C, H, W, dtype=torch.int64)
    # Sprinkle a few non-zero errors
    for c in range(C):
        for _ in range(50):
            error[c, torch.randint(0, H, (1,)), torch.randint(0, W, (1,))] = torch.randint(-3, 4, (1,)).item()

    # Build pass_mask (use a checkerboard for fairness)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pass_mask = ((yy + xx) % 2 == 1)  # Dot

    # Capacity check: count zeros on Dot positions
    capacity = ((error == 0) & pass_mask.unsqueeze(0)).sum().item()
    print(f"  Available embed candidates: {capacity}")

    # Random payload smaller than capacity
    n_payload = capacity // 2
    payload_bits = torch.randint(0, 2, (n_payload,), dtype=torch.int64)
    print(f"  Trying to embed {n_payload} bits")

    delta, n_consumed, loc, nc = embed_one_pass_noshift(error, pass_mask, payload_bits, T_select=0)
    print(f"  Consumed {n_consumed} bits, location bits: {sum(loc)}/{len(loc)} flagged")
    e_prime = error + delta

    # Extract
    rec_e, rec_payload = extract_one_pass_noshift(e_prime, pass_mask, loc, nc, T_select=0)

    # Reversibility check
    assert torch.equal(rec_e, error), "no-shift PEE error recovery failed"
    assert rec_payload == payload_bits.tolist(), "payload recovery failed"
    print("  reversibility: BIT-EXACT  |  payload recovery: BIT-EXACT")

    # PSNR-like quick check: how much delta did we add?
    n_modified = int((delta != 0).sum().item())
    print(f"  Pixels modified: {n_modified} (out of {C * H * W // 2} dot positions)")
    print("noshift_pee self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
