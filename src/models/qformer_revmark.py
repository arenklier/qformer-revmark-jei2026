"""
End-to-end QFormer-RevMark: a two-pass reversible watermarking wrapper that
combines a deterministic deep predictor with the Adaptive Quaternion PEE.

Checkerboard split
------------------
We use the classical "Cross / Dot" pixel partition:

    Cross : (i + j) % 2 == 0
    Dot   : (i + j) % 2 == 1

Embedding (two passes)
----------------------
Pass 1 -- embed in Dot:
    1.  mask Dot pixels to zero        ->  I_cross_only
    2.  pred_dot = Predictor(I_cross_only)
    3.  e_dot = I[Dot] - round(pred_dot)[Dot]
    4.  e_dot_prime, bits_used_1 = APEE.embed(e_dot, bits_batch_1)
    5.  I_mid[Dot] = round(pred_dot)[Dot] + e_dot_prime
        I_mid[Cross] = I[Cross]                      (unchanged)

Pass 2 -- embed in Cross:
    6.  mask Cross pixels to zero      ->  I_dot_only  (using I_mid's Dot)
    7.  pred_cross = Predictor(I_dot_only)
    8.  e_cross = I_mid[Cross] - round(pred_cross)[Cross]
    9.  e_cross_prime, bits_used_2 = APEE.embed(e_cross, bits_batch_2)
   10.  I_w[Cross] = round(pred_cross)[Cross] + e_cross_prime
        I_w[Dot]   = I_mid[Dot]

Extraction is the exact inverse in reverse pass order.

Bit-exactness guarantee
-----------------------
Because the predictor is fully deterministic and operates on the SAME input
during embedding-pass-k and extraction-pass-k, both sides compute identical
integer predictions, and the integer-arithmetic PEE inverts losslessly. Hence
the entire pipeline is bit-exact for arbitrary predictor weights (trained or
not).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from .qformer import QFormerPredictor
from ..pee.adaptive_pee import AdaptiveQuaternionPEE


# ============================================================
# Pixel-domain helpers
# ============================================================
def float_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[0,1] float -> int64 in [0,255] using round (NOT trunc)."""
    return torch.round(x.clamp(0.0, 1.0) * 255.0).to(torch.int64)


def uint8_to_float(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32) / 255.0


def checkerboard_mask(H: int, W: int, parity: int, device, dtype=torch.bool) -> torch.Tensor:
    """parity=0 -> Cross  (i+j even),  parity=1 -> Dot  (i+j odd)"""
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    return ((yy + xx) % 2 == parity).to(dtype)


# ============================================================
# End-to-end watermark module
# ============================================================
@dataclass
class EmbedResult:
    watermarked_uint8: torch.Tensor    # (B, 3, H, W) int64 in [0, 255 + delta]
    n_bits_pass1: int
    n_bits_pass2: int
    capacity_bpp: float


@dataclass
class ExtractResult:
    recovered_uint8: torch.Tensor      # (B, 3, H, W) int64
    bits_pass1: torch.Tensor           # extracted in Pass 1 order (Cross-prediction step)
    bits_pass2: torch.Tensor           # extracted in Pass 2 order (Dot-prediction step)
    carrier_pass1: torch.Tensor
    carrier_pass2: torch.Tensor


class QFormerRevMark(nn.Module):
    """Stitches the QFormer predictor and Adaptive Q-PEE into a reversible
    watermarking model."""

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        d_quaternion: int = 48,
        depth: int = 6,
        num_heads: int = 4,
        init_T: float = 5.0,
    ):
        super().__init__()
        self.predictor = QFormerPredictor(
            image_size=image_size,
            patch_size=patch_size,
            d_quaternion=d_quaternion,
            depth=depth,
            num_heads=num_heads,
        )
        self.apee = AdaptiveQuaternionPEE(init_T=init_T)
        self.image_size = image_size

    # ---------------- internal predictor wrapper (integer round) -----------
    def _predict_int(self, image_uint8: torch.Tensor) -> torch.Tensor:
        """Run predictor on a uint8 image, return rounded int64 prediction."""
        rgb_float = uint8_to_float(image_uint8)
        with torch.no_grad():
            pred_float = self.predictor(rgb_float)
        return float_to_uint8(pred_float)

    # ---------------- embedding -------------------------------------------
    def embed(
        self,
        image_uint8: torch.Tensor,
        bits: torch.Tensor,
    ) -> EmbedResult:
        """
        Parameters
        ----------
        image_uint8 : (B, 3, H, W) int64 in [0, 255]
        bits        : (B, 3, H, W) int64 in {0, 1} -- arranged channelwise.
                      We split this tensor into pass-1 (Dot positions) and
                      pass-2 (Cross positions) by the checkerboard mask.

        Returns
        -------
        EmbedResult
        """
        assert image_uint8.dim() == 4 and image_uint8.size(1) == 3
        assert image_uint8.dtype == torch.int64
        B, C, H, W = image_uint8.shape

        cross_mask = checkerboard_mask(H, W, parity=0, device=image_uint8.device, dtype=torch.bool)
        dot_mask = ~cross_mask  # (H, W) bool

        # ---------------- Pass 1: predict Dot from Cross --------------------
        I_cross_only = image_uint8 * cross_mask.to(torch.int64)
        pred_dot = self._predict_int(I_cross_only)             # (B, 3, H, W)

        e_dot = image_uint8 - pred_dot                          # full error
        # restrict the embed call to dot positions by masking bits/error to 0
        # elsewhere; the APEE will still process them but we will overwrite
        # them with originals before re-composition.
        bits_dot_input = bits * dot_mask.to(torch.int64)
        e_dot_input = e_dot * dot_mask.to(torch.int64)
        e_dot_prime, stats1 = self.apee.embed(e_dot_input, bits_dot_input)

        # Build the intermediate image
        I_mid = image_uint8.clone()
        modified_dot = pred_dot + e_dot_prime
        I_mid = torch.where(dot_mask.unsqueeze(0).unsqueeze(0), modified_dot, I_mid)

        # ---------------- Pass 2: predict Cross from (modified) Dot --------
        I_dot_only = I_mid * dot_mask.to(torch.int64)
        pred_cross = self._predict_int(I_dot_only)              # (B, 3, H, W)

        e_cross = I_mid - pred_cross
        bits_cross_input = bits * cross_mask.to(torch.int64)
        e_cross_input = e_cross * cross_mask.to(torch.int64)
        e_cross_prime, stats2 = self.apee.embed(e_cross_input, bits_cross_input)

        I_w = I_mid.clone()
        modified_cross = pred_cross + e_cross_prime
        I_w = torch.where(cross_mask.unsqueeze(0).unsqueeze(0), modified_cross, I_w)

        # Count bits *actually* placed on the masked positions
        # (APEEStats counts entire tensor; restrict to mask intersected with expandable.)
        T_i, T_j, T_k = self.apee.T_int
        n_bits_pass1 = 0
        n_bits_pass2 = 0
        for c, T in enumerate((T_i, T_j, T_k)):
            expandable_dot = (e_dot[:, c] >= -T) & (e_dot[:, c] <= T) & dot_mask.unsqueeze(0)
            expandable_cross = (e_cross[:, c] >= -T) & (e_cross[:, c] <= T) & cross_mask.unsqueeze(0)
            n_bits_pass1 += int(expandable_dot.sum().item())
            n_bits_pass2 += int(expandable_cross.sum().item())
        total_bits = n_bits_pass1 + n_bits_pass2
        capacity_bpp = total_bits / float(B * H * W)

        return EmbedResult(
            watermarked_uint8=I_w,
            n_bits_pass1=n_bits_pass1,
            n_bits_pass2=n_bits_pass2,
            capacity_bpp=capacity_bpp,
        )

    # ---------------- extraction ------------------------------------------
    def extract(self, image_w_uint8: torch.Tensor) -> ExtractResult:
        assert image_w_uint8.dim() == 4 and image_w_uint8.size(1) == 3
        assert image_w_uint8.dtype == torch.int64
        B, C, H, W = image_w_uint8.shape

        cross_mask = checkerboard_mask(H, W, parity=0, device=image_w_uint8.device, dtype=torch.bool)
        dot_mask = ~cross_mask

        # --- Reverse Pass 2: recover original Cross + bits_cross -----------
        I_dot_only = image_w_uint8 * dot_mask.to(torch.int64)
        pred_cross = self._predict_int(I_dot_only)              # same as during embed
        e_cross_prime = image_w_uint8 - pred_cross
        e_cross_prime = e_cross_prime * cross_mask.to(torch.int64)
        e_cross_rec, bits_cross_rec, carrier_cross = self.apee.extract(e_cross_prime)

        # Reconstruct intermediate image: Cross restored, Dot is still modified
        restored_cross = pred_cross + e_cross_rec
        I_mid_recon = image_w_uint8.clone()
        I_mid_recon = torch.where(cross_mask.unsqueeze(0).unsqueeze(0), restored_cross, I_mid_recon)

        # --- Reverse Pass 1: recover original Dot + bits_dot --------------
        I_cross_only = I_mid_recon * cross_mask.to(torch.int64)
        pred_dot = self._predict_int(I_cross_only)              # same as during embed
        e_dot_prime = I_mid_recon - pred_dot
        e_dot_prime = e_dot_prime * dot_mask.to(torch.int64)
        e_dot_rec, bits_dot_rec, carrier_dot = self.apee.extract(e_dot_prime)

        restored_dot = pred_dot + e_dot_rec
        I_rec = I_mid_recon.clone()
        I_rec = torch.where(dot_mask.unsqueeze(0).unsqueeze(0), restored_dot, I_rec)

        return ExtractResult(
            recovered_uint8=I_rec,
            bits_pass1=bits_dot_rec,    # pass-1 was embedded into Dot positions
            bits_pass2=bits_cross_rec,  # pass-2 was embedded into Cross positions
            carrier_pass1=carrier_dot,
            carrier_pass2=carrier_cross,
        )


# ============================================================
# Smoke test
# ============================================================
def _smoke_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    H = W = 32
    model = QFormerRevMark(
        image_size=H,
        patch_size=8,
        d_quaternion=16,
        depth=2,
        num_heads=4,
        init_T=10.0,
    ).to(device).eval()

    # Random RGB image, integer domain
    img_uint8 = torch.randint(0, 256, (1, 3, H, W), dtype=torch.int64, device=device)
    bits = torch.randint(0, 2, (1, 3, H, W), dtype=torch.int64, device=device)

    er = model.embed(img_uint8, bits)
    xr = model.extract(er.watermarked_uint8)

    # 1) image must round-trip bit-exactly
    if torch.equal(xr.recovered_uint8, img_uint8):
        print("  Image round-trip      : BIT-EXACT")
    else:
        diff = (xr.recovered_uint8 - img_uint8).abs()
        print(f"  Image round-trip FAILED. max|diff|={diff.max().item()}, mean={diff.float().mean().item():.3f}")

    # 2) bits embedded on Dot must equal extracted Dot bits, on Cross likewise
    # Reconstruct the input-side bit selections
    cross_mask = checkerboard_mask(H, W, parity=0, device=device, dtype=torch.bool)
    dot_mask = ~cross_mask
    bits_pass1_in = (bits * dot_mask.to(torch.int64))
    bits_pass2_in = (bits * cross_mask.to(torch.int64))

    eq_dot = ((xr.bits_pass1 == bits_pass1_in) | (~xr.carrier_pass1)).all().item()
    eq_cross = ((xr.bits_pass2 == bits_pass2_in) | (~xr.carrier_pass2)).all().item()
    print(f"  Dot bits on carriers  : {'OK' if eq_dot else 'FAIL'}")
    print(f"  Cross bits on carriers: {'OK' if eq_cross else 'FAIL'}")

    print(f"  Capacity (random img) : {er.capacity_bpp:.4f} bpp  "
          f"(pass1={er.n_bits_pass1}, pass2={er.n_bits_pass2})")
    print(f"  Threshold (i, j, k)   : {model.apee.T_int}")


if __name__ == "__main__":
    print("QFormer-RevMark smoke test")
    _smoke_test()
