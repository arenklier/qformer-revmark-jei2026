"""
End-to-end QFormer-RevMark with no-shift PEE for high-PSNR low-payload regime.

Differences from QFormerRevMark
-------------------------------
* Embedding only modifies pixels in [-T_select, T_select] error range; pixels
  outside this range are NEVER touched (no histogram shift).
* A side-channel "location map" is produced to disambiguate at extraction.
* PSNR at small payloads jumps dramatically (typically +5 to +15 dB).

Pipeline (embed):
    Pass 1: predict Dot from Cross-only -> e_dot  -> embed via noshift -> watermarked Dot
    Pass 2: predict Cross from (modified) Dot -> e_cross -> embed via noshift -> watermarked Cross
    Side data: list of location bitmaps + T_select + n_payload

Extraction is the exact inverse with the same predictor outputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn

from .qformer_revmark import (
    QFormerRevMark,
    float_to_uint8,
    uint8_to_float,
    checkerboard_mask,
)
from ..pee.noshift_pee import (
    embed_one_pass_noshift,
    extract_one_pass_noshift,
)


@dataclass
class NoShiftEmbedResult:
    watermarked_uint8: torch.Tensor          # (1, 3, H, W) int64
    side_data: dict                          # {T_select, n_payload, loc_p1, nc_p1, loc_p2, nc_p2}
    n_payload_bits: int
    psnr_quick: float


@dataclass
class NoShiftExtractResult:
    recovered_uint8: torch.Tensor
    payload_bits: torch.Tensor               # (n_payload,)


class QFormerNoShiftRevMark(QFormerRevMark):
    """QFormerRevMark with no-shift PEE."""

    @torch.no_grad()
    def embed_noshift(
        self,
        image_uint8: torch.Tensor,           # (1, 3, H, W) int64
        payload_bits: torch.Tensor,          # (N,) int64 in {0,1}
        T_select: int = 0,
    ) -> NoShiftEmbedResult:
        assert image_uint8.dim() == 4 and image_uint8.size(0) == 1 and image_uint8.size(1) == 3
        assert image_uint8.dtype == torch.int64
        B, C, H, W = image_uint8.shape

        device = image_uint8.device
        cross_mask = checkerboard_mask(H, W, parity=0, device=device, dtype=torch.bool)
        dot_mask = ~cross_mask

        # --------- Pass 1: predict Dot from Cross-only ---------
        I_cross_only = image_uint8 * cross_mask.to(torch.int64)
        pred_dot = self._predict_int(I_cross_only)               # (1,3,H,W)
        e_dot = (image_uint8 - pred_dot)[0]                       # (3,H,W)
        delta_dot, n_consumed_1, loc_p1, nc_p1 = embed_one_pass_noshift(
            e_dot, dot_mask, payload_bits, T_select=T_select, pred_int=pred_dot[0]
        )
        I_mid = image_uint8.clone()
        modified_dot = (pred_dot[0] + e_dot + delta_dot).unsqueeze(0)
        I_mid = torch.where(dot_mask.unsqueeze(0).unsqueeze(0), modified_dot, I_mid)

        # --------- Pass 2: predict Cross from (modified) Dot ---------
        I_dot_only = I_mid * dot_mask.to(torch.int64)
        pred_cross = self._predict_int(I_dot_only)
        e_cross = (I_mid - pred_cross)[0]
        remaining = payload_bits[n_consumed_1:]
        delta_cross, n_consumed_2_added, loc_p2, nc_p2 = embed_one_pass_noshift(
            e_cross, cross_mask, remaining, T_select=T_select, pred_int=pred_cross[0]
        )
        I_w = I_mid.clone()
        modified_cross = (pred_cross[0] + e_cross + delta_cross).unsqueeze(0)
        I_w = torch.where(cross_mask.unsqueeze(0).unsqueeze(0), modified_cross, I_w)

        total_consumed = n_consumed_1 + n_consumed_2_added
        diff = (I_w - image_uint8).float()
        mse = (diff ** 2).mean().item()
        psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)

        side = {
            "T_select": T_select,
            "n_payload_bits": total_consumed,
            "loc_p1": loc_p1,
            "nc_p1": nc_p1,
            "loc_p2": loc_p2,
            "nc_p2": nc_p2,
        }
        return NoShiftEmbedResult(
            watermarked_uint8=I_w,
            side_data=side,
            n_payload_bits=total_consumed,
            psnr_quick=psnr,
        )

    @torch.no_grad()
    def extract_noshift(
        self,
        image_w_uint8: torch.Tensor,
        side_data: dict,
    ) -> NoShiftExtractResult:
        assert image_w_uint8.dim() == 4
        B, C, H, W = image_w_uint8.shape
        device = image_w_uint8.device
        T_select = side_data["T_select"]

        cross_mask = checkerboard_mask(H, W, parity=0, device=device, dtype=torch.bool)
        dot_mask = ~cross_mask

        # --------- Reverse Pass 2: recover original Cross + bits_pass2 ---------
        I_dot_only = image_w_uint8 * dot_mask.to(torch.int64)
        pred_cross = self._predict_int(I_dot_only)
        e_prime_cross = (image_w_uint8 - pred_cross)[0]
        rec_e_cross, payload_p2 = extract_one_pass_noshift(
            e_prime_cross, cross_mask,
            side_data["loc_p2"], side_data["nc_p2"],
            T_select=T_select,
        )
        # Restore Cross pixels
        restored_cross = (pred_cross[0] + rec_e_cross).unsqueeze(0)
        I_mid_recon = image_w_uint8.clone()
        I_mid_recon = torch.where(cross_mask.unsqueeze(0).unsqueeze(0), restored_cross, I_mid_recon)

        # --------- Reverse Pass 1: recover original Dot + bits_pass1 ---------
        I_cross_only = I_mid_recon * cross_mask.to(torch.int64)
        pred_dot = self._predict_int(I_cross_only)
        e_prime_dot = (I_mid_recon - pred_dot)[0]
        rec_e_dot, payload_p1 = extract_one_pass_noshift(
            e_prime_dot, dot_mask,
            side_data["loc_p1"], side_data["nc_p1"],
            T_select=T_select,
        )
        restored_dot = (pred_dot[0] + rec_e_dot).unsqueeze(0)
        I_rec = I_mid_recon.clone()
        I_rec = torch.where(dot_mask.unsqueeze(0).unsqueeze(0), restored_dot, I_rec)

        # Concatenate payload bits (pass 1 first, then pass 2)
        all_bits = torch.tensor(payload_p1 + payload_p2, dtype=torch.int64, device=device)
        return NoShiftExtractResult(
            recovered_uint8=I_rec,
            payload_bits=all_bits,
        )


def _smoke_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    H = W = 64
    model = QFormerNoShiftRevMark(
        image_size=H,
        patch_size=8,
        d_quaternion=16,
        depth=2,
        num_heads=4,
        init_T=5.0,
    ).to(device).eval()

    img = torch.randint(0, 256, (1, 3, H, W), dtype=torch.int64, device=device)
    # try embedding 200 bits
    payload = torch.randint(0, 2, (200,), dtype=torch.int64, device=device)

    er = model.embed_noshift(img, payload, T_select=0)
    xr = model.extract_noshift(er.watermarked_uint8, er.side_data)

    bit_exact_img = torch.equal(xr.recovered_uint8, img)
    bit_exact_payload = torch.equal(xr.payload_bits[:er.n_payload_bits], payload[:er.n_payload_bits])

    print(f"  Image round-trip:    {bit_exact_img}")
    print(f"  Payload round-trip:  {bit_exact_payload}  (bits embedded = {er.n_payload_bits} / asked {len(payload)})")
    print(f"  PSNR (random untrained model): {er.psnr_quick:.2f} dB")


if __name__ == "__main__":
    _smoke_test()
