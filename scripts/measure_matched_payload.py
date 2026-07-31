"""
Matched-payload comparison: standard PEE versus no-shift PEE at the SAME payload.

The paper currently contrasts no-shift at 1000 bits (74 dB) with standard PEE at its
full T=0 capacity (~25k bits, 51 dB). Those are different payloads, so the comparison
understates nothing but is not like-for-like.

The point being tested here: in canonical PEE the histogram shift is applied to every
pixel with |e| > T regardless of how many bits are embedded, so its distortion is
essentially payload-independent. If that holds, standard PEE asked for 1000 bits still
pays ~51 dB, and the honest matched-payload gap to no-shift is ~23 dB.

Standard PEE at threshold T (canonical Thodi-Rodriguez form):
    e' = 2e + b     |e| <= T     (expansion, only b=1 actually moves the pixel)
    e' = e + T + 1  e > T        (shift)
    e' = e - T      e < -T       (shift; a no-op at T=0)
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ayhan/qformer-revmark")
sys.path.insert(0, str(ROOT / "scripts"))

from verify_revision_experiments import (  # noqa: E402
    KODAK_DIR, load_gray_uint8, embed_noshift_gray, extract_noshift_gray,
    checker_masks, rhombus_predict, psnr,
)

torch.manual_seed(0)
np.random.seed(0)


def standard_pee_embed(img, payload, T=0):
    """Canonical two-pass standard PEE. Returns (marked, n_embedded).

    Unused expansion capacity is filled with zero bits, which for e=0 leaves the
    pixel unchanged, so this is the most favourable reading for standard PEE.
    """
    H, W = img.shape
    cross, dot = checker_masks(H, W)
    current = img.clone()
    cursor = 0
    pay = payload.tolist()

    for pass_mask in (dot, cross):
        pred = rhombus_predict(current, pass_mask)
        e = current - pred
        interior = torch.zeros_like(pass_mask)
        interior[1:H - 1, 1:W - 1] = True
        active = pass_mask & interior
        new = current.clone()

        for pos in active.flatten().nonzero(as_tuple=False).flatten().tolist():
            yi, xi = pos // W, pos % W
            ev = int(e[yi, xi].item())
            pv = int(pred[yi, xi].item())
            if abs(ev) <= T:
                b = pay[cursor] if cursor < len(pay) else 0
                if cursor < len(pay):
                    cursor += 1
                enew = 2 * ev + b
            elif ev > T:
                enew = ev + T + 1
            else:
                enew = ev - T
            val = pv + enew
            if 0 <= val <= 255:                 # skip pixels that would overflow
                new[yi, xi] = val
        current = new
    return current, cursor


def main():
    paths = sorted(KODAK_DIR.glob("kodim*.png"))[:24]
    size = 256
    payloads = (1000, 5000, 10000, 25000)

    print(f"Kodak {len(paths)} images, gray {size}x{size}, classical rhombus predictor, T=0")
    print()
    print(f"{'payload':>8} | {'standard PEE':>22} | {'no-shift PEE':>22}")
    print(f"{'':>8} | {'PSNR (dB)':>11} {'embedded':>10} | {'PSNR (dB)':>11} {'embedded':>10}")
    print("-" * 60)

    for N in payloads:
        sp, se, np_, ne = [], [], [], []
        for p in paths:
            img = load_gray_uint8(p, size)
            payload = torch.randint(0, 2, (N,), dtype=torch.int64)

            m_std, emb_std = standard_pee_embed(img, payload, T=0)
            sp.append(psnr(m_std, img))
            se.append(emb_std)

            m_ns, side, emb_ns = embed_noshift_gray(img, payload, T_sel=0)
            np_.append(psnr(m_ns, img))
            ne.append(emb_ns)

        print(f"{N:>8} | {np.mean(sp):>11.2f} {int(np.mean(se)):>10} | "
              f"{np.mean(np_):>11.2f} {int(np.mean(ne)):>10}")

    print()
    print("If standard-PEE PSNR is flat across payloads, its distortion is dominated by")
    print("the histogram shift, which no-shift PEE removes. The gap at the SAME payload")
    print("is then the honest measure of what the mechanism buys in the image domain.")


if __name__ == "__main__":
    main()
