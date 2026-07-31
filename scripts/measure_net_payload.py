"""
Net payload accounting for no-shift PEE, measured rather than assumed.

For each payload target we record, from real embeddings on Kodak:
    C        candidate count the map has to cover
    N        bits actually embedded
    H(map)   entropy of the location bitmap, C * Hb(N/C), which is what an
             arithmetic coder achieves on it
    net      N - H(map), the bits gained minus the bits shipped on the side

The crossover, where net turns positive, is where the payload fills enough of the
candidate set that the map becomes cheaper than the payload it indexes.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ayhan/qformer-revmark")
sys.path.insert(0, str(ROOT / "scripts"))

from verify_revision_experiments import (  # noqa: E402
    KODAK_DIR, load_gray_uint8, embed_noshift_gray, psnr,
)

torch.manual_seed(0)


def Hb(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def main():
    paths = sorted(KODAK_DIR.glob("kodim*.png"))[:24]
    size = 256
    targets = (1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000)

    print(f"Kodak {len(paths)} images, gray {size}x{size}, T_sel=0")
    print()
    print(f"{'target':>7} | {'embedded':>9} | {'cands':>7} | {'fill':>6} | "
          f"{'map bits':>9} | {'net bits':>9} | {'PSNR':>6}")
    print("-" * 72)

    rows = []
    for T in targets:
        emb, cand, mapb, ps = [], [], [], []
        for p in paths:
            img = load_gray_uint8(p, size)
            payload = torch.randint(0, 2, (T,), dtype=torch.int64)
            marked, side, n = embed_noshift_gray(img, payload, T_sel=0)
            C = len(side["loc"])
            emb.append(n); cand.append(C)
            mapb.append(C * Hb(n / C) if C else 0.0)
            ps.append(psnr(marked, img))
        N, C, M, P = np.mean(emb), np.mean(cand), np.mean(mapb), np.mean(ps)
        rows.append((T, N, C, M, N - M, P))
        print(f"{T:>7} | {N:>9.0f} | {C:>7.0f} | {N/C:>6.3f} | {M:>9.0f} | "
              f"{N - M:>+9.0f} | {P:>6.2f}")

    print()
    best = max(rows, key=lambda r: r[4])
    print(f"Best net payload in this range: {best[4]:+.0f} bits at a {best[0]}-bit target "
          f"({best[5]:.2f} dB).")
    C = rows[0][2]
    lo, hi = 0.5, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if Hb(mid) > mid:
            lo = mid
        else:
            hi = mid
    print(f"Analytic crossover: net turns positive once the payload fills "
          f"{lo*100:.1f}% of the candidate set, about {lo*C:.0f} bits at C={C:.0f}.")
    print("Reaching that fill factor requires the candidate set to be close to the")
    print("payload size, which is a design choice about how candidates are selected.")


if __name__ == "__main__":
    main()
