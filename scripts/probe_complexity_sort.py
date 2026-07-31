"""
Feasibility probe for complexity-ordered candidate selection.

The map currently has to cover every position with |e| <= T_sel+1, about 26 000 of
them, while only about 41 percent are actually embeddable. That fill factor caps the
net payload below zero for any target.

If we instead order positions by a complexity measure computed from the UNCHANGED
context (so the extractor can reproduce the order) and only use a prefix, the map
shrinks to cover the prefix alone. The question this probe answers: does complexity
ordering concentrate the |e|=0 positions near the front?

Reported per prefix length k: the fraction of the top-k that is embeddable, and the
map cost that would follow, against the payload it could carry.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ayhan/qformer-revmark")
sys.path.insert(0, str(ROOT / "scripts"))

from verify_revision_experiments import (  # noqa: E402
    KODAK_DIR, load_gray_uint8, checker_masks, rhombus_predict,
)

torch.manual_seed(0)


def Hb(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def local_complexity(img, pass_mask):
    """Spread of the four orthogonal neighbours. Uses only context pixels, which are
    unchanged during this pass, so embedder and extractor compute the same value."""
    H, W = img.shape
    up = torch.zeros_like(img); up[1:, :] = img[:-1, :]
    dn = torch.zeros_like(img); dn[:-1, :] = img[1:, :]
    lf = torch.zeros_like(img); lf[:, 1:] = img[:, :-1]
    rt = torch.zeros_like(img); rt[:, :-1] = img[:, 1:]
    stack = torch.stack([up, dn, lf, rt]).float()
    return stack.max(0).values - stack.min(0).values      # range of the neighbourhood


def main():
    paths = sorted(KODAK_DIR.glob("kodim*.png"))[:24]
    size = 256
    prefixes = (1250, 2500, 5000, 10000, 20000)

    print(f"Kodak {len(paths)} images, gray {size}x{size}, Dot pass, T_sel=0")
    print()
    print(f"{'prefix k':>9} | {'embeddable in top-k':>20} | {'fill':>6} | "
          f"{'map bits':>9} | {'payload':>8} | {'net':>8}")
    print("-" * 74)

    agg = {k: [] for k in prefixes}
    baseline = []
    for p in paths:
        img = load_gray_uint8(p, size)
        H, W = img.shape
        cross, dot = checker_masks(H, W)
        pred = rhombus_predict(img, dot)
        e = img - pred
        cx = local_complexity(img, dot)

        interior = torch.zeros_like(dot)
        interior[1:H - 1, 1:W - 1] = True
        sel = dot & interior
        idx = sel.flatten().nonzero(as_tuple=False).flatten()
        cvals = cx.flatten()[idx]
        evals = e.flatten()[idx]
        pvals = (pred + e).flatten()[idx]              # pixel value, for the 255 guard

        order = torch.argsort(cvals)                   # smooth regions first
        emb_ok = ((evals.abs() == 0) & (pvals < 255)).to(torch.int64)[order]

        baseline.append(float(emb_ok.float().mean()))
        for k in prefixes:
            kk = min(k, len(emb_ok))
            agg[k].append(float(emb_ok[:kk].float().mean()))

    print(f"{'(all)':>9} | {np.mean(baseline):>19.1%} | {'':>6} | {'':>9} | {'':>8} | {'':>8}")
    for k in prefixes:
        fill = np.mean(agg[k])
        payload = k * fill
        mapbits = k * Hb(fill)
        print(f"{k:>9} | {fill:>19.1%} | {fill:>6.3f} | {mapbits:>9.0f} | "
              f"{payload:>8.0f} | {payload - mapbits:>+8.0f}")

    print()
    print("Net turns positive where the fill factor exceeds 77.3 percent, since that is")
    print("where Hb(p) drops below p. A prefix that reaches such a fill would make the")
    print("side channel cheaper than the payload it carries.")


if __name__ == "__main__":
    main()
