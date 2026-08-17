"""Side-channel cost of the resolution sweep, at the same fixed 500-bit payload."""
import math, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from verify_revision_experiments import (KODAK_DIR, load_gray_uint8, embed_noshift_gray, psnr)


def Hb(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


paths = sorted(KODAK_DIR.glob("kodim*.png"))[:24]
g = torch.Generator().manual_seed(0)
N = 500
print(f"{'size':>10} {'PSNR':>8} {'cand (raw map)':>15} {'fill':>7} {'coded map':>10} {'net':>9}")
print("-" * 68)
rows = {}
for size in (128, 256, 512, 1024):
    ps, cands, embs = [], [], []
    for p in paths:
        img = load_gray_uint8(p, size)
        payload = torch.randint(0, 2, (N,), generator=g, dtype=torch.int64)
        marked, side, n_emb = embed_noshift_gray(img, payload, T_sel=0)
        ps.append(psnr(img, marked))
        cands.append(sum(side["ncand"]))
        embs.append(n_emb)
    C, E, P = float(np.mean(cands)), float(np.mean(embs)), float(np.mean(ps))
    coded = C * Hb(E / C)
    rows[size] = dict(psnr=P, cand=C, fill=E / C, coded=coded, net=E - coded, emb=E)
    print(f"{size:>5}x{size:<4} {P:>8.2f} {C:>15,.0f} {E/C:>7.4f} {coded:>10,.0f} {E-coded:>+9,.0f}")

json.dump(rows, open("/home/ayhan/qformer-revmark/results/resolution_side.json", "w"), indent=1)
