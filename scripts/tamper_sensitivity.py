"""Tamper sensitivity: how small a modification does the extractor detect?

Answers the practical question the 0/24 attack table leaves open, namely where the
detection boundary sits and whether the extractor ever returns wrong bits silently.
Outcomes per image: REFUSED (consistency check fires), CLEAN (payload recovered
exactly), or SILENT (returned a payload that differs from the embedded one).
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from verify_revision_experiments import (KODAK_DIR, load_gray_uint8,
                                         embed_noshift_gray, extract_noshift_gray)

paths = sorted(KODAK_DIR.glob("kodim*.png"))[:24]
g = torch.Generator().manual_seed(0)
rng = np.random.default_rng(0)
SIZE, N = 256, 5000
KS = [1, 2, 4, 8, 16, 32, 64, 256, 1024]

print(f"Kodak {len(paths)} images, gray {SIZE}x{SIZE}, {N}-bit payload, T_sel=0")
print(f"{'pixels changed':>15} {'refused':>9} {'clean':>7} {'SILENT':>8} {'detect rate':>12}")
print("-" * 56)
rows = {}
for k in KS:
    ref = clean = silent = 0
    for p in paths:
        img = load_gray_uint8(p, SIZE)
        payload = torch.randint(0, 2, (N,), generator=g, dtype=torch.int64)
        marked, side, n_emb = embed_noshift_gray(img, payload, T_sel=0)
        t = marked.clone()
        H, W = t.shape
        idx = rng.choice(H * W, size=k, replace=False)
        for j in idx:
            y, x = int(j // W), int(j % W)
            t[y, x] = int(np.clip(int(t[y, x]) + (1 if rng.random() < 0.5 else -1), 0, 255))
        try:
            _, rec = extract_noshift_gray(t, side)
            ok = len(rec) == n_emb and all(int(a) == int(b) for a, b in zip(rec, payload[:n_emb].tolist()))
            if ok:
                clean += 1
            else:
                silent += 1
        except Exception:
            ref += 1
    n = len(paths)
    rows[k] = dict(refused=ref, clean=clean, silent=silent, detect=(ref + silent) / n)
    print(f"{k:>15} {ref:>6}/{n:<2} {clean:>4}/{n:<2} {silent:>5}/{n:<2} {(ref+silent)/n:>11.1%}")

json.dump(rows, open("/home/ayhan/qformer-revmark/results/tamper_sensitivity.json", "w"), indent=1)
print("\nSILENT is the dangerous column: a wrong payload returned without any warning.")
