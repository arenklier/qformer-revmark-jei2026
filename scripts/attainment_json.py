"""Per-image attainment data + Jensen diagnostics for Table 2."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from measure_attainment import load_rgb, build_design, lin_mmse_var, normal_scores, cond_mi_bits, KODAK
from scipy.stats import spearmanr, pearsonr

rows = []
for p in sorted(KODAK.glob("kodim*.png"))[:24]:
    img = load_rgb(p, 256)
    for c in range(3):
        Xc, Nc, Nmc = build_design(img, c)
        s2s = lin_mmse_var(Xc, Nc)
        s2j = lin_mmse_var(Xc, np.column_stack([Nc, Nmc]))
        Z = normal_scores(np.column_stack([Xc, Nc, Nmc]))
        I = cond_mi_bits(Z, [0], [5, 6], [1, 2, 3, 4])
        rows.append(dict(img=p.stem, ch="RGB"[c], s2_scalar=s2s, s2_joint=s2j,
                         achieved=s2j / s2s, I_bits=I, floor=2.0 ** (-2.0 * I)))

json.dump(rows, open("/home/ayhan/qformer-revmark/results/attainment_kodak.json", "w"), indent=1)

print("=" * 70)
print("1) PER-IMAGE IDENTITY CHECK  floor_i == 2^(-2 I_i)")
err = max(abs(r["floor"] - 2.0 ** (-2.0 * r["I_bits"])) for r in rows)
print(f"   max |floor_i - 2^-2I_i| = {err:.3e}   -> exact by construction\n")

print("2) JENSEN GAP  (why the printed table rows do not satisfy the identity)")
print(f"   {'Ch':>4} {'mean I':>8} {'2^-2*meanI':>11} {'mean(2^-2I)':>12} {'gap':>7} {'achieved':>9}")
allI, allF, allA = [], [], []
for c in "RGB":
    sub = [r for r in rows if r["ch"] == c]
    mI = np.mean([r["I_bits"] for r in sub]); mF = np.mean([r["floor"] for r in sub])
    mA = np.mean([r["achieved"] for r in sub])
    print(f"   {c:>4} {mI:8.3f} {2**(-2*mI):11.3f} {mF:12.3f} {mF-2**(-2*mI):+7.3f} {mA:9.3f}")
    allI += [r["I_bits"] for r in sub]; allF += [r["floor"] for r in sub]; allA += [r["achieved"] for r in sub]
mI, mF, mA = np.mean(allI), np.mean(allF), np.mean(allA)
print(f"   {'Mean':>4} {mI:8.3f} {2**(-2*mI):11.3f} {mF:12.3f} {mF-2**(-2*mI):+7.3f} {mA:9.3f}\n")

print("3) ACHIEVED vs FLOOR per image  (Corollary 1 predicts equality under Gaussianity)")
d = [r["achieved"] - r["floor"] for r in rows]
below = sum(1 for x in d if x < 0)
print(f"   achieved < floor in {below}/{len(rows)} pairs   mean diff = {np.mean(d):+.4f}")
print(f"   ratio achieved/floor: mean {np.mean([r['achieved']/r['floor'] for r in rows]):.4f}"
      f"  min {min(r['achieved']/r['floor'] for r in rows):.4f}"
      f"  max {max(r['achieved']/r['floor'] for r in rows):.4f}\n")

print("4) CORRELATIONS")
rho, prho = spearmanr(allI, allA); r_, pr = pearsonr(allI, allA)
print(f"   Spearman rho(I, achieved) = {rho:+.3f}  p={prho:.2e}")
print(f"   Pearson    r(I, achieved) = {r_:+.3f}  p={pr:.2e}")
rho2, p2 = spearmanr(allF, allA)
print(f"   Spearman rho(floor, achieved) = {rho2:+.3f}  p={p2:.2e}\n")

print("5) CLUSTERED CHECK (R1: 72 pairs are 24 images x 3 channels, not independent)")
per = []
for im in sorted(set(r["img"] for r in rows)):
    sub = [r for r in rows if r["img"] == im]
    per.append((np.mean([r["I_bits"] for r in sub]), np.mean([r["achieved"] for r in sub])))
rho3, p3 = spearmanr([x[0] for x in per], [x[1] for x in per])
print(f"   per-image means, n=24:  Spearman rho = {rho3:+.3f}  p = {p3:.2e}")
wr = []
for c in "RGB":
    sub = [r for r in rows if r["ch"] == c]
    wr.append(spearmanr([r["I_bits"] for r in sub], [r["achieved"] for r in sub])[0])
print(f"   within-channel rho (n=24 each): R {wr[0]:+.3f}  G {wr[1]:+.3f}  B {wr[2]:+.3f}")
