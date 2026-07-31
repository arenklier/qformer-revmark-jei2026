"""
Empirical attainment of the entropy-power floor on real images.

For each colour channel c at Dot positions:
    X_c        the pixel to predict
    N_c        within-channel context (4 orthogonal neighbours in channel c)
    N_-c       cross-channel context (co-located pixels of the other two channels)

We measure
    sigma2_scalar   linear MMSE residual variance from N_c alone      (exact, no assumptions)
    sigma2_joint    linear MMSE residual variance from N_c u N_-c     (exact, no assumptions)
    achieved ratio  sigma2_joint / sigma2_scalar
    floor ratio     2^(-2 I(X_c ; N_-c | N_c)), the entropy-power floor ratio

The conditional MI is estimated with a Gaussian-copula (normal-scores) estimator,
which handles non-Gaussian marginals. Attainment = how close the achieved ratio
comes to the floor ratio (1.0 means the predictor extracts all the cross-channel
information that the floor allows).
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import norm, rankdata

KODAK = Path("/home/ayhan/qformer-revmark/data/kodak")
rng = np.random.default_rng(0)


def load_rgb(p, size=256):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float64)


def build_design(img, c):
    """Return X_c, N_c (4 cols), N_-c (2 cols) at Dot positions, interior only."""
    H, W, _ = img.shape
    yy, xx = np.mgrid[1:H - 1, 1:W - 1]
    dot = ((yy + xx) % 2 == 1)
    ys, xs = yy[dot], xx[dot]

    Xc = img[ys, xs, c]
    up = img[ys - 1, xs, c]
    dn = img[ys + 1, xs, c]
    lf = img[ys, xs - 1, c]
    rt = img[ys, xs + 1, c]
    Nc = np.column_stack([up, dn, lf, rt])

    others = [k for k in range(3) if k != c]
    Nmc = np.column_stack([img[ys, xs, k] for k in others])
    return Xc, Nc, Nmc


def lin_mmse_var(X, C):
    A = np.column_stack([C, np.ones(len(C))])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return float(((X - A @ coef) ** 2).mean())


def normal_scores(M):
    """Rank-based transform of each column to standard normal scores."""
    M = np.atleast_2d(M.T).T
    out = np.empty_like(M, dtype=np.float64)
    n = M.shape[0]
    for j in range(M.shape[1]):
        r = rankdata(M[:, j], method="average")
        out[:, j] = norm.ppf(r / (n + 1.0))
    return out


def cond_cov(Sig, i_idx, a_idx):
    """Conditional covariance of block i given block a."""
    Sii = Sig[np.ix_(i_idx, i_idx)]
    if len(a_idx) == 0:
        return Sii
    Sia = Sig[np.ix_(i_idx, a_idx)]
    Saa = Sig[np.ix_(a_idx, a_idx)]
    return Sii - Sia @ np.linalg.pinv(Saa) @ Sia.T


def cond_mi_bits(Z, x_idx, b_idx, a_idx):
    """I(X;B|A) in bits under a Gaussian copula (Z already normal scores)."""
    Sig = np.cov(Z, rowvar=False)
    Sxa = cond_cov(Sig, x_idx, a_idx)
    Sba = cond_cov(Sig, b_idx, a_idx)
    Sxba = cond_cov(Sig, x_idx + b_idx, a_idx)
    val = (np.linalg.slogdet(Sxa)[1] + np.linalg.slogdet(Sba)[1]
           - np.linalg.slogdet(Sxba)[1])
    return 0.5 * val / math.log(2.0)


def main():
    paths = sorted(KODAK.glob("kodim*.png"))[:24]
    print(f"Kodak images: {len(paths)}   (RGB 256x256, Dot positions, interior)")
    print()
    print(f"{'Ch':>3} | {'s2_scalar':>10} | {'s2_joint':>10} | {'achieved':>9} | "
          f"{'I(bits)':>8} | {'floor':>8} | {'attain':>7}")
    print("-" * 74)

    agg = {c: dict(ach=[], flo=[], s2s=[], s2j=[], mi=[]) for c in range(3)}
    for p in paths:
        img = load_rgb(p, 256)
        for c in range(3):
            Xc, Nc, Nmc = build_design(img, c)
            s2s = lin_mmse_var(Xc, Nc)
            s2j = lin_mmse_var(Xc, np.column_stack([Nc, Nmc]))
            achieved = s2j / s2s

            Z = normal_scores(np.column_stack([Xc, Nc, Nmc]))
            I = cond_mi_bits(Z, [0], [5, 6], [1, 2, 3, 4])
            floor = 2.0 ** (-2.0 * I)

            agg[c]["ach"].append(achieved)
            agg[c]["flo"].append(floor)
            agg[c]["s2s"].append(s2s)
            agg[c]["s2j"].append(s2j)
            agg[c]["mi"].append(I)

    names = "RGB"
    all_ach, all_flo, all_mi = [], [], []
    for c in range(3):
        a = np.mean(agg[c]["ach"]); f = np.mean(agg[c]["flo"])
        mi = np.mean(agg[c]["mi"])
        print(f"{names[c]:>3} | {np.mean(agg[c]['s2s']):10.3f} | {np.mean(agg[c]['s2j']):10.3f} | "
              f"{a:9.4f} | {mi:8.4f} | {f:8.4f} |")
        all_ach.append(a); all_flo.append(f); all_mi.append(mi)

    print("-" * 74)
    A, F, M = np.mean(all_ach), np.mean(all_flo), np.mean(all_mi)
    print(f"{'avg':>3} | {'':>10} | {'':>10} | {A:9.4f} | {M:8.4f} | {F:8.4f} |")

    # Does the quantity the theorem names predict the observed gain?
    from scipy.stats import spearmanr, pearsonr
    mi_all = np.concatenate([agg[c]["mi"] for c in range(3)])
    ach_all = np.concatenate([agg[c]["ach"] for c in range(3)])
    rho, prho = spearmanr(mi_all, ach_all)
    r, pr = pearsonr(mi_all, ach_all)
    print()
    print(f"Across all {len(mi_all)} (image, channel) pairs:")
    print(f"  Spearman rho(I, achieved ratio) = {rho:+.3f}   p = {prho:.2e}")
    print(f"  Pearson   r  (I, achieved ratio) = {r:+.3f}   p = {pr:.2e}")
    print()
    print("Reading:")
    print("  A negative correlation means: the larger the conditional mutual information")
    print("  that the theorem names, the larger the observed variance reduction.")
    print("  Note the entropy-power floor bounds each variance separately, so the achieved")
    print("  ratio is not constrained to lie above the floor ratio; the floor states what")
    print("  cross-channel context makes possible, not what a given predictor attains.")


if __name__ == "__main__":
    main()
