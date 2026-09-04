"""
Q-Q plot of prediction errors against Gaussian for the Gaussian assumption
sanity check (Lemma 1, Assumption A1).

Loads a trained predictor, computes prediction errors on Kodak / CLIC /
USC-SIPI, and produces a 3-panel Q-Q plot. Saves to paper/fig_qq.pdf.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.qformer_revmark import (  # noqa: E402
    QFormerRevMark, checkerboard_mask, float_to_uint8,
)


@torch.no_grad()
def collect_errors(model, loader, device, max_samples=200000):
    """Return concat of i,j,k channel errors from checkerboard pass-1."""
    image_size = model.image_size
    cross_mask = checkerboard_mask(image_size, image_size, parity=0, device=device, dtype=torch.float32)
    dot_mask = 1.0 - cross_mask
    out = {"E_R": [], "E_G": [], "E_B": []}
    n = 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img = float_to_uint8(rgb)
        # Pass 1
        cross_only = img * cross_mask.unsqueeze(0).unsqueeze(0).to(torch.int64)
        pred_dot = model._predict_int(cross_only)
        err = (img - pred_dot)[:, :, dot_mask.bool()].cpu().numpy()  # (B, 3, n_dot)
        # err.shape will be (B, 3, n_dot_pixels)
        for c, name in enumerate(("E_R", "E_G", "E_B")):
            out[name].append(err[:, c].ravel())
        n += err.shape[0]
        if sum(len(x) for x in out["E_R"]) >= max_samples:
            break
    return {k: np.concatenate(v)[:max_samples] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/qformer_v5_heldout_best.pt")
    ap.add_argument("--dataset", type=str, default="kodak")
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    sa = ckpt.get("args", {})
    model = QFormerRevMark(
        image_size=sa.get("image_size", args.image_size),
        patch_size=sa.get("patch_size", 16),
        d_quaternion=sa.get("d_quat", 48),
        depth=sa.get("depth", 6),
        num_heads=sa.get("heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = ImageFolderFlat(str(ROOT / "data" / args.dataset), image_size=args.image_size, crop="center")
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    errs = collect_errors(model, loader, device)

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(6.28, 2.55), dpi=400)
    channels = ["E_R", "E_G", "E_B"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    for ax, ch, col in zip(axes, channels, colors):
        sample = errs[ch].astype(np.float64)
        # Clip extreme outliers to focus on the support that matters for PEE
        clip = 30.0
        sample_clip = sample[np.abs(sample) <= clip]
        n = len(sample_clip)
        # Empirical quantiles
        emp = np.sort(sample_clip)
        # Theoretical normal quantiles (same mean/std)
        mu = sample_clip.mean()
        sd = sample_clip.std()
        from scipy import stats as scs
        theo = scs.norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n), loc=mu, scale=sd)
        ax.scatter(theo, emp, s=2, alpha=0.4, c=col, rasterized=True)
        lo, hi = min(theo.min(), emp.min()), max(theo.max(), emp.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        # Shaded |e|<=8 region
        ax.axvspan(-8, 8, facecolor="0.88", edgecolor="0.55", linewidth=0.5, zorder=0, label=r"PEE-relevant $|e|\!\leq\!8$")
        ax.set_title(f"{ch}   n={n:,}\n$\\mu$={mu:.2f},  $\\sigma$={sd:.2f}", fontsize=8, linespacing=1.2)
        ax.set_xlabel(r"Theoretical quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.grid(False)
        ax.set_facecolor("white")
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=8, framealpha=1.0)

    pass  # suptitle kaldirildi
    fig.tight_layout()
    out_path = ROOT / "paper" / "fig_qq"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    print(f"Saved: {out_path}.pdf  and  .png")
    print(f"Stats (clipped to |e|<=30):")
    for ch in channels:
        sample = errs[ch][np.abs(errs[ch]) <= 30]
        print(f"  {ch}: n={len(sample):,}, mean={sample.mean():.3f}, std={sample.std():.3f}, "
              f"frac |e|<=8 = {(np.abs(sample) <= 8).mean():.3f}")


if __name__ == "__main__":
    main()
