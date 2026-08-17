"""
Plot PSNR-vs-capacity curves from results/capacity_curve_*.json.

Produces figures suitable for inclusion in the manuscript:
    results/fig_psnr_vs_bpp.png   (and .pdf)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="kodak,clic")
    ap.add_argument("--out", type=str, default="results/fig_psnr_vs_bpp")
    args = ap.parse_args()

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    markers = ["o", "s", "^", "D", "v"]
    for i, name in enumerate(datasets):
        path = ROOT / "results" / f"capacity_curve_{name}.json"
        if not path.exists():
            print(f"  WARN: missing {path}", file=sys.stderr)
            continue
        data = json.load(open(path))
        xs = [r["capacity_bpp_mean"] for r in data["results"]]
        ys = [r["psnr_mean"] for r in data["results"]]
        Ts = [r["T"] for r in data["results"]]
        ax.plot(xs, ys, marker=markers[i % len(markers)], linewidth=1.7,
                markersize=6, label=name.upper())
        # T labels next to a few points
        for x, y, T in zip(xs, ys, Ts):
            if T in (0, 1, 5, 8, 12, 20):
                ax.annotate(f"T={T}", (x, y), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, alpha=0.7)

    # Reference horizontal lines for JEI-style PSNR floors
    for psnr_floor, lab in []:
        ax.axhline(y=psnr_floor, linestyle="--", linewidth=0.8, alpha=0.5, color="gray")
        ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else 1.0,
                psnr_floor + 0.3, lab, ha="right", va="bottom", fontsize=7, alpha=0.6, color="gray")

    ax.set_xlabel("Embedding capacity (bpp)")
    ax.set_ylabel("PSNR (dB) -- watermarked vs original")
    ax.set_title("QFormer-RevMark: PSNR-vs-capacity (T sweep)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"Saved: {out_path.with_suffix('.png')}  and  .pdf")


if __name__ == "__main__":
    main()
