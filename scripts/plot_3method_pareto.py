"""
Multi-method PSNR-vs-bpp Pareto plot for the manuscript.

Reads capacity_curve_*.json for Classical, QFormer-v3, QFormer-v4
and produces a single overlaid figure suitable as Figure 2 of the paper.

Output: paper/fig_pareto.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    # 4-neighbour classical T-sweep was computed inline; recreate from inline numbers.
    classical_T = [0, 1, 2, 3, 5]
    classical_psnr = [51.56, 45.66, 42.47, 40.44, 37.86]
    classical_bpp  = [0.554, 1.284, 1.697, 1.954, 2.271]

    # QFormer-v3 (saved JSON)
    p_v3 = ROOT / "results" / "capacity_curve_qformer_v3_best_kodak.json"
    if p_v3.exists():
        d = json.load(open(p_v3))
        v3 = sorted([(r["capacity_bpp_mean"], r["psnr_mean"], r["T"]) for r in d["results"]])
    else:
        v3 = []

    # QFormer-v4 (int-aware)
    p_v4 = ROOT / "results" / "capacity_curve_qformer_v4_intaware_best_kodak.json"
    d4 = json.load(open(p_v4))
    v4 = sorted([(r["capacity_bpp_mean"], r["psnr_mean"], r["T"]) for r in d4["results"]])

    # No-shift highlights (Q-v4 on Kodak)
    p_ns_v4 = ROOT / "results" / "noshift_qformer_v4_intaware_best_kodak_T0.json"
    d_ns = json.load(open(p_ns_v4))
    H = 256
    pix = H * H * 3
    ns_points = []
    for r in d_ns["results"]:
        bpp = r["embedded_mean"] / pix
        ns_points.append((bpp, r["psnr_mean"]))

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=140)

    ax.plot(classical_bpp, classical_psnr, marker="s", color="tab:blue", linewidth=1.8,
            markersize=7, label="Classical 4-neigh.  (standard PEE)")
    if v3:
        ax.plot([x[0] for x in v3], [x[1] for x in v3], marker="o", color="tab:green",
                linewidth=1.8, markersize=7, label="QFormer-v3  (standard PEE)")
    ax.plot([x[0] for x in v4], [x[1] for x in v4], marker="^", color="tab:red",
            linewidth=1.8, markersize=7, label="QFormer-v4 int-aware  (standard PEE)")

    # No-shift points highlighted
    if ns_points:
        ax.plot([x[0] for x in ns_points], [x[1] for x in ns_points], marker="*",
                color="tab:purple", linewidth=2.5, markersize=14,
                linestyle="--", label="QFormer-v4 no-shift PEE  (this work)")

    # T annotations
    for x, y, T in v4:
        if T in (0, 1, 5):
            ax.annotate(f"T={T}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=10, alpha=0.7, color="tab:red")

    ax.set_xlabel("Embedding capacity (bpp)")
    ax.set_ylabel("PSNR (dB)  --  watermarked vs original")
    ax.set_title("QFormer-RevMark vs classical baseline (Kodak, 24 images)")
    ax.grid(False)
    ax.set_facecolor("white")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.set_xlim(left=0, right=2.5)
    ax.set_ylim(bottom=30, top=78)
    fig.tight_layout()

    out_path = ROOT / "paper" / "fig_pareto"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_path}.pdf  and  .png")


if __name__ == "__main__":
    main()
