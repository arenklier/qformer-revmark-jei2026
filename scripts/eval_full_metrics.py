"""
Full-metric evaluation: per-image PSNR / SSIM / MS-SSIM / LPIPS + 95% CI.

Reports mean +/- std and 95% confidence interval across the test set.
Replaces the saturated SSIM-only reporting with a perceptual metric (LPIPS)
and a multi-scale metric (MS-SSIM) plus uncertainty bands suitable for the
JEI-style results tables.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.qformer_noshift_revmark import QFormerNoShiftRevMark  # noqa: E402
from src.models.qformer_revmark import float_to_uint8  # noqa: E402
from src.utils.metrics import ssim as ssim_fn  # noqa: E402

import lpips
from pytorch_msssim import ms_ssim


def _ci95(values: list) -> float:
    """95% confidence interval half-width via t-distribution approximation."""
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    std = arr.std(ddof=1)
    return 1.96 * std / math.sqrt(len(arr))


@torch.no_grad()
def measure(model, lpips_model, loader, n_payload, device):
    psnrs, ssims, msssims, lps, embs, sides = [], [], [], [], [], []
    n_bit_exact = 0
    n_imgs = 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img = float_to_uint8(rgb)
        for i in range(rgb.size(0)):
            payload = torch.randint(0, 2, (n_payload,), dtype=torch.int64, device=device)
            er = model.embed_noshift(img[i:i+1], payload, T_select=0)
            xr = model.extract_noshift(er.watermarked_uint8, er.side_data)

            orig_f = img[i:i+1].float() / 255.0
            mark_f = er.watermarked_uint8.float() / 255.0
            diff = (er.watermarked_uint8 - img[i:i+1]).float()
            mse = (diff ** 2).mean().item()
            psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)

            ssim_val = ssim_fn(mark_f, orig_f).item()
            msssim_val = ms_ssim(mark_f, orig_f, data_range=1.0).item()
            # LPIPS expects [-1, 1] range
            lp = lpips_model(mark_f * 2 - 1, orig_f * 2 - 1).item()

            n_loc_bits = len(er.side_data["loc_p1"]) + len(er.side_data["loc_p2"])

            psnrs.append(psnr if not math.isinf(psnr) else 100.0)
            ssims.append(ssim_val)
            msssims.append(msssim_val)
            lps.append(lp)
            embs.append(er.n_payload_bits)
            sides.append(n_loc_bits)
            n_bit_exact += int(torch.equal(xr.recovered_uint8, img[i:i+1]))
            n_imgs += 1

    def stats(vals):
        a = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "std":  float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "ci95": float(_ci95(vals)),
            "min":  float(a.min()),
            "max":  float(a.max()),
        }

    return {
        "n_payload": n_payload,
        "n_imgs": n_imgs,
        "n_bit_exact": n_bit_exact,
        "PSNR": stats(psnrs),
        "SSIM": stats(ssims),
        "MS_SSIM": stats(msssims),
        "LPIPS": stats(lps),
        "embedded": stats(embs),
        "side_bits": stats(sides),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/qformer_v5_heldout_best.pt")
    ap.add_argument("--dataset", type=str, default="kodak")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--payloads", type=str, default="1000,5000,10000")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    sa = ckpt.get("args", {})
    model = QFormerNoShiftRevMark(
        image_size=sa.get("image_size", args.image_size),
        patch_size=sa.get("patch_size", 16),
        d_quaternion=sa.get("d_quat", 48),
        depth=sa.get("depth", 6),
        num_heads=sa.get("heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
    lpips_model.eval()

    ds = ImageFolderFlat(str(ROOT / "data" / args.dataset), image_size=args.image_size, crop="center")
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset: {args.dataset}  ({len(ds)} images)")
    print()

    payloads = [int(x) for x in args.payloads.split(",")]
    rows = []
    print("| Payload | PSNR (dB)              | MS-SSIM             | LPIPS (alex)         | Side bits           |")
    print("|--------:|-----------------------:|--------------------:|---------------------:|--------------------:|")
    for n in payloads:
        r = measure(model, lpips_model, loader, n, device)
        rows.append(r)
        ps, ms_, lp, sd = r["PSNR"], r["MS_SSIM"], r["LPIPS"], r["side_bits"]
        print(f"| {n:>7,d} | {ps['mean']:5.2f} ± {ps['ci95']:.2f} ({ps['std']:.2f}) "
              f"| {ms_['mean']:.5f} ± {ms_['ci95']:.5f} "
              f"| {lp['mean']:.5f} ± {lp['ci95']:.5f} "
              f"| {sd['mean']:>7,.0f} ± {sd['ci95']:.0f} |")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.checkpoint).stem
    out_path = out_dir / f"fullmetrics_{tag}_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(args.checkpoint), "dataset": args.dataset,
                   "image_size": args.image_size, "results": rows}, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
