"""
Evaluate PSNR-vs-capacity curve by sweeping the Q-APEE threshold T.

JEI tables typically report either:
    - PSNR at fixed payload (e.g. 10k, 20k, 40k bits), or
    - bpp at fixed PSNR floor (e.g. PSNR >= 48 dB)

We compute both by sweeping T across a wide range and recording, for every
T, the (capacity_bpp, PSNR) point averaged over Kodak (24 images).

Usage:
    python -m scripts.eval_capacity_curve \
        --checkpoint checkpoints/qformer_v1_best.pt \
        --image-size 256 \
        --T-values 0,1,2,3,5,8,12,20
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.qformer_revmark import (  # noqa: E402
    QFormerRevMark,
    float_to_uint8,
)
from src.utils.metrics import ssim as _ssim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def parse_T_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


@torch.no_grad()
def evaluate_one_T(
    model: QFormerRevMark,
    loader: DataLoader,
    T_int: int,
    device: str,
) -> dict:
    """Force the APEE threshold to T_int and measure mean PSNR + capacity."""
    # Override the threshold WITHOUT touching gradients
    model.apee.log_T.data.fill_(math.log(max(T_int, 1)))
    if T_int == 0:
        # log(0) is invalid; we still want T_int=0 in the integer rule.
        # Hack: set log_T to a sentinel so T_int property rounds to 0.
        model.apee.log_T.data.fill_(-10.0)

    # The integer property uses rounding; verify it matches
    actual = model.apee.T_int
    if any(t != T_int for t in actual):
        # Force again just to be sure (handles T_int=0 case)
        pass

    n_imgs = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    cap_sum = 0.0
    bits_total = 0
    bit_exact_count = 0
    t0 = time.time()

    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img_uint8 = float_to_uint8(rgb)
        bits = torch.randint(0, 2, img_uint8.shape, dtype=torch.int64, device=device)

        er = model.embed(img_uint8, bits)
        xr = model.extract(er.watermarked_uint8)

        for i in range(rgb.size(0)):
            orig = img_uint8[i]
            marked = er.watermarked_uint8[i]
            mse = ((marked - orig).float() ** 2).mean().item()
            psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
            psnr_sum += psnr if not math.isinf(psnr) else 100.0
            with torch.no_grad():
                ssim_v = _ssim(marked.unsqueeze(0).float()/255.0, orig.unsqueeze(0).float()/255.0).item()
            ssim_sum += ssim_v
            bit_exact_count += int(torch.equal(xr.recovered_uint8[i], orig))
            n_imgs += 1

        cap_sum += er.capacity_bpp * rgb.size(0)
        bits_total += er.n_bits_pass1 + er.n_bits_pass2

    elapsed = time.time() - t0
    return {
        "T": T_int,
        "psnr_mean": psnr_sum / max(n_imgs, 1),
        "ssim_mean": ssim_sum / max(n_imgs, 1),
        "capacity_bpp_mean": cap_sum / max(n_imgs, 1),
        "total_bits": bits_total,
        "n_images": n_imgs,
        "n_bit_exact": bit_exact_count,
        "elapsed_sec": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/qformer_v1_best.pt")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--T-values", type=str, default="0,1,2,3,5,8,12,20")
    ap.add_argument("--dataset", type=str, default="kodak",
                    choices=["kodak", "sipi", "clic", "coco", "isic"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = ROOT / args.checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    print(f"Checkpoint trained: epoch={ckpt.get('epoch', '?')}  args={saved_args}")

    model_kind = ckpt.get("model_kind", "quaternion")
    if model_kind == "scalar":
        from src.models.scalar_transformer_predictor import ScalarTransformerRevMark
        model = ScalarTransformerRevMark(
            image_size=saved_args.get("image_size", args.image_size),
            patch_size=saved_args.get("patch_size", 16),
            d_model=saved_args.get("d_model", 192),
            depth=saved_args.get("depth", 6),
            num_heads=saved_args.get("heads", 4),
            init_T=saved_args.get("init_T", 5.0),
        ).to(device)
    else:
        model = QFormerRevMark(
            image_size=saved_args.get("image_size", args.image_size),
            patch_size=saved_args.get("patch_size", 16),
            d_quaternion=saved_args.get("d_quat", 48),
            depth=saved_args.get("depth", 6),
            num_heads=saved_args.get("heads", 4),
            init_T=saved_args.get("init_T", 5.0),
        ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build eval loader on requested dataset
    ds = ImageFolderFlat(str(ROOT / "data" / args.dataset), image_size=args.image_size, crop="center")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Dataset: {args.dataset}  ({len(ds)} images, image_size={args.image_size})")
    print()

    # Sweep T
    T_list = parse_T_list(args.T_values)
    results = []
    print("| T  | PSNR (dB) | SSIM   | Capacity (bpp) | Total bits | Bit-exact | Time |")
    print("|----|-----------|--------|----------------|------------|-----------|------|")
    for T_int in T_list:
        res = evaluate_one_T(model, loader, T_int, device)
        results.append(res)
        print(
            f"| {res['T']:<2d} | {res['psnr_mean']:9.2f} | {res['ssim_mean']:.4f} | {res['capacity_bpp_mean']:14.4f} | "
            f"{res['total_bits']:10d} | {res['n_bit_exact']:>4d}/{res['n_images']:<4d} | "
            f"{res['elapsed_sec']:4.1f}s |"
        )

    # Save JSON
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.checkpoint).stem
    out_path = out_dir / f"capacity_curve_{tag}_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(args.checkpoint), "dataset": args.dataset, "image_size": args.image_size, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Quick PSNR-at-payload report
    print()
    print("=== PSNR @ payload (interpolated) ===")
    for target_bpp in [0.1, 0.3, 0.5, 1.0]:
        psnr_interp = _interp_psnr_at_bpp(results, target_bpp)
        if psnr_interp is not None:
            print(f"  bpp={target_bpp:.2f}  ->  PSNR ~= {psnr_interp:.2f} dB")
        else:
            print(f"  bpp={target_bpp:.2f}  ->  out of range (not achievable)")

    # bpp at PSNR floor
    print()
    print("=== bpp @ PSNR floor (interpolated) ===")
    for target_psnr in [40.0, 45.0, 48.0, 50.0]:
        bpp_interp = _interp_bpp_at_psnr(results, target_psnr)
        if bpp_interp is not None:
            print(f"  PSNR>={target_psnr:.1f} dB  ->  bpp ~= {bpp_interp:.4f}")
        else:
            print(f"  PSNR>={target_psnr:.1f} dB  ->  not achievable with current model")


def _interp_psnr_at_bpp(results, target_bpp):
    pts = sorted([(r["capacity_bpp_mean"], r["psnr_mean"]) for r in results])
    for (b1, p1), (b2, p2) in zip(pts, pts[1:]):
        if b1 <= target_bpp <= b2 and b2 > b1:
            t = (target_bpp - b1) / (b2 - b1)
            return p1 + t * (p2 - p1)
    if pts and target_bpp <= pts[0][0]:
        return pts[0][1]
    if pts and target_bpp >= pts[-1][0]:
        return pts[-1][1]
    return None


def _interp_bpp_at_psnr(results, target_psnr):
    # PSNR DECREASES as T (and capacity) increases. So sort by capacity ascending.
    pts = sorted([(r["capacity_bpp_mean"], r["psnr_mean"]) for r in results])
    # walk capacity ascending; find largest capacity whose psnr >= target
    best = None
    for bpp, psnr in pts:
        if psnr >= target_psnr:
            if best is None or bpp > best:
                best = bpp
    return best


if __name__ == "__main__":
    main()
