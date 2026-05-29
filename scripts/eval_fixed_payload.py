"""
Evaluate PSNR at fixed payload sizes (JEI-style reporting).

For each target N (in bits), find the smallest integer threshold T whose
capacity is >= N, then embed exactly N bits (rest of the input bit stream
zero-padded) and measure the watermarked PSNR on the eval dataset.

This mirrors the way most reversible-WM papers report "PSNR @ 10k / 20k /
40k bits" rather than continuous capacity curves.

Usage:
    python -m scripts.eval_fixed_payload \
        --checkpoint checkpoints/qformer_v2_best.pt \
        --dataset kodak \
        --payloads 10000,20000,40000,80000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.qformer_revmark import (  # noqa: E402
    QFormerRevMark,
    float_to_uint8,
)


def _build_model_from_ckpt(ckpt, device, image_size):
    saved_args = ckpt.get("args", {})
    model_kind = ckpt.get("model_kind", "quaternion")
    if model_kind == "scalar":
        from src.models.scalar_transformer_predictor import ScalarTransformerRevMark
        model = ScalarTransformerRevMark(
            image_size=saved_args.get("image_size", image_size),
            patch_size=saved_args.get("patch_size", 16),
            d_model=saved_args.get("d_model", 192),
            depth=saved_args.get("depth", 6),
            num_heads=saved_args.get("heads", 4),
        ).to(device)
    else:
        model = QFormerRevMark(
            image_size=saved_args.get("image_size", image_size),
            patch_size=saved_args.get("patch_size", 16),
            d_quaternion=saved_args.get("d_quat", 48),
            depth=saved_args.get("depth", 6),
            num_heads=saved_args.get("heads", 4),
        ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_kind


@torch.no_grad()
def measure_single_image_at_T(model: QFormerRevMark, img_uint8, bits, T_int: int):
    model.apee.log_T.data.fill_(math.log(max(T_int, 1)))
    if T_int == 0:
        model.apee.log_T.data.fill_(-10.0)
    er = model.embed(img_uint8.unsqueeze(0), bits.unsqueeze(0))
    marked = er.watermarked_uint8[0]
    diff = (marked - img_uint8).float()
    mse = (diff ** 2).mean().item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
    total_bits = er.n_bits_pass1 + er.n_bits_pass2
    return {"psnr": psnr, "capacity_bits": total_bits, "T": T_int, "mse": mse}


@torch.no_grad()
def find_smallest_T_for_payload(model, img_uint8, n_target, T_max=20, device="cuda"):
    """Return the smallest T whose capacity reaches n_target on this image."""
    bits_full = torch.randint(0, 2, img_uint8.shape, dtype=torch.int64, device=device)
    best = None
    for T_int in range(T_max + 1):
        r = measure_single_image_at_T(model, img_uint8, bits_full, T_int)
        if r["capacity_bits"] >= n_target:
            r["target_bits"] = n_target
            return r
        best = r
    # If we never reach n_target, return the maximum-T result
    best["target_bits"] = n_target
    best["unreachable"] = True
    return best


@torch.no_grad()
def measure_at_fixed_payload(
    model,
    loader: DataLoader,
    n_target: int,
    device: str,
) -> dict:
    psnr_sum = 0.0
    cap_sum = 0
    n_imgs = 0
    chosen_Ts = []
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img_uint8 = float_to_uint8(rgb)
        for i in range(rgb.size(0)):
            r = find_smallest_T_for_payload(model, img_uint8[i], n_target, device=device)
            psnr_sum += r["psnr"] if not math.isinf(r["psnr"]) else 100.0
            cap_sum += r["capacity_bits"]
            chosen_Ts.append(r["T"])
            n_imgs += 1
    return {
        "n_target": n_target,
        "psnr_mean": psnr_sum / max(n_imgs, 1),
        "capacity_mean": cap_sum / max(n_imgs, 1),
        "n_imgs": n_imgs,
        "T_min": int(min(chosen_Ts)) if chosen_Ts else None,
        "T_max": int(max(chosen_Ts)) if chosen_Ts else None,
        "T_med": int(sorted(chosen_Ts)[len(chosen_Ts) // 2]) if chosen_Ts else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/qformer_v2_best.pt")
    ap.add_argument("--dataset", type=str, default="kodak")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--payloads", type=str, default="5000,10000,20000,40000,80000")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = ROOT / args.checkpoint
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, kind = _build_model_from_ckpt(ckpt, device, args.image_size)
    print(f"Model kind: {kind}")

    ds = ImageFolderFlat(str(ROOT / "data" / args.dataset), image_size=args.image_size, crop="center")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset: {args.dataset}  ({len(ds)} images, image_size={args.image_size})")
    print()

    targets = [int(x) for x in args.payloads.split(",") if x.strip()]
    results = []
    print("| Payload (bits) | bpp | PSNR (dB) | Used T (min/med/max) | Avg cap (bits) |")
    print("|---------------:|----:|----------:|---------------------:|---------------:|")
    pix_per_img = args.image_size * args.image_size * 3
    for n in targets:
        r = measure_at_fixed_payload(model, loader, n, device)
        results.append(r)
        bpp = n / pix_per_img
        print(
            f"| {n:>14,d} | {bpp:.4f} | {r['psnr_mean']:9.2f} | "
            f"{r['T_min']:>3d} / {r['T_med']:>3d} / {r['T_max']:>3d}"
            f"            | {int(r['capacity_mean']):>12,d} |"
        )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.checkpoint).stem
    out_path = out_dir / f"fixed_payload_{tag}_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "dataset": args.dataset,
            "image_size": args.image_size,
            "pixels_per_image": pix_per_img,
            "results": results,
        }, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
