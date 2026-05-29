"""
Evaluate the no-shift PEE variant on a dataset.

For each image we embed N random bits using the no-shift wrapper and measure
the watermarked PSNR. Side-channel (location bitmap) is computed but NOT
counted against PSNR (it is reported separately as overhead in bytes).
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
from src.models.qformer_noshift_revmark import QFormerNoShiftRevMark  # noqa: E402
from src.models.qformer_revmark import QFormerRevMark, float_to_uint8  # noqa: E402
from src.utils.metrics import ssim as _ssim  # noqa: E402


def _build_model_from_ckpt(ckpt, device, image_size):
    saved = ckpt.get("args", {})
    if ckpt.get("model_kind") == "scalar":
        raise NotImplementedError("no-shift wrapper currently only supports quaternion checkpoints")
    model = QFormerNoShiftRevMark(
        image_size=saved.get("image_size", image_size),
        patch_size=saved.get("patch_size", 16),
        d_quaternion=saved.get("d_quat", 48),
        depth=saved.get("depth", 6),
        num_heads=saved.get("heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def measure_noshift(model, loader, n_target, T_select, device):
    psnr_sum = 0.0
    ssim_sum = 0.0
    embedded_sum = 0
    sidebytes_sum = 0
    n_imgs = 0
    bit_exact_count = 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img_uint8 = float_to_uint8(rgb)
        for i in range(rgb.size(0)):
            payload = torch.randint(0, 2, (n_target,), dtype=torch.int64, device=device)
            er = model.embed_noshift(img_uint8[i:i+1], payload, T_select=T_select)
            xr = model.extract_noshift(er.watermarked_uint8, er.side_data)
            bit_exact_count += int(torch.equal(xr.recovered_uint8, img_uint8[i:i+1]))
            psnr_sum += er.psnr_quick if not math.isinf(er.psnr_quick) else 100.0
            with torch.no_grad():
                ssim_val = _ssim(er.watermarked_uint8.float()/255.0, img_uint8[i:i+1].float()/255.0).item()
            ssim_sum += ssim_val
            embedded_sum += er.n_payload_bits
            # side data: each location bit is 1 byte in python list; for paper we report bits
            n_loc_bits = len(er.side_data["loc_p1"]) + len(er.side_data["loc_p2"])
            sidebytes_sum += (n_loc_bits + 7) // 8
            n_imgs += 1
    return {
        "n_target": n_target,
        "T_select": T_select,
        "psnr_mean": psnr_sum / max(n_imgs, 1),
        "ssim_mean": ssim_sum / max(n_imgs, 1),
        "embedded_mean": embedded_sum / max(n_imgs, 1),
        "side_bytes_mean": sidebytes_sum / max(n_imgs, 1),
        "n_imgs": n_imgs,
        "n_bit_exact": bit_exact_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/qformer_v2_best.pt")
    ap.add_argument("--dataset", type=str, default="kodak")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--payloads", type=str, default="1000,5000,10000,20000,40000")
    ap.add_argument("--T-select", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model = _build_model_from_ckpt(ckpt, device, args.image_size)
    ds = ImageFolderFlat(str(ROOT / "data" / args.dataset), image_size=args.image_size, crop="center")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset: {args.dataset}  ({len(ds)} images, image_size={args.image_size})")
    print(f"T_select: {args.T_select}")
    print()

    targets = [int(x) for x in args.payloads.split(",") if x.strip()]
    print("| Payload requested | Embedded actual | PSNR (dB) | SSIM   | Side bytes | Bit-exact |")
    print("|------------------:|---------------:|----------:|-------:|-----------:|----------:|")
    results = []
    for n in targets:
        r = measure_noshift(model, loader, n, args.T_select, device)
        results.append(r)
        print(
            f"| {r['n_target']:>17,d} | {r['embedded_mean']:>14,.0f} | "
            f"{r['psnr_mean']:9.2f} | {r['ssim_mean']:.4f} | {r['side_bytes_mean']:>10,.0f} | "
            f"{r['n_bit_exact']:>4d}/{r['n_imgs']:<4d} |"
        )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.checkpoint).stem
    out_path = out_dir / f"noshift_{tag}_{args.dataset}_T{args.T_select}.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "dataset": args.dataset,
            "image_size": args.image_size,
            "T_select": args.T_select,
            "results": results,
        }, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
