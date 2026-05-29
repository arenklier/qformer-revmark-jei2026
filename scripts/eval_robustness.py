"""
Robustness analysis for QFormer-RevMark (no-shift mode).

Reversible watermarking is fragile by design -- the integer-arithmetic PEE
inverse relies on bit-exact pixel values. We characterise the bit-recovery
rate (BER) under three standard image-processing attacks:

    - JPEG re-compression at Q in {50, 70, 90}
    - Additive Gaussian noise of sigma in {2, 5, 10}
    - Random spatial crop of {5%, 10%}

For each attack we attempt extraction and compute:

    BER        = fraction of payload bits that flip
    PSNR_w     = PSNR of attacked image vs original (= attack severity)
    n_recover  = fraction of test images where the full extraction
                 pipeline did not crash (defensive check)

This is reported as a transparency measure; reversible WM is not expected
to survive these attacks in the strict sense, but the BER tells the reader
how graceful the failure is.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.qformer_noshift_revmark import QFormerNoShiftRevMark  # noqa: E402
from src.models.qformer_revmark import float_to_uint8, uint8_to_float  # noqa: E402


def jpeg_compress(img_uint8: torch.Tensor, quality: int) -> torch.Tensor:
    """img_uint8: (1, 3, H, W) int64 in [0, 255].  Returns attacked img_uint8."""
    arr = img_uint8[0].cpu().to(torch.uint8).numpy().transpose(1, 2, 0)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    attacked = Image.open(buf).convert("RGB")
    import numpy as np
    np_arr = np.array(attacked).transpose(2, 0, 1)
    return torch.tensor(np_arr, dtype=torch.int64, device=img_uint8.device).unsqueeze(0)


def add_gaussian_noise(img_uint8: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(img_uint8.float()) * float(sigma)
    return (img_uint8.float() + noise).round().clamp(0, 255).to(torch.int64)


def random_crop_pad(img_uint8: torch.Tensor, crop_frac: float) -> torch.Tensor:
    """Crop crop_frac of edges and zero-pad back to original size."""
    _, _, H, W = img_uint8.shape
    crop = int(min(H, W) * crop_frac)
    if crop == 0:
        return img_uint8.clone()
    out = img_uint8.clone()
    out[:, :, :crop, :] = 0
    out[:, :, -crop:, :] = 0
    out[:, :, :, :crop] = 0
    out[:, :, :, -crop:] = 0
    return out


def attack_psnr(orig, attacked):
    d = (attacked.float() - orig.float())
    mse = (d ** 2).mean().item()
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def _build_model(ckpt, device, image_size):
    sa = ckpt.get("args", {})
    m = QFormerNoShiftRevMark(
        image_size=sa.get("image_size", image_size),
        patch_size=sa.get("patch_size", 16),
        d_quaternion=sa.get("d_quat", 48),
        depth=sa.get("depth", 6),
        num_heads=sa.get("heads", 4),
    ).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


def measure_one_attack(model, loader, n_payload, attack_fn, attack_name, device):
    bit_err_sum = 0
    total_bits = 0
    attack_psnr_sum = 0.0
    n_imgs = 0
    n_crashed = 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img = float_to_uint8(rgb)
        for i in range(rgb.size(0)):
            payload = torch.randint(0, 2, (n_payload,), dtype=torch.int64, device=device)
            with torch.no_grad():
                er = model.embed_noshift(img[i:i+1], payload, T_select=0)
            attacked = attack_fn(er.watermarked_uint8)
            attack_psnr_sum += attack_psnr(er.watermarked_uint8, attacked)

            try:
                with torch.no_grad():
                    xr = model.extract_noshift(attacked, er.side_data)
                if xr.payload_bits.size(0) > 0:
                    n_compare = min(er.n_payload_bits, xr.payload_bits.size(0))
                    flips = (xr.payload_bits[:n_compare] != payload[:n_compare]).sum().item()
                    bit_err_sum += flips
                    total_bits += n_compare
            except Exception as e:
                n_crashed += 1
            n_imgs += 1
    ber = bit_err_sum / max(total_bits, 1)
    return {
        "attack": attack_name,
        "n_target_bits": n_payload,
        "ber": ber,
        "attack_psnr_mean": attack_psnr_sum / max(n_imgs, 1),
        "n_imgs": n_imgs,
        "n_crashed": n_crashed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/qformer_v4_intaware_best.pt")
    ap.add_argument("--dataset", type=str, default="kodak")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-payload", type=int, default=5000)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model = _build_model(ckpt, device, args.image_size)
    ds = ImageFolderFlat(str(ROOT / "data" / args.dataset), image_size=args.image_size, crop="center")
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset: {args.dataset}  ({len(ds)} imgs)  payload={args.n_payload}")
    print()

    attacks = [
        (lambda im: jpeg_compress(im, 90), "JPEG Q=90"),
        (lambda im: jpeg_compress(im, 70), "JPEG Q=70"),
        (lambda im: jpeg_compress(im, 50), "JPEG Q=50"),
        (lambda im: add_gaussian_noise(im, 2), "Gauss sigma=2"),
        (lambda im: add_gaussian_noise(im, 5), "Gauss sigma=5"),
        (lambda im: add_gaussian_noise(im, 10), "Gauss sigma=10"),
        (lambda im: random_crop_pad(im, 0.05), "crop 5%"),
        (lambda im: random_crop_pad(im, 0.10), "crop 10%"),
    ]
    results = []
    print("| Attack             | Attack PSNR (dB) | BER     | n_crashed |")
    print("|--------------------|------------------|---------|-----------|")
    for fn, name in attacks:
        r = measure_one_attack(model, loader, args.n_payload, fn, name, device)
        results.append(r)
        print(f"| {r['attack']:<19s}| {r['attack_psnr_mean']:>16.2f} | {r['ber']:.4f}  | {r['n_crashed']:>3d}/{r['n_imgs']:<3d} |")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.checkpoint).stem
    out_path = out_dir / f"robustness_{tag}_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(args.checkpoint), "dataset": args.dataset,
                   "n_payload": args.n_payload, "results": results}, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
