"""
Sample image visualisation figures for the manuscript.

Produces 2x3 grid figures:
    [original | watermarked | 30x abs(diff)]    -- standard PEE mode
    [original | watermarked | 30x abs(diff)]    -- no-shift mode

Each row annotates PSNR / SSIM / payload.

Usage:
    python -m scripts.make_sample_figures --checkpoint checkpoints/qformer_v4_intaware_best.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.qformer_revmark import QFormerRevMark, float_to_uint8, uint8_to_float  # noqa: E402
from src.models.qformer_noshift_revmark import QFormerNoShiftRevMark  # noqa: E402
from src.utils.metrics import ssim as ssim_fn  # noqa: E402


def _load_image(path, image_size=256):
    pil = Image.open(path).convert("RGB")
    tfm = T.Compose([T.Resize(image_size + 16), T.CenterCrop(image_size), T.ToTensor()])
    return tfm(pil)


def _to_uint8_np(t: torch.Tensor):
    """(C, H, W) tensor 0..255 int64 -> H, W, C uint8 numpy."""
    return t.detach().cpu().numpy().astype(np.uint8).transpose(1, 2, 0)


def _psnr_ssim(marked_uint8, orig_uint8):
    diff = (marked_uint8.float() - orig_uint8.float())
    mse = (diff ** 2).mean().item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
    sv = ssim_fn(marked_uint8.unsqueeze(0).float() / 255.0,
                  orig_uint8.unsqueeze(0).float() / 255.0).item()
    return psnr, sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/qformer_v4_intaware_best.pt")
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    sa = ckpt.get("args", {})

    model_std = QFormerRevMark(
        image_size=sa.get("image_size", args.image_size),
        patch_size=sa.get("patch_size", 16),
        d_quaternion=sa.get("d_quat", 48),
        depth=sa.get("depth", 6),
        num_heads=sa.get("heads", 4),
    ).to(device)
    model_std.load_state_dict(ckpt["model"])
    model_std.eval()

    model_ns = QFormerNoShiftRevMark(
        image_size=sa.get("image_size", args.image_size),
        patch_size=sa.get("patch_size", 16),
        d_quaternion=sa.get("d_quat", 48),
        depth=sa.get("depth", 6),
        num_heads=sa.get("heads", 4),
    ).to(device)
    model_ns.load_state_dict(ckpt["model"])
    model_ns.eval()

    # Pick representative images
    sample_paths = [
        (ROOT / "data" / "kodak" / "kodim01.png", "Kodak-01 (people)"),
        (ROOT / "data" / "kodak" / "kodim23.png", "Kodak-23 (lion)"),
        (ROOT / "data" / "sipi" / "4.2.03.tiff", "Baboon"),
        (ROOT / "data" / "sipi" / "4.2.07.tiff", "Peppers"),
    ]

    rgb_imgs = []
    for p, name in sample_paths:
        if not p.exists():
            print(f"  skip missing: {p}")
            continue
        rgb_imgs.append((_load_image(p, args.image_size), name))

    # Build a figure: for each image, three columns (orig | watermarked | 30*|diff|)
    # We do TWO rows per image: standard PEE (T=0) and no-shift T=0 at 5K bits.
    n_img = len(rgb_imgs)
    plt.rcParams.update({"font.size": 8.5})
    fig, axes = plt.subplots(n_img, 5, figsize=(6.28, 1.92 * n_img), dpi=360)
    if n_img == 1:
        axes = axes.reshape(1, -1)

    torch.manual_seed(0)  # sabit payload cekilisi, yeniden uretilebilirlik
    # Configure standard PEE at T=0
    model_std.apee.log_T.data.fill_(-10.0)

    for row, (rgb_float, name) in enumerate(rgb_imgs):
        rgb_float = rgb_float.unsqueeze(0).to(device)
        img_uint8 = float_to_uint8(rgb_float)

        # Standard PEE
        bits = torch.randint(0, 2, img_uint8.shape, dtype=torch.int64, device=device)
        with torch.no_grad():
            er_std = model_std.embed(img_uint8, bits)
        marked_std = er_std.watermarked_uint8[0]
        diff_std = (marked_std.float() - img_uint8[0].float()).abs()
        psnr_std, ssim_std = _psnr_ssim(marked_std, img_uint8[0])
        n_bits_std = er_std.n_bits_pass1 + er_std.n_bits_pass2

        # No-shift PEE at 5K bits target
        payload = torch.randint(0, 2, (5000,), dtype=torch.int64, device=device)
        with torch.no_grad():
            er_ns = model_ns.embed_noshift(img_uint8, payload, T_select=0)
        marked_ns = er_ns.watermarked_uint8[0]
        diff_ns = (marked_ns.float() - img_uint8[0].float()).abs()
        psnr_ns, ssim_ns = _psnr_ssim(marked_ns, img_uint8[0])

        # Cells 0-2: standard PEE
        axes[row, 0].imshow(_to_uint8_np(img_uint8[0]))
        axes[row, 0].set_title(f"{name}\noriginal", fontsize=8.5, linespacing=1.15)
        axes[row, 1].imshow(_to_uint8_np(marked_std))
        axes[row, 1].set_title(f"Std PEE  T=0\nPSNR {psnr_std:.1f} dB\nSSIM {ssim_std:.3f}\n{n_bits_std:,} bits", fontsize=8.5, linespacing=1.15)
        amp_std = (diff_std * 30).clamp(0, 255)
        axes[row, 2].imshow(_to_uint8_np(amp_std.to(torch.int64)))
        axes[row, 2].set_title("30x  |diff|\n(Std PEE)", fontsize=8.5, linespacing=1.15)

        # Cells 3-5: no-shift PEE
        # 4. sutun (yinelenen original) kaldirildi: 1. sutunla bayt-bayt ayniydi
        axes[row, 3].imshow(_to_uint8_np(marked_ns))
        axes[row, 3].set_title(f"No-shift  T=0\nPSNR {psnr_ns:.1f} dB\nSSIM {ssim_ns:.3f}\n{er_ns.n_payload_bits:,} bits", fontsize=8.5, linespacing=1.15)
        amp_ns = (diff_ns * 30).clamp(0, 255)
        axes[row, 4].imshow(_to_uint8_np(amp_ns.to(torch.int64)))
        axes[row, 4].set_title("30x  |diff|\n(no-shift)", fontsize=8.5, linespacing=1.15)

        for c in range(5):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    pass  # suptitle kaldirildi (SPIE)
    plt.tight_layout(pad=0.8, h_pad=1.9, w_pad=0.35)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_samples"
    fig.savefig(out_path.with_suffix(".png"), dpi=360, bbox_inches="tight", pad_inches=0.07)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.07)
    print(f"Saved: {out_path}.png  and  .pdf")


if __name__ == "__main__":
    main()
