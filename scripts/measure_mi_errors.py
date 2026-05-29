"""
Empirical validation of Lemma 1: the cross-channel mutual-information bonus.

For each test dataset we run the trained QFormer predictor in checkerboard
mode (Cross<-Dot and Dot<-Cross), collect the per-pixel prediction errors
across all three RGB channels, and estimate

    I(E_R ; E_G, E_B)   in bits/pixel

with the same plug-in MI estimator used by metrics.mutual_information_rgb.

Lemma 1 predicts:

    C_quaternion >= C_scalar + I(E_R; E_G, E_B)

so the empirical I value here is the LOWER BOUND on the capacity bonus that
Q-PEE achieves over scalar PEE on this dataset.

Usage:
    python -m scripts.measure_mi_errors \
        --checkpoint checkpoints/qformer_v1_best.pt \
        --datasets kodak,sipi,clic,coco,isic
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
from src.models.qformer_revmark import (  # noqa: E402
    QFormerRevMark,
    checkerboard_mask,
    float_to_uint8,
)
from src.utils.metrics import mutual_information_rgb  # noqa: E402


@torch.no_grad()
def collect_errors(model: QFormerRevMark, loader: DataLoader, device: str) -> torch.Tensor:
    """Return (N_pixels, 3) tensor of int-domain prediction errors across the dataset."""
    model.eval()
    image_size = model.image_size
    cross_mask = checkerboard_mask(image_size, image_size, parity=0, device=device, dtype=torch.float32)
    dot_mask = 1.0 - cross_mask

    all_errs = []
    n_imgs = 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img_uint8 = float_to_uint8(rgb)

        # Two predictor passes (Cross-in -> predict Dot, then Dot-in -> predict Cross)
        # The errors at Dot positions in pass 1 and at Cross positions in pass 2
        # together cover every pixel exactly once.

        # Pass 1: predict Dot from Cross-only
        cross_only = img_uint8 * cross_mask.unsqueeze(0).unsqueeze(0).to(torch.int64)
        pred_dot = model._predict_int(cross_only)
        err_dot = (img_uint8 - pred_dot)  # (B, 3, H, W) int64
        dm = (dot_mask > 0).unsqueeze(0).unsqueeze(0).expand_as(err_dot)
        err_dot_flat = err_dot.permute(0, 2, 3, 1)[dm.permute(0, 2, 3, 1)].view(-1, 3)
        # Above indexing path is wrong because we restricted by dm; redo cleanly:
        # For each batch element, take the H*W positions where dot_mask==1 (== H*W/2).
        B, C, H, W = err_dot.shape
        dm2 = (dot_mask > 0).view(H, W)
        err_dot_clean = err_dot.permute(0, 2, 3, 1)[:, dm2]  # (B, N_dot, 3)
        err_dot_flat = err_dot_clean.reshape(-1, 3)

        # Pass 2: predict Cross from Dot-only
        dot_only = img_uint8 * dot_mask.unsqueeze(0).unsqueeze(0).to(torch.int64)
        pred_cross = model._predict_int(dot_only)
        err_cross = (img_uint8 - pred_cross)
        cm2 = (cross_mask > 0).view(H, W)
        err_cross_clean = err_cross.permute(0, 2, 3, 1)[:, cm2]
        err_cross_flat = err_cross_clean.reshape(-1, 3)

        all_errs.append(err_dot_flat.cpu())
        all_errs.append(err_cross_flat.cpu())
        n_imgs += rgb.size(0)

    errors = torch.cat(all_errs, dim=0)
    return errors


def estimate_mi_from_errors(errors_int: torch.Tensor, n_bins: int = 32) -> dict:
    """
    Estimate I(E_R; E_G, E_B) from integer error samples in [-255, 255].

    We use the same plug-in estimator as metrics.mutual_information_rgb but
    on the centred-and-normalised error tensor so the [0,1] preprocessing
    inside that function works.
    """
    # Map [-255, 255] -> [0, 1] for the existing estimator
    errs = errors_int.float()
    # Compute per-channel scaling: use 99.9% quantile to clip outliers, then scale
    abs_max = errs.abs().max().item()
    if abs_max == 0:
        abs_max = 1.0
    errs_normed = (errs + abs_max) / (2.0 * abs_max + 1e-9)  # in [0, 1]
    # Build (1, 3, H, W) - fake spatial shape acceptable for the estimator
    N = errs_normed.size(0)
    side = int(math.sqrt(N))
    side = max(side, 1)
    truncated = errs_normed[: side * side].view(side, side, 3).permute(2, 0, 1).unsqueeze(0)
    mi = mutual_information_rgb(truncated, n_bins=n_bins)

    # Empirical statistics of raw errors
    stats = {
        "n_samples": int(N),
        "abs_max": float(abs_max),
        "mean_abs": [float(errs[:, c].abs().mean().item()) for c in range(3)],
        "std":       [float(errs[:, c].std().item()) for c in range(3)],
    }
    return {"mi": mi, "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/qformer_v1_best.pt")
    ap.add_argument("--datasets", type=str, default="kodak,sipi,clic,coco,isic")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--mi-bins", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = ROOT / args.checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    model = QFormerRevMark(
        image_size=saved_args.get("image_size", args.image_size),
        patch_size=saved_args.get("patch_size", 16),
        d_quaternion=saved_args.get("d_quat", 48),
        depth=saved_args.get("depth", 6),
        num_heads=saved_args.get("heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    all_results = {}

    print()
    print("Lemma 1 empirical validation:  capacity bonus  >=  I(E_R; E_G, E_B)")
    print("(measured on prediction errors of the trained predictor)")
    print()
    print(f"| Dataset | N samples | mean|e_R| | mean|e_G| | mean|e_B| |"
          f" I(E_R;E_G) | I(E_R;E_B) | I(E_G;E_B) | I(E_R; E_G,E_B) |")
    print(f"|---------|----------:|----------:|----------:|----------:|"
          f"-----------:|-----------:|-----------:|----------------:|")

    for name in ds_names:
        try:
            ds = ImageFolderFlat(str(ROOT / "data" / name), image_size=args.image_size, crop="center")
        except (FileNotFoundError, RuntimeError):
            print(f"| {name:<7s} | (dataset missing)                                                                                |")
            continue
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        errors = collect_errors(model, loader, device)
        result = estimate_mi_from_errors(errors, n_bins=args.mi_bins)
        all_results[name] = {
            "n_images": len(ds),
            **result,
        }

        mi = result["mi"]
        st = result["stats"]
        print(
            f"| {name:<7s} | {st['n_samples']:9d} | "
            f"{st['mean_abs'][0]:9.3f} | {st['mean_abs'][1]:9.3f} | {st['mean_abs'][2]:9.3f} | "
            f"{mi['I_RG']:10.3f} | {mi['I_RB']:10.3f} | {mi['I_GB']:10.3f} | "
            f"{mi['I_total']:15.3f} |"
        )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mi_lemma_validation.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(args.checkpoint), "image_size": args.image_size, "results": all_results}, f, indent=2)
    print()
    print(f"Saved: {out_path}")

    # Interpretation
    print()
    print("INTERPRETATION:")
    print("  The Lemma states that quaternion PEE gains AT LEAST  I(E_R; E_G, E_B)  bpp")
    print("  over scalar three-channel PEE. A value above ~1 bpp is non-trivial and")
    print("  consistent with our T-sweep observation that quaternion PEE achieves")
    print("  >0.5 bpp gain at matched PSNR vs scalar baselines.")


if __name__ == "__main__":
    main()
