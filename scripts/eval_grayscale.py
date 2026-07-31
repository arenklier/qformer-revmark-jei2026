"""
Fair comparison to Hu 2021 (CNNP) and ICNNP on the grayscale 512x512 setup.

Loads images, converts to grayscale 512x512 (Y luminance), and applies
the classical 4-neighbour rhombus predictor + no-shift PEE. Reports
per-image PSNR/SSIM/MS-SSIM/LPIPS at the standard Hu-2021 payload sizes
(10k, 20k, 40k bits), with 95% CI.

The single-channel pipeline reuses the same Adaptive Q-PEE module by
treating the gray image as a 3-channel image with G = B = R (i.e.,
processing only the R channel and discarding the duplicates). This is
the simplest pathway that does not require retraining; cleaner
single-channel variants are left for future work.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.classical_predictor import ClassicalNoShiftRevMark  # noqa: E402
from src.models.qformer_revmark import float_to_uint8  # noqa: E402
from src.utils.metrics import ssim as ssim_fn  # noqa: E402

try:
    import lpips as lpips_mod
    from pytorch_msssim import ms_ssim
    _HAVE_FULL = True
except Exception:
    _HAVE_FULL = False


_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class GrayscaleAt512(Dataset):
    """Load images, convert to grayscale, resize/crop to 512x512.
    Returns a triple-replicated 'RGB' tensor compatible with the existing
    3-channel pipeline."""

    def __init__(self, root, size=512):
        self.root = Path(root)
        self.size = size
        self.files = sorted(
            f for f in self.root.rglob("*")
            if f.is_file() and f.suffix.lower() in _EXTS
            and not f.name.startswith(".")
            and "__MACOSX" not in str(f)
        )
        self.tfm = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(size + 32),
            T.CenterCrop(size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        g = self.tfm(img)  # (1, H, W)
        # replicate so existing 3-channel API works; we will only process channel 0
        rgb_like = g.expand(3, -1, -1).contiguous()
        return rgb_like, str(self.files[idx])


def _ci95(values):
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return 1.96 * arr.std(ddof=1) / math.sqrt(len(arr))


@torch.no_grad()
def measure(model, lpips_model, loader, n_payload, device, gray_only=True):
    psnrs, ssims, msssims, lps, embs, sides = [], [], [], [], [], []
    n_imgs = 0
    n_bit_exact = 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img = float_to_uint8(rgb)
        for i in range(rgb.size(0)):
            # Random payload — embed via the 3-channel API (all 3 will get bits)
            payload = torch.randint(0, 2, (n_payload,), dtype=torch.int64, device=device)
            er = model.embed_noshift(img[i:i+1], payload, T_select=0)
            xr = model.extract_noshift(er.watermarked_uint8, er.side_data)

            if gray_only:
                # Use only channel 0 for metrics (others are identical duplicates)
                mark_g = er.watermarked_uint8[:, :1].float() / 255.0
                orig_g = img[i:i+1, :1].float() / 255.0
                # Build a 3-channel tensor for LPIPS / MS-SSIM
                mark_3 = mark_g.expand(-1, 3, -1, -1)
                orig_3 = orig_g.expand(-1, 3, -1, -1)
                diff = (er.watermarked_uint8[:, :1] - img[i:i+1, :1]).float()
            else:
                mark_3 = er.watermarked_uint8.float() / 255.0
                orig_3 = img[i:i+1].float() / 255.0
                diff = (er.watermarked_uint8 - img[i:i+1]).float()

            mse = (diff ** 2).mean().item()
            psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
            ssim_val = ssim_fn(mark_3, orig_3).item()
            msssim_val = ms_ssim(mark_3, orig_3, data_range=1.0).item() if _HAVE_FULL else float("nan")
            lp = lpips_model(mark_3 * 2 - 1, orig_3 * 2 - 1).item() if _HAVE_FULL else float("nan")

            psnrs.append(psnr if not math.isinf(psnr) else 100.0)
            ssims.append(ssim_val)
            msssims.append(msssim_val)
            lps.append(lp)
            embs.append(er.n_payload_bits)
            n_loc_bits = len(er.side_data["loc_p1"]) + len(er.side_data["loc_p2"])
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
    ap.add_argument("--dataset", type=str, default="kodak",
                    help="Folder name under data/ (kodak, sipi, clic)")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--payloads", type=str, default="10000,20000,40000")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  image_size={args.image_size}  gray-replicated 3-channel pipeline")

    ds = GrayscaleAt512(ROOT / "data" / args.dataset, size=args.image_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset: {args.dataset}  ({len(ds)} images, grayscale 512x512)")
    print()

    model = ClassicalNoShiftRevMark(image_size=args.image_size, patch_size=16).to(device).eval()
    if _HAVE_FULL:
        lpips_model = lpips_mod.LPIPS(net="alex", verbose=False).to(device).eval()
    else:
        class _Dummy:
            def __call__(self, *a, **k):
                return torch.tensor(float("nan"))
        lpips_model = _Dummy()
        print("WARN: lpips / pytorch-msssim missing, those columns will be NaN")

    payloads = [int(x) for x in args.payloads.split(",")]
    rows = []
    print("| Payload | PSNR (dB)              | MS-SSIM  | LPIPS    | Bit-exact |")
    print("|--------:|-----------------------:|---------:|---------:|----------:|")
    for n in payloads:
        r = measure(model, lpips_model, loader, n, device, gray_only=True)
        rows.append(r)
        ps, ms_, lp = r["PSNR"], r["MS_SSIM"], r["LPIPS"]
        print(f"| {n:>7,d} | {ps['mean']:5.2f} ± {ps['ci95']:.2f} (sd {ps['std']:.2f}) "
              f"| {ms_['mean']:.5f} "
              f"| {lp['mean']:.5f} "
              f"| {r['n_bit_exact']:>3d}/{r['n_imgs']:<3d} |")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grayscale512_classical_noshift_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "image_size": args.image_size,
                   "method": "Classical 4-neigh. + no-shift PEE on grayscale 512x512",
                   "results": rows}, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
