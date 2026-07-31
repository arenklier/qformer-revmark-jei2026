"""
Plug Hu & Xiang (IEEE SPL 2021) CNNP pretrained predictor into our
no-shift PEE pipeline on grayscale 512x512 images for a direct
predictor-vs-predictor comparison with our classical and quaternion
predictors.

This is NOT a full reproduction of Hu 2021's MATLAB pipeline (which uses
their own histogram-shifting / expansion-embedding MATLAB scripts).
Instead, we hold the embedding stage fixed at our no-shift PEE (T_sel=0)
and substitute only the predictor, so the comparison isolates the
predictor's contribution.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Hu 2021 model architecture (verbatim from their repo's model/cnnp.py)
class HuCNNP(nn.Module):
    def __init__(self):
        super().__init__()
        channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channel, 3, 1, 0), nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, channel, 5, 1, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, channel, 7, 1, 2), nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, 1, 3, 1, 1),
        )

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(x)
        o3 = self.conv3(x)
        o4 = self.conv4(o1 + o2 + o3)
        o5 = self.conv5(o1 + o2 + o3 + o4)
        return o5


# ============================================================
# Wrapper: Hu CNNP as a 3-channel grayscale predictor for our pipeline
# ============================================================
class HuPredictorWrapper(nn.Module):
    """Wrap Hu's 1-channel CNNP to accept 3-channel grayscale-replicated RGB."""

    def __init__(self, hu_ckpt_path: str, image_size: int = 512, patch_size: int = 16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.cnnp = HuCNNP()
        ckpt = torch.load(hu_ckpt_path, map_location="cpu", weights_only=False)
        # Hu's checkpoint stores under 'network' key
        state = ckpt.get("network", ckpt)
        self.cnnp.load_state_dict(state)
        self.cnnp.eval()

    @torch.no_grad()
    def forward(self, rgb_float: torch.Tensor) -> torch.Tensor:
        # rgb_float: (B, 3, H, W) in [0,1], replicate-grayscale convention
        gray = rgb_float[:, :1]   # use channel 0
        # Hu's conv1 has padding=0 -> output shrinks by 2 pixels.
        # Reflect-pad by 1 pixel on each side so output matches input size.
        gray_padded = torch.nn.functional.pad(gray, (1, 1, 1, 1), mode="reflect")
        pred1 = self.cnnp(gray_padded)  # (B, 1, H, W) matches input H, W
        # Clamp to valid pixel range and replicate to 3 channels
        return pred1.clamp(0.0, 1.0).expand(-1, 3, -1, -1).contiguous()


# ============================================================
# Eval driver -- mirrors eval_grayscale.py but with the Hu predictor
# ============================================================
import io
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset

from src.pee.adaptive_pee import AdaptiveQuaternionPEE
from src.models.qformer_revmark import (
    QFormerRevMark, float_to_uint8, checkerboard_mask, uint8_to_float,
)
from src.models.qformer_noshift_revmark import QFormerNoShiftRevMark
from src.utils.metrics import ssim as ssim_fn

try:
    import lpips as lpips_mod
    from pytorch_msssim import ms_ssim
    _HAVE_FULL = True
except Exception:
    _HAVE_FULL = False


_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm"}


class GrayscaleAt512(Dataset):
    def __init__(self, root, size=512):
        self.root = Path(root)
        self.size = size
        self.files = sorted(
            f for f in self.root.rglob("*")
            if f.is_file() and f.suffix.lower() in _EXTS
            and not f.name.startswith(".") and "__MACOSX" not in str(f)
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
        g = self.tfm(img)
        return g.expand(3, -1, -1).contiguous(), str(self.files[idx])


# Build a no-shift wrapper that uses a custom predictor
class CustomPredictorNoShiftRevMark(QFormerNoShiftRevMark):
    def __init__(self, predictor: nn.Module, image_size: int, patch_size: int = 16, init_T: float = 5.0):
        nn.Module.__init__(self)
        self.predictor = predictor
        self.apee = AdaptiveQuaternionPEE(init_T=init_T)
        self.image_size = image_size


def _ci95(values):
    if len(values) < 2: return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return 1.96 * arr.std(ddof=1) / math.sqrt(len(arr))


@torch.no_grad()
def measure(model, lpips_model, loader, n_payload, device):
    psnrs, msssims, lps, embs = [], [], [], []
    n_imgs, n_bit_exact = 0, 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img = float_to_uint8(rgb)
        for i in range(rgb.size(0)):
            payload = torch.randint(0, 2, (n_payload,), dtype=torch.int64, device=device)
            er = model.embed_noshift(img[i:i+1], payload, T_select=0)
            try:
                xr = model.extract_noshift(er.watermarked_uint8, er.side_data)
                be = int(torch.equal(xr.recovered_uint8, img[i:i+1]))
            except Exception:
                be = 0
            mark_g = er.watermarked_uint8[:, :1].float() / 255.0
            orig_g = img[i:i+1, :1].float() / 255.0
            mark_3 = mark_g.expand(-1, 3, -1, -1)
            orig_3 = orig_g.expand(-1, 3, -1, -1)
            diff = (er.watermarked_uint8[:, :1] - img[i:i+1, :1]).float()
            mse = (diff ** 2).mean().item()
            psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
            msssim_val = ms_ssim(mark_3, orig_3, data_range=1.0).item() if _HAVE_FULL else float("nan")
            lp = lpips_model(mark_3 * 2 - 1, orig_3 * 2 - 1).item() if _HAVE_FULL else float("nan")
            psnrs.append(psnr if not math.isinf(psnr) else 100.0)
            msssims.append(msssim_val)
            lps.append(lp)
            embs.append(er.n_payload_bits)
            n_bit_exact += be
            n_imgs += 1

    def stats(v):
        a = np.asarray(v, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a)>1 else 0),
                "ci95": float(_ci95(v))}
    return {
        "n_payload": n_payload, "n_imgs": n_imgs, "n_bit_exact": n_bit_exact,
        "PSNR": stats(psnrs), "MS_SSIM": stats(msssims),
        "LPIPS": stats(lps), "embedded": stats(embs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hu-ckpt", type=str,
                    default=str(ROOT / "external/hu2021/CNN-Prediction-Based-Reversible-Data-Hiding/model_parameter/model_state.pth"))
    ap.add_argument("--dataset", type=str, default="kodak")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--payloads", type=str, default="10000,20000")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Hu ckpt: {args.hu_ckpt}")

    hu_pred = HuPredictorWrapper(args.hu_ckpt, image_size=args.image_size).to(device).eval()
    model = CustomPredictorNoShiftRevMark(hu_pred, image_size=args.image_size).to(device).eval()

    ds = GrayscaleAt512(ROOT / "data" / args.dataset, size=args.image_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"Dataset: {args.dataset}  ({len(ds)} images, grayscale 512x512)")

    if _HAVE_FULL:
        lpips_model = lpips_mod.LPIPS(net="alex", verbose=False).to(device).eval()
    else:
        class _D:
            def __call__(self, *a, **k): return torch.tensor(float("nan"))
        lpips_model = _D()

    print()
    print("Predictor: Hu & Xiang CNNP (IEEE SPL 2021), pretrained weights, no-shift PEE T_sel=0")
    print()
    print("| Payload | PSNR (dB)              | MS-SSIM  | LPIPS    | bit-exact |")
    print("|--------:|-----------------------:|---------:|---------:|----------:|")
    payloads = [int(x) for x in args.payloads.split(",")]
    rows = []
    for n in payloads:
        r = measure(model, lpips_model, loader, n, device)
        rows.append(r)
        ps, ms_, lp = r["PSNR"], r["MS_SSIM"], r["LPIPS"]
        print(f"| {n:>7,d} | {ps['mean']:5.2f} ± {ps['ci95']:.2f} (sd {ps['std']:.2f}) "
              f"| {ms_['mean']:.5f} "
              f"| {lp['mean']:.5f} "
              f"| {r['n_bit_exact']:>3d}/{r['n_imgs']:<3d} |")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hu2021_predictor_noshift_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({"hu_ckpt": args.hu_ckpt, "dataset": args.dataset,
                   "image_size": args.image_size, "results": rows}, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
