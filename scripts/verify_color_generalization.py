"""
Two proactive strengthening experiments for the JEI revision:

  1. CIEDE2000 (Delta-E) colour-difference metric between original and marked
     images, for the classical no-shift predictor and the deep v5 predictor
     at RGB 256x256 Kodak. Reports colour fidelity, complementing PSNR/SSIM.

  2. Five-dataset generalisation table: classical no-shift at a fixed 5000-bit
     payload on Kodak, USC-SIPI, CLIC-2020, COCO val-2017, and ISIC dermoscopy,
     reporting PSNR + Delta-E + bit-exact.

Runs on L40S. Classical path needs no deep model; the v5 Delta-E uses the
released QFormerNoShiftRevMark checkpoint.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000

ROOT = Path("/home/ayhan/qformer-revmark")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_revision_experiments import (  # noqa: E402
    embed_noshift_gray, extract_noshift_gray, psnr,
)

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXTS = (".png", ".tiff", ".tif", ".jpg", ".jpeg", ".bmp")


def list_images(root, limit=None):
    root = Path(root)
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in EXTS]
    return files[:limit] if limit else files


def load_rgb(path, size=256):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
    return torch.tensor(np.array(im), dtype=torch.int64).permute(2, 0, 1)  # (3,H,W) int64


def rgb_noshift_classical(img_rgb, total_payload, T=0):
    per = total_payload // 3
    marked_ch, embs, allok = [], 0, True
    for c in range(3):
        pay = torch.randint(0, 2, (per,), dtype=torch.int64)
        m, side, n = embed_noshift_gray(img_rgb[c], pay, T_sel=T)
        rec, rp = extract_noshift_gray(m, side)
        allok = allok and torch.equal(rec, img_rgb[c]) and rp[:n] == pay.tolist()[:n]
        embs += n
        marked_ch.append(m)
    return torch.stack(marked_ch), embs, allok


def delta_e(orig_rgb, marked_rgb):
    """orig/marked: (3,H,W) int64 [0,255]. Returns mean CIEDE2000 Delta-E."""
    a = orig_rgb.permute(1, 2, 0).numpy().astype(np.float64) / 255.0
    b = marked_rgb.permute(1, 2, 0).numpy().astype(np.float64) / 255.0
    lab_a = rgb2lab(a)
    lab_b = rgb2lab(b)
    de = deltaE_ciede2000(lab_a, lab_b)
    return float(de.mean()), float(de.max())


# ----------------------------------------------------------------------------
# Experiment 1: CIEDE2000 on Kodak RGB 256 (classical + v5)
# ----------------------------------------------------------------------------
def exp_delta_e_classical():
    print("\n=== CIEDE2000 (Delta-E), classical no-shift, RGB 256 Kodak ===")
    paths = list_images(ROOT / "data" / "kodak", 24)
    print(f"{'Payload':>8} | {'PSNR':>7} | {'dE mean':>8} | {'dE max':>7} | bit-exact")
    for tp in (1000, 5000, 10000):
        des, dmax, ps, ex = [], [], [], 0
        for p in paths:
            img = load_rgb(p, 256)
            marked, embs, ok = rgb_noshift_classical(img, tp)
            if ok:
                ex += 1
            dme, dmx = delta_e(img, marked)
            des.append(dme); dmax.append(dmx); ps.append(psnr(marked, img))
        print(f"{tp:>8} | {np.mean(ps):>7.2f} | {np.mean(des):>8.4f} | {np.mean(dmax):>7.3f} | {ex}/{len(paths)}")


def exp_delta_e_v5():
    print("\n=== CIEDE2000 (Delta-E), QFormer-v5 deep no-shift, RGB 256 Kodak, 10000-bit ===")
    try:
        from src.models.qformer_noshift_revmark import QFormerNoShiftRevMark
        from src.models.qformer_revmark import float_to_uint8
        ckpt = torch.load(ROOT / "checkpoints" / "qformer_v5_heldout_best.pt", map_location=DEVICE, weights_only=False)
        sa = ckpt.get("args", {})
        model = QFormerNoShiftRevMark(
            image_size=sa.get("image_size", 256), patch_size=sa.get("patch_size", 16),
            d_quaternion=sa.get("d_quat", 48), depth=sa.get("depth", 6), num_heads=sa.get("heads", 4),
        ).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
    except Exception as e:
        print(f"  (skipped v5 Delta-E: {e})")
        return
    paths = list_images(ROOT / "data" / "kodak", 24)
    des, dmax = [], []
    for p in paths:
        img = load_rgb(p, 256)
        rgb01 = (img.float() / 255.0).unsqueeze(0).to(DEVICE)
        img_uint8 = float_to_uint8(rgb01)
        payload = torch.randint(0, 2, (10000,), dtype=torch.int64, device=DEVICE)
        with torch.no_grad():
            er = model.embed_noshift(img_uint8, payload, T_select=0)
        marked = er.watermarked_uint8[0].cpu().to(torch.int64)
        dme, dmx = delta_e(img, marked)
        des.append(dme); dmax.append(dmx)
    print(f"  v5 10000-bit: dE mean={np.mean(des):.4f}  dE max={np.mean(dmax):.3f}")


# ----------------------------------------------------------------------------
# Experiment 2: 5-dataset generalisation (classical, RGB 256, 5000-bit)
# ----------------------------------------------------------------------------
def exp_five_dataset():
    print("\n=== Five-dataset generalisation, classical no-shift, RGB 256, 5000-bit ===")
    ds = {
        "Kodak": ("kodak", 24), "USC-SIPI": ("sipi", 14), "CLIC-2020": ("clic", 41),
        "COCO val-2017": ("coco", 50), "ISIC dermo.": ("isic", 50),
    }
    print(f"{'Dataset':>14} | {'N':>4} | {'PSNR':>7} | {'dE mean':>8} | bit-exact")
    for name, (sub, lim) in ds.items():
        paths = list_images(ROOT / "data" / sub, lim)
        if not paths:
            print(f"{name:>14} | (no images found)")
            continue
        des, ps, ex = [], [], 0
        for p in paths:
            img = load_rgb(p, 256)
            marked, embs, ok = rgb_noshift_classical(img, 5000)
            if ok:
                ex += 1
            dme, _ = delta_e(img, marked)
            des.append(dme); ps.append(psnr(marked, img))
        print(f"{name:>14} | {len(paths):>4} | {np.mean(ps):>7.2f} | {np.mean(des):>8.4f} | {ex}/{len(paths)}")


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    exp_delta_e_classical()
    exp_delta_e_v5()
    exp_five_dataset()
    print("\n=== DONE ===")
