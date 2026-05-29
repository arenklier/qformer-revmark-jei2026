"""
Sanity smoke test of QFormer-RevMark on a real Kodak image.
Run: python -m scripts.smoke_test_kodak
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.qformer_revmark import QFormerRevMark, float_to_uint8


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 256

    # Load Kodak image 1
    img_path = ROOT / "data" / "kodak" / "kodim01.png"
    pil = Image.open(img_path).convert("RGB")
    tfm = T.Compose([T.Resize(img_size + 16), T.CenterCrop(img_size), T.ToTensor()])
    rgb_float = tfm(pil).unsqueeze(0).to(device)
    img_uint8 = float_to_uint8(rgb_float)

    print(f"Image: {img_path.name}  shape={tuple(img_uint8.shape)}  device={device}")

    # Build model (untrained -- random predictor, but PEE math is exact)
    model = QFormerRevMark(
        image_size=img_size,
        patch_size=16,
        d_quaternion=48,
        depth=6,
        num_heads=4,
        init_T=5.0,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Random bit payload spread across the image (one bit per pixel-channel slot)
    torch.manual_seed(7)
    bits = torch.randint(0, 2, img_uint8.shape, dtype=torch.int64, device=device)

    er = model.embed(img_uint8, bits)
    xr = model.extract(er.watermarked_uint8)

    # Bit-exact check
    is_bit_exact = bool(torch.equal(xr.recovered_uint8, img_uint8))

    # Compute PSNR vs original
    diff = (er.watermarked_uint8 - img_uint8).float()
    mse = (diff ** 2).mean().item()
    if mse > 0:
        psnr = 10.0 * (255.0 ** 2 / mse).__abs__().__rfloordiv__(1)  # placeholder
        psnr = 10.0 * torch.log10(torch.tensor(255.0 ** 2 / mse)).item()
    else:
        psnr = float("inf")

    print()
    print("=== Results ===")
    print(f"  Round-trip bit-exact   : {is_bit_exact}")
    print(f"  PSNR (orig vs marked)  : {psnr:.2f} dB")
    print(f"  Capacity               : {er.capacity_bpp:.4f} bpp")
    print(f"  Bits Pass1 (Dot)       : {er.n_bits_pass1}")
    print(f"  Bits Pass2 (Cross)     : {er.n_bits_pass2}")
    print(f"  Total bits             : {er.n_bits_pass1 + er.n_bits_pass2}")
    print(f"  Pixels                 : {img_size * img_size * 3}")
    print(f"  Threshold T            : {model.apee.T_int}")

    # Sanity-bound: with untrained predictor, capacity will be low and PSNR
    # will look bad. After training PSNR should rise above 50 dB and capacity
    # above 1 bpp on natural images.


if __name__ == "__main__":
    main()
