"""
Cross-method comparison on Kodak (256x256).

Tests four predictor variants under both standard PEE and no-shift PEE:
  - Classical 4-neighbour rhombus      (no training, ~0 params)
  - QFormer-v2  (d_quat=48, 50 epoch)
  - QFormer-v3  (d_quat=64, 100 epoch)
  - Scalar Transformer matched-params (d_model=104, 50 epoch)

Reports:
  - PSNR @ payload curve (T sweep)
  - PSNR @ 1k / 5k / 10k payload (no-shift, T_select=0)

Output: results/baseline_comparison_kodak.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.qformer_revmark import QFormerRevMark, float_to_uint8  # noqa: E402
from src.models.qformer_noshift_revmark import QFormerNoShiftRevMark  # noqa: E402
from src.models.scalar_transformer_predictor import ScalarTransformerRevMark  # noqa: E402
from src.models.classical_predictor import ClassicalRevMark, ClassicalNoShiftRevMark  # noqa: E402


# ============================================================
# Model factories
# ============================================================
def build_qformer_from_ckpt(ckpt, device, noshift=False):
    sa = ckpt.get("args", {})
    cls = QFormerNoShiftRevMark if noshift else QFormerRevMark
    m = cls(
        image_size=sa.get("image_size", 256),
        patch_size=sa.get("patch_size", 16),
        d_quaternion=sa.get("d_quat", 48),
        depth=sa.get("depth", 6),
        num_heads=sa.get("heads", 4),
    ).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


def build_scalar_from_ckpt(ckpt, device):
    sa = ckpt.get("args", {})
    m = ScalarTransformerRevMark(
        image_size=sa.get("image_size", 256),
        patch_size=sa.get("patch_size", 16),
        d_model=sa.get("d_model", 192),
        depth=sa.get("depth", 6),
        num_heads=sa.get("heads", 4),
    ).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


def build_classical(device, noshift=False, image_size=256):
    cls = ClassicalNoShiftRevMark if noshift else ClassicalRevMark
    return cls(image_size=image_size, patch_size=16).to(device).eval()


# ============================================================
# Standard-PEE T sweep on a method
# ============================================================
@torch.no_grad()
def t_sweep(model, loader, T_values, device):
    rows = []
    for T in T_values:
        model.apee.log_T.data.fill_(math.log(max(T, 1)))
        if T == 0:
            model.apee.log_T.data.fill_(-10.0)

        psnr_sum, cap_sum, ne = 0.0, 0.0, 0
        for rgb, _ in loader:
            rgb = rgb.to(device, non_blocking=True)
            img = float_to_uint8(rgb)
            bits = torch.randint(0, 2, img.shape, dtype=torch.int64, device=device)
            er = model.embed(img, bits)
            for i in range(rgb.size(0)):
                d = (er.watermarked_uint8[i] - img[i]).float()
                mse = (d ** 2).mean().item()
                psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
                psnr_sum += psnr if not math.isinf(psnr) else 100.0
                cap_sum += er.capacity_bpp
                ne += 1
        rows.append({"T": T, "psnr": psnr_sum / ne, "bpp": cap_sum / ne})
    return rows


# ============================================================
# No-shift fixed payload
# ============================================================
@torch.no_grad()
def noshift_payloads(model, loader, payloads, device, T_select=0):
    rows = []
    for N in payloads:
        psnr_sum, emb_sum, ne, be = 0.0, 0, 0, 0
        for rgb, _ in loader:
            rgb = rgb.to(device, non_blocking=True)
            img = float_to_uint8(rgb)
            for i in range(rgb.size(0)):
                payload = torch.randint(0, 2, (N,), dtype=torch.int64, device=device)
                er = model.embed_noshift(img[i:i+1], payload, T_select=T_select)
                xr = model.extract_noshift(er.watermarked_uint8, er.side_data)
                be += int(torch.equal(xr.recovered_uint8, img[i:i+1]))
                psnr_sum += er.psnr_quick if not math.isinf(er.psnr_quick) else 100.0
                emb_sum += er.n_payload_bits
                ne += 1
        rows.append({"N": N, "psnr": psnr_sum / ne, "embedded": emb_sum / ne, "n_img": ne, "n_bit_exact": be})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--dataset", type=str, default="kodak")
    ap.add_argument("--T-values", type=str, default="0,1,2,3,5")
    ap.add_argument("--payloads", type=str, default="1000,5000,10000")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds = ImageFolderFlat(str(ROOT / "data" / args.dataset), image_size=args.image_size, crop="center")
    loader_t = DataLoader(ds, batch_size=4, shuffle=False, num_workers=args.num_workers)
    loader_n = DataLoader(ds, batch_size=2, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset: {args.dataset}  ({len(ds)} images)")

    T_values = [int(x) for x in args.T_values.split(",")]
    payloads = [int(x) for x in args.payloads.split(",")]

    # ---- Standard PEE T-sweep ----
    print()
    print("=" * 80)
    print("  Standard PEE T-sweep")
    print("=" * 80)

    methods_std = {}
    # Classical
    print("  [1/4] Classical predictor + standard PEE ...")
    m = build_classical(device, noshift=False, image_size=args.image_size)
    methods_std["classical"] = t_sweep(m, loader_t, T_values, device)

    # Q-v2
    print("  [2/4] QFormer-v2 + standard PEE ...")
    ck = torch.load(ROOT / "checkpoints/qformer_v2_best.pt", map_location=device, weights_only=False)
    m = build_qformer_from_ckpt(ck, device, noshift=False)
    methods_std["qformer_v2"] = t_sweep(m, loader_t, T_values, device)

    # Q-v3
    print("  [3/4] QFormer-v3 + standard PEE ...")
    ck = torch.load(ROOT / "checkpoints/qformer_v3_best.pt", map_location=device, weights_only=False)
    m = build_qformer_from_ckpt(ck, device, noshift=False)
    methods_std["qformer_v3"] = t_sweep(m, loader_t, T_values, device)

    # Scalar matched-params
    print("  [4/4] Scalar matched-params + standard PEE ...")
    ck = torch.load(ROOT / "checkpoints/scalar_v2_matchparams_best.pt", map_location=device, weights_only=False)
    m = build_scalar_from_ckpt(ck, device)
    methods_std["scalar_matchparams"] = t_sweep(m, loader_t, T_values, device)

    # ---- Print combined table ----
    print()
    methods = list(methods_std.keys())
    header = "| T  | " + " | ".join(f"{n:>21}" for n in methods) + " |"
    print(header)
    sep = "|----|-" + "-|-".join(["-" * 21 for _ in methods]) + "|"
    print(sep)
    for i, T in enumerate(T_values):
        cells = []
        for name in methods:
            r = methods_std[name][i]
            cells.append(f"PSNR {r['psnr']:5.2f}  bpp {r['bpp']:5.3f}")
        print(f"| {T:>2d} | " + " | ".join(cells) + " |")

    # ---- No-shift comparison (Classical, Q-v2, Q-v3) ----
    print()
    print("=" * 80)
    print("  No-shift PEE (T_select=0) at fixed payload")
    print("=" * 80)

    methods_ns = {}
    print("  [1/3] Classical + no-shift ...")
    m = build_classical(device, noshift=True, image_size=args.image_size)
    methods_ns["classical_noshift"] = noshift_payloads(m, loader_n, payloads, device, 0)

    print("  [2/3] Q-v2 + no-shift ...")
    ck = torch.load(ROOT / "checkpoints/qformer_v2_best.pt", map_location=device, weights_only=False)
    m = build_qformer_from_ckpt(ck, device, noshift=True)
    methods_ns["qformer_v2_noshift"] = noshift_payloads(m, loader_n, payloads, device, 0)

    print("  [3/3] Q-v3 + no-shift ...")
    ck = torch.load(ROOT / "checkpoints/qformer_v3_best.pt", map_location=device, weights_only=False)
    m = build_qformer_from_ckpt(ck, device, noshift=True)
    methods_ns["qformer_v3_noshift"] = noshift_payloads(m, loader_n, payloads, device, 0)

    print()
    header = "| Payload | " + " | ".join(f"{n:>27}" for n in methods_ns.keys()) + " |"
    print(header)
    sep = "|---------|-" + "-|-".join(["-" * 27 for _ in methods_ns]) + "|"
    print(sep)
    for i, N in enumerate(payloads):
        cells = []
        for name in methods_ns.keys():
            r = methods_ns[name][i]
            cells.append(f"PSNR {r['psnr']:5.2f}  emb {int(r['embedded']):>6,d}")
        print(f"| {N:>7,d} | " + " | ".join(cells) + " |")

    # ---- Save ----
    out = {"image_size": args.image_size, "dataset": args.dataset,
           "standard_pee": methods_std, "no_shift_pee": methods_ns}
    out_path = ROOT / "results" / f"baseline_comparison_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
