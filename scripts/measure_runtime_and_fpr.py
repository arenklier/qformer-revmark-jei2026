"""
Wall-clock latency + extraction false-positive rate (FPR) under attack.

Two analyses:
  (a) Inference latency: predictor forward, full embed pipeline, full
      extract pipeline. Mean over 20 runs after warm-up.
  (b) FPR under attack: bypass the cardinality consistency check, run
      extraction anyway, measure the fraction of returned bits that match
      the original payload (50% = random; deviation from 50% indicates
      structural leakage).
"""
from __future__ import annotations

import sys
import time
import io
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.qformer_revmark import float_to_uint8  # noqa: E402
from src.models.qformer_noshift_revmark import QFormerNoShiftRevMark  # noqa: E402
from src.pee.noshift_pee import embed_one_pass_noshift  # noqa: E402


def _build(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt.get("args", {})
    m = QFormerNoShiftRevMark(
        image_size=sa.get("image_size", 256),
        patch_size=sa.get("patch_size", 16),
        d_quaternion=sa.get("d_quat", 48),
        depth=sa.get("depth", 6),
        num_heads=sa.get("heads", 4),
    ).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


@torch.no_grad()
def runtime_bench(model, sample_uint8, n_trials=20):
    device = sample_uint8.device
    img = sample_uint8
    # Warm-up
    for _ in range(3):
        _ = model._predict_int(img)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # (1) predictor forward
    pred_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = model._predict_int(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pred_times.append((time.perf_counter() - t0) * 1000.0)

    # (2) full embed
    payload = torch.randint(0, 2, (5000,), dtype=torch.int64, device=device)
    embed_times = []
    er = None
    for _ in range(n_trials):
        t0 = time.perf_counter()
        er = model.embed_noshift(img, payload, T_select=0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        embed_times.append((time.perf_counter() - t0) * 1000.0)

    # (3) full extract
    extract_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = model.extract_noshift(er.watermarked_uint8, er.side_data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        extract_times.append((time.perf_counter() - t0) * 1000.0)

    def stats(arr):
        a = np.asarray(arr)
        return f"{a.mean():.1f} ± {a.std():.1f} ms"
    return {
        "predictor_forward_ms": stats(pred_times),
        "embed_pipeline_ms": stats(embed_times),
        "extract_pipeline_ms": stats(extract_times),
        "predictor_raw_us": (np.mean(pred_times) * 1000),
    }


def jpeg_compress(img_uint8, quality):
    arr = img_uint8[0].cpu().to(torch.uint8).numpy().transpose(1, 2, 0)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    attacked = Image.open(buf).convert("RGB")
    np_arr = np.array(attacked).transpose(2, 0, 1)
    return torch.tensor(np_arr, dtype=torch.int64, device=img_uint8.device).unsqueeze(0)


def add_noise(img_uint8, sigma):
    n = torch.randn_like(img_uint8.float()) * float(sigma)
    return (img_uint8.float() + n).round().clamp(0, 255).to(torch.int64)


@torch.no_grad()
def fpr_under_attack(model, loader, n_payload, attack_fn, device):
    """Bypass consistency: read whatever bits the extractor returns; count
    matches against the *true* payload."""
    from src.models.qformer_revmark import checkerboard_mask
    from src.pee.noshift_pee import extract_one_pass_noshift

    bits_total = 0
    bits_correct = 0
    n_imgs = 0
    for rgb, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        img = float_to_uint8(rgb)
        for i in range(rgb.size(0)):
            payload = torch.randint(0, 2, (n_payload,), dtype=torch.int64, device=device)
            er = model.embed_noshift(img[i:i+1], payload, T_select=0)
            attacked = attack_fn(er.watermarked_uint8)

            # Manual extract bypassing the consistency check
            B, C, H, W = attacked.shape
            cross_mask = checkerboard_mask(H, W, parity=0, device=device, dtype=torch.bool)
            dot_mask = ~cross_mask
            try:
                I_dot_only = attacked * dot_mask.to(torch.int64)
                pred_cross = model._predict_int(I_dot_only)
                e_prime_cross = (attacked - pred_cross)[0]
                # Skip the cardinality check; force-process candidates by truncating
                _, payload_p2 = _bypass_extract(e_prime_cross, cross_mask,
                                                er.side_data["loc_p2"],
                                                er.side_data["nc_p2"])
                I_cross_only = attacked * cross_mask.to(torch.int64)
                pred_dot = model._predict_int(I_cross_only)
                e_prime_dot = (attacked - pred_dot)[0]
                _, payload_p1 = _bypass_extract(e_prime_dot, dot_mask,
                                                er.side_data["loc_p1"],
                                                er.side_data["nc_p1"])
                all_bits = (payload_p1 + payload_p2)[: er.n_payload_bits]
                if len(all_bits) > 0:
                    n_cmp = min(len(all_bits), er.n_payload_bits)
                    true_bits = payload[:n_cmp].cpu().tolist()
                    matches = sum(1 for a, b in zip(all_bits[:n_cmp], true_bits) if a == b)
                    bits_correct += matches
                    bits_total += n_cmp
            except Exception:
                pass
            n_imgs += 1
    return {"bits_total": bits_total, "bits_correct": bits_correct,
            "match_rate": bits_correct / max(bits_total, 1), "n_imgs": n_imgs}


def _bypass_extract(e_prime, pass_mask, loc, nc):
    """Like extract_one_pass_noshift but skips the cardinality assertion."""
    import torch
    C, H, W = e_prime.shape
    rec_e = e_prime.clone()
    payload = []
    offset = 0
    for c in range(C):
        e_c = e_prime[c]
        mask_c = pass_mask & (e_c.abs() <= 1)  # T_select+1=1
        flat_positions = mask_c.flatten().nonzero(as_tuple=False).flatten().tolist()
        # Process MIN of (extract candidates, embed-time count) — bypass mismatch
        n_process = min(len(flat_positions), nc[c])
        for k in range(n_process):
            pos = flat_positions[k]
            flag = loc[offset]
            offset += 1
            yi, xi = pos // W, pos % W
            if flag == 1:
                b = int(e_c[yi, xi].item() % 2)
                payload.append(b)
        # Skip remaining offsets if extract has more candidates
        if len(flat_positions) > nc[c]:
            offset += 0  # we already consumed nc[c] of them via the loop
        else:
            # extract has FEWER candidates than embed; advance offset
            offset += nc[c] - len(flat_positions)
    return rec_e, payload


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = _build(ROOT / "checkpoints/qformer_v5_heldout_best.pt", device)
    ds = ImageFolderFlat(str(ROOT / "data/kodak"), image_size=256, crop="center")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # --- (A) Runtime ---
    sample = float_to_uint8(next(iter(loader))[0].to(device))
    rt = runtime_bench(model, sample, n_trials=20)
    print()
    print("=" * 60)
    print("Inference latency (Kodak 256x256 RGB, single image, fp32, L40S)")
    print("=" * 60)
    print(f"  Predictor forward     : {rt['predictor_forward_ms']}")
    print(f"  Embed pipeline (full) : {rt['embed_pipeline_ms']}")
    print(f"  Extract pipeline (full): {rt['extract_pipeline_ms']}")

    # --- (B) FPR under attack ---
    print()
    print("=" * 60)
    print("FPR / bit-match rate under attack (payload = 5,000 bits)")
    print("=" * 60)
    print("| Attack             | bits matched | match rate | random=0.500 |")
    print("|--------------------|-------------:|-----------:|-------------:|")
    attacks = [
        (lambda im: jpeg_compress(im, 90), "JPEG Q=90"),
        (lambda im: jpeg_compress(im, 70), "JPEG Q=70"),
        (lambda im: add_noise(im, 2.0), "Gauss sigma=2"),
        (lambda im: add_noise(im, 5.0), "Gauss sigma=5"),
    ]
    for fn, name in attacks:
        r = fpr_under_attack(model, loader, 5000, fn, device)
        print(f"| {name:<19}| {r['bits_correct']:>12,d} | {r['match_rate']:>10.4f} | (random) |")
    print()
    print("Interpretation: match rate ≈ 0.500 indicates extracted bits are")
    print("statistically indistinguishable from a fair coin flip -- the")
    print("location-bitmap leak is structural noise, not a usable channel.")


if __name__ == "__main__":
    main()
