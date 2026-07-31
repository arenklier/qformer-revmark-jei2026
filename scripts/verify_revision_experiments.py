"""
Standalone verification for the JEI major-revision experiments.

Runs entirely with the CLASSICAL 4-neighbour rhombus predictor (no deep model
or L40S needed) on the Kodak dataset:

  1. Multi-resolution no-shift PEE at 128/256/512/1024, 10000-bit payload
     -> reports marked-image PSNR and bit-exact recovery.
  2. Robustness / fragility under impulse (salt-pepper) noise + median filter
     + extra Gaussian sigma=1, to complete Table 5.

Kodak is downloaded on first run (~50 MB) to a local cache.
"""
from __future__ import annotations

import io
import math
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

KODAK_DIR = Path(os.environ.get("KODAK_DIR", "/home/ayhan/qformer-revmark/data/kodak"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


# ----------------------------------------------------------------------------
# Kodak (already present on the server)
# ----------------------------------------------------------------------------
def ensure_kodak():
    imgs = sorted(KODAK_DIR.glob("kodim*.png"))
    if len(imgs) < 24:
        imgs = sorted(KODAK_DIR.glob("*.png"))
    return imgs[:24]


def load_gray_uint8(path, size):
    """Load image, center-crop to square, resize, return (H,W) int64 grayscale in [0,255]."""
    im = Image.open(path).convert("L")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((size, size), Image.BICUBIC)
    return torch.tensor(np.array(im), dtype=torch.int64)


# ----------------------------------------------------------------------------
# Classical 4-neighbour rhombus predictor (integer, exact)
# ----------------------------------------------------------------------------
def rhombus_predict(img, pass_mask):
    """Predict each pixel in pass_mask as floor-average of 4 orthogonal neighbours."""
    H, W = img.shape
    pred = img.clone()
    up = torch.zeros_like(img); up[1:, :] = img[:-1, :]
    dn = torch.zeros_like(img); dn[:-1, :] = img[1:, :]
    lf = torch.zeros_like(img); lf[:, 1:] = img[:, :-1]
    rt = torch.zeros_like(img); rt[:, :-1] = img[:, 1:]
    avg = (up + dn + lf + rt) // 4
    pred[pass_mask] = avg[pass_mask]
    return pred


def checker_masks(H, W):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cross = ((yy + xx) % 2 == 0)
    dot = ~cross
    return cross, dot


# ----------------------------------------------------------------------------
# No-shift embed / extract (single channel, two-pass) -- integer exact
# ----------------------------------------------------------------------------
def embed_noshift_gray(img, payload, T_sel=0):
    """img: (H,W) int64. payload: 1-D int64 {0,1}. Returns (marked, side, n_embedded)."""
    H, W = img.shape
    cross, dot = checker_masks(H, W)
    current = img.clone()
    cursor = 0
    side = {"loc": [], "ncand": [], "T": T_sel, "npay": int(payload.numel())}
    payload_list = payload.tolist()

    for pass_mask in (dot, cross):
        pred = rhombus_predict(current, pass_mask)
        e = current - pred
        interior = torch.zeros_like(pass_mask)
        interior[1:H-1, 1:W-1] = True
        cand_mask = pass_mask & interior & (e.abs() <= T_sel + 1)
        positions = cand_mask.flatten().nonzero(as_tuple=False).flatten().tolist()
        side["ncand"].append(len(positions))
        delta = torch.zeros_like(current)
        for pos in positions:
            yi, xi = pos // W, pos % W
            e_val = int(e[yi, xi].item())
            pixel = int(pred[yi, xi].item()) + e_val
            embeddable = abs(e_val) <= T_sel and pixel < 255 and cursor < len(payload_list)
            if embeddable:
                b = payload_list[cursor]
                cursor += 1
                side["loc"].append(1)
                if b == 1:
                    delta[yi, xi] = 1
            else:
                side["loc"].append(0)
        current = current + delta
    return current, side, cursor


def extract_noshift_gray(marked, side):
    H, W = marked.shape
    cross, dot = checker_masks(H, W)
    current = marked.clone()
    payload = []
    loc = side["loc"]
    T_sel = side["T"]
    offset_total = 0
    # extraction inverts pass order: cross first, then dot
    # need per-pass candidate counts; embed did dot then cross
    ncand_dot, ncand_cross = side["ncand"]
    # We must undo cross pass first (it was applied last).
    # But cross prediction used the dot-modified image. To invert we recover cross
    # then dot. Recompute predictions in reverse using the fact that only |e|<=T pixels
    # of each pass moved by +1.
    # --- undo cross pass ---
    for pass_mask, ncand, loc_slice in _reverse_passes(cross, dot, ncand_cross, ncand_dot, loc):
        pred = rhombus_predict(current, pass_mask)
        e = current - pred
        interior = torch.zeros_like(pass_mask)
        interior[1:H-1, 1:W-1] = True
        cand_mask = pass_mask & interior & (e.abs() <= T_sel + 1)
        positions = cand_mask.flatten().nonzero(as_tuple=False).flatten().tolist()
        if len(positions) != ncand:
            raise RuntimeError(f"candidate drift: embed={ncand} extract={len(positions)}")
        recovered = current.clone()
        local_payload = []
        for k, pos in enumerate(positions):
            yi, xi = pos // W, pos % W
            flag = loc_slice[k]
            if flag == 1:
                b = int(e[yi, xi].item())
                local_payload.append(b)
                recovered[yi, xi] = current[yi, xi] - b
        current = recovered
        payload = local_payload + payload  # prepend (reverse order)
    return current, payload


def _reverse_passes(cross, dot, ncand_cross, ncand_dot, loc):
    """Yield (pass_mask, ncand, loc_slice) in extraction order: cross then dot."""
    # loc was built dot-first then cross. So dot occupies [0:ncand_dot], cross [ncand_dot:].
    loc_dot = loc[:ncand_dot]
    loc_cross = loc[ncand_dot:ncand_dot + ncand_cross]
    yield cross, ncand_cross, loc_cross
    yield dot, ncand_dot, loc_dot


def psnr(a, b):
    d = (a.float() - b.float())
    mse = (d ** 2).mean().item()
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


# ----------------------------------------------------------------------------
# Attacks
# ----------------------------------------------------------------------------
def att_gaussian(img, sigma):
    n = torch.randn_like(img.float()) * sigma
    return (img.float() + n).round().clamp(0, 255).to(torch.int64)


def att_impulse(img, density):
    out = img.clone()
    H, W = img.shape
    n = int(H * W * density)
    ys = torch.randint(0, H, (n,))
    xs = torch.randint(0, W, (n,))
    vals = torch.randint(0, 2, (n,)) * 255
    for y, x, v in zip(ys.tolist(), xs.tolist(), vals.tolist()):
        out[y, x] = v
    return out


def att_median(img):
    x = img.float().unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    patches = x.unfold(2, 3, 1).unfold(3, 3, 1)  # (1,1,H,W,3,3)
    med = patches.reshape(*patches.shape[:4], 9).median(dim=-1).values
    return med.squeeze(0).squeeze(0).round().clamp(0, 255).to(torch.int64)


# ----------------------------------------------------------------------------
# Approximate (map-free) extraction
# ----------------------------------------------------------------------------
def approximate_extract_gray(marked, T_sel=0):
    """Map-free extractor: re-run predictor, read e' mod 2 at ALL candidates in
    embed order (dot then cross). Returns the guessed bit stream."""
    H, W = marked.shape
    cross, dot = checker_masks(H, W)
    current = marked.clone()
    guessed = []
    for pass_mask in (dot, cross):
        pred = rhombus_predict(current, pass_mask)
        e = current - pred
        interior = torch.zeros_like(pass_mask)
        interior[1:H-1, 1:W-1] = True
        cand_mask = pass_mask & interior & (e.abs() <= T_sel + 1)
        positions = cand_mask.flatten().nonzero(as_tuple=False).flatten().tolist()
        for pos in positions:
            yi, xi = pos // W, pos % W
            guessed.append(int(e[yi, xi].item()) % 2)
    return guessed


def _load_rgb(path, size):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
    return torch.tensor(np.array(im), dtype=torch.int64).permute(2, 0, 1)


def _rgb_noshift(img_rgb, total_payload, T=0):
    per = total_payload // 3
    embs = 0
    marked_ch = []
    allok = True
    for c in range(3):
        pay = torch.randint(0, 2, (per,), dtype=torch.int64)
        m, side, n = embed_noshift_gray(img_rgb[c], pay, T_sel=T)
        rec, rp = extract_noshift_gray(m, side)
        allok = allok and torch.equal(rec, img_rgb[c]) and rp[:n] == pay.tolist()[:n]
        embs += n
        marked_ch.append(m)
    return torch.stack(marked_ch), embs, allok


def exp_hecai_comparison(paths, size=512):
    # Head-to-head with He and Cai 2024 (Mathematics), who report colour 512x512
    # PSNR 63.61 dB @ 20000 bits and 60.53 dB @ 40000 bits.
    print("\n=== EXPERIMENT 4: He-Cai 2024 comparison (classical no-shift, RGB 512) ===")
    print(f"{'Payload':>8} | {'PSNR mean':>10} | {'embedded':>10} | {'bit-exact':>9} | He-Cai")
    hecai = {20000: 63.61, 40000: 60.53}
    for tp in (20000, 40000):
        ps = []
        exact = 0
        em = []
        for p in paths:
            img = _load_rgb(p, size)
            marked, embs, ok = _rgb_noshift(img, tp)
            if ok:
                exact += 1
            em.append(embs)
            ps.append(psnr(marked, img))
        a = np.array(ps)
        ci = 1.96 * a.std(ddof=1) / math.sqrt(len(a))
        print(f"{tp:>8} | {a.mean():>7.2f}+/-{ci:.2f} | {int(np.mean(em)):>5}/{tp} | {exact:>4}/{len(paths):<3} | {hecai[tp]:.2f}")


def exp_approximate_extraction(paths, payload_bits=5000, size=256):
    print("\n=== EXPERIMENT 3: Map-free approximate extraction (classical) ===")
    print("Reviewer R1.f: 'what if the map is lost? any approximate way?'")
    match_no_side = []
    for p in paths:
        img = load_gray_uint8(p, size)
        payload = torch.randint(0, 2, (payload_bits,), dtype=torch.int64)
        marked, side, n_emb = embed_noshift_gray(img, payload, T_sel=0)
        guessed = approximate_extract_gray(marked, T_sel=0)
        # Best-effort positional match over the true payload length.
        n = min(n_emb, len(guessed), payload_bits)
        pay = payload.tolist()[:n]
        gs = guessed[:n]
        m = sum(1 for a, b in zip(pay, gs) if a == b) / max(n, 1)
        match_no_side.append(m)
    arr = np.array(match_no_side)
    print(f"  Map-free bit-match rate (mean over {len(paths)} imgs): {arr.mean():.4f}")
    print(f"  min={arr.min():.4f}  max={arr.max():.4f}  (0.5 = chance)")
    return arr.mean()


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def exp_multi_resolution(paths, payload_bits=500):
    # Fixed 500-bit payload fits within the candidate budget at every resolution,
    # isolating the resolution effect from candidate saturation. A 10000-bit
    # target does NOT fit at 128 (~2600 fit) or 256 (~8600 fit), which would
    # confound the comparison.
    print("\n=== EXPERIMENT 1: Multi-resolution no-shift, fixed payload (classical) ===")
    print(f"Fixed payload = {payload_bits} bits")
    print(f"{'Size':>6} | {'PSNR mean':>10} | {'PSNR CI95':>9} | {'embedded':>9} | {'bit-exact':>9}")
    results = {}
    for size in (128, 256, 512, 1024):
        psnrs = []
        exact = 0
        embs = []
        for p in paths:
            img = load_gray_uint8(p, size)
            payload = torch.randint(0, 2, (payload_bits,), dtype=torch.int64)
            marked, side, n_emb = embed_noshift_gray(img, payload, T_sel=0)
            embs.append(n_emb)
            rec, rec_pay = extract_noshift_gray(marked, side)
            ok = torch.equal(rec, img) and rec_pay[:n_emb] == payload.tolist()[:n_emb]
            if ok:
                exact += 1
            psnrs.append(psnr(marked, img))
        arr = np.array(psnrs)
        ci = 1.96 * arr.std(ddof=1) / math.sqrt(len(arr))
        results[size] = (arr.mean(), ci, exact, len(paths))
        print(f"{size:>6} | {arr.mean():>10.2f} | {ci:>9.2f} | {int(np.mean(embs)):>4}/{payload_bits:<4} | {exact:>4}/{len(paths):<4}")
    return results


def exp_robustness(paths, payload_bits=5000, size=256):
    print("\n=== EXPERIMENT 2: Fragility under attacks (classical predictor) ===")
    attacks = [
        ("Gaussian sigma=1", lambda im: att_gaussian(im, 1)),
        ("Gaussian sigma=2", lambda im: att_gaussian(im, 2)),
        ("Impulse 0.1%", lambda im: att_impulse(im, 0.001)),
        ("Impulse 0.5%", lambda im: att_impulse(im, 0.005)),
        ("Impulse 1.0%", lambda im: att_impulse(im, 0.010)),
        ("Median 3x3", att_median),
    ]
    print(f"{'Attack':>18} | {'Att.PSNR':>8} | {'success':>8} | {'refuse':>8}")
    results = {}
    for name, fn in attacks:
        apsnr_sum = 0.0
        n_success = 0
        n_refuse = 0
        for p in paths:
            img = load_gray_uint8(p, size)
            payload = torch.randint(0, 2, (payload_bits,), dtype=torch.int64)
            marked, side, n_emb = embed_noshift_gray(img, payload, T_sel=0)
            attacked = fn(marked)
            apsnr_sum += psnr(marked, attacked)
            try:
                rec, rec_pay = extract_noshift_gray(attacked, side)
                # success = full bit-exact host + payload
                if torch.equal(rec, img) and rec_pay[:n_emb] == payload.tolist()[:n_emb]:
                    n_success += 1
                else:
                    n_refuse += 1
            except RuntimeError:
                n_refuse += 1
        results[name] = (apsnr_sum / len(paths), n_success, n_refuse, len(paths))
        print(f"{name:>18} | {apsnr_sum/len(paths):>8.2f} | {n_success:>4}/{len(paths):<3} | {n_refuse:>4}/{len(paths):<3}")
    return results


if __name__ == "__main__":
    paths = ensure_kodak()
    print(f"Kodak images: {len(paths)}  |  device: {DEVICE}")
    r1 = exp_multi_resolution(paths)
    r2 = exp_robustness(paths)
    r3 = exp_approximate_extraction(paths)
    r4 = exp_hecai_comparison(paths)
    print("\n=== DONE ===")
