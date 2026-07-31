"""
Does quaternion attention actually exploit cross-channel structure better than a
matched-parameter scalar Transformer?

Two predictors at matched parameter count:
    QFormer-v2                 d_quat=48, depth 6   (~926 k params)
    scalar_v2_matchparams      d_model=104, depth 6 (~948 k params)

For each we measure, on the residual prediction errors:
  1. per-channel error variance and the |e|=0 fraction (PEE capacity proxy)
  2. residual cross-channel mutual information I(E_R; E_G, E_B)

Reading:
  LOWER residual cross-channel MI  => the predictor has consumed more of the
  cross-channel structure, i.e. it did the job quaternion algebra is meant to do.
  HIGHER or equal                  => quaternion attention leaves as much on the
  table as scalar attention, an honest negative result.

Also reports inference latency and peak memory for the deployment argument.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ayhan/qformer-revmark")
sys.path.insert(0, str(ROOT))

from src.data.datasets import ImageFolderFlat  # noqa: E402
from src.models.qformer_revmark import QFormerRevMark, float_to_uint8  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


def build(ckpt_path):
    ck = torch.load(ROOT / ckpt_path, map_location=DEVICE, weights_only=False)
    a = ck.get("args", {})
    kind = ck.get("model_kind", "quaternion")
    if kind == "scalar":
        from src.models.scalar_transformer_predictor import ScalarTransformerRevMark as M
        m = M(image_size=a.get("image_size", 256), patch_size=a.get("patch_size", 16),
              d_model=a.get("d_model", 104), depth=a.get("depth", 6),
              num_heads=a.get("heads", 4)).to(DEVICE)
    else:
        m = QFormerRevMark(image_size=a.get("image_size", 256), patch_size=a.get("patch_size", 16),
                           d_quaternion=a.get("d_quat", 48), depth=a.get("depth", 6),
                           num_heads=a.get("heads", 4)).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return m, kind, n


def checker(H, W, device):
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    return ((yy + xx) % 2 == 0), ((yy + xx) % 2 == 1)


@torch.no_grad()
def residual_errors(model, loader, max_imgs=24):
    """Collect integer prediction errors on Dot positions for all 3 channels."""
    errs = []
    n = 0
    for rgb, _ in loader:
        rgb = rgb.to(DEVICE)
        img = float_to_uint8(rgb)
        for i in range(rgb.size(0)):
            if n >= max_imgs:
                break
            x = img[i:i + 1]
            H, W = x.shape[-2:]
            cross, dot = checker(H, W, DEVICE)
            inp = x.float() * cross.unsqueeze(0).unsqueeze(0)
            pred = model.predictor(inp / 255.0) * 255.0
            pred_int = torch.round(pred).clamp(0, 255).to(torch.int64)
            e = (x - pred_int)[0]                     # (3,H,W)
            sel = dot.unsqueeze(0).expand_as(e)
            errs.append(e[sel].reshape(3, -1).cpu().numpy())
            n += 1
    return np.concatenate(errs, axis=1)               # (3, N)


def mi_plugin(a, b, bins=32):
    """Plug-in MI (bits) between 1-D a and (possibly 2-D) b via joint histogram."""
    def disc(v):
        lo, hi = np.percentile(v, [0.5, 99.5])
        return np.clip(((v - lo) / max(hi - lo, 1e-9) * (bins - 1)).astype(int), 0, bins - 1)
    A = disc(a)
    if b.ndim == 1:
        B = disc(b)
    else:
        B = disc(b[0]) * bins + disc(b[1])
    ab = np.histogram2d(A, B, bins=[bins, B.max() + 1])[0]
    pab = ab / ab.sum()
    pa = pab.sum(1, keepdims=True)
    pb = pab.sum(0, keepdims=True)
    nz = pab > 0
    return float((pab[nz] * np.log2(pab[nz] / (pa @ pb)[nz])).sum())


@torch.no_grad()
def latency(model, reps=20):
    x = torch.rand(1, 3, 256, 256, device=DEVICE)
    for _ in range(5):
        model.predictor(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        model.predictor(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t.append((time.perf_counter() - s) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else float("nan")
    return float(np.mean(t)), float(np.std(t)), peak


def main():
    ds = ImageFolderFlat(str(ROOT / "data" / "kodak"), image_size=256, crop="center")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)

    models = {
        "Q v2 (quat, 50ep)": "checkpoints/qformer_v2_best.pt",
        "Q v4 (quat, 80ep)": "checkpoints/qformer_v4_intaware_best.pt",
        "Q v5 (quat, 80ep HO)": "checkpoints/qformer_v5_heldout_best.pt",
        "Scalar (50ep, matched)": "checkpoints/scalar_v2_matchparams_best.pt",
    }
    print(f"{'Model':<26} | {'params':>9} | {'mean s2':>8} | {'|e|=0 %':>8} | "
          f"{'I(ER;EG,EB)':>11} | {'ms':>6} | {'MB':>6}")
    print("-" * 92)
    out = {}
    for name, ck in models.items():
        m, kind, npar = build(ck)
        E = residual_errors(m, loader)
        s2 = float(np.mean(E.astype(np.float64) ** 2))
        zero = float((E == 0).mean() * 100)
        mi = mi_plugin(E[0].astype(float), E[1:].astype(float))
        ms, sd, mb = latency(m)
        out[name] = dict(params=npar, s2=s2, zero=zero, mi=mi, ms=ms, mb=mb)
        print(f"{name:<26} | {npar:>9,} | {s2:>8.2f} | {zero:>7.2f}% | {mi:>11.4f} | "
              f"{ms:>6.1f} | {mb:>6.1f}")

    s = out["Scalar (50ep, matched)"]
    print()
    print("Residual cross-channel MI, lower = consumed more cross-channel structure:")
    for k,v in out.items():
        if k.startswith("Q"):
            verdict = "BETTER than scalar" if v["mi"] < s["mi"] else "worse than scalar"
            print(f"  {k:<24} MI={v['mi']:.4f}  e2={v['s2']:7.1f}  {verdict}")
    print(f"  {'Scalar baseline':<24} MI={s['mi']:.4f}  e2={s['s2']:7.1f}")
    q = out["Q v2 (quat, 50ep)"]
    print()
    print("Residual cross-channel MI (lower = more cross-channel structure consumed):")
    print(f"  quaternion {q['mi']:.4f}  vs  scalar {s['mi']:.4f}   "
          f"=> quaternion leaves {'LESS' if q['mi'] < s['mi'] else 'MORE OR EQUAL'}")
    print(f"Latency: quaternion {q['ms']:.1f} ms vs scalar {s['ms']:.1f} ms "
          f"({s['ms']/q['ms']:.2f}x)")
    print(f"Peak memory: quaternion {q['mb']:.1f} MB vs scalar {s['mb']:.1f} MB")


if __name__ == "__main__":
    main()
