"""Table 8, single-estimator version.

Model construction from measure_quaternion_role.build (handles quaternion AND scalar),
error collection and MI estimation from measure_mi_errors (the estimator that produced
Table 9), so the shared QFormer-v2/Kodak cell agrees between the two tables by construction.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).parent))
from measure_quaternion_role import build, latency, DEVICE, ROOT
from measure_mi_errors import collect_errors, estimate_mi_from_errors
from src.data.datasets import ImageFolderFlat

MODELS = [
    ("QFormer-v2 (quaternion)",     "50 ep", "checkpoints/qformer_v2_best.pt"),
    ("QFormer-v4 (quaternion)",     "80 ep", "checkpoints/qformer_v4_intaware_best.pt"),
    ("QFormer-v5 (quaternion, HO)", "80 ep", "checkpoints/qformer_v5_heldout_best.pt"),
    ("Scalar, matched params",      "50 ep", "checkpoints/scalar_v2_matchparams_best.pt"),
    ("Scalar, matched params",      "80 ep", "checkpoints/scalar_v3_80ep_best.pt"),
]

ds = ImageFolderFlat(str(ROOT / "data" / "kodak"), image_size=256, crop="center")
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

print(f"{'Model':<28} {'ep':>6} {'params':>10} {'mean e2':>9} {'|e|=0 %':>8} {'I(ER;EG,EB)':>12} {'ms':>7}")
print("-" * 88)
out = {}
for name, ep, ck in MODELS:
    m, kind, npar = build(ck)
    E = collect_errors(m, loader, DEVICE)          # (N, 3) int errors, both passes
    e2 = float((E.float() ** 2).mean())
    zero = float((E == 0).float().mean() * 100)
    _mi = estimate_mi_from_errors(E, n_bins=32)["mi"]
    if isinstance(_mi, dict):
        if not out: print("  MI dict keys:", list(_mi.keys()))
        mi = _mi["I_total"]
    else:
        mi = _mi
    ms, sd, mb = latency(m)
    key = f"{name} [{ep}]"
    out[key] = dict(params=npar, mean_e2=e2, zero_pct=zero, mi=float(mi), ms=ms, mb=mb,
                    n_samples=int(E.shape[0]))
    print(f"{name:<28} {ep:>6} {npar:>10,} {e2:>9.1f} {zero:>7.2f}% {mi:>12.4f} {ms:>6.1f}")

json.dump(out, open(ROOT / "results" / "quatrole_final_kodak.json", "w"), indent=1)
n = list(out.values())[0]["n_samples"]
print(f"\nsamples = {n:,}  (Table 9 Kodak reports 1,572,864)")
print("mean e2 is the per-sample mean squared error pooled over the three channels.")
