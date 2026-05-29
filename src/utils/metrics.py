"""
Metrics for QFormer-RevMark.

Includes:
  - PSNR / SSIM (image quality)
  - NC (normalized correlation, for watermark detection)
  - Capacity in bpp (bits per pixel)
  - Reversibility check (must be bit-exact)
  - Empirical mutual information I(R;G;B) for Lemma 1 validation
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Image quality
# ============================================================
def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio in dB. Returns scalar (averaged over batch).

    x, y: (B, C, H, W) tensors with values in [0, data_range].
    Returns +inf if x == y exactly (used as reversibility check).
    """
    mse = F.mse_loss(x, y, reduction="mean")
    if mse.item() == 0.0:
        return torch.tensor(float("inf"), device=x.device)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def psnr_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Returns a (B,) tensor of per-image PSNR values."""
    mse = F.mse_loss(x, y, reduction="none").mean(dim=(1, 2, 3))
    return torch.where(
        mse > 0,
        10.0 * torch.log10((data_range ** 2) / mse.clamp_min(1e-20)),
        torch.full_like(mse, float("inf")),
    )


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    x = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Structural similarity index. Returns scalar averaged over batch + channels.

    x, y: (B, C, H, W).
    """
    if x.shape != y.shape:
        raise ValueError("inputs must share shape")
    C = x.size(1)
    win = _gaussian_window(window_size, sigma, x.device, x.dtype).repeat(C, 1, 1, 1)

    def _filter(t):
        return F.conv2d(t, win, padding=window_size // 2, groups=C)

    mu_x = _filter(x)
    mu_y = _filter(y)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2 = _filter(x * x) - mu_x2
    sig_y2 = _filter(y * y) - mu_y2
    sig_xy = _filter(x * y) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
    )
    return ssim_map.mean()


# ============================================================
# Watermark detection
# ============================================================
def normalized_correlation(w_true: torch.Tensor, w_hat: torch.Tensor) -> torch.Tensor:
    """
    NC for binary watermarks. Inputs treated as flat bit tensors in {0, 1}.
    NC = sum(w_true * w_hat) / sqrt(sum(w_true^2) * sum(w_hat^2)).
    """
    a = w_true.float().flatten()
    b = w_hat.float().flatten()
    num = (a * b).sum()
    den = torch.sqrt((a * a).sum() * (b * b).sum()).clamp_min(1e-12)
    return num / den


def bit_error_rate(w_true: torch.Tensor, w_hat: torch.Tensor) -> float:
    """BER = fraction of bits differing between w_true and w_hat."""
    a = (w_true > 0.5).float().flatten()
    b = (w_hat > 0.5).float().flatten()
    return (a != b).float().mean().item()


# ============================================================
# Capacity
# ============================================================
def capacity_bpp(payload_bits: int, height: int, width: int) -> float:
    """Embedding capacity in bits per pixel."""
    return payload_bits / float(height * width)


# ============================================================
# Reversibility verification
# ============================================================
def is_bit_exact(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Strict equality test; required for reversible-WM correctness."""
    return torch.equal(x, y)


def reversibility_report(x: torch.Tensor, y: torch.Tensor) -> dict:
    diff = (x - y).abs()
    return {
        "bit_exact": is_bit_exact(x, y),
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "n_mismatched_pixels": int((diff > 0).sum().item()),
    }


# ============================================================
# Empirical I(R;G;B) for Lemma 1 validation
# ============================================================
def _bin_to_indices(x_flat: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Quantise values in [0,1] to integer bin indices in [0, n_bins-1]."""
    x = np.clip(x_flat, 0.0, 1.0)
    idx = np.minimum((x * n_bins).astype(np.int64), n_bins - 1)
    return idx


def mutual_information_rgb(image_rgb: torch.Tensor, n_bins: int = 32) -> dict:
    """
    Empirical pairwise and total cross-channel mutual information.

    Returns
    -------
    dict with keys:
        I_RG, I_RB, I_GB    : pairwise MI estimates (bits)
        I_total             : I(R; G, B) (bits) -- chain-rule total cross-channel info
        H_R, H_G, H_B       : marginal entropies (bits)

    The Lemma claims that quaternion-PEE gains at least I_total over scalar PEE.
    """
    if image_rgb.dim() == 4:
        # batch: stack pixels across batch and spatial
        image_rgb = image_rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    elif image_rgb.dim() == 3 and image_rgb.size(0) == 3:
        # (3, H, W)
        image_rgb = image_rgb.permute(1, 2, 0).reshape(-1, 3)
    else:
        raise ValueError("expected (B,3,H,W) or (3,H,W)")

    rgb = image_rgb.detach().cpu().numpy()
    r_idx = _bin_to_indices(rgb[:, 0], n_bins)
    g_idx = _bin_to_indices(rgb[:, 1], n_bins)
    b_idx = _bin_to_indices(rgb[:, 2], n_bins)

    def H(idx, n_bins):
        counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
        p = counts / counts.sum()
        nz = p > 0
        return -np.sum(p[nz] * np.log2(p[nz]))

    def H_joint2(a, b, n_bins):
        joint = a * n_bins + b
        counts = np.bincount(joint, minlength=n_bins * n_bins).astype(np.float64)
        p = counts / counts.sum()
        nz = p > 0
        return -np.sum(p[nz] * np.log2(p[nz]))

    def H_joint3(a, b, c, n_bins):
        joint = (a * n_bins + b) * n_bins + c
        counts = np.bincount(joint, minlength=n_bins ** 3).astype(np.float64)
        p = counts / counts.sum()
        nz = p > 0
        return -np.sum(p[nz] * np.log2(p[nz]))

    HR = H(r_idx, n_bins)
    HG = H(g_idx, n_bins)
    HB = H(b_idx, n_bins)
    HRG = H_joint2(r_idx, g_idx, n_bins)
    HRB = H_joint2(r_idx, b_idx, n_bins)
    HGB = H_joint2(g_idx, b_idx, n_bins)
    HRGB = H_joint3(r_idx, g_idx, b_idx, n_bins)

    I_RG = HR + HG - HRG
    I_RB = HR + HB - HRB
    I_GB = HG + HB - HGB
    # I(R; G, B) = H(R) + H(G,B) - H(R,G,B)
    HGB_joint = HGB
    I_total = HR + HGB_joint - HRGB

    return {
        "I_RG": float(I_RG),
        "I_RB": float(I_RB),
        "I_GB": float(I_GB),
        "I_total": float(I_total),
        "H_R": float(HR),
        "H_G": float(HG),
        "H_B": float(HB),
        "n_pixels": int(rgb.shape[0]),
        "n_bins": n_bins,
    }


def _self_test():
    torch.manual_seed(0)

    # PSNR / SSIM on identical images -> inf / 1
    x = torch.rand(2, 3, 64, 64)
    assert torch.isinf(psnr(x, x))
    assert abs(ssim(x, x).item() - 1.0) < 1e-4

    # Small noise gives finite PSNR
    y = (x + 0.01 * torch.randn_like(x)).clamp(0, 1)
    p = psnr(x, y).item()
    s = ssim(x, y).item()
    assert 30 < p < 70, f"PSNR={p}"
    assert 0.5 < s < 1.0, f"SSIM={s}"

    # NC
    w = torch.randint(0, 2, (1024,))
    assert abs(normalized_correlation(w, w).item() - 1.0) < 1e-6
    assert bit_error_rate(w, w) == 0.0

    # Reversibility
    rep = reversibility_report(x, x)
    assert rep["bit_exact"]
    assert rep["max_abs_error"] == 0

    # Mutual information: independent channels -> ~0
    indep = torch.rand(1, 3, 256, 256)
    mi_indep = mutual_information_rgb(indep, n_bins=16)

    # Correlated channels -> large I(R;G,B)
    base = torch.rand(1, 1, 256, 256)
    correlated = torch.cat([base, base + 0.05 * torch.randn_like(base), base + 0.05 * torch.randn_like(base)], dim=1).clamp(0, 1)
    mi_corr = mutual_information_rgb(correlated, n_bins=16)

    print("metrics self-test: ALL PASS")
    print(f"  PSNR(noise=0.01) = {p:.2f} dB,  SSIM = {s:.4f}")
    print(f"  MI independent  : I_total = {mi_indep['I_total']:.3f} bits")
    print(f"  MI correlated   : I_total = {mi_corr['I_total']:.3f} bits")
    assert mi_corr["I_total"] > mi_indep["I_total"], "MI should be larger for correlated channels"


if __name__ == "__main__":
    _self_test()
