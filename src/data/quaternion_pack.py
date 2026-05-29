"""
RGB <-> Quaternion conversion helpers.

A pure quaternion image is encoded as q = 0 + R*i + G*j + B*k.
PyTorch tensor convention: shape (B, 4, H, W) where channels are (real, i, j, k).
"""
import torch


def rgb_to_quaternion(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor (B, 3, H, W) -> quaternion tensor (B, 4, H, W).
    Real component is zero; (R, G, B) become (i, j, k).
    """
    assert rgb.dim() == 4 and rgb.size(1) == 3, f"expected (B,3,H,W), got {tuple(rgb.shape)}"
    real = torch.zeros_like(rgb[:, :1])
    return torch.cat([real, rgb], dim=1).contiguous()


def quaternion_to_rgb(q: torch.Tensor) -> torch.Tensor:
    """Inverse of rgb_to_quaternion. Discards real part."""
    assert q.dim() == 4 and q.size(1) == 4, f"expected (B,4,H,W), got {tuple(q.shape)}"
    return q[:, 1:].contiguous()


def pack_unpack_roundtrip_check(rgb: torch.Tensor) -> dict:
    """Sanity check that pack -> unpack reproduces the input bit-exactly."""
    q = rgb_to_quaternion(rgb)
    rgb_back = quaternion_to_rgb(q)
    max_abs_err = (rgb - rgb_back).abs().max().item()
    return {
        "input_shape": tuple(rgb.shape),
        "quat_shape": tuple(q.shape),
        "max_abs_error": max_abs_err,
        "bit_exact": max_abs_err == 0.0,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    r = pack_unpack_roundtrip_check(x)
    print(r)
    assert r["bit_exact"], "Pack/unpack must be bit-exact!"
    print("OK: RGB<->quaternion pack/unpack is bit-exact.")
