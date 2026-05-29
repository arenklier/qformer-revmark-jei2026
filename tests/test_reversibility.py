"""
Unit tests verifying STRICT bit-exact reversibility primitives.

These tests MUST pass before any reversible-watermarking experiment is run.
A failure here invalidates the central claim of the manuscript.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from src.data.quaternion_pack import (
    rgb_to_quaternion,
    quaternion_to_rgb,
    pack_unpack_roundtrip_check,
)


def test_pack_unpack_random():
    torch.manual_seed(0)
    rgb = torch.rand(4, 3, 64, 64)
    res = pack_unpack_roundtrip_check(rgb)
    assert res["bit_exact"], f"Pack/unpack not bit-exact: {res}"


def test_pack_unpack_extreme_values():
    """Boundary cases at 0 and 1."""
    cases = [
        torch.zeros(1, 3, 8, 8),
        torch.ones(1, 3, 8, 8),
        torch.full((1, 3, 8, 8), 0.5),
    ]
    for rgb in cases:
        q = rgb_to_quaternion(rgb)
        rgb_back = quaternion_to_rgb(q)
        assert torch.equal(rgb, rgb_back), "extreme value pack/unpack failed"


def test_pack_unpack_uint8():
    """Bit-exact on the integer range we actually publish on."""
    torch.manual_seed(1)
    rgb = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8).float() / 255.0
    q = rgb_to_quaternion(rgb)
    rgb_back = quaternion_to_rgb(q)
    assert torch.equal(rgb, rgb_back), "uint8 pack/unpack not bit-exact"


def test_quaternion_shape_invariant():
    """rgb_to_quaternion always yields 4 channels with real=0."""
    rgb = torch.rand(3, 3, 16, 16)
    q = rgb_to_quaternion(rgb)
    assert q.shape == (3, 4, 16, 16)
    assert torch.all(q[:, 0] == 0)


def test_quaternion_components_unchanged():
    """The i, j, k components must equal R, G, B respectively."""
    rgb = torch.rand(2, 3, 16, 16)
    q = rgb_to_quaternion(rgb)
    assert torch.equal(q[:, 1], rgb[:, 0])  # i == R
    assert torch.equal(q[:, 2], rgb[:, 1])  # j == G
    assert torch.equal(q[:, 3], rgb[:, 2])  # k == B


if __name__ == "__main__":
    test_pack_unpack_random()
    test_pack_unpack_extreme_values()
    test_pack_unpack_uint8()
    test_quaternion_shape_invariant()
    test_quaternion_components_unchanged()
    print("All reversibility primitive tests PASSED.")
