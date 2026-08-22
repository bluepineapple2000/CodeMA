"""Tests for seg_pipeline loaders. All synthetic — no real data."""
import numpy as np
import pytest
import torch

import h5py

from seg_pipeline.loaders import estimate_beam_center


def _make_raw_h5(path, cy: int, cx: int, H: int = 256, W: int = 256, n_frames: int = 5):
    """Write a minimal raw HDF5 with a bright spot at (cy, cx) in each frame."""
    data = np.zeros((n_frames, H, W), dtype=np.uint16)
    # Bright region ~20px wide — large enough to survive the Gaussian blur argmax
    r = 10
    data[:, max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 60000
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)


def test_estimate_beam_center_near_center(tmp_path):
    h5 = tmp_path / "raw.h5"
    _make_raw_h5(h5, cy=128, cx=130)
    result = estimate_beam_center(h5, "data", n_sample=5)
    assert result is not None
    cy, cx, beam_half_height, beam_half_width = result
    assert abs(cy - 128) < 5
    assert abs(cx - 130) < 5
    assert beam_half_height > 0
    assert beam_half_width > 0


def test_estimate_beam_center_rejects_far_from_center(tmp_path):
    """Beam way off-center → estimator returns None (sanity guard)."""
    h5 = tmp_path / "raw.h5"
    # Place bright spot at row 10 — far from geometric center 128 of a 256px frame
    _make_raw_h5(h5, cy=10, cx=128)
    result = estimate_beam_center(h5, "data", n_sample=5)
    assert result is None
