"""Tests for seg_pipeline passes. All synthetic — no real data, no imports from pipeline/."""
import numpy as np
import pytest
import torch

from seg_pipeline.passes import (
    _ncc_centered,
    _max_weight_matching,
    pass1_extract,
    pass2_pair_friedel,
    pass3_gate,
)
from seg_pipeline.seg_types import SegBlob, SegPipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = SegPipelineConfig(
    min_blob_area=4,
    max_blob_area=500,
    segmentation_source="seg_vol",
    seg_vol_binarize_threshold=0.5,
    friedel_omega_offset_frames=100,
    friedel_omega_tolerance_frames=5,
    min_area_ratio=0.5,
    min_ncc=0.3,
    frame_height=20,   # matches _make_blob's 20×20 synthetic frames
    beam_center_y=9.5, # geometric centre of the 20-row synthetic frame
)


def _gaussian_blob(h: int, w: int, cy: float, cx: float, sigma: float) -> torch.Tensor:
    """Gaussian intensity profile — non-flat, so NCC is always well-defined."""
    ys = torch.arange(h).float().unsqueeze(1).expand(h, w)
    xs = torch.arange(w).float().unsqueeze(0).expand(h, w)
    return torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))


def _disk_frame(h: int = 64, w: int = 64, cy: float = 32, cx: float = 32, r: int = 5) -> torch.Tensor:
    """Gaussian blob frame. Values > 0.5 (the _CFG threshold) form approximately a disk of radius r."""
    return _gaussian_blob(h, w, cy, cx, sigma=r / 1.5)


def _disk_mask(h: int, w: int, cy: float, cx: float, r: int) -> torch.Tensor:
    ys = torch.arange(h).float().unsqueeze(1).expand(h, w)
    xs = torch.arange(w).float().unsqueeze(0).expand(h, w)
    return ((ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2)


def _make_blob(blob_id: int, frame_idx: int, cy: float = 10.0, cx: float = 10.0, r: int = 4) -> SegBlob:
    mask = _disk_mask(20, 20, cy, cx, r)
    intensity = _gaussian_blob(20, 20, cy, cx, sigma=r / 1.5)
    return SegBlob(
        blob_id=blob_id,
        frame_idx=frame_idx,
        centroid=(cx, cy),
        area=int(mask.sum()),
        bbox=(0, 0, 20, 20),
        mask=mask,
        intensity=intensity,
    )


# ---------------------------------------------------------------------------
# Pass 1
# ---------------------------------------------------------------------------

def test_pass1_extracts_single_blob():
    frame = _disk_frame(64, 64, cy=32, cx=32, r=5)
    blobs = pass1_extract(frame, frame_idx=0, config=_CFG)
    assert len(blobs) == 1
    b = blobs[0]
    assert abs(b.centroid[0] - 32) < 1.5
    assert abs(b.centroid[1] - 32) < 1.5
    assert b.frame_idx == 0


def test_pass1_blob_has_intensity():
    frame = _disk_frame(64, 64, cy=32, cx=32, r=5)
    blobs = pass1_extract(frame, frame_idx=0, config=_CFG)
    assert len(blobs) == 1
    b = blobs[0]
    assert b.intensity is not None
    assert b.intensity.shape == b.mask.shape
    assert b.intensity.dtype == torch.float32


def test_pass1_filters_by_min_area():
    cfg = SegPipelineConfig(min_blob_area=200, seg_vol_binarize_threshold=0.5)
    frame = _disk_frame(64, 64, cy=32, cx=32, r=5)  # area ~ pi*25 ≈ 78 pixels
    blobs = pass1_extract(frame, 0, cfg)
    assert len(blobs) == 0


def test_pass1_filters_by_max_area():
    cfg = SegPipelineConfig(max_blob_area=10, seg_vol_binarize_threshold=0.5)
    frame = _disk_frame(64, 64, cy=32, cx=32, r=5)
    blobs = pass1_extract(frame, 0, cfg)
    assert len(blobs) == 0


def test_pass1_empty_frame():
    blobs = pass1_extract(torch.zeros(64, 64), 0, _CFG)
    assert blobs == []


def test_pass1_start_id_offset():
    frame = _disk_frame()
    blobs = pass1_extract(frame, 0, _CFG, start_id=42)
    assert blobs[0].blob_id == 42


# ---------------------------------------------------------------------------
# NCC
# ---------------------------------------------------------------------------

def _ncc_patches(mask_a, int_a, mask_b, int_b):
    """Helper: call _ncc_centered with numpy arrays and CoM from mask."""
    np_mask_a = mask_a.numpy()
    np_int_a = int_a.numpy()
    np_mask_b = mask_b.numpy()
    np_int_b = int_b.numpy()
    ys_a, xs_a = np.where(np_mask_a)
    ys_b, xs_b = np.where(np_mask_b)
    com_a = (float(ys_a.mean()), float(xs_a.mean()))
    com_b = (float(ys_b.mean()), float(xs_b.mean()))
    return _ncc_centered(np_int_a, np_mask_a, com_a, np_int_b, np_mask_b, com_b)


def test_ncc_identical_patches():
    # Gaussian intensity: values vary across disk → std > 0 → NCC well-defined
    mask = _disk_mask(20, 20, 10, 10, 5)
    intensity = _gaussian_blob(20, 20, 10, 10, sigma=3.0)
    ncc = _ncc_patches(mask, intensity, mask, intensity)
    assert ncc == pytest.approx(1.0, abs=1e-6)


def test_ncc_flat_signal_returns_zero():
    mask = _disk_mask(20, 20, 10, 10, 5)
    flat = torch.ones(20, 20)
    ncc = _ncc_patches(mask, flat, mask, flat)
    assert ncc == pytest.approx(0.0, abs=1e-6)


def test_ncc_anticorrelated_patches():
    # A: Gaussian peak (high center, low edges); B: inverted (low center, high edges)
    mask = _disk_mask(20, 20, 10, 10, 5)
    intensity_a = _gaussian_blob(20, 20, 10, 10, sigma=2.0)
    intensity_b = 1.0 - intensity_a  # anticorrelated
    ncc = _ncc_patches(mask, intensity_a, mask, intensity_b)
    assert ncc < 0.0


# ---------------------------------------------------------------------------
# Max-weight matching
# ---------------------------------------------------------------------------

def test_mwm_empty():
    assert _max_weight_matching(np.zeros((0, 0))) == []


def test_mwm_all_forbidden():
    assert _max_weight_matching(np.zeros((3, 3))) == []


def test_mwm_single_valid_pair():
    m = np.zeros((2, 2))
    m[0, 1] = 0.8
    result = _max_weight_matching(m)
    assert (0, 1) in result


def test_mwm_optimal_not_greedy():
    """Greedy (pick 0.8 first) would assign row0→col0 and leave row1 unmatched.
    Optimal assigns row0→col1 (0.5) + row1→col0 (0.9) = 1.4 > 0.8."""
    m = np.array([[0.8, 0.5],
                  [0.9, 0.0]])
    result = dict(_max_weight_matching(m))
    assert result.get(1) == 0  # row1 gets col0 (score=0.9)
    assert result.get(0) == 1  # row0 gets col1 (score=0.5)


# ---------------------------------------------------------------------------
# Pass 2
# ---------------------------------------------------------------------------

def test_pass2_pairs_at_correct_offset():
    a = _make_blob(0, frame_idx=0)
    b = _make_blob(1, frame_idx=100)
    pass2_pair_friedel([a, b], _CFG)
    assert a.friedel_partner_id == 1
    assert b.friedel_partner_id == 0
    assert a.ncc_score > 0


def test_pass2_rejects_wrong_offset():
    a = _make_blob(0, frame_idx=0)
    b = _make_blob(1, frame_idx=50)  # offset=50, expected ~100
    pass2_pair_friedel([a, b], _CFG)
    assert a.friedel_partner_id is None
    assert b.friedel_partner_id is None


def test_pass2_rejects_low_ncc():
    cfg = SegPipelineConfig(
        min_blob_area=4, max_blob_area=500,
        seg_vol_binarize_threshold=0.5,
        friedel_omega_offset_frames=100, friedel_omega_tolerance_frames=5,
        min_area_ratio=0.5,
        min_ncc=0.99,  # near-perfect match required
    )
    a = _make_blob(0, 0, cy=10, cx=10, r=4)
    # b: tiny 2×2 square — completely different shape and intensity pattern
    tiny_mask = torch.zeros(20, 20, dtype=torch.bool)
    tiny_mask[9:11, 9:11] = True
    tiny_int = tiny_mask.float() * 2.0
    b = SegBlob(blob_id=1, frame_idx=100, centroid=(10, 10), area=4,
                bbox=(0, 0, 20, 20), mask=tiny_mask, intensity=tiny_int)
    pass2_pair_friedel([a, b], cfg)
    assert a.friedel_partner_id is None


def test_pass2_no_double_assignment():
    # Two source blobs competing for one target
    a = _make_blob(0, 0, cy=10, cx=10, r=4)
    b = _make_blob(1, 0, cy=30, cx=10, r=4)  # different position, same frame
    c = _make_blob(2, 100, cy=10, cx=10, r=4)  # target at offset
    pass2_pair_friedel([a, b, c], _CFG)
    # c can only have one partner
    partners = [x for x in [a, b] if x.friedel_partner_id == c.blob_id]
    assert len(partners) <= 1
    assert c.friedel_partner_id is None or c.friedel_partner_id in (0, 1)


# ---------------------------------------------------------------------------
# Pass 3
# ---------------------------------------------------------------------------

def test_pass3_requires_friedel_partner():
    a = _make_blob(0, 0)
    b = _make_blob(1, 0)
    b.friedel_partner_id = 0
    b.ncc_score = 0.9
    spots = pass3_gate([a, b], _CFG)
    ids = {s.blob_id for s in spots}
    assert 1 in ids
    assert 0 not in ids


def test_pass3_patch_is_2d():
    b = _make_blob(0, 0)
    b.friedel_partner_id = 1
    b.ncc_score = 0.9
    spots = pass3_gate([b], _CFG)
    assert len(spots) == 1
    assert spots[0].patch.ndim == 2
    assert spots[0].mask.ndim == 2



def test_pass2_area_weight_rejects_small_blob():
    # Two identical blobs (NCC ≈ 1.0) but tiny area — area_weight should push threshold above 1.0
    # making it impossible to match, while a large-area identical pair still matches.
    cfg_weighted = SegPipelineConfig(
        min_blob_area=4, max_blob_area=5000,
        seg_vol_binarize_threshold=0.5,
        friedel_omega_offset_frames=100, friedel_omega_tolerance_frames=5,
        min_area_ratio=0.5, min_ncc=0.5,
        ncc_area_weight=5.0,   # 5/sqrt(~20px) ≈ 1.1 → effective threshold > 1.0 for small blobs
        frame_height=20, beam_center_y=9.5,
    )
    small_a = _make_blob(0, frame_idx=0, r=2)   # area ≈ 12 px
    small_b = _make_blob(1, frame_idx=100, r=2)
    pass2_pair_friedel([small_a, small_b], cfg_weighted)
    assert small_a.friedel_partner_id is None  # threshold too high for this size

    cfg_weighted_large = SegPipelineConfig(
        min_blob_area=4, max_blob_area=5000,
        seg_vol_binarize_threshold=0.5,
        friedel_omega_offset_frames=100, friedel_omega_tolerance_frames=5,
        min_area_ratio=0.5, min_ncc=0.5,
        ncc_area_weight=5.0,   # 5/sqrt(~1250px) ≈ 0.14 → effective threshold ≈ 0.64 for large blobs
        frame_height=20, beam_center_y=9.5,
    )
    large_a = _make_blob(2, frame_idx=0, r=20)   # area ≈ 1257 px
    large_b = _make_blob(3, frame_idx=100, r=20)
    pass2_pair_friedel([large_a, large_b], cfg_weighted_large)
    assert large_a.friedel_partner_id == large_b.blob_id  # large blobs still match


# ---------------------------------------------------------------------------
# Pass 2 — X tolerance
# ---------------------------------------------------------------------------

def test_pass2_x_difference_rejects_large_dx():
    # |x_A - x_B| = |5 - 15| = 10 > 2*3 = 6 → reject
    cfg = SegPipelineConfig(
        min_blob_area=4, max_blob_area=500,
        seg_vol_binarize_threshold=0.5,
        friedel_omega_offset_frames=100, friedel_omega_tolerance_frames=5,
        min_area_ratio=0.5, min_ncc=0.3,
        frame_height=20, beam_center_y=9.5,
        beam_half_width=3.0,
    )
    a = _make_blob(0, frame_idx=0,   cx=5.0)
    b = _make_blob(1, frame_idx=100, cx=15.0)
    pass2_pair_friedel([a, b], cfg)
    assert a.friedel_partner_id is None


def test_pass2_x_difference_accepts_small_dx():
    # |x_A - x_B| = |10 - 11| = 1 < 2*3 = 6 → accept
    cfg = SegPipelineConfig(
        min_blob_area=4, max_blob_area=500,
        seg_vol_binarize_threshold=0.5,
        friedel_omega_offset_frames=100, friedel_omega_tolerance_frames=5,
        min_area_ratio=0.5, min_ncc=0.3,
        frame_height=20, beam_center_y=9.5,
        beam_half_width=3.0,
    )
    a = _make_blob(0, frame_idx=0,   cx=10.0)
    b = _make_blob(1, frame_idx=100, cx=11.0)
    pass2_pair_friedel([a, b], cfg)
    assert a.friedel_partner_id == b.blob_id


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_end_to_end_synthetic():
    """10 disk pairs at correct offset + 5 unpaired; expect some spots out."""
    cfg = SegPipelineConfig(
        seg_vol_binarize_threshold=0.5,
        friedel_omega_offset_frames=100,
        friedel_omega_tolerance_frames=5,
        min_area_ratio=0.5,
        min_ncc=0.4,
        min_blob_area=4,
        max_blob_area=500,
        frame_height=128, beam_center_y=63.5,
    )

    n_frames = 210
    h, w = 128, 128
    frames = [torch.zeros(h, w) for _ in range(n_frames)]

    pairs: list[tuple[int, int]] = []
    for k in range(10):
        t_a = k * 10
        t_b = t_a + 100
        cy, cx = 30 + k * 5, 30 + k * 3
        r = 5
        blob = _gaussian_blob(h, w, cy, cx, sigma=r / 1.5)
        frames[t_a] = frames[t_a] + blob
        frames[t_b] = frames[t_b] + blob  # same pattern → NCC ≈ 1.0
        pairs.append((t_a, t_b))

    # 5 unpaired blobs
    for k in range(5):
        t = k * 3 + 200
        if t < n_frames:
            frames[t] = frames[t] + _gaussian_blob(h, w, 60, 60, sigma=5 / 1.5)

    all_blobs = []
    next_id = 0
    for t, frame in enumerate(frames):
        blobs = pass1_extract(frame, t, cfg, start_id=next_id)
        all_blobs.extend(blobs)
        next_id += len(blobs)

    pass2_pair_friedel(all_blobs, cfg)
    spots = pass3_gate(all_blobs, cfg)

    assert len(spots) > 0, "Expected some isolated spots"
    assert len(spots) <= len(pairs) * 2, "Should not exceed total injected blobs"
