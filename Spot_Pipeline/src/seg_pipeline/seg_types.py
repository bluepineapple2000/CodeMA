from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class SegPipelineConfig:
    # Pass 1
    min_blob_area: int = 4
    max_blob_area: int = 65536

    # Source + binarization (both sources carry intensity; threshold to get binary mask)
    segmentation_source: str = "seg_vol"     # "seg_vol" | "log_tif"
    seg_vol_binarize_threshold: float = 1.0  # frame > thr → foreground (uint8 scale)
    log_binarize_threshold: float = 0.1      # frame > thr → foreground (float32 scale)

    # Pass 2 — Friedel matching
    friedel_omega_offset_frames: int = 1800  # derive from n_frames at call site
    friedel_omega_tolerance_frames: int = 5
    min_area_ratio: float = 0.8  # area(smaller)/area(larger) pre-filter; same grain → similar size
    min_ncc: float = 0.85        # NCC threshold; real Friedel pair → mirrored intensity profile
    ncc_area_weight: float = 0.0 # adds k/sqrt(min_area) to threshold — tightens requirement for small blobs; 0 = disabled

    # Spatial Friedel constraints — auto-derived from raw frames via registry
    frame_height: int = 2048
    beam_center_y: float = 1023.5
    beam_center_x: float = 1023.5
    beam_half_height: Optional[float] = None  # Y midpoint of pair must be within this of beam_center_y
    beam_half_width: Optional[float] = None   # X midpoint of pair must be within this of beam_center_x

    # Pass 3 — conservative gating
    require_friedel_partner: bool = True


@dataclass
class SegBlob:
    blob_id: int
    frame_idx: int
    centroid: tuple[float, float]        # (x, y) global frame coords
    area: int
    bbox: tuple[int, int, int, int]      # (min_r, min_c, max_r, max_c)
    mask: torch.Tensor = field(repr=False)       # (H, W) bool, local crop
    intensity: torch.Tensor = field(repr=False)  # (H, W) float32, same crop
    friedel_partner_id: Optional[int] = None
    ncc_score: float = 0.0


@dataclass
class SegIsolatedSpot:
    blob_id: int
    patch: torch.Tensor = field(repr=False)  # (H, W) float32 intensity crop
    mask: torch.Tensor = field(repr=False)   # (H, W) bool
    centroid: tuple[float, float]
    frame_idx: int
    friedel_partner_id: Optional[int]
    ncc_score: float
