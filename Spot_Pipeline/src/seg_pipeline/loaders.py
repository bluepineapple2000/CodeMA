"""Frame iterators for HDF5 seg_vol files and LoG TIF stacks (TAR archive or directory).

hdf5plugin is imported here to register ESRF Blosc/Bitshuffle/LZ4 codecs before h5py opens any file.
"""
from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401 — registers ESRF codecs
except ImportError:
    pass  # tolerated in environments without ESRF data

import h5py
from PIL import Image


def _frame_index_from_name(name: str) -> int:
    """Parse frame index from last contiguous digit run in a filename stem."""
    digits = re.findall(r"\d+", Path(name).stem)
    if not digits:
        raise ValueError(f"Cannot parse frame index from {name!r}")
    return int(digits[-1])


def iter_frames_hdf5(
    h5_path: Path,
    dataset_key: str,
    frame_indices: set[int] | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield (frame_idx, float32 HW tensor). If frame_indices is given, only those frames.

    Reads HDF5 chunks in one slice each to avoid re-decompressing the same chunk once per frame.
    Critical for gzip-compressed datasets where the chunk depth spans many frames (e.g. 100),
    because h5py's default chunk cache is far too small to hold a decompressed chunk.
    """
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_key]
        n_total = ds.shape[0]
        indices = sorted(frame_indices) if frame_indices is not None else range(n_total)

        chunk_depth = ds.chunks[0] if ds.chunks is not None else 1

        # Group requested indices by which chunk they fall in, preserving sorted order.
        i = 0
        indices_list = list(indices)
        while i < len(indices_list):
            t = indices_list[i]
            chunk_start = (t // chunk_depth) * chunk_depth
            chunk_end   = min(chunk_start + chunk_depth, n_total)

            # Collect all requested indices in this chunk (they are sorted, so contiguous).
            j = i
            while j < len(indices_list) and indices_list[j] < chunk_end:
                j += 1
            batch_indices = indices_list[i:j]

            # One HDF5 read → one decompression for the whole chunk.
            batch = ds[chunk_start:chunk_end].astype(np.float32)
            for t in batch_indices:
                yield t, torch.from_numpy(batch[t - chunk_start])

            i = j


def iter_frames_tar(
    tar_path: Path,
    frame_indices: set[int] | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield (frame_idx, float32 HW tensor) for each .tif in a TAR archive, sorted by name."""
    with tarfile.open(tar_path, "r") as tf:
        members = sorted(
            (m for m in tf.getmembers() if m.name.lower().endswith(".tif")),
            key=lambda m: m.name,
        )
        for member in members:
            idx = _frame_index_from_name(member.name)
            if frame_indices is not None and idx not in frame_indices:
                continue
            raw = tf.extractfile(member)
            if raw is None:
                continue
            img = Image.open(io.BytesIO(raw.read()))
            yield idx, torch.from_numpy(np.array(img, dtype=np.float32))


def iter_frames_dir(
    dir_path: Path,
    frame_indices: set[int] | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield (frame_idx, float32 HW tensor) for each .tif in a directory, sorted by name."""
    tifs = sorted(dir_path.glob("*.tif"))
    for path in tifs:
        idx = _frame_index_from_name(path.name)
        if frame_indices is not None and idx not in frame_indices:
            continue
        img = Image.open(path)
        yield idx, torch.from_numpy(np.array(img, dtype=np.float32))


def estimate_beam_center(
    raw_path: Path,
    raw_key: str,
    n_sample: int = 10,
    max_deviation_frac: float = 0.10,
) -> tuple[float, float, float, float] | None:
    """Estimate beam center and half-extents from a max-projection of evenly-sampled raw frames.

    Loads n_sample frames, max-projects, then Gaussian-blurs with a large kernel to
    suppress diffraction spots. Beam center is the weighted centroid of the blurred image.
    Beam half-height and half-width are derived from the row and column profiles respectively:
    half the extent above the midpoint between background and peak intensity.

    Returns (cy, cx, beam_half_height, beam_half_width) in pixel coordinates, or None if cy
    or cx deviates more than max_deviation_frac of the frame dimension from the geometric
    center (implausible calibration or wrong file — caller should fall back to frame/2).
    """
    import kornia

    with h5py.File(raw_path, "r") as f:
        ds = f[raw_key]
        n_total, H, W = ds.shape[0], ds.shape[1], ds.shape[2]
        indices = [int(i) for i in np.linspace(0, n_total - 1, n_sample)]
        frames = [torch.from_numpy(ds[i].astype(np.float32)) for i in indices]

    projection = torch.stack(frames).max(dim=0).values  # (H, W)

    blurred = kornia.filters.gaussian_blur2d(
        projection[None, None],
        kernel_size=(101, 101),
        sigma=(20.0, 20.0),
    )[0, 0]

    # Weighted centroid over the blurred image — more robust than argmax for a
    # thick beam whose peak intensity may be off-centre.
    rows = torch.arange(H, dtype=torch.float32)
    cols = torch.arange(W, dtype=torch.float32)
    total = blurred.sum()
    cy = float((blurred.sum(dim=1) * rows).sum() / total)
    cx = float((blurred.sum(dim=0) * cols).sum() / total)

    if abs(cy - H / 2) > max_deviation_frac * H or abs(cx - W / 2) > max_deviation_frac * W:
        return None

    # Beam half-height from the row profile: find vertical extent at 50% between
    # background (10th percentile) and peak — i.e. a half-maximum relative to background.
    row_profile = blurred.sum(dim=1)  # (H,) — sum intensity over all columns per row
    bg = float(torch.quantile(row_profile, 0.10))
    peak = float(row_profile.max())
    threshold = bg + 0.5 * (peak - bg)
    beam_rows = torch.where(row_profile > threshold)[0]
    if len(beam_rows) >= 2:
        beam_half_height = float((beam_rows[-1] - beam_rows[0]) / 2)
    else:
        beam_half_height = float(H / 4)  # fallback: quarter frame

    # Beam half-width from the column profile: same half-maximum method.
    col_profile = blurred.sum(dim=0)  # (W,) — sum intensity over all rows per column
    bg_x = float(torch.quantile(col_profile, 0.10))
    peak_x = float(col_profile.max())
    threshold_x = bg_x + 0.5 * (peak_x - bg_x)
    beam_cols = torch.where(col_profile > threshold_x)[0]
    if len(beam_cols) >= 2:
        beam_half_width = float((beam_cols[-1] - beam_cols[0]) / 2)
    else:
        beam_half_width = float(W / 4)  # fallback: quarter frame

    return cy, cx, beam_half_height, beam_half_width


_KNOWN_SCAN_SIZES = [3600, 7200]


def infer_scan_shape(
    seg_path: Path,
    seg_type: str,
    seg_key: str,
) -> tuple[int, int]:
    """Return (n_frames, frame_height) by peeking at the segmentation source.

    n_frames is snapped down to the nearest recognised scan size (3600 or 7200)
    to exclude trailing dark/empty frames that some scans have appended.
    """
    p = Path(seg_path)
    if seg_type == "seg_vol":
        with h5py.File(p, "r") as f:
            shape = f[seg_key].shape  # (T, H, W)
        total, frame_height = shape[0], shape[1]
    elif p.is_dir():
        tifs = sorted(p.glob("*.tif"))
        if not tifs:
            raise ValueError(f"No .tif files found in {seg_path}")
        total = len(tifs)
        frame_height = np.array(Image.open(tifs[0])).shape[0]
    else:
        with tarfile.open(p, "r") as tf:
            members = sorted(
                [m for m in tf.getmembers() if m.name.lower().endswith(".tif")],
                key=lambda m: m.name,
            )
            if not members:
                raise ValueError(f"No .tif files found in {seg_path}")
            raw = tf.extractfile(members[0])
            frame_height = np.array(Image.open(io.BytesIO(raw.read()))).shape[0]  # type: ignore[arg-type]
        total = len(members)

    valid = [s for s in _KNOWN_SCAN_SIZES if s <= total]
    if not valid:
        raise ValueError(
            f"Source has only {total} frames — expected at least {_KNOWN_SCAN_SIZES[0]}."
        )
    n_frames = max(valid)
    if total > n_frames:
        print(f"  Inferred scan size: {n_frames} frames (ignoring {total - n_frames} trailing frames).")
    return n_frames, frame_height


def friedel_frame_set(
    start: int,
    end: int,
    n_frames: int,
    offset: int,
    tol: int,
) -> set[int]:
    """Frames [start, end) plus their Friedel partner window, clipped to [0, n_frames)."""
    base = set(range(start, end))
    partner = set(range(start + offset - tol, end + offset + tol + 1))
    partner |= set(range(start - offset - tol, end - offset + tol + 1))
    return {i for i in base | partner if 0 <= i < n_frames}
