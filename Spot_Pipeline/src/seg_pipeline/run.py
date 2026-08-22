"""CLI entry point for the segmentation-based isolated spot pipeline.

Usage:
    python -m seg_pipeline.run --config configs/al_logtif.toml
    python -m seg_pipeline.run --config configs/al_logtif.toml --frames 0:100
    python -m seg_pipeline.run \\
        --seg-path /data/Al/seg_vol.h5 --seg-key entry/data/data --seg-type seg_vol \\
        --n-frames 3600 --output /data/Al/isolated_spots.h5
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import h5py
import numpy as np
import torch

from seg_pipeline.loaders import (
    estimate_beam_center,
    friedel_frame_set,
    infer_scan_shape,
    iter_frames_dir,
    iter_frames_hdf5,
    iter_frames_tar,
)
from seg_pipeline.passes import pass1_extract, pass2_pair_friedel, pass3_gate
from seg_pipeline.seg_types import SegPipelineConfig


def _friedel_offset(n_frames: int) -> int:
    if n_frames == 3600:
        return 1800
    if n_frames == 7200:
        return 3600
    raise ValueError(
        f"n_frames={n_frames} is not a recognised scan size (expected 3600 or 7200). "
        "Set --n-frames to the number of valid rotation frames in your scan."
    )


def _iter_seg(args: argparse.Namespace, frame_indices: set[int] | None):
    p = Path(args.seg_path)
    if args.seg_type == "seg_vol":
        for idx, frame in iter_frames_hdf5(p, args.seg_key, frame_indices):
            # seg_vol frames are stored transposed vs LoG TIF orientation.
            # Matches numpy: rot90(frame[::-1], k=-1)
            yield idx, torch.rot90(torch.flip(frame, [0]), k=-1, dims=[0, 1])
        return
    if p.is_dir():
        yield from iter_frames_dir(p, frame_indices)
    else:
        yield from iter_frames_tar(p, frame_indices)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segmentation-based Friedel-paired spot extractor")
    parser.add_argument("--config", default=None,
                        help="Path to a TOML config file. All parameters can be set there; "
                             "any CLI argument provided explicitly overrides the file.")
    parser.add_argument("--seg-path", default=None,
                        help="Path to segmentation source: HDF5 file (seg_vol) or TIF directory / TAR (log_tif).")
    parser.add_argument("--seg-key", default="entry/data/data",
                        help="HDF5 dataset key inside --seg-path. Only used when --seg-type=seg_vol.")
    parser.add_argument("--seg-type", choices=["seg_vol", "log_tif"], default="seg_vol",
                        help="Segmentation source format: 'seg_vol' (HDF5) or 'log_tif' (per-frame TIFs).")
    parser.add_argument("--output", default=None,
                        help="Path for the output HDF5 file containing isolated spots.")
    parser.add_argument("--frames", default=None,
                        help="Frame range to process, e.g. '0:20'. "
                             "Friedel partner frames are added automatically.")
    parser.add_argument("--omega-tolerance", type=int, default=5,
                        help="Search window around the expected Friedel frame offset (±frames). Default: 5.")
    parser.add_argument("--min-blob-area", type=int, default=4,
                        help="Minimum blob area in pixels to keep after Pass 1. Default: 4.")
    parser.add_argument("--max-blob-area", type=int, default=65536,
                        help="Maximum blob area in pixels to keep after Pass 1. "
                             "Default: 65536 (1/64th of a 2048×2048 frame).")
    parser.add_argument("--seg-vol-threshold", type=float, default=1.0,
                        help="Binarization threshold for seg_vol frames (uint8 scale). Default: 1.0.")
    parser.add_argument("--log-threshold", type=float, default=0.1,
                        help="Binarization threshold for log_tif frames (float32 scale). Default: 0.1.")
    parser.add_argument("--min-area-ratio", type=float, default=0.8,
                        help="Minimum area(small)/area(large) ratio for a candidate pair to proceed to NCC. "
                             "Rejects pairs where one blob is much larger than the other. Default: 0.8.")
    parser.add_argument("--min-ncc", type=float, default=0.85,
                        help="Minimum NCC score to accept a Friedel pair. "
                             "Pearson correlation of intensity patches after h-flip and CoM alignment. Default: 0.85.")
    parser.add_argument("--ncc-area-weight", type=float, default=0.0,
                        help="Area-dependent NCC tightening: adds k/sqrt(min_area) to the threshold per pair. "
                             "Compensates for high NCC variance on small blobs. 0 = disabled. Default: 0.")
    parser.add_argument("--registry", default=None,
                        help="Path to scan_registry.toml. Required — used to locate the raw HDF5 "
                             "for beam-center estimation.")
    parser.add_argument("--scan-name", default=None,
                        help="Scan name as it appears in scan_registry.toml (e.g. 'Al_big_grains'). Required.")
    parser.add_argument("--raw-dir", default=None,
                        help="Directory containing the raw HDF5 file. Overrides the default of using "
                             "the parent of --seg-path (needed when LoG TIFs live in a subdirectory).")
    return parser


def main(argv: list[str] | None = None) -> None:
    # Pre-parse to find --config before setting defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre, _ = pre_parser.parse_known_args(argv)

    parser = _build_parser()

    if pre.config:
        with open(pre.config, "rb") as f:
            toml_cfg = tomllib.load(f)
        parser.set_defaults(**toml_cfg)

    args = parser.parse_args(argv)

    # Required args that may come from config file rather than CLI.
    missing = [flag for flag, val in [
        ("--seg-path",   args.seg_path),
        ("--output",     args.output),
        ("--registry",   args.registry),
        ("--scan-name",  args.scan_name),
    ] if val is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)} "
                     f"(provide on the command line or via --config)")

    print("Inferring scan shape from segmentation source...", flush=True)
    n_frames, frame_height = infer_scan_shape(Path(args.seg_path), args.seg_type, args.seg_key)
    print(f"  n_frames={n_frames}, frame_height={frame_height}", flush=True)

    with open(args.registry, "rb") as f:
        registry = tomllib.load(f)
    scan_entry = registry.get("scans", {}).get(args.scan_name)
    if not scan_entry:
        raise SystemExit(f"Scan '{args.scan_name}' not found in registry {args.registry}")
    raw_filename = scan_entry.get("raw")
    if not raw_filename:
        raise SystemExit(f"Scan '{args.scan_name}' has no 'raw' entry in registry {args.registry}")
    raw_key = registry.get("hdf5_keys", {}).get("raw", "instrument/detector_0/data")
    scan_dir = Path(args.raw_dir) if args.raw_dir else Path(args.seg_path).parent
    raw_path = scan_dir / raw_filename

    print(f"Estimating beam center from {raw_path.name}...", flush=True)
    result = estimate_beam_center(raw_path, raw_key)
    if result is None:
        raise SystemExit(
            f"Beam center estimate out of plausible range for {raw_path}. "
            "Check that the raw HDF5 path and key are correct."
        )
    beam_center_y, beam_center_x, beam_half_height, beam_half_width = result
    print(f"  beam_center_y={beam_center_y:.1f}  beam_center_x={beam_center_x:.1f}  "
          f"beam_half_height={beam_half_height:.1f}px  beam_half_width={beam_half_width:.1f}px", flush=True)


    offset = _friedel_offset(n_frames)
    frame_indices: set[int] | None = None
    if args.frames:
        start_s, _, end_s = args.frames.partition(":")
        start, end = int(start_s), int(end_s)
        frame_indices = friedel_frame_set(start, end, n_frames, offset, args.omega_tolerance)
        print(f"Frame subset: {len(frame_indices)} frames "
              f"(requested {start}:{end} + Friedel partners ±{args.omega_tolerance})", flush=True)

    config = SegPipelineConfig(
        segmentation_source=args.seg_type,
        seg_vol_binarize_threshold=args.seg_vol_threshold,
        log_binarize_threshold=args.log_threshold,
        min_blob_area=args.min_blob_area,
        max_blob_area=args.max_blob_area,
        friedel_omega_offset_frames=offset,
        friedel_omega_tolerance_frames=args.omega_tolerance,
        min_area_ratio=args.min_area_ratio,
        min_ncc=args.min_ncc,
        ncc_area_weight=args.ncc_area_weight,
        frame_height=frame_height,
        beam_center_y=beam_center_y,
        beam_center_x=beam_center_x,
        beam_half_height=beam_half_height,
        beam_half_width=beam_half_width,
    )

    import time
    t_start = time.time()

    t0 = t_start
    print("Pass 1: extracting blobs...", flush=True)
    all_blobs = []
    next_id = 0
    for frame_idx, frame in _iter_seg(args, frame_indices):
        blobs = pass1_extract(frame, frame_idx, config, start_id=next_id)
        all_blobs.extend(blobs)
        next_id += len(blobs)
        print(f"  frame {frame_idx:4d}: {len(blobs):4d} blobs  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  total: {len(all_blobs)} blobs", flush=True)
    t_pass1 = time.time() - t0

    print("Pass 2: Friedel pairing...", flush=True)
    pass2_pair_friedel(all_blobs, config)
    n_paired = sum(1 for b in all_blobs if b.friedel_partner_id is not None)
    print(f"  {n_paired} blobs paired ({n_paired // 2} Friedel pairs)", flush=True)
    t_pass2 = time.time() - t_start - t_pass1

    print("Pass 3: conservative gating...", flush=True)
    spots = pass3_gate(all_blobs, config)
    print(f"  {len(spots)} isolated spots retained", flush=True)
    t_pass3 = time.time() - t_start - t_pass1 - t_pass2

    scan_name = Path(args.seg_path).stem
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        grp = f.create_group(scan_name)

        cfg = grp.create_group("config")
        cfg["seg_path"]               = str(Path(args.seg_path).resolve())
        cfg["seg_key"]                = str(args.seg_key)
        cfg["seg_type"]               = str(args.seg_type)
        cfg["n_frames"]               = n_frames
        cfg["frames"]                 = str(args.frames) if args.frames else "all"
        cfg["omega_tolerance"]        = args.omega_tolerance
        cfg["min_blob_area"]          = args.min_blob_area
        cfg["max_blob_area"]          = args.max_blob_area
        cfg["seg_vol_threshold"]      = args.seg_vol_threshold
        cfg["log_threshold"]          = args.log_threshold
        cfg["min_area_ratio"]         = args.min_area_ratio
        cfg["min_ncc"]                = args.min_ncc
        cfg["ncc_area_weight"]        = args.ncc_area_weight
        cfg["frame_height"]           = frame_height
        cfg["beam_center_y"]          = beam_center_y
        cfg["beam_center_x"]          = beam_center_x
        cfg["beam_half_height"]       = beam_half_height if beam_half_height is not None else -1
        cfg["beam_half_width"]        = beam_half_width if beam_half_width is not None else -1

        t_total = time.time() - t_start
        tim = grp.create_group("timings")
        tim["pass1_s"] = t_pass1
        tim["pass2_s"] = t_pass2
        tim["pass3_s"] = t_pass3
        tim["total_s"] = t_total

        # Columnar storage — one array per field, flat concatenated patches/masks
        # with an offset index. Avoids per-blob HDF5 object overhead which
        # dominated file size in the old per-group format.
        grp.attrs["format_version"] = 2
        N = len(spots)
        sg = grp.create_group("spots")
        sg.create_dataset("blob_id",
            data=np.array([s.blob_id for s in spots], dtype=np.int32))
        sg.create_dataset("frame_idx",
            data=np.array([s.frame_idx for s in spots], dtype=np.int32))
        sg.create_dataset("friedel_partner_id",
            data=np.array([-1 if s.friedel_partner_id is None else s.friedel_partner_id
                           for s in spots], dtype=np.int32))
        sg.create_dataset("ncc_score",
            data=np.array([s.ncc_score for s in spots], dtype=np.float32))
        sg.create_dataset("centroid",
            data=np.array([list(s.centroid) for s in spots], dtype=np.float32))
        shapes = np.array([list(s.patch.shape) for s in spots], dtype=np.int32)
        sg.create_dataset("shape", data=shapes)
        offsets = np.zeros(N, dtype=np.int64)
        if N > 0:
            offsets[1:] = np.cumsum(shapes[:-1, 0] * shapes[:-1, 1])
        sg.create_dataset("offset", data=offsets)
        patches_flat = (np.concatenate([s.patch.numpy().ravel() for s in spots]).astype(np.float32)
                        if N > 0 else np.empty(0, dtype=np.float32))
        masks_flat   = (np.concatenate([s.mask.numpy().ravel().view(np.uint8) for s in spots])
                        if N > 0 else np.empty(0, dtype=np.uint8))
        if N > 0:
            chunk = min(65536, len(patches_flat))
            sg.create_dataset("patches", data=patches_flat, chunks=(chunk,),
                              compression="gzip", compression_opts=4, shuffle=True)
            sg.create_dataset("masks", data=masks_flat, chunks=(chunk,),
                              compression="gzip", compression_opts=4, shuffle=True)
        else:
            sg.create_dataset("patches", data=patches_flat)
            sg.create_dataset("masks", data=masks_flat)

    print(f"Output written to {out_path}  ({len(spots)} spots)")
    print(
        f"\nTime summary:\n"
        f"  Pass 1 (blob extraction): {t_pass1:6.1f}s\n"
        f"  Pass 2 (Friedel pairing): {t_pass2:6.1f}s\n"
        f"  Pass 3 (gating):          {t_pass3:6.1f}s\n"
        f"  Total:                    {t_total:6.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
