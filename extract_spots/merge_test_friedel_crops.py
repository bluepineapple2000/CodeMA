#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


COMMON_DATASETS = ("image", "input_mask", "spot_images", "spot_masks")
STRING_ATTRS = ("scan", "joint_normalization_scope", "raw_path", "seg_path")
FLOAT_ATTRS = (
    "shared_normalization_scale",
    "min_mask_intensity",
    "input_fit_margin_fraction",
    "target_fit_margin_fraction",
)
INT_ATTRS = (
    "rotation_k_90deg",
    "crop_size",
    "input_frame",
    "input_spot_number",
)
BOOL_ATTRS = (
    "align_ground_truth_to_input",
    "black_out_input_background",
    "black_out_target_background",
    "flip_ground_truth_horizontally",
    "normalize_intensity_with_mask",
)
ARRAY_ATTRS = (
    "target_frames",
    "target_spot_numbers",
    "target_alignment_shift_rc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge all HDF5 files from a Friedel-crop directory into one combined HDF5 file."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/test_friedel_crops",
        type=Path,
        help="Directory containing source .h5 files.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="data/test_friedel_crops/all_test_friedel_crops.h5",
        type=Path,
        help="Path for the merged output .h5 file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()



def source_files(input_dir: Path, output_path: Path) -> list[Path]:
    files = sorted(
        p for p in input_dir.glob("*.h5")
        if p.is_file() and p.resolve() != output_path.resolve()
    )
    if not files:
        raise FileNotFoundError(f"No source .h5 files found in {input_dir}")
    return files



def require_common_schema(files: Iterable[Path]) -> None:
    reference = None
    for path in files:
        with h5py.File(path, "r") as handle:
            schema = {name: (handle[name].shape, str(handle[name].dtype)) for name in COMMON_DATASETS}
        if reference is None:
            reference = schema
            continue
        if schema != reference:
            raise ValueError(f"Common dataset schema mismatch in {path.name}: {schema} != {reference}")



def copy_node(source: h5py.Group, target: h5py.Group) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value
    for name, obj in source.items():
        if isinstance(obj, h5py.Dataset):
            source.copy(obj, target, name=name)
        else:
            group = target.create_group(name)
            copy_node(obj, group)



def as_bytes_array(values: list[str]) -> np.ndarray:
    return np.asarray([value.encode("utf-8") for value in values])



def create_stacked_group(output: h5py.File, files: list[Path]) -> None:
    group = output.create_group("stacked")
    group.create_dataset("sample_ids", data=as_bytes_array([path.stem for path in files]))
    group.create_dataset("source_files", data=as_bytes_array([path.name for path in files]))

    for dataset_name in COMMON_DATASETS:
        stacked = []
        dtype = None
        for path in files:
            with h5py.File(path, "r") as handle:
                data = handle[dataset_name][...]
                stacked.append(data)
                dtype = handle[dataset_name].dtype
        group.create_dataset(
            dataset_name,
            data=np.stack(stacked, axis=0),
            compression="gzip",
            dtype=dtype,
        )

    metadata = output.create_group("metadata")

    for attr_name in STRING_ATTRS:
        values = []
        for path in files:
            with h5py.File(path, "r") as handle:
                values.append(str(handle.attrs[attr_name]))
        metadata.create_dataset(attr_name, data=as_bytes_array(values))

    for attr_name in INT_ATTRS:
        values = []
        for path in files:
            with h5py.File(path, "r") as handle:
                values.append(int(handle.attrs[attr_name]))
        metadata.create_dataset(attr_name, data=np.asarray(values, dtype=np.int32))

    for attr_name in BOOL_ATTRS:
        values = []
        for path in files:
            with h5py.File(path, "r") as handle:
                values.append(bool(handle.attrs[attr_name]))
        metadata.create_dataset(attr_name, data=np.asarray(values, dtype=np.uint8))

    for attr_name in FLOAT_ATTRS:
        values = []
        for path in files:
            with h5py.File(path, "r") as handle:
                raw = handle.attrs.get(attr_name, np.nan)
                values.append(float(raw))
        metadata.create_dataset(attr_name, data=np.asarray(values, dtype=np.float32))

    for attr_name in ARRAY_ATTRS:
        stacked = []
        dtype = None
        for path in files:
            with h5py.File(path, "r") as handle:
                value = np.asarray(handle.attrs[attr_name])
                stacked.append(value)
                dtype = value.dtype
        metadata.create_dataset(attr_name, data=np.stack(stacked, axis=0), dtype=dtype)

    target_names = []
    for path in files:
        with h5py.File(path, "r") as handle:
            raw = np.asarray(handle.attrs["target_names"])
            target_names.append([item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw])
    metadata.create_dataset(
        "target_names",
        data=np.asarray([[item.encode("utf-8") for item in row] for row in target_names]),
    )



def merge_files(input_dir: Path, output_path: Path, overwrite: bool) -> Path:
    files = source_files(input_dir, output_path)
    require_common_schema(files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
        output_path.unlink()

    with h5py.File(output_path, "w") as output:
        output.attrs["merged_from_dir"] = str(input_dir)
        output.attrs["sample_count"] = len(files)
        output.attrs["source_files"] = as_bytes_array([path.name for path in files])

        samples_group = output.create_group("samples")
        for path in files:
            with h5py.File(path, "r") as source:
                sample_group = samples_group.create_group(path.stem)
                sample_group.attrs["source_filename"] = path.name
                copy_node(source, sample_group)

        create_stacked_group(output, files)

    return output_path



def main() -> None:
    args = parse_args()
    merged = merge_files(args.input_dir, args.output, args.overwrite)
    print(f"Merged file written to {merged}")


if __name__ == "__main__":
    main()
