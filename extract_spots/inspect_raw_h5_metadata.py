"""Read-only metadata inspection for ESRF-style raw HDF5 scan files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def preview(dataset: h5py.Dataset, limit: int = 3) -> str:
    """Return a small, decoded preview without loading image stacks."""
    if dataset.size == 0:
        return "<empty>"
    if dataset.ndim >= 2:
        return "<omitted: multidimensional data>"

    values = dataset[: min(dataset.shape[0], limit)] if dataset.ndim else [dataset[()]]
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="replace"))
        elif isinstance(value, np.ndarray):
            decoded.append(value.tolist())
        else:
            decoded.append(value.item() if isinstance(value, np.generic) else value)
    return repr(decoded)


def inspect(path: Path) -> None:
    print(f"\n=== {path} ===")
    with h5py.File(path, "r") as handle:
        print("root keys:", list(handle.keys()))
        if handle.attrs:
            print("root attrs:", dict(handle.attrs))

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                if obj.attrs:
                    print(f"GROUP   {name} attrs={dict(obj.attrs)}")
                return
            print(
                f"DATASET {name} shape={obj.shape} dtype={obj.dtype}"
                f" attrs={dict(obj.attrs)}"
            )
            if obj.ndim <= 1:
                print("        preview:", preview(obj))

        handle.visititems(visitor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        inspect(path)


if __name__ == "__main__":
    main()
