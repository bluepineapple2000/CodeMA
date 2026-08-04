import argparse
import csv
import json
import logging
from collections import deque
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers ESRF compression filters
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_overlap_classifier import FourLayerOverlapCNN, H5OverlapClassifierDataset


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_DIR.parent / "data_esrf"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "real_overlap_predictions"

DATASETS = {
    "Al": {
        "main": "_iris_iris_500_cont.h5",
        "pre": "full.h5",
    },
    "Al_big_grains": {
        "main": "Al1050_dct7p5_Al1050_dct7p.h5",
    },
    "Al_deformed_LoG": {
        "main": "A2050_as_annealed_DT_W370_0p12_strain_dct3_A2050_as_annealed_DT_W370_0p12_strain_dct.h5",
        "log_tar": "A2050_as_annealed_DT_W370_0p12_strain_dct3.tar",
    },
    "Al_small_grains": {
        "main": "iris_dct_450C_0003_iris_dct_450C.h5",
    },
    "Cu": {
        "main": "copper3_dct_load1_redo_copper3_dct_load1_redo.h5",
    },
    "IN718_twins": {
        "main": "ep_sync_12_dct_7p5_3_ep_sync_12_dct_7p5.h5",
    },
    "Iron": {
        "main": "ind_Iron_dct_load1_z1_ind_Iron_dct_load1_z.h5",
    },
    "Iron_deformed": {
        "main": "pureIron2_dct_2nd_load_z1_pureIron2_dct_2nd_load_z.h5",
    },
    "Ti7Al": {
        "main": "sam_23_dct4_sam_23_dct.h5",
    },
}


def connected_components(binary: np.ndarray, min_pixels: int) -> list[dict]:
    height, width = binary.shape
    coords = set(np.flatnonzero(binary).tolist())
    components = []
    component_id = 0

    while coords:
        start = coords.pop()
        queue = deque([start])
        pixels = [start]

        while queue:
            current = queue.popleft()
            y, x = divmod(current, width)
            for dy, dx in (
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1),
            ):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor = ny * width + nx
                    if neighbor in coords:
                        coords.remove(neighbor)
                        queue.append(neighbor)
                        pixels.append(neighbor)

        if len(pixels) < min_pixels:
            continue

        pixels = np.asarray(pixels, dtype=np.int64)
        ys = pixels // width
        xs = pixels % width
        component_id += 1
        components.append(
            {
                "component_id": component_id,
                "pixels": int(len(pixels)),
                "y0": int(ys.min()),
                "y1": int(ys.max() + 1),
                "x0": int(xs.min()),
                "x1": int(xs.max() + 1),
                "cy": float(ys.mean()),
                "cx": float(xs.mean()),
            }
        )

    return sorted(components, key=lambda item: item["pixels"], reverse=True)


def padded_bbox(component: dict, frame_shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    height, width = frame_shape
    return (
        max(0, int(component["y0"]) - padding),
        max(0, int(component["x0"]) - padding),
        min(height, int(component["y1"]) + padding),
        min(width, int(component["x1"]) + padding),
    )


def center_pad(array: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    target_height, target_width = target_shape
    height, width = array.shape
    if height > target_height or width > target_width:
        raise ValueError(f"Crop shape {array.shape} does not fit target shape {target_shape}")
    output = np.zeros(target_shape, dtype=array.dtype)
    top = (target_height - height) // 2
    left = (target_width - width) // 2
    output[top:top + height, left:left + width] = array
    return output, (top, left)


def prepare_component_patch(
    frame: np.ndarray,
    component: dict,
    patch_shape: tuple[int, int],
    padding: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    y0, x0, y1, x1 = padded_bbox(component, frame.shape, padding)
    crop = frame[y0:y1, x0:x1].astype(np.float32, copy=False)
    mask = crop > 0
    signal = crop.copy()
    signal[~mask] = 0.0
    signal[~np.isfinite(signal)] = 0.0
    signal = np.maximum(signal, 0.0)
    padded, pad_top_left = center_pad(signal, patch_shape)
    padded_mask, _ = center_pad(mask.astype(np.uint8), patch_shape)
    return padded, padded_mask, (y0, x0, y1, x1), pad_top_left


def normalize_for_model(image: np.ndarray) -> torch.Tensor:
    normalized = H5OverlapClassifierDataset.normalize_image(image)
    return torch.from_numpy(normalized).unsqueeze(0)


def load_model(checkpoint_path: Path, device: torch.device, base_channels: int, dropout: float) -> FourLayerOverlapCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_args = checkpoint.get("args") or {}
        base_channels = int(saved_args.get("base_channels", base_channels))
        dropout = float(saved_args.get("dropout", dropout))

    model = FourLayerOverlapCNN(base_channels=base_channels, dropout=dropout)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def selected_frame_indices(n_frames: int, frame_count: int, frame_step: int, frame_start: int) -> list[int]:
    indices = [frame_start + i * frame_step for i in range(frame_count)]
    return [idx for idx in indices if 0 <= idx < n_frames]


def write_preview_png(path: Path, patch: np.ndarray, mask: np.ndarray, record: dict) -> None:
    positive_values = patch[np.isfinite(patch) & (patch > 0)]
    if positive_values.size:
        vmin, vmax = np.percentile(positive_values, [1, 99.8])
        vmax = max(float(vmax), float(vmin) + 1e-6)
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    axes[0].imshow(patch, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{record['dataset']} frame {record['frame']}")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"p={record['overlap_probability']:.3f}")
    for axis in axes:
        axis.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predict_batch(model, images: list[np.ndarray], device: torch.device, img_scale: float, amp: bool) -> np.ndarray:
    tensors = []
    for image in images:
        tensor = normalize_for_model(image)
        if img_scale != 1.0:
            new_size = (
                max(1, int(round(tensor.shape[-2] * img_scale))),
                max(1, int(round(tensor.shape[-1] * img_scale))),
            )
            tensor = F.interpolate(tensor.unsqueeze(0), size=new_size, mode="bilinear", align_corners=False).squeeze(0)
        tensors.append(tensor)

    batch = torch.stack(tensors).to(device=device, dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        logits = model(batch)
        probabilities = torch.sigmoid(logits)
    return probabilities.detach().cpu().numpy()


def append_predictions_to_h5(h5_file: h5py.File, records: list[dict], patches: list[np.ndarray], masks: list[np.ndarray]) -> None:
    for record, patch, mask in zip(records, patches, masks):
        group_name = f"{record['dataset']}/frame_{record['frame']:05d}/component_{record['component_id']:05d}"
        group = h5_file.create_group(group_name)
        group.create_dataset("patch", data=patch, compression="gzip", compression_opts=4)
        group.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
        for key, value in record.items():
            if isinstance(value, (tuple, list)):
                group.attrs[key] = json.dumps(list(value))
            else:
                group.attrs[key] = value


def scan_dataset(name: str, entry: dict, args, model, device: torch.device, h5_out: h5py.File, csv_writer) -> tuple[int, int]:
    scan_dir = args.data_root / name
    seg_path = scan_dir / "segvol.h5"
    if not seg_path.exists():
        logging.warning("Skipping %s: missing %s", name, seg_path)
        return 0, 0

    predicted_count = 0
    scanned_count = 0
    preview_dir = args.output_dir / "previews" / name
    preview_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(seg_path, "r") as seg_h5:
        seg = seg_h5[args.seg_key]
        frames = selected_frame_indices(seg.shape[0], args.frames_per_dataset, args.frame_step, args.frame_start)
        logging.info("%s: scanning frames %s", name, frames)

        for frame_idx in tqdm(frames, desc=name, unit="frame"):
            frame = seg[frame_idx]
            components = connected_components(frame > args.seg_threshold, min_pixels=args.min_pixels)

            pending_records = []
            pending_patches = []
            pending_masks = []
            for component in components:
                try:
                    patch, mask, bbox, pad_top_left = prepare_component_patch(
                        frame,
                        component,
                        patch_shape=(args.patch_size, args.patch_size),
                        padding=args.padding,
                    )
                except ValueError:
                    continue

                scanned_count += 1
                record = {
                    "dataset": name,
                    "frame": int(frame_idx),
                    "component_id": int(component["component_id"]),
                    "pixels": int(component["pixels"]),
                    "bbox_y0x0y1x1": bbox,
                    "pad_top_left": pad_top_left,
                    "centroid_yx": (float(component["cy"]), float(component["cx"])),
                    "seg_path": str(seg_path),
                    "seg_key": args.seg_key,
                    "main_path": str(scan_dir / entry["main"]),
                }
                pending_records.append(record)
                pending_patches.append(patch)
                pending_masks.append(mask)

                if len(pending_patches) >= args.batch_size:
                    predicted_count += flush_predictions(
                        model, device, args, h5_out, csv_writer, preview_dir,
                        pending_records, pending_patches, pending_masks,
                    )
                    pending_records, pending_patches, pending_masks = [], [], []

            predicted_count += flush_predictions(
                model, device, args, h5_out, csv_writer, preview_dir,
                pending_records, pending_patches, pending_masks,
            )

    return scanned_count, predicted_count


def flush_predictions(
    model,
    device: torch.device,
    args,
    h5_out: h5py.File,
    csv_writer,
    preview_dir: Path,
    records: list[dict],
    patches: list[np.ndarray],
    masks: list[np.ndarray],
) -> int:
    if not patches:
        return 0

    probabilities = predict_batch(model, patches, device, args.scale, args.amp)
    kept_records = []
    kept_patches = []
    kept_masks = []
    for record, patch, mask, probability in zip(records, patches, masks, probabilities):
        probability = float(probability)
        if probability < args.threshold:
            continue
        record["overlap_probability"] = probability
        kept_records.append(record)
        kept_patches.append(patch)
        kept_masks.append(mask)
        csv_writer.writerow(
            {
                "dataset": record["dataset"],
                "frame": record["frame"],
                "component_id": record["component_id"],
                "overlap_probability": f"{probability:.6f}",
                "pixels": record["pixels"],
                "bbox_y0": record["bbox_y0x0y1x1"][0],
                "bbox_x0": record["bbox_y0x0y1x1"][1],
                "bbox_y1": record["bbox_y0x0y1x1"][2],
                "bbox_x1": record["bbox_y0x0y1x1"][3],
                "seg_path": record["seg_path"],
                "main_path": record["main_path"],
            }
        )

        preview_path = preview_dir / (
            f"p{probability:.3f}_frame_{record['frame']:05d}_component_{record['component_id']:05d}.png"
        )
        write_preview_png(preview_path, patch, mask, record)

    append_predictions_to_h5(h5_out, kept_records, kept_patches, kept_masks)
    return len(kept_records)


def run(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.base_channels, args.dropout)

    out_h5 = args.output_dir / "predicted_overlaps.h5"
    out_csv = args.output_dir / "predicted_overlaps.csv"
    if out_h5.exists() and not args.overwrite:
        raise FileExistsError(f"{out_h5} exists. Pass --overwrite to replace it.")
    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{out_csv} exists. Pass --overwrite to replace it.")

    fieldnames = [
        "dataset",
        "frame",
        "component_id",
        "overlap_probability",
        "pixels",
        "bbox_y0",
        "bbox_x0",
        "bbox_y1",
        "bbox_x1",
        "seg_path",
        "main_path",
    ]

    total_scanned = 0
    total_predicted = 0
    with h5py.File(out_h5, "w") as h5_out, out_csv.open("w", newline="") as csv_file:
        h5_out.attrs["checkpoint"] = str(args.checkpoint)
        h5_out.attrs["threshold"] = float(args.threshold)
        h5_out.attrs["frames_per_dataset"] = int(args.frames_per_dataset)
        h5_out.attrs["frame_step"] = int(args.frame_step)
        h5_out.attrs["frame_start"] = int(args.frame_start)
        h5_out.attrs["patch_size"] = int(args.patch_size)
        h5_out.attrs["scale"] = float(args.scale)

        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        selected_names = args.datasets or list(DATASETS)
        for name in selected_names:
            if name not in DATASETS:
                raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")
            scanned, predicted = scan_dataset(name, DATASETS[name], args, model, device, h5_out, csv_writer)
            total_scanned += scanned
            total_predicted += predicted
            logging.info("%s: scanned %d spot crops, kept %d predicted overlaps", name, scanned, predicted)

    logging.info("Done. Scanned %d spot crops and saved %d predicted overlaps.", total_scanned, total_predicted)
    logging.info("CSV: %s", out_csv)
    logging.info("HDF5: %s", out_h5)
    logging.info("Preview PNGs: %s", args.output_dir / "previews")


def get_args():
    parser = argparse.ArgumentParser(description="Run the synthetic overlap classifier on real segvol spots.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Best or final .pth file from train_overlap_classifier.py")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional subset of dataset names to scan.")
    parser.add_argument("--seg-key", type=str, default="segvol")
    parser.add_argument("--seg-threshold", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum overlap probability to save.")
    parser.add_argument("--frames-per-dataset", type=int, default=20)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--min-pixels", type=int, default=3)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument("--scale", type=float, default=0.5, help="Must match the scale used during classifier training.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(get_args())
