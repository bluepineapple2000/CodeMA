from __future__ import annotations

from collections import deque
from csv import DictWriter
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers ESRF compression filters
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ESRF = PROJECT_DIR.parent / "data_esrf"
SCAN = "IN718_twins"
FRAME = 1148
SEG_KEY = "segvol"
RAW_KEY = "instrument/detector_0/data"

SEG_FILE = DATA_ESRF / SCAN / "segvol.h5"
RAW_FILE = DATA_ESRF / SCAN / "ep_sync_12_dct_7p5_3_ep_sync_12_dct_7p5.h5"
OUT_DIR = PROJECT_DIR / "prediction_previews" / "IN718_twins_frame_1148_overlap"
OUT_H5 = OUT_DIR / "IN718_twins_frame_1148_spots.h5"
OUT_CSV = OUT_DIR / "IN718_twins_frame_1148_spots.csv"

# Corrected frame coordinates. This captures the large overlapping blob that
# sits in the lower-right raw-frame region marked in the frame-1148 screenshot.
OVERLAP_HINT_BBOX_YX = (1720, 1580, 1940, 1810)


def fix_seg_orientation(frame: np.ndarray) -> np.ndarray:
    return np.rot90(frame[::-1], k=-1)


def connected_components(binary: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    components: list[dict] = []
    label = 0

    for start_y, start_x in np.argwhere(binary):
        start_y = int(start_y)
        start_x = int(start_x)
        if labels[start_y, start_x] != 0:
            continue

        label += 1
        queue = deque([(start_y, start_x)])
        labels[start_y, start_x] = label
        y0 = y1 = start_y
        x0 = x1 = start_x
        area = 0

        while queue:
            y, x = queue.popleft()
            area += 1
            y0 = min(y0, y)
            y1 = max(y1, y)
            x0 = min(x0, x)
            x1 = max(x1, x)

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = y + dy
                nx = x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and binary[ny, nx]
                    and labels[ny, nx] == 0
                ):
                    labels[ny, nx] = label
                    queue.append((ny, nx))

        components.append(
            {
                "label": label,
                "area": area,
                "y0": y0,
                "y1": y1 + 1,
                "x0": x0,
                "x1": x1 + 1,
                "cy": 0.5 * (y0 + y1 + 1),
                "cx": 0.5 * (x0 + x1 + 1),
            }
        )

    return labels, components


def padded_bbox(component: dict, shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    height, width = shape
    return (
        max(0, component["y0"] - padding),
        max(0, component["x0"] - padding),
        min(height, component["y1"] + padding),
        min(width, component["x1"] + padding),
    )


def component_intensity_stats(component: dict, labels: np.ndarray, raw: np.ndarray) -> dict:
    y0, y1, x0, x1 = component["y0"], component["y1"], component["x0"], component["x1"]
    mask = labels[y0:y1, x0:x1] == component["label"]
    values = raw[y0:y1, x0:x1][mask]
    return {
        "raw_min": int(values.min()),
        "raw_max": int(values.max()),
        "raw_mean": float(values.mean()),
        "raw_sum": int(values.sum()),
    }


def choose_overlap_component(components: list[dict]) -> dict:
    y0, x0, y1, x1 = OVERLAP_HINT_BBOX_YX
    hinted = [
        comp
        for comp in components
        if y0 <= comp["cy"] <= y1 and x0 <= comp["cx"] <= x1 and comp["area"] >= 100
    ]
    if hinted:
        return max(hinted, key=lambda comp: comp["area"])
    return max(components, key=lambda comp: comp["area"])


def smooth_closed_patch(
    points_xy: list[tuple[float, float]],
    color: str,
    linewidth: float,
    alpha: float = 0.98,
    fill_alpha: float = 0.0,
) -> PathPatch:
    points = np.asarray(points_xy, dtype=np.float64)
    vertices = []
    codes = []
    n_points = len(points)

    for idx, point in enumerate(points):
        previous_point = points[(idx - 1) % n_points]
        next_point = points[(idx + 1) % n_points]
        next_next_point = points[(idx + 2) % n_points]

        if idx == 0:
            vertices.append(tuple(point))
            codes.append(MplPath.MOVETO)

        control_1 = point + (next_point - previous_point) / 6.0
        control_2 = next_point - (next_next_point - point) / 6.0
        vertices.extend([tuple(control_1), tuple(control_2), tuple(next_point)])
        codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

    vertices.append(tuple(points[0]))
    codes.append(MplPath.CLOSEPOLY)
    return PathPatch(
        MplPath(vertices, codes),
        fill=fill_alpha > 0,
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        hatch=None,
        antialiased=True,
    ) if fill_alpha <= 0 else PathPatch(
        MplPath(vertices, codes),
        fill=True,
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
        alpha=fill_alpha,
        capstyle="round",
        joinstyle="round",
        antialiased=True,
    )


def scaled_points(
    raw_crop: np.ndarray, points_unit: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    height, width = raw_crop.shape
    return [(x * width, y * height) for x, y in points_unit]


def polygon_mask(shape: tuple[int, int], points_xy: list[tuple[float, float]]) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return MplPath(points_xy).contains_points(points).reshape(shape)


def smooth_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    values = mask.astype(np.float32)
    for _ in range(iterations):
        padded = np.pad(values, 1, mode="edge")
        values = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
    return values


def spot_alpha_mask(
    raw_crop: np.ndarray,
    region_points: list[tuple[float, float]],
    lower_percentile: float,
    upper_percentile: float,
    max_alpha: float,
    support_smoothing: int = 2,
    alpha_smoothing: int = 1,
) -> np.ndarray:
    support = polygon_mask(raw_crop.shape, scaled_points(raw_crop, region_points))
    values = raw_crop[support]
    low, high = np.percentile(values, [lower_percentile, upper_percentile])
    scaled = np.clip((raw_crop.astype(np.float32) - low) / max(high - low, 1.0), 0.0, 1.0)
    signal = support & (raw_crop >= low)
    soft_support = smooth_mask(signal, iterations=support_smoothing)
    alpha = soft_support * (0.08 + max_alpha * scaled)
    alpha = smooth_mask(alpha, iterations=alpha_smoothing)
    alpha[~smooth_mask(support, iterations=1).astype(bool)] = 0.0
    return np.clip(alpha, 0.0, max_alpha)


def add_colored_spot(
    ax,
    raw_crop: np.ndarray,
    region_points: list[tuple[float, float]],
    color: tuple[float, float, float],
    lower_percentile: float,
    upper_percentile: float,
    max_alpha: float,
    outline_width: float,
    support_smoothing: int = 2,
    alpha_smoothing: int = 1,
    contour_fraction: float = 0.18,
) -> np.ndarray:
    alpha = spot_alpha_mask(
        raw_crop,
        region_points,
        lower_percentile,
        upper_percentile,
        max_alpha,
        support_smoothing=support_smoothing,
        alpha_smoothing=alpha_smoothing,
    )
    rgba = np.zeros((*raw_crop.shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = alpha
    ax.imshow(rgba, interpolation="bilinear")
    ax.contour(alpha, levels=[max_alpha * contour_fraction], colors=[color], linewidths=outline_width, alpha=0.82)
    return alpha


def weighted_two_cluster_mask(raw_crop: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(object_mask)
    values = raw_crop[ys, xs].astype(np.float64)
    if len(values) < 2:
        return np.zeros_like(raw_crop, dtype=np.uint8)

    threshold = np.percentile(values, 68)
    keep = values >= threshold
    points = np.column_stack([ys[keep], xs[keep]]).astype(np.float64)
    weights = values[keep] - values[keep].min() + 1.0
    if len(points) < 2:
        return np.zeros_like(raw_crop, dtype=np.uint8)

    centers = np.array([points[np.argmin(points[:, 0])], points[np.argmax(points[:, 0])]], dtype=np.float64)
    for _ in range(40):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignment = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for cluster_id in (0, 1):
            cluster = assignment == cluster_id
            if cluster.any():
                next_centers[cluster_id] = np.average(points[cluster], axis=0, weights=weights[cluster])
        if np.allclose(next_centers, centers):
            break
        centers = next_centers

    all_points = np.column_stack([ys, xs]).astype(np.float64)
    all_distances = ((all_points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    all_assignment = np.argmin(all_distances, axis=1) + 1
    split = np.zeros_like(raw_crop, dtype=np.uint8)
    split[ys, xs] = all_assignment.astype(np.uint8)
    return split


def save_spot_archive(raw: np.ndarray, seg: np.ndarray, labels: np.ndarray, components: list[dict]) -> None:
    with h5py.File(OUT_H5, "w") as h5:
        h5.attrs["scan"] = SCAN
        h5.attrs["frame"] = FRAME
        h5.attrs["raw_file"] = str(RAW_FILE)
        h5.attrs["seg_file"] = str(SEG_FILE)
        h5.attrs["seg_orientation"] = "np.rot90(frame[::-1], k=-1)"
        h5.attrs["raw_orientation"] = "unchanged"
        h5.attrs["component_connectivity"] = "4-neighbour"

        for comp in components:
            y0, y1, x0, x1 = comp["y0"], comp["y1"], comp["x0"], comp["x1"]
            group = h5.create_group(f"spot_{comp['label']:04d}")
            group.attrs["bbox_yx"] = np.array([y0, x0, y1, x1], dtype=np.int32)
            group.attrs["area"] = int(comp["area"])
            group.attrs["centroid_yx"] = np.array([comp["cy"], comp["cx"]], dtype=np.float32)
            group.create_dataset("raw_crop", data=raw[y0:y1, x0:x1], compression="gzip")
            group.create_dataset("seg_crop", data=seg[y0:y1, x0:x1], compression="gzip")
            group.create_dataset(
                "component_mask",
                data=(labels[y0:y1, x0:x1] == comp["label"]).astype(np.uint8),
                compression="gzip",
            )


def save_metadata_csv(components: list[dict]) -> None:
    fieldnames = [
        "label",
        "area",
        "y0",
        "x0",
        "y1",
        "x1",
        "cy",
        "cx",
        "raw_min",
        "raw_max",
        "raw_mean",
        "raw_sum",
        "is_overlap_candidate",
    ]
    with OUT_CSV.open("w", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comp in components:
            writer.writerow({key: comp[key] for key in fieldnames})


def save_overview(raw: np.ndarray, labels: np.ndarray, components: list[dict], overlap: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=220)
    vmin, vmax = np.percentile(raw, [0.2, 99.85])
    ax.imshow(raw, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    for comp in components:
        if comp["area"] < 100:
            continue
        color = "#f6c445" if comp["label"] != overlap["label"] else "#00b7ff"
        rect = Rectangle(
            (comp["x0"], comp["y0"]),
            comp["x1"] - comp["x0"],
            comp["y1"] - comp["y0"],
            fill=False,
            edgecolor=color,
            linewidth=0.45 if comp["label"] != overlap["label"] else 1.6,
        )
        ax.add_patch(rect)
    ax.set_xlim(0, raw.shape[1])
    ax.set_ylim(raw.shape[0], 0)
    ax.set_axis_off()
    fig.savefig(OUT_DIR / "IN718_twins_frame_1148_overview_spot_boxes.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_overlap_figures(raw: np.ndarray, seg: np.ndarray, labels: np.ndarray, overlap: dict) -> None:
    y0, x0, y1, x1 = padded_bbox(overlap, raw.shape, padding=36)
    raw_crop = raw[y0:y1, x0:x1]
    seg_crop = seg[y0:y1, x0:x1]
    object_mask = labels[y0:y1, x0:x1] == overlap["label"]
    split = weighted_two_cluster_mask(raw_crop, object_mask)
    vmin, vmax = np.percentile(raw_crop, [1, 99.75])

    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=700)
    ax.imshow(raw_crop, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bicubic")

    yellow_spot = [
        (0.06, 0.20), (0.16, 0.12), (0.34, 0.10), (0.60, 0.09),
        (0.78, 0.13), (0.90, 0.26), (0.96, 0.46), (0.92, 0.61),
        (0.78, 0.70), (0.61, 0.76), (0.47, 0.76), (0.36, 0.70),
        (0.30, 0.60), (0.25, 0.50), (0.18, 0.42), (0.08, 0.35),
    ]
    green_spot = [
        (0.18, 0.51), (0.31, 0.52), (0.46, 0.57), (0.58, 0.68),
        (0.67, 0.80), (0.64, 0.91), (0.45, 0.92), (0.26, 0.86),
        (0.10, 0.77), (0.06, 0.66), (0.09, 0.57),
    ]

    yellow_alpha = add_colored_spot(
        ax, raw_crop, yellow_spot, (1.0, 0.95, 0.0),
        lower_percentile=31, upper_percentile=97.8, max_alpha=0.42, outline_width=1.0,
        support_smoothing=6, alpha_smoothing=3, contour_fraction=0.16,
    )
    green_alpha = add_colored_spot(
        ax, raw_crop, green_spot, (0.05, 1.0, 0.12),
        lower_percentile=26, upper_percentile=97.0, max_alpha=0.40, outline_width=1.2,
        support_smoothing=3, alpha_smoothing=1, contour_fraction=0.18,
    )

    ax.set_xlim(0, raw_crop.shape[1])
    ax.set_ylim(raw_crop.shape[0], 0)
    ax.set_axis_off()
    fig.savefig(OUT_DIR / "IN718_twins_frame_1148_overlap_annotated.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_DIR / "IN718_twins_frame_1148_overlap_annotated.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=700)
    ax.imshow(raw_crop, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bicubic")
    ax.set_axis_off()
    fig.savefig(OUT_DIR / "IN718_twins_frame_1148_overlap_raw_crop.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    np.savez_compressed(
        OUT_DIR / "IN718_twins_frame_1148_overlap_crop.npz",
        raw_crop=raw_crop,
        seg_crop=seg_crop,
        component_mask=object_mask.astype(np.uint8),
        visual_split_mask=split,
        visual_yellow_alpha=yellow_alpha,
        visual_green_alpha=green_alpha,
        bbox_yx=np.array([y0, x0, y1, x1], dtype=np.int32),
        component_bbox_yx=np.array(
            [overlap["y0"], overlap["x0"], overlap["y1"], overlap["x1"]], dtype=np.int32
        ),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(SEG_FILE, "r") as h5:
        seg = fix_seg_orientation(np.asarray(h5[SEG_KEY][FRAME]))
    with h5py.File(RAW_FILE, "r") as h5:
        raw = np.asarray(h5[RAW_KEY][FRAME])

    labels, components = connected_components(seg > 0)
    for comp in components:
        comp.update(component_intensity_stats(comp, labels, raw))
        comp["is_overlap_candidate"] = False

    overlap = choose_overlap_component(components)
    overlap["is_overlap_candidate"] = True

    save_spot_archive(raw, seg, labels, components)
    save_metadata_csv(components)
    save_overview(raw, labels, components, overlap)
    save_overlap_figures(raw, seg, labels, overlap)

    print(f"Saved {len(components)} frame-{FRAME} components to {OUT_H5}")
    print(f"Metadata: {OUT_CSV}")
    print(
        "Overlap candidate:",
        f"spot_{overlap['label']:04d}",
        f"bbox_yx={[overlap['y0'], overlap['x0'], overlap['y1'], overlap['x1']]}",
        f"area={overlap['area']}",
    )
    print(f"Figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
