from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np
import torch

from seg_pipeline.seg_types import SegBlob, SegIsolatedSpot, SegPipelineConfig


def _cc_sparse(binary: torch.Tensor) -> np.ndarray:
    """BFS connected components on a sparse binary mask.
    Only touches nonzero pixels — O(k) where k << H*W for typical segmentation frames.
    Returns int32 label array of the same shape (0 = background).
    """
    H, W = binary.shape
    b = binary.numpy()
    ys, xs = np.where(b)
    if len(ys) == 0:
        return np.zeros((H, W), dtype=np.int32)

    visited = np.zeros((H, W), dtype=bool)
    labels = np.zeros((H, W), dtype=np.int32)
    label = 0

    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        label += 1
        queue: deque[tuple[int, int]] = deque([(y0, x0)])
        visited[y0, x0] = True
        while queue:
            y, x = queue.popleft()
            labels[y, x] = label
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and b[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    return labels


# ---------------------------------------------------------------------------
# Pass 1
# ---------------------------------------------------------------------------

def pass1_extract(
    frame: torch.Tensor,
    frame_idx: int,
    config: SegPipelineConfig,
    start_id: int = 0,
) -> list[SegBlob]:
    """Extract blobs from one intensity frame via binarization + CC + regionprops."""
    thr = (
        config.seg_vol_binarize_threshold
        if config.segmentation_source == "seg_vol"
        else config.log_binarize_threshold
    )
    binary = frame > thr
    if not binary.any():
        return []

    label_np = _cc_sparse(binary)

    # Vectorised regionprops: sort all nonzero pixels by label, then slice groups.
    # Avoids O(n_blobs × H×W) scans from the naive `labels == lab` approach.
    frame_np = frame.numpy()
    ys_all, xs_all = np.where(label_np > 0)
    if len(ys_all) == 0:
        return []

    labs_all = label_np[ys_all, xs_all]
    order = np.argsort(labs_all)
    ys_s = ys_all[order]
    xs_s = xs_all[order]
    labs_s = labs_all[order]
    _, starts, counts = np.unique(labs_s, return_index=True, return_counts=True)

    blobs: list[SegBlob] = []
    bid = start_id
    for start, count in zip(starts, counts):
        area = int(count)
        if area < config.min_blob_area or area > config.max_blob_area:
            continue
        ys = ys_s[start:start + count]
        xs = xs_s[start:start + count]
        cy = float(ys.mean())
        cx = float(xs.mean())
        min_r, max_r = int(ys.min()), int(ys.max()) + 1
        min_c, max_c = int(xs.min()), int(xs.max()) + 1
        crop_mask = np.zeros((max_r - min_r, max_c - min_c), dtype=bool)
        crop_mask[ys - min_r, xs - min_c] = True
        crop_int = frame_np[min_r:max_r, min_c:max_c].astype(np.float32)
        blobs.append(SegBlob(
            blob_id=bid,
            frame_idx=frame_idx,
            centroid=(cx, cy),
            area=area,
            bbox=(min_r, min_c, max_r, max_c),
            mask=torch.from_numpy(crop_mask),
            intensity=torch.from_numpy(crop_int),
        ))
        bid += 1
    return blobs


# ---------------------------------------------------------------------------
# Pass 2 helpers
# ---------------------------------------------------------------------------

def _ncc_centered(
    int_a: np.ndarray, mask_a: np.ndarray, com_a: tuple[float, float],
    int_b: np.ndarray, mask_b: np.ndarray, com_b: tuple[float, float],
) -> float:
    """Pearson NCC between intensity patches after CoM-aligning on common canvas.
    Evaluated over the union of the two masks.
    B is assumed already h-flipped by caller before passing here.
    Returns 0.0 when either patch is flat (no signal to correlate).
    """
    ha, wa = int_a.shape
    hb, wb = int_b.shape
    ch = max(ha, hb) * 2 + 4
    cw = max(wa, wb) * 2 + 4
    cy, cx = ch // 2, cw // 2

    def _place(arr: np.ndarray, mask: np.ndarray, com: tuple[float, float]):
        h, w = arr.shape
        y0 = cy - int(round(com[0]))
        x0 = cx - int(round(com[1]))
        c_arr = np.zeros((ch, cw), dtype=np.float32)
        c_mask = np.zeros((ch, cw), dtype=bool)
        sy, ey = max(0, y0), min(ch, y0 + h)
        sx, ex = max(0, x0), min(cw, x0 + w)
        my, ny = max(0, -y0), h - max(0, y0 + h - ch)
        mx, nx = max(0, -x0), w - max(0, x0 + w - cw)
        c_arr[sy:ey, sx:ex] = arr[my:ny, mx:nx]
        c_mask[sy:ey, sx:ex] = mask[my:ny, mx:nx]
        return c_arr, c_mask

    ca, ma = _place(int_a, mask_a, com_a)
    cb, mb = _place(int_b, mask_b, com_b)

    eval_region = ma | mb
    if not eval_region.any():
        return 0.0

    vals_a = ca[eval_region]
    vals_b = cb[eval_region]
    vals_a = vals_a - vals_a.mean()
    vals_b = vals_b - vals_b.mean()
    norm = np.sqrt(np.dot(vals_a, vals_a) * np.dot(vals_b, vals_b))
    if norm < 1e-8:
        return 0.0
    return float(np.dot(vals_a, vals_b) / norm)


def _com_in_crop(blob: SegBlob) -> tuple[float, float]:
    """Centre-of-mass of blob relative to its local crop (row, col)."""
    return (blob.centroid[1] - blob.bbox[0], blob.centroid[0] - blob.bbox[1])


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Hungarian algorithm (Kuhn-Munkres) for minimum-cost assignment on n×n matrix.
    Returns assignment[i] = j.
    """
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.intp)
    way = np.zeros(n + 1, dtype=np.intp)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, -1
            for j in range(1, n + 1):
                if not used[j]:
                    val = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if val < minv[j]:
                        minv[j] = val
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    ans = np.empty(n, dtype=np.intp)
    for j in range(1, n + 1):
        if p[j] != 0:
            ans[p[j] - 1] = j - 1
    return ans


def _max_weight_matching(score_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Maximum weight bipartite matching on a (n, m) score matrix.
    Entries = 0 mean forbidden. Returns list of (row, col) index pairs.
    Uses the Hungarian algorithm on a pruned square cost matrix where
    forbidden/padding cost = 1.0 and valid cost = 1 - score.
    """
    if not (score_matrix > 0).any():
        return []

    # Prune to rows/cols that have at least one valid entry — keeps the
    # Hungarian matrix small even when the full matrix is large.
    row_has = (score_matrix > 0).any(axis=1)
    col_has = (score_matrix > 0).any(axis=0)
    active_r = np.where(row_has)[0]
    active_c = np.where(col_has)[0]

    sub = score_matrix[np.ix_(active_r, active_c)]
    nr, nc = len(active_r), len(active_c)
    size = max(nr, nc)
    valid = sub > 0
    cost = np.ones((size, size))
    cost[:nr, :nc][valid] = 1.0 - sub[valid]

    assignment = _hungarian(cost)
    return [
        (int(active_r[i]), int(active_c[int(assignment[i])]))
        for i in range(nr)
        if assignment[i] < nc and valid[i, assignment[i]]
    ]


# ---------------------------------------------------------------------------
# Pass 2
# ---------------------------------------------------------------------------

def pass2_pair_friedel(blobs: list[SegBlob], config: SegPipelineConfig) -> list[SegBlob]:
    """Pair blobs by Friedel ω-offset using CoM-centred NCC + max-weight bipartite matching.
    Mutates blobs in place (sets friedel_partner_id and ncc_score). Returns same list.
    """
    if len(blobs) < 2:
        return blobs

    offset = config.friedel_omega_offset_frames
    tol = config.friedel_omega_tolerance_frames

    by_frame: defaultdict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(blobs):
        by_frame[b.frame_idx].append(i)

    matched: set[int] = set()
    ar = config.min_area_ratio
    min_ncc = config.min_ncc
    ncc_area_weight = config.ncc_area_weight
    mid_tol      = config.beam_half_height
    x_half_width = config.beam_half_width
    t0 = time.time()

    for t in sorted(by_frame):
        src_idx = [i for i in by_frame[t] if i not in matched]
        if not src_idx:
            continue

        # Collect candidates from both Friedel directions
        tgt_seen: set[int] = set()
        for dt in range(-tol, tol + 1):
            for t2 in (t + offset + dt, t - offset + dt):
                if t2 in by_frame and t2 != t:
                    for i in by_frame[t2]:
                        if i not in matched:
                            tgt_seen.add(i)
        tgt_idx = list(tgt_seen)
        if not tgt_idx:
            print(f"  frame {t:4d}: {len(src_idx):3d} src blobs, no Friedel candidates", flush=True)
            continue

        n, m = len(src_idx), len(tgt_idx)
        ncc_mat = np.zeros((n, m))
        n_skipped = n_computed = 0

        # Precompute per-target: h-flipped numpy intensity+mask arrays, CoM-in-crop, centroid row.
        tgt_data: list[tuple] = []
        for ti in tgt_idx:
            tb = blobs[ti]
            np_mask = tb.mask.numpy()
            np_int = tb.intensity.numpy()
            tgt_data.append((
                tb.area,
                tb.bbox[2] - tb.bbox[0],
                tb.bbox[3] - tb.bbox[1],
                np_int[:, ::-1],   # h-flipped intensity
                np_mask[:, ::-1],  # h-flipped mask
                _com_in_crop(tb),
                tb.centroid[1],    # row (y) for Y spatial pre-filter
                tb.centroid[0],    # col (x) for X spatial pre-filter
            ))

        for r, si in enumerate(src_idx):
            sb = blobs[si]
            com_a = _com_in_crop(sb)
            np_a_int = sb.intensity.numpy()
            np_a_mask = sb.mask.numpy()
            ha = sb.bbox[2] - sb.bbox[0]
            wa = sb.bbox[3] - sb.bbox[1]
            for c, (tb_area, hb, wb, flipped_int, flipped_mask, com_b, tgt_cy, tgt_cx) in enumerate(tgt_data):
                # Midpoint of pair = implied grain row; must lie within the illuminated slab.
                if mid_tol is not None:
                    if abs((sb.centroid[1] + tgt_cy) / 2 - config.beam_center_y) > mid_tol:
                        n_skipped += 1
                        continue
                # X difference pre-filter: 180° rotation preserves spot X side,
                # so |x_A - x_B| = 2*|g0x| < 2*specimen_radius.
                if x_half_width is not None:
                    if abs(sb.centroid[0] - tgt_cx) > 2 * x_half_width:
                        n_skipped += 1
                        continue
                # Area-ratio pre-filter: same grain → similar blob size. Independent of min_ncc.
                if min(sb.area, tb_area) < ar * max(sb.area, tb_area):
                    n_skipped += 1
                    continue
                # Dimension check (h-flip preserves rows, swaps nothing; just size compat).
                if min(ha, hb) < ar * max(ha, hb):
                    n_skipped += 1
                    continue
                if min(wa, wb) < ar * max(wa, wb):
                    n_skipped += 1
                    continue
                ncc = _ncc_centered(np_a_int, np_a_mask, com_a, flipped_int, flipped_mask, com_b)
                n_computed += 1
                effective_threshold = min_ncc
                if ncc_area_weight > 0.0:
                    effective_threshold += ncc_area_weight / np.sqrt(min(sb.area, tb_area))
                if ncc >= effective_threshold:
                    ncc_mat[r, c] = ncc

        frame_pairs = 0
        for r, c in _max_weight_matching(ncc_mat):
            si, ti = src_idx[r], tgt_idx[c]
            if ti in matched:
                continue
            score = ncc_mat[r, c]
            blobs[si].friedel_partner_id = blobs[ti].blob_id
            blobs[ti].friedel_partner_id = blobs[si].blob_id
            blobs[si].ncc_score = score
            blobs[ti].ncc_score = score
            matched.add(si)
            matched.add(ti)
            frame_pairs += 1

        print(
            f"  frame {t:4d}: {n:3d}×{m:3d}  "
            f"computed {n_computed:5d}  skipped {n_skipped:5d}  "
            f"pairs {frame_pairs:3d}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    return blobs


# ---------------------------------------------------------------------------
# Pass 3
# ---------------------------------------------------------------------------

def pass3_gate(
    blobs: list[SegBlob],
    config: SegPipelineConfig,
) -> list[SegIsolatedSpot]:
    """Materialise paired blobs as SegIsolatedSpot; NCC already gated in pass 2."""
    if not blobs:
        return []

    paired = [b for b in blobs if b.friedel_partner_id is not None]
    if not paired:
        return []

    spots: list[SegIsolatedSpot] = []
    for b in paired:
        spots.append(SegIsolatedSpot(
            blob_id=b.blob_id,
            patch=b.intensity,
            mask=b.mask,
            centroid=b.centroid,
            frame_idx=b.frame_idx,
            friedel_partner_id=b.friedel_partner_id,
            ncc_score=b.ncc_score,
        ))
    return spots
