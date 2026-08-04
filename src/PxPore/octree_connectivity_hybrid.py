"""Adaptive coarse-grid/octree connectivity for directional transport.

This module is intentionally separate from ``octree_conectivity`` so the
legacy octree/any implementation remains unchanged.  Refined root voxels are
removed from the coarse graph, represented by their void octree leaves, and
then reattached through explicit face-sharing interface edges.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .connectivity_multicore import (
    INVALID,
    LABEL_MASK_ACC,
    LABEL_MASK_TRAP,
    TX0,
    TX1,
    TY0,
    TY1,
    TZ0,
    TZ1,
    _connectivity_build_nonperiodic_roots,
    _periodic_collect_seam_edges,
    _periodic_winding_from_edges,
)
from .octree import OCC_ACC, OCC_TRAP, OCC_VOID
from .octree_conectivity import (
    _extract_void_leaves,
    _face_touch_boxes,
    _idx1d,
    _idx3d,
)


_WIND_X = np.uint8(1 << 0)
_WIND_Y = np.uint8(1 << 1)
_WIND_Z = np.uint8(1 << 2)
_WIND_ANY = _WIND_X | _WIND_Y | _WIND_Z


@njit(inline="always", cache=True)
def _uf_find(parent, node):
    root = node
    while parent[root] != root:
        root = parent[root]
    current = node
    while parent[current] != current:
        next_node = parent[current]
        parent[current] = root
        current = next_node
    return root


@njit(inline="always", cache=True)
def _uf_find_readonly(parent, node):
    root = node
    while parent[root] != root:
        root = parent[root]
    return root


@njit(inline="always", cache=True)
def _uf_union(parent, rank, left, right):
    root_left = _uf_find(parent, left)
    root_right = _uf_find(parent, right)
    if root_left == root_right:
        return root_left
    if rank[root_left] < rank[root_right]:
        root_left, root_right = root_right, root_left
    parent[root_right] = root_left
    if rank[root_left] == rank[root_right]:
        rank[root_left] += np.uint8(1)
    return root_left


@njit(inline="always", cache=True)
def _bucket_lookup(bucket_root_ids, root_id):
    position = np.searchsorted(bucket_root_ids, root_id)
    if (
        position < bucket_root_ids.shape[0]
        and bucket_root_ids[position] == root_id
    ):
        return position
    return -1


def _half_tables(grid_info):
    _, _, _, dgx, dgy, dgz = grid_info
    levels = np.arange(16, dtype=np.float64)
    tables = np.empty((16, 3), dtype=np.float64)
    tables[:, 0] = 0.5 * dgx / (2.0**levels)
    tables[:, 1] = 0.5 * dgy / (2.0**levels)
    tables[:, 2] = 0.5 * dgz / (2.0**levels)
    return tables


@njit(parallel=True, cache=True)
def _union_leaves_inside_roots(
    oct_soa_tuple,
    leaf_linear,
    leaf_bucket,
    half_tables,
    parent_leaf,
    rank_leaf,
    tolerance,
):
    x, y, z, _, _, _, level, _ = oct_soa_tuple
    bucket_count = leaf_bucket.shape[0] - 1
    # Buckets are disjoint, so every parallel iteration writes to a disjoint
    # parent/rank interval.
    for bucket in prange(bucket_count):
        start = leaf_bucket[bucket]
        stop = leaf_bucket[bucket + 1]
        for left in range(start, stop):
            left_node = leaf_linear[left]
            left_half = half_tables[level[left_node]]
            for right in range(left + 1, stop):
                right_node = leaf_linear[right]
                right_half = half_tables[level[right_node]]
                if _face_touch_boxes(
                    x[left_node],
                    y[left_node],
                    z[left_node],
                    left_half,
                    x[right_node],
                    y[right_node],
                    z[right_node],
                    right_half,
                    tolerance,
                ):
                    _uf_union(
                        parent_leaf,
                        rank_leaf,
                        np.uint32(left),
                        np.uint32(right),
                    )


@njit(parallel=True, cache=True)
def _count_cross_root_leaf_edges(
    grid_info,
    oct_soa_tuple,
    leaf_linear,
    leaf_bucket,
    bucket_root_ids,
    half_tables,
    tolerance,
):
    gx, gy, gz, _, _, _ = grid_info
    x, y, z, _, _, _, level, _ = oct_soa_tuple
    bucket_count = bucket_root_ids.shape[0]
    counts = np.zeros(bucket_count, dtype=np.int64)
    for bucket in prange(bucket_count):
        root_id = bucket_root_ids[bucket]
        ix, iy, iz = _idx1d(root_id, gy, gz)
        start = leaf_bucket[bucket]
        stop = leaf_bucket[bucket + 1]
        count = np.int64(0)

        for axis in range(3):
            tx, ty, tz = ix, iy, iz
            if axis == 0:
                tx += 1
                if tx >= gx:
                    continue
            elif axis == 1:
                ty += 1
                if ty >= gy:
                    continue
            else:
                tz += 1
                if tz >= gz:
                    continue
            neighbor_id = _idx3d(tx, ty, tz, gy, gz)
            neighbor_bucket = _bucket_lookup(
                bucket_root_ids, neighbor_id
            )
            if neighbor_bucket < 0:
                continue
            neighbor_start = leaf_bucket[neighbor_bucket]
            neighbor_stop = leaf_bucket[neighbor_bucket + 1]
            for left in range(start, stop):
                left_node = leaf_linear[left]
                left_half = half_tables[level[left_node]]
                for right in range(neighbor_start, neighbor_stop):
                    right_node = leaf_linear[right]
                    right_half = half_tables[level[right_node]]
                    if _face_touch_boxes(
                        x[left_node],
                        y[left_node],
                        z[left_node],
                        left_half,
                        x[right_node],
                        y[right_node],
                        z[right_node],
                        right_half,
                        tolerance,
                    ):
                        count += 1
        counts[bucket] = count
    return counts


@njit(parallel=True, cache=True)
def _fill_cross_root_leaf_edges(
    grid_info,
    oct_soa_tuple,
    leaf_linear,
    leaf_bucket,
    bucket_root_ids,
    half_tables,
    tolerance,
    offsets,
    edge_left,
    edge_right,
):
    gx, gy, gz, _, _, _ = grid_info
    x, y, z, _, _, _, level, _ = oct_soa_tuple
    bucket_count = bucket_root_ids.shape[0]
    for bucket in prange(bucket_count):
        root_id = bucket_root_ids[bucket]
        ix, iy, iz = _idx1d(root_id, gy, gz)
        start = leaf_bucket[bucket]
        stop = leaf_bucket[bucket + 1]
        position = offsets[bucket]

        for axis in range(3):
            tx, ty, tz = ix, iy, iz
            if axis == 0:
                tx += 1
                if tx >= gx:
                    continue
            elif axis == 1:
                ty += 1
                if ty >= gy:
                    continue
            else:
                tz += 1
                if tz >= gz:
                    continue
            neighbor_id = _idx3d(tx, ty, tz, gy, gz)
            neighbor_bucket = _bucket_lookup(
                bucket_root_ids, neighbor_id
            )
            if neighbor_bucket < 0:
                continue
            neighbor_start = leaf_bucket[neighbor_bucket]
            neighbor_stop = leaf_bucket[neighbor_bucket + 1]
            for left in range(start, stop):
                left_node = leaf_linear[left]
                left_half = half_tables[level[left_node]]
                for right in range(neighbor_start, neighbor_stop):
                    right_node = leaf_linear[right]
                    right_half = half_tables[level[right_node]]
                    if _face_touch_boxes(
                        x[left_node],
                        y[left_node],
                        z[left_node],
                        left_half,
                        x[right_node],
                        y[right_node],
                        z[right_node],
                        right_half,
                        tolerance,
                    ):
                        edge_left[position] = np.uint32(left)
                        edge_right[position] = np.uint32(right)
                        position += 1


def _prefix_offsets(counts):
    offsets = np.empty(counts.shape[0], dtype=np.int64)
    if counts.size == 0:
        return offsets, 0
    offsets[0] = 0
    if counts.size > 1:
        offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return offsets, int(offsets[-1] + counts[-1])


@njit(cache=True)
def _union_edge_list(parent, rank, edge_left, edge_right):
    for edge in range(edge_left.shape[0]):
        _uf_union(parent, rank, edge_left[edge], edge_right[edge])


@njit(parallel=True, cache=True)
def _flatten_leaf_roots(parent_leaf):
    roots = np.empty(parent_leaf.shape[0], dtype=np.uint32)
    for index in prange(parent_leaf.shape[0]):
        roots[index] = _uf_find_readonly(
            parent_leaf, np.uint32(index)
        )
    return roots


def _build_leaf_components(
    grid_info,
    oct_soa_tuple,
    leaf_linear,
    leaf_root_linear,
    leaf_bucket,
    tolerance,
):
    n_leaf = leaf_linear.shape[0]
    if n_leaf == 0:
        return np.empty(0, dtype=np.uint32)
    bucket_root_ids = np.asarray(
        leaf_root_linear[leaf_bucket[:-1]], dtype=np.int32
    )
    if np.any(np.diff(bucket_root_ids.astype(np.int64)) <= 0):
        raise ValueError(
            "Octree leaf buckets must be strictly ordered by root voxel"
        )

    parent_leaf = np.arange(n_leaf, dtype=np.uint32)
    rank_leaf = np.zeros(n_leaf, dtype=np.uint8)
    half_tables = _half_tables(grid_info)
    _union_leaves_inside_roots(
        oct_soa_tuple,
        leaf_linear,
        leaf_bucket,
        half_tables,
        parent_leaf,
        rank_leaf,
        tolerance,
    )
    counts = _count_cross_root_leaf_edges(
        grid_info,
        oct_soa_tuple,
        leaf_linear,
        leaf_bucket,
        bucket_root_ids,
        half_tables,
        tolerance,
    )
    offsets, edge_count = _prefix_offsets(counts)
    edge_left = np.empty(edge_count, dtype=np.uint32)
    edge_right = np.empty(edge_count, dtype=np.uint32)
    _fill_cross_root_leaf_edges(
        grid_info,
        oct_soa_tuple,
        leaf_linear,
        leaf_bucket,
        bucket_root_ids,
        half_tables,
        tolerance,
        offsets,
        edge_left,
        edge_right,
    )
    _union_edge_list(parent_leaf, rank_leaf, edge_left, edge_right)
    return _flatten_leaf_roots(parent_leaf)


def _coarse_boundary_touch(
    coarse_void,
    coarse_root,
    active_roots,
):
    touch = np.zeros(active_roots.shape[0], dtype=np.uint8)
    faces = (
        (coarse_void[0, :, :], coarse_root[0, :, :], TX0),
        (coarse_void[-1, :, :], coarse_root[-1, :, :], TX1),
        (coarse_void[:, 0, :], coarse_root[:, 0, :], TY0),
        (coarse_void[:, -1, :], coarse_root[:, -1, :], TY1),
        (coarse_void[:, :, 0], coarse_root[:, :, 0], TZ0),
        (coarse_void[:, :, -1], coarse_root[:, :, -1], TZ1),
    )
    for mask, roots, bit in faces:
        face_roots = np.unique(roots[mask != 0])
        if face_roots.size:
            nodes = np.searchsorted(active_roots, face_roots)
            touch[nodes] |= bit
    return touch


def _leaf_boundary_touch(
    grid_info,
    oct_soa_tuple,
    leaf_linear,
    leaf_component_roots,
    active_leaf_roots,
    tolerance,
):
    gx, gy, gz, dgx, dgy, dgz = grid_info
    lengths = np.asarray((gx * dgx, gy * dgy, gz * dgz))
    x, y, z, _, _, _, level, _ = oct_soa_tuple
    centers = (x[leaf_linear], y[leaf_linear], z[leaf_linear])
    half = _half_tables(grid_info)[level[leaf_linear]]
    touch = np.zeros(active_leaf_roots.shape[0], dtype=np.uint8)
    bits = ((TX0, TX1), (TY0, TY1), (TZ0, TZ1))
    for axis in range(3):
        low = np.abs(centers[axis] - half[:, axis]) <= tolerance
        high = (
            np.abs(centers[axis] + half[:, axis] - lengths[axis])
            <= tolerance
        )
        for mask, bit in ((low, bits[axis][0]), (high, bits[axis][1])):
            roots = np.unique(leaf_component_roots[mask])
            if roots.size:
                nodes = np.searchsorted(active_leaf_roots, roots)
                touch[nodes] |= bit
    return touch


@njit(parallel=True, cache=True)
def _count_leaf_coarse_edges(
    periodic,
    grid_info,
    coarse_void,
    oct_soa_tuple,
    leaf_linear,
    leaf_root_linear,
    half_tables,
    tolerance,
):
    gx, gy, gz, dgx, dgy, dgz = grid_info
    x, y, z, _, _, _, level, _ = oct_soa_tuple
    counts = np.zeros(leaf_linear.shape[0], dtype=np.int64)
    for leaf in prange(leaf_linear.shape[0]):
        node = leaf_linear[leaf]
        root_id = leaf_root_linear[leaf]
        ix, iy, iz = _idx1d(root_id, gy, gz)
        h = half_tables[level[node]]
        count = np.int64(0)

        if abs((x[node] - h[0]) - ix * dgx) <= tolerance:
            if ix > 0:
                count += coarse_void[ix - 1, iy, iz] != 0
            elif periodic:
                count += coarse_void[gx - 1, iy, iz] != 0
        if abs((x[node] + h[0]) - (ix + 1) * dgx) <= tolerance:
            if ix + 1 < gx:
                count += coarse_void[ix + 1, iy, iz] != 0
            elif periodic:
                count += coarse_void[0, iy, iz] != 0
        if abs((y[node] - h[1]) - iy * dgy) <= tolerance:
            if iy > 0:
                count += coarse_void[ix, iy - 1, iz] != 0
            elif periodic:
                count += coarse_void[ix, gy - 1, iz] != 0
        if abs((y[node] + h[1]) - (iy + 1) * dgy) <= tolerance:
            if iy + 1 < gy:
                count += coarse_void[ix, iy + 1, iz] != 0
            elif periodic:
                count += coarse_void[ix, 0, iz] != 0
        if abs((z[node] - h[2]) - iz * dgz) <= tolerance:
            if iz > 0:
                count += coarse_void[ix, iy, iz - 1] != 0
            elif periodic:
                count += coarse_void[ix, iy, gz - 1] != 0
        if abs((z[node] + h[2]) - (iz + 1) * dgz) <= tolerance:
            if iz + 1 < gz:
                count += coarse_void[ix, iy, iz + 1] != 0
            elif periodic:
                count += coarse_void[ix, iy, 0] != 0
        counts[leaf] = count
    return counts


@njit(parallel=True, cache=True)
def _fill_leaf_coarse_edges(
    periodic,
    grid_info,
    coarse_void,
    coarse_root,
    active_coarse_roots,
    oct_soa_tuple,
    leaf_linear,
    leaf_root_linear,
    leaf_component_roots,
    active_leaf_roots,
    half_tables,
    tolerance,
    offsets,
    edge_a,
    edge_b,
    edge_shift,
):
    gx, gy, gz, dgx, dgy, dgz = grid_info
    x, y, z, _, _, _, level, _ = oct_soa_tuple
    coarse_count = active_coarse_roots.shape[0]
    for leaf in prange(leaf_linear.shape[0]):
        node = leaf_linear[leaf]
        root_id = leaf_root_linear[leaf]
        ix, iy, iz = _idx1d(root_id, gy, gz)
        h = half_tables[level[node]]
        leaf_graph_node = np.uint32(
            coarse_count
            + np.searchsorted(active_leaf_roots, leaf_component_roots[leaf])
        )
        position = offsets[leaf]

        # Each tuple is (neighbor coordinates, shift, leaf_is_edge_a).
        if abs((x[node] - h[0]) - ix * dgx) <= tolerance:
            if ix > 0 and coarse_void[ix - 1, iy, iz] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix - 1, iy, iz]
                ))
                edge_a[position], edge_b[position] = leaf_graph_node, coarse_node
                position += 1
            elif periodic and ix == 0 and coarse_void[gx - 1, iy, iz] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[gx - 1, iy, iz]
                ))
                edge_a[position], edge_b[position] = leaf_graph_node, coarse_node
                edge_shift[position, 0] = -1
                position += 1
        if abs((x[node] + h[0]) - (ix + 1) * dgx) <= tolerance:
            if ix + 1 < gx and coarse_void[ix + 1, iy, iz] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix + 1, iy, iz]
                ))
                edge_a[position], edge_b[position] = leaf_graph_node, coarse_node
                position += 1
            elif periodic and ix == gx - 1 and coarse_void[0, iy, iz] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[0, iy, iz]
                ))
                edge_a[position], edge_b[position] = coarse_node, leaf_graph_node
                edge_shift[position, 0] = -1
                position += 1

        if abs((y[node] - h[1]) - iy * dgy) <= tolerance:
            if iy > 0 and coarse_void[ix, iy - 1, iz] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix, iy - 1, iz]
                ))
                edge_a[position], edge_b[position] = leaf_graph_node, coarse_node
                position += 1
            elif periodic and iy == 0 and coarse_void[ix, gy - 1, iz] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix, gy - 1, iz]
                ))
                edge_a[position], edge_b[position] = leaf_graph_node, coarse_node
                edge_shift[position, 1] = -1
                position += 1
        if abs((y[node] + h[1]) - (iy + 1) * dgy) <= tolerance:
            if iy + 1 < gy and coarse_void[ix, iy + 1, iz] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix, iy + 1, iz]
                ))
                edge_a[position], edge_b[position] = leaf_graph_node, coarse_node
                position += 1
            elif periodic and iy == gy - 1 and coarse_void[ix, 0, iz] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix, 0, iz]
                ))
                edge_a[position], edge_b[position] = coarse_node, leaf_graph_node
                edge_shift[position, 1] = -1
                position += 1

        if abs((z[node] - h[2]) - iz * dgz) <= tolerance:
            if iz > 0 and coarse_void[ix, iy, iz - 1] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix, iy, iz - 1]
                ))
                edge_a[position], edge_b[position] = leaf_graph_node, coarse_node
                position += 1
            elif periodic and iz == 0 and coarse_void[ix, iy, gz - 1] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix, iy, gz - 1]
                ))
                edge_a[position], edge_b[position] = leaf_graph_node, coarse_node
                edge_shift[position, 2] = -1
                position += 1
        if abs((z[node] + h[2]) - (iz + 1) * dgz) <= tolerance:
            if iz + 1 < gz and coarse_void[ix, iy, iz + 1] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix, iy, iz + 1]
                ))
                edge_a[position], edge_b[position] = leaf_graph_node, coarse_node
                position += 1
            elif periodic and iz == gz - 1 and coarse_void[ix, iy, 0] != 0:
                coarse_node = np.uint32(np.searchsorted(
                    active_coarse_roots, coarse_root[ix, iy, 0]
                ))
                edge_a[position], edge_b[position] = coarse_node, leaf_graph_node
                edge_shift[position, 2] = -1


def _leaf_coarse_edges(
    periodic,
    grid_info,
    coarse_void,
    coarse_root,
    active_coarse_roots,
    oct_soa_tuple,
    leaf_linear,
    leaf_root_linear,
    leaf_component_roots,
    active_leaf_roots,
    tolerance,
):
    half_tables = _half_tables(grid_info)
    counts = _count_leaf_coarse_edges(
        periodic,
        grid_info,
        coarse_void,
        oct_soa_tuple,
        leaf_linear,
        leaf_root_linear,
        half_tables,
        tolerance,
    )
    offsets, edge_count = _prefix_offsets(counts)
    edge_a = np.empty(edge_count, dtype=np.uint32)
    edge_b = np.empty(edge_count, dtype=np.uint32)
    edge_shift = np.zeros((edge_count, 3), dtype=np.int8)
    _fill_leaf_coarse_edges(
        periodic,
        grid_info,
        coarse_void,
        coarse_root,
        active_coarse_roots,
        oct_soa_tuple,
        leaf_linear,
        leaf_root_linear,
        leaf_component_roots,
        active_leaf_roots,
        half_tables,
        tolerance,
        offsets,
        edge_a,
        edge_b,
        edge_shift,
    )
    return edge_a, edge_b, edge_shift


@njit(parallel=True, cache=True)
def _count_periodic_leaf_seams(
    grid_info,
    oct_soa_tuple,
    leaf_linear,
    leaf_bucket,
    bucket_root_ids,
    half_tables,
    tolerance,
):
    gx, gy, gz, dgx, dgy, dgz = grid_info
    lengths = (gx * dgx, gy * dgy, gz * dgz)
    x, y, z, _, _, _, level, _ = oct_soa_tuple
    counts = np.zeros(bucket_root_ids.shape[0], dtype=np.int64)
    for bucket in prange(bucket_root_ids.shape[0]):
        root_id = bucket_root_ids[bucket]
        ix, iy, iz = _idx1d(root_id, gy, gz)
        start = leaf_bucket[bucket]
        stop = leaf_bucket[bucket + 1]
        count = np.int64(0)
        for axis in range(3):
            if (
                (axis == 0 and ix != 0)
                or (axis == 1 and iy != 0)
                or (axis == 2 and iz != 0)
            ):
                continue
            tx, ty, tz = ix, iy, iz
            if axis == 0:
                tx = gx - 1
            elif axis == 1:
                ty = gy - 1
            else:
                tz = gz - 1
            opposite_id = _idx3d(tx, ty, tz, gy, gz)
            opposite_bucket = _bucket_lookup(
                bucket_root_ids, opposite_id
            )
            if opposite_bucket < 0:
                continue
            opposite_start = leaf_bucket[opposite_bucket]
            opposite_stop = leaf_bucket[opposite_bucket + 1]
            for left in range(start, stop):
                left_node = leaf_linear[left]
                left_half = half_tables[level[left_node]]
                for right in range(opposite_start, opposite_stop):
                    right_node = leaf_linear[right]
                    right_half = half_tables[level[right_node]]
                    right_x = x[right_node]
                    right_y = y[right_node]
                    right_z = z[right_node]
                    if axis == 0:
                        right_x -= lengths[0]
                    elif axis == 1:
                        right_y -= lengths[1]
                    else:
                        right_z -= lengths[2]
                    if _face_touch_boxes(
                        x[left_node],
                        y[left_node],
                        z[left_node],
                        left_half,
                        right_x,
                        right_y,
                        right_z,
                        right_half,
                        tolerance,
                    ):
                        count += 1
        counts[bucket] = count
    return counts


@njit(parallel=True, cache=True)
def _fill_periodic_leaf_seams(
    grid_info,
    oct_soa_tuple,
    leaf_linear,
    leaf_bucket,
    bucket_root_ids,
    leaf_component_roots,
    active_leaf_roots,
    coarse_count,
    half_tables,
    tolerance,
    offsets,
    edge_a,
    edge_b,
    edge_shift,
):
    gx, gy, gz, dgx, dgy, dgz = grid_info
    lengths = (gx * dgx, gy * dgy, gz * dgz)
    x, y, z, _, _, _, level, _ = oct_soa_tuple
    for bucket in prange(bucket_root_ids.shape[0]):
        root_id = bucket_root_ids[bucket]
        ix, iy, iz = _idx1d(root_id, gy, gz)
        start = leaf_bucket[bucket]
        stop = leaf_bucket[bucket + 1]
        position = offsets[bucket]
        for axis in range(3):
            if (
                (axis == 0 and ix != 0)
                or (axis == 1 and iy != 0)
                or (axis == 2 and iz != 0)
            ):
                continue
            tx, ty, tz = ix, iy, iz
            if axis == 0:
                tx = gx - 1
            elif axis == 1:
                ty = gy - 1
            else:
                tz = gz - 1
            opposite_id = _idx3d(tx, ty, tz, gy, gz)
            opposite_bucket = _bucket_lookup(
                bucket_root_ids, opposite_id
            )
            if opposite_bucket < 0:
                continue
            opposite_start = leaf_bucket[opposite_bucket]
            opposite_stop = leaf_bucket[opposite_bucket + 1]
            for left in range(start, stop):
                left_node = leaf_linear[left]
                left_half = half_tables[level[left_node]]
                for right in range(opposite_start, opposite_stop):
                    right_node = leaf_linear[right]
                    right_half = half_tables[level[right_node]]
                    right_x = x[right_node]
                    right_y = y[right_node]
                    right_z = z[right_node]
                    if axis == 0:
                        right_x -= lengths[0]
                    elif axis == 1:
                        right_y -= lengths[1]
                    else:
                        right_z -= lengths[2]
                    if _face_touch_boxes(
                        x[left_node],
                        y[left_node],
                        z[left_node],
                        left_half,
                        right_x,
                        right_y,
                        right_z,
                        right_half,
                        tolerance,
                    ):
                        edge_a[position] = np.uint32(
                            coarse_count
                            + np.searchsorted(
                                active_leaf_roots,
                                leaf_component_roots[left],
                            )
                        )
                        edge_b[position] = np.uint32(
                            coarse_count
                            + np.searchsorted(
                                active_leaf_roots,
                                leaf_component_roots[right],
                            )
                        )
                        edge_shift[position, axis] = -1
                        position += 1


def _periodic_leaf_seam_edges(
    grid_info,
    oct_soa_tuple,
    leaf_linear,
    leaf_bucket,
    leaf_root_linear,
    leaf_component_roots,
    active_leaf_roots,
    coarse_count,
    tolerance,
):
    bucket_root_ids = np.asarray(
        leaf_root_linear[leaf_bucket[:-1]], dtype=np.int32
    )
    half_tables = _half_tables(grid_info)
    counts = _count_periodic_leaf_seams(
        grid_info,
        oct_soa_tuple,
        leaf_linear,
        leaf_bucket,
        bucket_root_ids,
        half_tables,
        tolerance,
    )
    offsets, edge_count = _prefix_offsets(counts)
    edge_a = np.empty(edge_count, dtype=np.uint32)
    edge_b = np.empty(edge_count, dtype=np.uint32)
    edge_shift = np.zeros((edge_count, 3), dtype=np.int8)
    _fill_periodic_leaf_seams(
        grid_info,
        oct_soa_tuple,
        leaf_linear,
        leaf_bucket,
        bucket_root_ids,
        leaf_component_roots,
        active_leaf_roots,
        coarse_count,
        half_tables,
        tolerance,
        offsets,
        edge_a,
        edge_b,
        edge_shift,
    )
    return edge_a, edge_b, edge_shift


def _periodic_coarse_seam_edges(
    coarse_void,
    coarse_root,
    active_coarse_roots,
):
    root_a, root_b, shift = _periodic_collect_seam_edges(
        coarse_void, coarse_root.ravel()
    )
    valid = root_a != INVALID
    if not np.any(valid):
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty((0, 3), dtype=np.int8),
        )
    return (
        np.searchsorted(active_coarse_roots, root_a[valid]).astype(
            np.uint32
        ),
        np.searchsorted(active_coarse_roots, root_b[valid]).astype(
            np.uint32
        ),
        shift[valid],
    )


def _nonperiodic_node_access(node_touch, edge_a, edge_b, direction):
    node_count = node_touch.shape[0]
    parent = np.arange(node_count, dtype=np.uint32)
    rank = np.zeros(node_count, dtype=np.uint8)
    _union_edge_list(parent, rank, edge_a, edge_b)
    component_touch = np.zeros(node_count, dtype=np.uint8)
    for node in range(node_count):
        root = _uf_find(parent, np.uint32(node))
        component_touch[root] |= node_touch[node]

    required = {
        "x": TX0 | TX1,
        "y": TY0 | TY1,
        "z": TZ0 | TZ1,
    }
    access = np.zeros(node_count, dtype=np.bool_)
    for node in range(node_count):
        bits = component_touch[_uf_find(parent, np.uint32(node))]
        if direction == "any":
            access[node] = (
                ((bits & (TX0 | TX1)) == (TX0 | TX1))
                or ((bits & (TY0 | TY1)) == (TY0 | TY1))
                or ((bits & (TZ0 | TZ1)) == (TZ0 | TZ1))
            )
        else:
            access[node] = (bits & required[direction]) == required[direction]
    return access


def _periodic_node_access(node_count, edge_a, edge_b, edge_shift, direction):
    if edge_a.size == 0:
        return np.zeros(node_count, dtype=np.bool_)
    winding = _periodic_winding_from_edges(
        node_count, edge_a, edge_b, edge_shift
    )
    masks = {
        "any": _WIND_ANY,
        "x": _WIND_X,
        "y": _WIND_Y,
        "z": _WIND_Z,
    }
    return (winding & masks[direction]) != 0


@njit(parallel=True, cache=True)
def _write_coarse_labels(
    coarse_void,
    coarse_root,
    active_coarse_roots,
    node_access,
    label_mask,
):
    flat_void = coarse_void.ravel()
    flat_root = coarse_root.ravel()
    flat_label = label_mask.ravel()
    for index in prange(flat_void.shape[0]):
        if flat_void[index] == 0:
            flat_label[index] = 0
        else:
            node = np.searchsorted(active_coarse_roots, flat_root[index])
            flat_label[index] = (
                LABEL_MASK_ACC if node_access[node] else LABEL_MASK_TRAP
            )


@njit(parallel=True, cache=True)
def _write_leaf_flags(
    oct_soa_tuple,
    leaf_linear,
    leaf_component_roots,
    active_leaf_roots,
    coarse_count,
    node_access,
):
    _, _, _, _, _, _, _, occ = oct_soa_tuple
    for leaf in prange(leaf_linear.shape[0]):
        node_id = leaf_linear[leaf]
        occ[node_id] &= np.uint8(~(OCC_ACC | OCC_TRAP))
        graph_node = (
            coarse_count
            + np.searchsorted(active_leaf_roots, leaf_component_roots[leaf])
        )
        if node_access[graph_node]:
            occ[node_id] |= OCC_ACC
        else:
            occ[node_id] |= OCC_TRAP


@njit(parallel=True, cache=True)
def _backfill_refined_roots(
    label_mask,
    oct_soa_tuple,
    leaf_linear,
    leaf_root_linear,
    leaf_bucket,
    grid_info,
):
    _, gy, gz, _, _, _ = grid_info
    _, _, _, _, _, _, _, occ = oct_soa_tuple
    for bucket in prange(leaf_bucket.shape[0] - 1):
        start = leaf_bucket[bucket]
        stop = leaf_bucket[bucket + 1]
        root_id = leaf_root_linear[start]
        ix, iy, iz = _idx1d(root_id, gy, gz)
        has_void = False
        accessible = False
        for leaf in range(start, stop):
            node_id = leaf_linear[leaf]
            if (occ[node_id] & OCC_VOID) != 0:
                has_void = True
                if (occ[node_id] & OCC_ACC) != 0:
                    accessible = True
                    break
        if accessible:
            label_mask[ix, iy, iz] = LABEL_MASK_ACC
        elif has_void:
            label_mask[ix, iy, iz] = LABEL_MASK_TRAP


def _concatenate_edges(edge_sets):
    nonempty = [edge_set for edge_set in edge_sets if edge_set[0].size]
    if not nonempty:
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty((0, 3), dtype=np.int8),
        )
    return tuple(np.concatenate(items, axis=0) for items in zip(*nonempty))


def percolation_masks_with_octree_hybrid(
    void,
    grid_mask,
    grid_info,
    oct_soa_tuple,
    connectivity="periodic",
    transport_direction="any",
):
    """Classify an adaptive coarse/leaf graph by direction or winding."""
    if connectivity not in ("legacy", "periodic"):
        raise ValueError("connectivity must be one of: legacy, periodic")
    if transport_direction not in ("any", "x", "y", "z"):
        raise ValueError("transport_direction must be one of: any, x, y, z")

    gx, gy, gz, dgx, dgy, dgz = grid_info
    # Octree centers are stored as float32.  At the far side of a large box,
    # their rounding error is governed by the box length rather than the leaf
    # size; a spacing-only tolerance can therefore sever valid interfaces.
    box_scale = max(gx * dgx, gy * dgy, gz * dgz, 1.0)
    tolerance = max(
        1.0e-6 * max(dgx, dgy, dgz),
        4.0 * np.finfo(np.float32).eps * box_scale,
    )
    octree_mask = (grid_mask & np.uint8(128)) == np.uint8(128)
    coarse_void = np.ascontiguousarray(void, dtype=np.uint8).copy()
    coarse_void[octree_mask] = 0
    coarse_void, coarse_root_flat = _connectivity_build_nonperiodic_roots(
        coarse_void
    )
    coarse_root = coarse_root_flat.reshape(coarse_void.shape)
    active_coarse_roots = np.unique(coarse_root[coarse_void != 0])

    leaf_linear, leaf_root_linear, leaf_bucket = _extract_void_leaves(
        grid_info, oct_soa_tuple
    )
    leaf_component_roots = _build_leaf_components(
        grid_info,
        oct_soa_tuple,
        leaf_linear,
        leaf_root_linear,
        leaf_bucket,
        tolerance,
    )
    active_leaf_roots = np.unique(leaf_component_roots)
    coarse_count = active_coarse_roots.shape[0]
    node_count = coarse_count + active_leaf_roots.shape[0]
    if node_count == 0:
        return np.zeros(void.shape, dtype=np.int8), None

    interface_edges = _leaf_coarse_edges(
        connectivity == "periodic",
        grid_info,
        coarse_void,
        coarse_root,
        active_coarse_roots,
        oct_soa_tuple,
        leaf_linear,
        leaf_root_linear,
        leaf_component_roots,
        active_leaf_roots,
        tolerance,
    )

    if connectivity == "periodic":
        coarse_seams = _periodic_coarse_seam_edges(
            coarse_void, coarse_root, active_coarse_roots
        )
        leaf_seams = _periodic_leaf_seam_edges(
            grid_info,
            oct_soa_tuple,
            leaf_linear,
            leaf_bucket,
            leaf_root_linear,
            leaf_component_roots,
            active_leaf_roots,
            coarse_count,
            tolerance,
        )
        edge_a, edge_b, edge_shift = _concatenate_edges(
            (interface_edges, coarse_seams, leaf_seams)
        )
        node_access = _periodic_node_access(
            node_count,
            edge_a,
            edge_b,
            edge_shift,
            transport_direction,
        )
    else:
        coarse_touch = _coarse_boundary_touch(
            coarse_void, coarse_root, active_coarse_roots
        )
        leaf_touch = _leaf_boundary_touch(
            grid_info,
            oct_soa_tuple,
            leaf_linear,
            leaf_component_roots,
            active_leaf_roots,
            tolerance,
        )
        node_touch = np.concatenate((coarse_touch, leaf_touch))
        edge_a, edge_b, _ = interface_edges
        node_access = _nonperiodic_node_access(
            node_touch, edge_a, edge_b, transport_direction
        )

    label_mask = np.zeros(void.shape, dtype=np.int8)
    if coarse_count:
        _write_coarse_labels(
            coarse_void,
            coarse_root,
            active_coarse_roots,
            node_access,
            label_mask,
        )
    if leaf_linear.size:
        _write_leaf_flags(
            oct_soa_tuple,
            leaf_linear,
            leaf_component_roots,
            active_leaf_roots,
            coarse_count,
            node_access,
        )
        _backfill_refined_roots(
            label_mask,
            oct_soa_tuple,
            leaf_linear,
            leaf_root_linear,
            leaf_bucket,
            grid_info,
        )
    return label_mask, None
