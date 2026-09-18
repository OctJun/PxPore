import numpy as np
from numba import njit, prange

INVALID = np.uint32(0xFFFFFFFF)

TX0 = np.uint8(1 << 0)
TX1 = np.uint8(1 << 1)
TY0 = np.uint8(1 << 2)
TY1 = np.uint8(1 << 3)
TZ0 = np.uint8(1 << 4)
TZ1 = np.uint8(1 << 5)


# 固定分块厚度，保证不同线程环境及 JIT 缓存使用相同的接缝位置。
SLAB_Z = 32

LABEL_MASK_ACC = np.int8(2)
LABEL_MASK_TRAP = np.int8(1)


@njit(inline='always', cache=True)
def _idx3d(x, y, z, gy, gz):
    return (x * gy + y) * gz + z


@njit(cache=True)
def _uf_find(parent, a):
    root = a
    while parent[root] != root:
        root = parent[root]

    # path compression
    cur = a
    while parent[cur] != cur:
        p = parent[cur]  # 读取当前节点的父节点
        parent[cur] = root  # 把当前节点直接连接到根节点
        cur = p  # 继续向上查找，直到到达根节点

    return root


@njit(cache=True)
def _uf_union(parent, rank, a, b):
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)

    if ra == rb:
        return ra

    # union by rank
    if rank[ra] < rank[rb]:
        tmp = ra
        ra = rb
        rb = tmp

    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += np.uint8(1)

    return ra


@njit(parallel=True, cache=True)
def _init_and_local_union_zslab(void_mask, parent, rank):
    """
    沿 z 方向分 slab，并行执行：
    1. 初始化 parent/rank
    2. slab 内局部 union
       - z 方向只在 slab 内 union 到 z+1
       - x/y 正向邻居正常 union
    """
    gx, gy, gz = void_mask.shape
    nslabs = (gz + SLAB_Z - 1) // SLAB_Z

    for s in prange(nslabs):
        zs = s * SLAB_Z
        ze = min(zs + SLAB_Z, gz)

        # init
        for x in range(gx):
            for y in range(gy):
                for z in range(zs, ze):
                    idx = _idx3d(x, y, z, gy, gz)
                    if void_mask[x, y, z] == 1:
                        parent[idx] = np.uint32(idx)
                        rank[idx] = np.uint8(0)
                    else:
                        parent[idx] = INVALID
                        rank[idx] = np.uint8(0)

        # slab 内局部 union，只看正向邻居避免重复
        for x in range(gx):
            for y in range(gy):
                for z in range(zs, ze):
                    if void_mask[x, y, z] != 1:
                        continue

                    a = np.uint32(_idx3d(x, y, z, gy, gz))

                    # +x
                    if x + 1 < gx and void_mask[x + 1, y, z] == 1:
                        b = np.uint32(_idx3d(x + 1, y, z, gy, gz))
                        _uf_union(parent, rank, a, b)

                    # +y
                    if y + 1 < gy and void_mask[x, y + 1, z] == 1:
                        b = np.uint32(_idx3d(x, y + 1, z, gy, gz))
                        _uf_union(parent, rank, a, b)

                    # +z 仅 slab 内
                    if z + 1 < ze and void_mask[x, y, z + 1] == 1:
                        b = np.uint32(_idx3d(x, y, z + 1, gy, gz))
                        _uf_union(parent, rank, a, b)


@njit(cache=True)
def _seam_union_z(void_mask, parent, rank):
    """
    串行合并相邻 z-slab 的接缝:
      z = ze-1  与  z = ze
    """
    gx, gy, gz = void_mask.shape
    nslabs = (gz + SLAB_Z - 1) // SLAB_Z

    for s in range(nslabs - 1):
        ze_left = min((s + 1) * SLAB_Z, gz)
        z0 = ze_left - 1
        z1 = ze_left

        if z1 >= gz:
            continue

        for x in range(gx):
            for y in range(gy):
                if void_mask[x, y, z0] == 1 and void_mask[x, y, z1] == 1:
                    a = np.uint32(_idx3d(x, y, z0, gy, gz))
                    b = np.uint32(_idx3d(x, y, z1, gy, gz))
                    _uf_union(parent, rank, a, b)


@njit(parallel=True, cache=True)
def _flatten_roots(void_mask, parent, root):
    """
    所有 union 完成后，统一把每个 void 点的最终 root 写出来
    之后不再重复 find
    """
    gx, gy, gz = void_mask.shape
    nslabs = (gz + SLAB_Z - 1) // SLAB_Z

    for s in prange(nslabs):
        zs = s * SLAB_Z
        ze = min(zs + SLAB_Z, gz)

        for x in range(gx):
            for y in range(gy):
                for z in range(zs, ze):
                    idx = _idx3d(x, y, z, gy, gz)
                    if void_mask[x, y, z] == 1:
                        root[idx] = _uf_find(parent, np.uint32(idx))
                    else:
                        root[idx] = INVALID


@njit(cache=True)
def _mark_boundary_touch_from_root(void_mask, root, root_touch):
    gx, gy, gz = void_mask.shape

    # x = 0
    x = 0
    for y in range(gy):
        for z in range(gz):
            if void_mask[x, y, z] == 1:
                idx = _idx3d(x, y, z, gy, gz)
                r = root[idx]
                root_touch[r] |= TX0

    # x = gx-1
    x = gx - 1
    for y in range(gy):
        for z in range(gz):
            if void_mask[x, y, z] == 1:
                idx = _idx3d(x, y, z, gy, gz)
                r = root[idx]
                root_touch[r] |= TX1

    # y = 0
    y = 0
    for x in range(gx):
        for z in range(gz):
            if void_mask[x, y, z] == 1:
                idx = _idx3d(x, y, z, gy, gz)
                r = root[idx]
                root_touch[r] |= TY0

    # y = gy-1
    y = gy - 1
    for x in range(gx):
        for z in range(gz):
            if void_mask[x, y, z] == 1:
                idx = _idx3d(x, y, z, gy, gz)
                r = root[idx]
                root_touch[r] |= TY1

    # z = 0
    z = 0
    for x in range(gx):
        for y in range(gy):
            if void_mask[x, y, z] == 1:
                idx = _idx3d(x, y, z, gy, gz)
                r = root[idx]
                root_touch[r] |= TZ0

    # z = gz-1
    z = gz - 1
    for x in range(gx):
        for y in range(gy):
            if void_mask[x, y, z] == 1:
                idx = _idx3d(x, y, z, gy, gz)
                r = root[idx]
                root_touch[r] |= TZ1


@njit(inline='always', cache=True)
def _is_accessible(bits):
    return ((bits & TX0) != 0 and (bits & TX1) != 0) or \
           ((bits & TY0) != 0 and (bits & TY1) != 0) or \
           ((bits & TZ0) != 0 and (bits & TZ1) != 0)

    # return bits != 0

@njit(parallel=True, cache=True)
def _write_label_mask_from_root(void_mask, root, root_touch, label_mask):
    gx, gy, gz = void_mask.shape
    nslabs = (gz + SLAB_Z - 1) // SLAB_Z

    for s in prange(nslabs):
        zs = s * SLAB_Z
        ze = min(zs + SLAB_Z, gz)

        for x in range(gx):
            for y in range(gy):
                for z in range(zs, ze):
                    if void_mask[x, y, z] != 1:
                        label_mask[x, y, z] = 0
                    else:
                        idx = _idx3d(x, y, z, gy, gz)
                        bits = root_touch[root[idx]]
                        label_mask[x, y, z] = LABEL_MASK_ACC if _is_accessible(bits) else LABEL_MASK_TRAP


def percolation_masks(void_mask):
    """
    3D non-periodic, 6-connectivity only

    Parameters
    ----------
    void_mask : ndarray, shape (gx, gy, gz), dtype uint8/bool
        1 = void
        0 = solid

    Returns
    -------
    label_mask : int8, shape (gx, gy, gz)
        0 = solid
        1 = trap
        2 = accessible
    parent_id : None
    """
    if void_mask.ndim != 3:
        raise ValueError("void_mask must be a 3D array")

    void_mask = np.ascontiguousarray(void_mask, dtype=np.uint8)
    gx, gy, gz = void_mask.shape
    total = gx * gy * gz

    if total >= 4294967295:
        raise ValueError("total voxel count too large for uint32 parent/root")

    parent = np.empty(total, dtype=np.uint32)
    rank = np.empty(total, dtype=np.uint8)
    root = np.empty(total, dtype=np.uint32)
    root_touch = np.zeros(total, dtype=np.uint8)
    label_mask = np.empty((gx, gy, gz), dtype=np.int8)

    # 1) z-slab 内并行初始化 + 局部 union
    _init_and_local_union_zslab(void_mask, parent, rank)

    # 2) z-slab 接缝串行 union
    _seam_union_z(void_mask, parent, rank)

    # 3) 一次性把每个 void 点的最终 root 压平
    _flatten_roots(void_mask, parent, root)

    # 4) 根据 root 统计 6 个外边界面的触边信息
    _mark_boundary_touch_from_root(void_mask, root, root_touch)

    # 5) 直接写最终 label
    _write_label_mask_from_root(void_mask, root, root_touch, label_mask)

    return label_mask, root


_PERIODIC_WIND_X = np.uint8(1 << 0)
_PERIODIC_WIND_Y = np.uint8(1 << 1)
_PERIODIC_WIND_Z = np.uint8(1 << 2)
_PERIODIC_WIND_ANY = _PERIODIC_WIND_X | _PERIODIC_WIND_Y | _PERIODIC_WIND_Z


@njit(parallel=True, cache=True)
def _periodic_collect_seam_edges(void_mask, root):
    """
    并行收集 x/y/z 三组周期面之间的连边。
    无效槽位使用 INVALID 标记，避免并行循环中共享计数器。
    """
    gx, gy, gz = void_mask.shape
    nx_edges = gy * gz
    ny_edges = gx * gz
    nz_edges = gx * gy
    max_edges = nx_edges + ny_edges + nz_edges
    edge_a = np.full(max_edges, INVALID, dtype=np.uint32)
    edge_b = np.full(max_edges, INVALID, dtype=np.uint32)
    edge_shift = np.zeros((max_edges, 3), dtype=np.int8)

    for eid in prange(max_edges):
        if eid < nx_edges:
            y = eid // gz
            z = eid - y * gz
            if void_mask[0, y, z] == 1 and void_mask[gx - 1, y, z] == 1:
                edge_a[eid] = root[_idx3d(0, y, z, gy, gz)]
                edge_b[eid] = root[_idx3d(gx - 1, y, z, gy, gz)]
                edge_shift[eid, 0] = -1
        elif eid < nx_edges + ny_edges:
            local = eid - nx_edges
            x = local // gz
            z = local - x * gz
            if void_mask[x, 0, z] == 1 and void_mask[x, gy - 1, z] == 1:
                edge_a[eid] = root[_idx3d(x, 0, z, gy, gz)]
                edge_b[eid] = root[_idx3d(x, gy - 1, z, gy, gz)]
                edge_shift[eid, 1] = -1
        else:
            local = eid - nx_edges - ny_edges
            x = local // gy
            y = local - x * gy
            if void_mask[x, y, 0] == 1 and void_mask[x, y, gz - 1] == 1:
                edge_a[eid] = root[_idx3d(x, y, 0, gy, gz)]
                edge_b[eid] = root[_idx3d(x, y, gz - 1, gy, gz)]
                edge_shift[eid, 2] = -1

    return edge_a, edge_b, edge_shift


@njit(inline='always', cache=True)
def _periodic_weighted_find(parent, displacement, node):
    """返回周期分量根节点以及 node 到根节点的晶胞位移。"""
    root = node
    dx = np.int32(0)
    dy = np.int32(0)
    dz = np.int32(0)
    while parent[root] != root:
        dx += displacement[root, 0]
        dy += displacement[root, 1]
        dz += displacement[root, 2]
        root = parent[root]
    return root, dx, dy, dz


@njit(cache=True)
def _periodic_winding_from_edges(n_nodes, edge_a, edge_b, edge_shift):
    """
    使用带位移的并查集判断周期绕行。
    并查集合并存在父节点写依赖，因此这里保持串行。
    """
    parent = np.arange(n_nodes, dtype=np.uint32)
    rank = np.zeros(n_nodes, dtype=np.uint8)
    displacement = np.zeros((n_nodes, 3), dtype=np.int32)
    winding = np.zeros(n_nodes, dtype=np.uint8)

    for i in range(edge_a.shape[0]):
        a = edge_a[i]
        b = edge_b[i]
        ra, ax, ay, az = _periodic_weighted_find(
            parent, displacement, a
        )
        rb, bx, by, bz = _periodic_weighted_find(
            parent, displacement, b
        )
        sx = np.int32(edge_shift[i, 0])
        sy = np.int32(edge_shift[i, 1])
        sz = np.int32(edge_shift[i, 2])

        # 同一分量内出现非零位移残差，说明对应方向存在周期绕行。
        if ra == rb:
            if bx - ax - sx != 0:
                winding[ra] |= _PERIODIC_WIND_X
            if by - ay - sy != 0:
                winding[ra] |= _PERIODIC_WIND_Y
            if bz - az - sz != 0:
                winding[ra] |= _PERIODIC_WIND_Z
            continue

        merged_winding = winding[ra] | winding[rb]
        if rank[ra] < rank[rb]:
            parent[ra] = rb
            displacement[ra, 0] = bx - ax - sx
            displacement[ra, 1] = by - ay - sy
            displacement[ra, 2] = bz - az - sz
            winding[rb] = merged_winding
        else:
            parent[rb] = ra
            displacement[rb, 0] = sx + ax - bx
            displacement[rb, 1] = sy + ay - by
            displacement[rb, 2] = sz + az - bz
            winding[ra] = merged_winding
            if rank[ra] == rank[rb]:
                rank[ra] += np.uint8(1)

    node_winding = np.zeros(n_nodes, dtype=np.uint8)
    for i in range(n_nodes):
        root, _, _, _ = _periodic_weighted_find(
            parent, displacement, np.uint32(i)
        )
        node_winding[i] = winding[root]
    return node_winding


@njit(parallel=True, cache=True)
def _periodic_write_label_mask(
    void_mask, root, root_winding, direction_mask, label_mask
):
    """根据周期绕行方向并行写入 accessible/trap 标签。"""
    gx, gy, gz = void_mask.shape
    nslabs = (gz + SLAB_Z - 1) // SLAB_Z

    for s in prange(nslabs):
        zs = s * SLAB_Z
        ze = min(zs + SLAB_Z, gz)
        for x in range(gx):
            for y in range(gy):
                for z in range(zs, ze):
                    if void_mask[x, y, z] != 1:
                        label_mask[x, y, z] = 0
                    else:
                        idx = _idx3d(x, y, z, gy, gz)
                        bits = root_winding[root[idx]]
                        label_mask[x, y, z] = (
                            LABEL_MASK_ACC
                            if (bits & direction_mask) != 0
                            else LABEL_MASK_TRAP
                        )


@njit(parallel=True, cache=True)
def _directional_write_label_mask(
    void_mask, root, root_touch, required_mask, label_mask
):
    """
    按指定的非周期边界方向并行写入 accessible/trap 标签。
    required_mask 同时包含一个方向的正负边界位。
    """
    gx, gy, gz = void_mask.shape
    nslabs = (gz + SLAB_Z - 1) // SLAB_Z

    for s in prange(nslabs):
        zs = s * SLAB_Z
        ze = min(zs + SLAB_Z, gz)
        for x in range(gx):
            for y in range(gy):
                for z in range(zs, ze):
                    if void_mask[x, y, z] != 1:
                        label_mask[x, y, z] = 0
                    else:
                        idx = _idx3d(x, y, z, gy, gz)
                        bits = root_touch[root[idx]]
                        label_mask[x, y, z] = (
                            LABEL_MASK_ACC
                            if (bits & required_mask) == required_mask
                            else LABEL_MASK_TRAP
                        )


def _connectivity_build_nonperiodic_roots(void_mask):
    """构建供新增方向/周期功能使用的非周期连通分量。"""
    if void_mask.ndim != 3:
        raise ValueError("void_mask must be a 3D array")

    void_mask = np.ascontiguousarray(void_mask, dtype=np.uint8)
    gx, gy, gz = void_mask.shape
    total = gx * gy * gz
    if total >= 4294967295:
        raise ValueError("total voxel count too large for uint32 parent/root")

    parent = np.empty(total, dtype=np.uint32)
    rank = np.empty(total, dtype=np.uint8)
    root = np.empty(total, dtype=np.uint32)

    _init_and_local_union_zslab(void_mask, parent, rank)
    _seam_union_z(void_mask, parent, rank)
    _flatten_roots(void_mask, parent, root)
    return void_mask, root


def percolation_masks_directional(void_mask, direction="any"):
    """
    非周期、六邻域、指定方向的连通性判断。
    direction 可选 any/x/y/z；any 保持原函数行为。
    """
    if direction == "any":
        return percolation_masks(void_mask)

    required_masks = {
        "x": TX0 | TX1,
        "y": TY0 | TY1,
        "z": TZ0 | TZ1,
    }
    try:
        required_mask = required_masks[direction]
    except KeyError as exc:
        raise ValueError("direction must be one of: any, x, y, z") from exc

    void_mask, root = _connectivity_build_nonperiodic_roots(void_mask)
    root_touch = np.zeros(void_mask.size, dtype=np.uint8)
    label_mask = np.empty(void_mask.shape, dtype=np.int8)
    _mark_boundary_touch_from_root(void_mask, root, root_touch)
    _directional_write_label_mask(
        void_mask, root, root_touch, required_mask, label_mask
    )
    return label_mask, root


def percolation_masks_periodic(void_mask, direction="any"):
    """
    周期、六邻域、指定方向的连通性判断。
    只有产生非零晶胞位移的周期绕行才判定为 accessible；
    单次跨周期面的有限团簇仍判定为 trap。
    """
    direction_masks = {
        "any": _PERIODIC_WIND_ANY,
        "x": _PERIODIC_WIND_X,
        "y": _PERIODIC_WIND_Y,
        "z": _PERIODIC_WIND_Z,
    }
    try:
        direction_mask = direction_masks[direction]
    except KeyError as exc:
        raise ValueError("direction must be one of: any, x, y, z") from exc

    void_mask, root = _connectivity_build_nonperiodic_roots(void_mask)
    edge_root_a, edge_root_b, edge_shift = _periodic_collect_seam_edges(
        void_mask, root
    )
    label_mask = np.empty(void_mask.shape, dtype=np.int8)

    valid_edges = edge_root_a != INVALID
    if not np.any(valid_edges):
        label_mask.fill(LABEL_MASK_TRAP)
        label_mask[void_mask == 0] = 0
        return label_mask, root

    edge_root_a = edge_root_a[valid_edges]
    edge_root_b = edge_root_b[valid_edges]
    edge_shift = edge_shift[valid_edges]
    active_roots = np.unique(np.concatenate((edge_root_a, edge_root_b)))
    edge_a = np.searchsorted(active_roots, edge_root_a).astype(np.uint32)
    edge_b = np.searchsorted(active_roots, edge_root_b).astype(np.uint32)
    node_winding = _periodic_winding_from_edges(
        active_roots.size, edge_a, edge_b, edge_shift
    )

    root_winding = np.zeros(void_mask.size, dtype=np.uint8)
    root_winding[active_roots] = node_winding
    _periodic_write_label_mask(
        void_mask, root, root_winding, direction_mask, label_mask
    )
    return label_mask, root
