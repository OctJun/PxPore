import numpy as np
from numba import njit, prange,get_num_threads

INVALID = np.uint32(0xFFFFFFFF)

TX0 = np.uint8(1 << 0)
TX1 = np.uint8(1 << 1)
TY0 = np.uint8(1 << 2)
TY1 = np.uint8(1 << 3)
TZ0 = np.uint8(1 << 4)
TZ1 = np.uint8(1 << 5)


SLAB_Z = max(1, get_num_threads() // 2)

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
