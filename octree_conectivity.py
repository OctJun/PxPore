import numpy as np
import numba
from numba import njit, prange


from .octree import OCC_VOID,OCC_SOLID,OCC_ACC,OCC_TRAP
from .connectivity_multicore import percolation_masks,LABEL_MASK_ACC,LABEL_MASK_TRAP

OCC_SOLID = np.uint8(0)
OCC_VOID = np.uint8(1)
OCC_SPLIT = np.uint8(2)

N_THREADS=numba.get_num_threads()

@njit(inline='always', cache=True)
def _idx3d(ix, iy, iz, gy, gz):
    return (ix * gy + iy) * gz + iz

@njit(inline='always', cache=True)
def _idx1d(idx, gy, gz):
    ix = idx // (gy * gz)
    rem = idx - ix * gy * gz
    iy = rem // gz
    iz = rem - iy * gz
    return ix, iy, iz

@njit(cache=True)
def _uf_find(parent, a):
    root = a
    while parent[root] != root:
        root = parent[root]
    # 路径压缩
    cur = a
    while parent[cur] != cur:
        p = parent[cur]
        parent[cur] = root
        cur = p
    return root

@njit(cache=True)
def _uf_find_ro(parent, a):
    root = a
    while parent[root] != root:
        root = parent[root]
    return root



@njit(cache=True)
def _uf_union(parent, rank, a, b):
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return ra

    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += np.uint8(1)
    return ra


@njit(cache=True)
def _interval_overlap_pos(a0, a1, b0, b1, tol):
    left = a0 if a0 > b0 else b0
    right = a1 if a1 < b1 else b1
    return (right - left) > tol


@njit(cache=True)
def _face_touch_boxes(cx1, cy1, cz1, h1,
                      cx2, cy2, cz2, h2, tol):
    hx1,hy1,hz1 = h1
    hx2,hy2,hz2 = h2

    x1min = cx1 - hx1
    x1max = cx1 + hx1
    y1min = cy1 - hy1
    y1max = cy1 + hy1
    z1min = cz1 - hz1
    z1max = cz1 + hz1

    x2min = cx2 - hx2
    x2max = cx2 + hx2
    y2min = cy2 - hy2
    y2max = cy2 + hy2
    z2min = cz2 - hz2
    z2max = cz2 + hz2

    # x-face
    if abs(x1max - x2min) <= tol or abs(x2max - x1min) <= tol:
        if _interval_overlap_pos(y1min, y1max, y2min, y2max, tol) and \
           _interval_overlap_pos(z1min, z1max, z2min, z2max, tol):
            return True

    # y-face
    if abs(y1max - y2min) <= tol or abs(y2max - y1min) <= tol:
        if _interval_overlap_pos(x1min, x1max, x2min, x2max, tol) and \
           _interval_overlap_pos(z1min, z1max, z2min, z2max, tol):
            return True

    # z-face
    if abs(z1max - z2min) <= tol or abs(z2max - z1min) <= tol:
        if _interval_overlap_pos(x1min, x1max, x2min, x2max, tol) and \
           _interval_overlap_pos(y1min, y1max, y2min, y2max, tol):
            return True

    return False


@njit(parallel=True, cache=True)
def _extract_void_leaves(grid_info, oct_soa_tuple):
    gx, gy, gz, dgx, dgy, dgz = grid_info
    x, y, z, d, parent, child, level, occ = oct_soa_tuple

    leaf_mask = (child == -1)
    leaf_nodes = np.where(leaf_mask)[0]
    void_leaf_nodes = leaf_nodes[occ[leaf_nodes] == OCC_VOID]
    n_leaf = void_leaf_nodes.shape[0]

    leaf_root_linear = np.empty(n_leaf, dtype=np.int32)
    leaf_linear = np.empty(n_leaf, dtype=np.int32)

    for i in prange(n_leaf):
        cur = void_leaf_nodes[i]
        while parent[cur] != -1:
            cur = parent[cur]

        ix = int(np.floor(x[cur] / dgx)) % gx
        iy = int(np.floor(y[cur] / dgy)) % gy
        iz = int(np.floor(z[cur] / dgz)) % gz

        leaf_linear[i] = void_leaf_nodes[i]
        leaf_root_linear[i] = _idx3d(ix, iy, iz, gy, gz)

    # 依赖当前建树顺序：同一 root 的叶子是连续分组的
    leaf_bucket = np.empty(n_leaf + 1, dtype=np.int32)

    if n_leaf == 0:
        leaf_bucket[0] = 0
        return leaf_linear, leaf_root_linear, leaf_bucket[:1]

    k = 0
    leaf_bucket[k] = 0
    k += 1
    prev_rid = leaf_root_linear[0]

    for i in range(1, n_leaf):
        rid = leaf_root_linear[i]
        if rid != prev_rid:
            leaf_bucket[k] = i
            k += 1
            prev_rid = rid

    leaf_bucket[k] = n_leaf
    return leaf_linear, leaf_root_linear, leaf_bucket[:k + 1]
        
@njit(parallel=True, cache=True)
def _union_all_leaf_components(
    grid_info, grid_mask, oct_soa_tuple,
    leaf_linear, leaf_root_linear, leaf_bucket,
    parent_leaf, rank_leaf, tol
):
    gx, gy, gz, dgx, dgy, dgz = grid_info
    x, y, z, d, parent, child, level, occ = oct_soa_tuple

    half_tables = np.empty((16,3),dtype=np.float64)
    half_tables[:, 0] = 0.5 * dgx / (2.0 ** np.arange(16, dtype=np.float32))
    half_tables[:, 1] = 0.5 * dgy / (2.0 ** np.arange(16, dtype=np.float32))
    half_tables[:, 2] = 0.5 * dgz / (2.0 ** np.arange(16, dtype=np.float32))

    n_root = gx * gy * gz
    root_to_bucket = -np.ones(n_root, dtype=np.int32)
    for k in range(leaf_bucket.shape[0] - 1):
        s = leaf_bucket[k]
        rid = leaf_root_linear[s]
        root_to_bucket[rid] = k

    n_leaf = leaf_linear.shape[0]

    # 每个线程分配 n_leaf * 12 个 uint32 空间（每个边对占 2 个）
    buf_capacity = (n_leaf * 6 * 2 // N_THREADS ) * 2
    thread_buffers = np.zeros(shape=(N_THREADS,buf_capacity),dtype=np.uint32)
    write_pos = np.zeros(N_THREADS, dtype=np.int32)

    for ix in prange(gx):
        tid = numba.get_thread_id()
        buf = thread_buffers[tid]
        pos = write_pos[tid]  # 当前写位置
        for iy in range(gy):
            for iz in range(gz):
                rid_a = _idx3d(ix, iy, iz, gy, gz)
                if (grid_mask[ix, iy, iz] & 128) != 128:
                    continue
                bk_a = root_to_bucket[rid_a]
                if bk_a == -1:
                    continue
                sa = leaf_bucket[bk_a]
                ea = leaf_bucket[bk_a + 1]
                if ea <= sa:
                    continue

                # 内部合并
                for p in range(sa, ea):
                    nid_p = leaf_linear[p]
                    hp = half_tables[level[nid_p]]
                    for q in range(p + 1, ea):
                        nid_q = leaf_linear[q]
                        hq = half_tables[level[nid_q]]
                        if _face_touch_boxes(
                            x[nid_p], y[nid_p], z[nid_p], hp,
                            x[nid_q], y[nid_q], z[nid_q], hq, tol
                        ):
                            # 写入边对，并检查越界（实际很少发生）
                            if pos + 2 > buf_capacity:
                                # 理论上不会进来，但安全起见可以扩大缓冲区或报错
                                continue
                            buf[pos] = np.uint32(p)
                            buf[pos + 1] = np.uint32(q)
                            pos += 2

                # x+/y+/z+ 邻居
                targets = np.array(((ix + 1, iy, iz), (ix, iy + 1, iz), (ix, iy, iz + 1)), dtype=np.int32)
                for t in targets:
                    tx, ty, tz = t
                    if tx >= gx or ty >= gy or tz >= gz:
                        continue
                    rid_b = _idx3d(tx, ty, tz, gy, gz)
                    if (grid_mask[tx, ty, tz] & 128) == 128:
                        bk_b = root_to_bucket[rid_b]
                        if bk_b != -1:
                            sb = leaf_bucket[bk_b]
                            eb = leaf_bucket[bk_b + 1]
                            for p in range(sa, ea):
                                nid_p = leaf_linear[p]
                                hp = half_tables[level[nid_p]]
                                for q in range(sb, eb):
                                    nid_q = leaf_linear[q]
                                    hq = half_tables[level[nid_q]]
                                    if _face_touch_boxes(
                                        x[nid_p], y[nid_p], z[nid_p], hp,
                                        x[nid_q], y[nid_q], z[nid_q], hq, tol
                                    ):
                                        if pos + 2 > buf_capacity:
                                            continue
                                        buf[pos] = np.uint32(p)
                                        buf[pos + 1] = np.uint32(q)
                                        pos += 2


        # 更新该线程的写位置
        write_pos[tid] = pos

    # 汇总所有线程的有效边对，执行并查集合并
    for tid in range(N_THREADS):
        buf = thread_buffers[tid]
        count = write_pos[tid]
        for i in range(0, count, 2):
            p = buf[i]
            q = buf[i + 1]
            _uf_union(parent_leaf, rank_leaf, p, q)


@njit(parallel=True,cache=True)
def _mark_leaf_flag_from_coarse(label_mask_coarse, grid_mask, grid_info,
                                oct_soa_tuple, leaf_linear, leaf_root_linear,
                                parent_leaf):
    """
    不做 leaf-face 几何判面。
    只要某个 leaf component 覆盖到的 root voxel 周围存在
    非模糊且 coarse label > 0 的邻居，就把该 component 全部标为 ACC。
    否则标为 TRAP。
    """
    gx, gy, gz, dgx, dgy, dgz = grid_info
    x, y, z, d, parent, child, level, occ = oct_soa_tuple

    n_leaf = leaf_linear.shape[0]
    comp_acc = np.zeros(n_leaf, dtype=np.uint8)

    # 先清 ACC/TRAP 位
    for i in prange(n_leaf):
        nid = leaf_linear[i]
        occ[nid] &= np.uint8(~(OCC_ACC | OCC_TRAP))

    # 第一遍：按 component 收集是否接触到 coarse accessible 邻居
    for i in prange(n_leaf):
        nid = leaf_linear[i]
        if (occ[nid] & OCC_VOID) != OCC_VOID:
            continue

        root = _uf_find_ro(parent_leaf, np.uint32(i))
        if comp_acc[root] == 1:
            continue

        rid = leaf_root_linear[i]
        ix, iy, iz = _idx1d(rid, gy, gz)
        if ix == 0 or ix == gx - 1 or iy == 0 or iy == gy - 1 or iz == 0 or iz == gz - 1:
            comp_acc[root] = 1
            continue
        # -x
        if ix > 0:
            if (
                label_mask_coarse[ix - 1, iy, iz] == LABEL_MASK_ACC):
                comp_acc[root] = 1
                continue

        # +x
        if ix + 1 < gx:
            if (
                label_mask_coarse[ix + 1, iy, iz] == LABEL_MASK_ACC):
                comp_acc[root] = 1
                continue

        # -y
        if iy > 0:
            if (
                label_mask_coarse[ix, iy - 1, iz] == LABEL_MASK_ACC):
                comp_acc[root] = 1
                continue

        # +y
        if iy + 1 < gy:
            if (
                label_mask_coarse[ix, iy + 1, iz] == LABEL_MASK_ACC):
                comp_acc[root] = 1
                continue

        # -z
        if iz > 0:
            if (
                label_mask_coarse[ix, iy, iz - 1] == LABEL_MASK_ACC):
                comp_acc[root] = 1
                continue

        # +z
        if iz + 1 < gz:
            if (
                label_mask_coarse[ix, iy, iz + 1] == LABEL_MASK_ACC):
                comp_acc[root] = 1
                continue
        # # -x
        # if ix > 0:
        #     if ((grid_mask[ix - 1, iy, iz] & 128) == 0 and
        #         label_mask_coarse[ix - 1, iy, iz] == LABEL_MASK_ACC):
        #         comp_acc[root] = 1

        # # +x
        # if ix + 1 < gx:
        #     if ((grid_mask[ix + 1, iy, iz] & 128) == 0 and
        #         label_mask_coarse[ix + 1, iy, iz] == LABEL_MASK_ACC):
        #         comp_acc[root] = 1

        # # -y
        # if iy > 0:
        #     if ((grid_mask[ix, iy - 1, iz] & 128) == 0 and
        #         label_mask_coarse[ix, iy - 1, iz] == LABEL_MASK_ACC):
        #         comp_acc[root] = 1

        # # +y
        # if iy + 1 < gy:
        #     if ((grid_mask[ix, iy + 1, iz] & 128) == 0 and
        #         label_mask_coarse[ix, iy + 1, iz] == LABEL_MASK_ACC):
        #         comp_acc[root] = 1

        # # -z
        # if iz > 0:
        #     if ((grid_mask[ix, iy, iz - 1] & 128) == 0 and
        #         label_mask_coarse[ix, iy, iz - 1] == LABEL_MASK_ACC):
        #         comp_acc[root] = 1

        # # +z
        # if iz + 1 < gz:
        #     if ((grid_mask[ix, iy, iz + 1] & 128) == 0 and
        #         label_mask_coarse[ix, iy, iz + 1] == LABEL_MASK_ACC):
        #         comp_acc[root] = 1



    # 第二遍：整 component 写回 occ
    for i in prange(n_leaf):
        nid = leaf_linear[i]
        if (occ[nid] & OCC_VOID) != OCC_VOID:
            continue

        root = _uf_find_ro(parent_leaf, np.uint32(i))
        if comp_acc[root]:
            occ[nid] |= OCC_ACC
        else:
            occ[nid] |= OCC_TRAP


@njit(cache=True)
def _backfill_grid_from_leaf_components(label_mask_final, leaf_linear, leaf_root_linear,
                                        oct_soa_tuple, grid_info):
    gx, gy, gz, dgx, dgy, dgz = grid_info
    x, y, z, d, parent, child, level, occ = oct_soa_tuple

    n_leaf = leaf_linear.shape[0]
    for i in range(n_leaf):
        nid = leaf_linear[i]
        if (occ[nid] & OCC_VOID) == 0:
            continue

        rid = leaf_root_linear[i]
        ix, iy, iz = _idx1d(rid, gy, gz)

        # 如果细尺度判定该叶子属于 ACC，则强制将粗网格体素设为 ACC (2)
        if (occ[nid] & OCC_ACC) != 0:
            label_mask_final[ix, iy, iz] = LABEL_MASK_ACC
        # 否则如果是 TRAP，且当前粗网格尚未标记为 ACC，则设为 TRAP (1)
        elif (occ[nid] & OCC_TRAP) != 0:
            if label_mask_final[ix, iy, iz] != LABEL_MASK_ACC:
                label_mask_final[ix, iy, iz] = LABEL_MASK_TRAP

    return label_mask_final


def percolation_masks_with_octree(void, grid_mask, grid_info, oct_soa_tuple):
    """
    1) 先原样跑 coarse percolation
    2) 抽出全部 void leaf，直接建立 leaf 连通分量
    3) 对每个 leaf component 判断附近是否有 coarse accessible
    4) 根据 subgrid component 回填 coarse grid
    """
    label_mask_coarse, _ = percolation_masks(void)
    label_mask_final = label_mask_coarse.copy()

    gx, gy, gz, dgx, dgy, dgz = grid_info
    tol = 1e-6 * max(dgx, dgy, dgz)   # 全局容差
    
    x, y, z, d, parent, child, level, occ = oct_soa_tuple

    leaf_linear,leaf_root_linear,leaf_bucket = _extract_void_leaves(grid_info, oct_soa_tuple)  # 返回 leaf_root_linear

    n_leaf = leaf_linear.shape[0]
    parent_leaf = np.arange(n_leaf, dtype=np.uint32)
    rank_leaf = np.zeros(n_leaf, dtype=np.uint8)
    # print("debug1")
    _union_all_leaf_components(grid_info,grid_mask,oct_soa_tuple,
                               leaf_linear,leaf_root_linear,leaf_bucket,
                               parent_leaf,rank_leaf,tol)
    # print("debug2")
    _mark_leaf_flag_from_coarse(label_mask_coarse,grid_mask,grid_info,oct_soa_tuple,leaf_linear,leaf_root_linear,parent_leaf)

    # print("debug3")
    label_mask_final = _backfill_grid_from_leaf_components(
        label_mask_final, leaf_linear, leaf_root_linear,
        oct_soa_tuple, grid_info
    )
    # print("debug4")

    return label_mask_final, None