
import logging
import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

from .geometry import pbc_delta

logger = logging.getLogger('PxPore')


OCC_SOLID = np.uint8(0 << 0)
OCC_VOID = np.uint8(1 << 0)
OCC_SPLIT = np.uint8(1 << 1)   # 2: node has children
OCC_ACC = np.uint8(1 << 2)   # 4: accessible
OCC_TRAP = np.uint8(1 << 3)   # 8: trapped

# FLAG_ACTIVE = np.uint8(1 << 0)
# FLAG_ROOT   = np.uint8(1 << 1)


TOP_N_ATOM = 3
# 8 个子节点相对父中心的符号（±1）

REFINE_INTERIOR_SHPERE = False

_CHILD_SIGNS = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [-1,  1, -1],
    [1,  1, -1],
    [-1, -1,  1],
    [1, -1,  1],
    [-1,  1,  1],
    [1,  1,  1],
], dtype=np.float32)


def alloc_soa(total_nodes: int):
    x = np.empty(total_nodes, np.float32)  # center position
    y = np.empty(total_nodes, np.float32)
    z = np.empty(total_nodes, np.float32)
    d = np.empty(total_nodes, np.float32)  # distance to surface

    # nnid   = np.empty(total_nodes, np.int32)
    parent = np.empty(total_nodes, np.int32)
    child = np.empty(total_nodes, np.int32)

    level = np.empty(total_nodes, np.uint8)  # level (0-based)
    # occupancy status (OCC_VOID, OCC_SOLID, or OCC_SPLIT)
    occ = np.empty(total_nodes, np.uint8)

    return (x, y, z, d, parent, child, level, occ)


@njit(fastmath=True, cache=True)
def eval_center(x, y, z, cand_atoms, pos, box, rad_nm, probe_nm):
    Lx, Ly, Lz = box
    min_dist = 1e9
    nnid = -1
    for j in cand_atoms:
        if j == -1:
            break
        dx = pbc_delta(x - pos[j, 0], Lx)
        dy = pbc_delta(y - pos[j, 1], Ly)
        dz = pbc_delta(z - pos[j, 2], Lz)
        dist_vdw = np.sqrt(dx*dx + dy*dy + dz*dz) - rad_nm[j]
        dist_probe = dist_vdw-probe_nm
        if dist_probe < min_dist:
            min_dist = dist_probe
            nnid = j

    occ0 = OCC_VOID if min_dist > 0 else OCC_SOLID
    return min_dist, nnid, occ0


@njit(cache=True)
def need_refine(d, occ, level, half, min_half, max_level):
    # if REFINE_INTERIOR_SHPERE:
    #     d_sphere = half + 1e-6 # 内接球
    # else:
    # d_sphere = half * 1.73205080756  + 1e-6 #  外接球
    d_sphere = half*1.4142136 + 1e-6  # 介于内接和外接之间的值

    if level+1 >= max_level or half <= min_half:
        return False
    return abs(d) <= d_sphere


@njit(cache=True)
def topN_insert(scores, ids, score_new, id_new):
    """
    维护一个“按分数升序”的 topN 列表（scores/ids 长度 = N）
    初值 scores 全是 +inf，ids 全是 -1
    """
    N = scores.shape[0]
    # 如果比当前最差还差，就直接丢
    if score_new >= scores[N-1]:
        return
    # 找插入位置
    k = N - 1
    while k > 0 and score_new < scores[k-1]:
        scores[k] = scores[k-1]
        ids[k] = ids[k-1]
        k -= 1
    scores[k] = score_new
    ids[k] = id_new


@njit(parallel=True, cache=True)
def build_root_candidates_topN(
    N,
    root_cxzy,    # (n_roots,)
    pos, rad_nm, box,
    probe_nm,
    cell_list_obj
):

    root_cx, root_cy, root_cz = root_cxzy[:,
                                          0], root_cxzy[:, 1], root_cxzy[:, 2]
    n_roots = root_cx.shape[0]
    head, next_atom, nx, ny, nz = cell_list_obj

    cand_atoms = np.full((n_roots, N), -1, dtype=np.int32)
    cand_score = np.full((n_roots, N), np.inf,
                         dtype=np.float32)  # dist_vdw 的 topN

    Lx, Ly, Lz = box

    for r in prange(n_roots):
        x = root_cx[r]
        y = root_cy[r]
        z = root_cz[r]
        cx = int(x / Lx * nx) % nx
        cy = int(y / Ly * ny) % ny
        cz = int(z / Lz * nz) % nz

        scores = cand_score[r]
        ids = cand_atoms[r]

        for dx_cell in (-1, 0, 1):
            ncx = (cx + dx_cell) % nx
            for dy_cell in (-1, 0, 1):
                ncy = (cy + dy_cell) % ny
                for dz_cell in (-1, 0, 1):
                    ncz = (cz + dz_cell) % nz
                    c = (ncz * ny + ncy) * nx + ncx
                    j = head[c]
                    while j != -1:
                        dx = pbc_delta(x - pos[j, 0], Lx)
                        dy = pbc_delta(y - pos[j, 1], Ly)
                        dz = pbc_delta(z - pos[j, 2], Lz)

                        dist_vdw = np.sqrt(dx*dx + dy*dy + dz*dz) - rad_nm[j]
                        dist_probe = dist_vdw-probe_nm

                        topN_insert(scores, ids, np.float32(
                            dist_probe), np.int32(j))
                        j = next_atom[j]

    return cand_atoms, cand_score


@njit(parallel=True, cache=True)
def pass1_count_nodes(
    root_cxyz, half_tables, cand_atom,
    max_level, min_half,
    stack_buf,
    pos, rad_nm, box, probe_nm
):
    root_cx = root_cxyz[:, 0]
    root_cy = root_cxyz[:, 1]
    root_cz = root_cxyz[:, 2]
    n_roots = root_cx.shape[0]
    counts = np.zeros(n_roots, np.int32)
    max_stack = stack_buf.shape[1]
    for r in prange(n_roots):
        tid = get_thread_id()
        stack = stack_buf[tid]          # 每个线程独立的栈视图

        top = 0
        # 压入根节点
        cx = root_cx[r]
        cy = root_cy[r]
        cz = root_cz[r]
        # half = root_half[r]
        d0, _, occ0 = eval_center(
            cx, cy, cz, cand_atom[r], pos, box, rad_nm, probe_nm)

        stack[top]['x'] = cx
        stack[top]['y'] = cy
        stack[top]['z'] = cz
        # stack[top]['half'] = half
        stack[top]['lv'] = np.uint8(0)
        stack[top]['d'] = d0
        stack[top]['occ'] = occ0
        top += 1

        splits = 0
        overflow = False

        while top > 0:
            top -= 1
            # 从栈顶读取所有字段（一次性命中缓存行）
            x = stack[top]['x']
            y = stack[top]['y']
            z = stack[top]['z']
            # half = stack[top]['half']
            lv = stack[top]['lv']
            d = stack[top]['d']
            occ = stack[top]['occ']
            half = half_tables[lv]  # (3)

            if not need_refine(d, occ, lv, max(half), min_half, max_level):
                continue

            if top + 8 > max_stack:
                overflow = True
                break

            splits += 1
            child_half = half * 0.5
            next_lv = np.uint8(lv + 1)

            for i in range(8):
                sx = x + _CHILD_SIGNS[i, 0] * child_half[0]
                sy = y + _CHILD_SIGNS[i, 1] * child_half[1]
                sz = z + _CHILD_SIGNS[i, 2] * child_half[2]

                d2, _, occ2 = eval_center(
                    sx, sy, sz, cand_atom[r], pos, box, rad_nm, probe_nm)

                # 写入子节点（所有字段连续写入）
                stack[top]['x'] = sx
                stack[top]['y'] = sy
                stack[top]['z'] = sz
                # stack[top]['half'] = child_half
                stack[top]['lv'] = next_lv
                stack[top]['d'] = d2
                stack[top]['occ'] = occ2
                top += 1

        if overflow:
            counts[r] = np.int32(-1)
        else:
            counts[r] = np.int32(1 + 8 * splits)

    return counts


def prefix_sum_offsets(counts: np.ndarray):
    counts = np.asarray(counts, dtype=np.int64)
    n = counts.shape[0]
    offsets = np.empty(n, dtype=np.int64)

    if np.any(counts == -1):
        raise OverflowError("STACK OVERFLOW")

    if n == 0:
        return offsets, np.int64(0)

    offsets[0] = 0
    if n > 1:
        offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)

    total = np.int64(offsets[-1] + counts[-1])
    return offsets, total


@njit(parallel=True, cache=True)
def pass2_build_forest(
    # SoA outputs (total_nodes,)
    oct_soa_tuple,
    # roots
    root_cxzy,          # (n_roots,3) float32
    half_tables,        # (lvs,3) float32
    cand_atom,          # (n_roots, N) int32  或者任何 eval_center 能吃的结构
    # prefix-sum results
    offsets,            # (n_roots,) int32
    counts,             # (n_roots,) int32
    # control
    max_level, min_half,
    # atom data for eval_center
    pos, rad_nm, box, probe_nm,
    err_flag
):
    x, y, z, d, parent, child, level, occ = oct_soa_tuple
    n_roots = root_cxzy.shape[0]
    # err_flag = np.zeros(n_roots,dtype=np.int8)
    for r in prange(n_roots):
        base = offsets[r]
        nmax = counts[r]
        ptr = 0

        # --------
        # alloc root
        # --------
        rid = base + ptr
        ptr += 1

        cx = root_cxzy[r, 0]
        cy = root_cxzy[r, 1]
        cz = root_cxzy[r, 2]

        x[rid] = cx
        y[rid] = cy
        z[rid] = cz
        level[rid] = np.int8(0)
        parent[rid] = np.int32(-1)
        child[rid] = np.int32(-1)

        # flags[rid]  = np.uint8(FLAG_ACTIVE | FLAG_ROOT)

        dd, nn, occ0 = eval_center(
            cx, cy, cz, cand_atom[r], pos, box, rad_nm, probe_nm)
        d[rid] = np.float32(dd)
        # nnid[rid] = np.int32(nn)
        occ[rid] = np.int8(occ0)

        # --------
        # stack for DFS: store node_id + half
        # capacity nmax is safe as long as pass1/pass2 decisions match
        # --------
        st_nid = np.empty(nmax, np.int32)
        # st_half = np.empty(nmax, np.float32)
        top = 0

        st_nid[top] = np.int32(rid)
        # st_half[top] = np.float32(root_half[r])
        top += 1

        while top > 0:
            top -= 1
            nid = st_nid[top]
            # half = st_half[top]

            lv = level[nid]
            dd = d[nid]
            oc = occ[nid]
            half = half_tables[lv]  # (3)
            # 是否继续细分（这里假设 need_refine 已包含 level/half 终止条件）
            if not need_refine(dd, oc, lv, max(half), min_half, max_level):
                # 作为叶子：确保 child=-1（root 初始化就是 -1，但父节点 split 后孩子也会进来）
                child[nid] = np.int32(-1)
                continue
            # --------
            # split: allocate 8 children contiguous
            # --------
            # occ[nid] = OCC_SPLIT

            child_half = half * 0.5
            cbase = base + ptr
            ptr += 8

            child[nid] = np.int32(cbase)

            px = x[nid]
            py = y[nid]
            pz = z[nid]

            next_lv = np.int16(lv + 1)

            for i in range(8):
                cid = cbase + i

                parent[cid] = np.int32(nid)
                child[cid] = np.int32(-1)
                level[cid] = next_lv
                # flags[cid]  = np.uint8(FLAG_ACTIVE)

                sx = px + _CHILD_SIGNS[i, 0] * child_half[0]
                sy = py + _CHILD_SIGNS[i, 1] * child_half[1]
                sz = pz + _CHILD_SIGNS[i, 2] * child_half[2]

                x[cid] = sx
                y[cid] = sy
                z[cid] = sz

                dd2, nn2, occ2 = eval_center(
                    sx, sy, sz, cand_atom[r], pos, box, rad_nm, probe_nm)
                d[cid] = np.float32(dd2)
                # nnid[cid] = np.int32(nn2)
                occ[cid] = np.int8(occ2)

                # push child
                st_nid[top] = np.int32(cid)
                # st_half[top] = np.float32(child_half)
                top += 1
            if ptr > nmax:
                err_flag[r] = 1


def build_octree_forest(grid_mask, grid_info,
                        max_level: int, min_size: float,
                        pos, rad_nm, box, probe_nm,
                        cell_list_obj):

    gx, gy, gz, dgx, dgy, dgz = grid_info
    min_half_size = np.float32(min_size * 0.5)
    octree_mask = (grid_mask & 128) == 128
    octree_root_count = np.sum(octree_mask)
    ix, iy, iz = np.where(octree_mask)  # 3 个 shape=(N,) 的数组
    # ix, iy, iz = np.where(grid_mask > -np.inf)   # 3 个 shape=(N,) 的数组

    root_centers_xyz = np.column_stack((
        (ix + 0.5) * dgx,
        (iy + 0.5) * dgy,
        (iz + 0.5) * dgz,
    )).astype(np.float32)
    half_tables = np.empty((16, 3), dtype=np.float64)
    half_tables[:, 0] = 0.5 * dgx / (2.0 ** np.arange(16, dtype=np.float32))
    half_tables[:, 1] = 0.5 * dgy / (2.0 ** np.arange(16, dtype=np.float32))
    half_tables[:, 2] = 0.5 * dgz / (2.0 ** np.arange(16, dtype=np.float32))
    # root_half_arr = np.full(root_centers_xyz.shape[0], np.array([dgx,dgy,dgz])*0.5, np.float32)

    cand_atom, _ = build_root_candidates_topN(
        TOP_N_ATOM, root_centers_xyz, pos, rad_nm, box, probe_nm, cell_list_obj)

    max_stack = 100000
    n_threads = get_num_threads()
    stack_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('d', np.float32),
        ('lv', np.uint8),
        ('occ', np.uint8),
    ])
    stack_buf = np.empty((n_threads, max_stack), dtype=stack_dtype)
    logger.info(
        f"[OCTREE] max level = {max_level}, min size = {min_half_size*2:.3f} nm, root = {octree_root_count}")

    counts = pass1_count_nodes(root_centers_xyz, half_tables, cand_atom,
                               max_level, min_half_size,
                               stack_buf,
                               pos, rad_nm, box, probe_nm
                               )
    offsets, total = prefix_sum_offsets(counts)
    logger.info(f"[OCTREE] total nodes = {total.sum()}")
    if total > np.iinfo(np.uint32).max:
        raise ValueError(
            f"Too many octree nodes {total} vs {np.iinfo(np.uint32).max}, decrease --oct-level, increase --oct-grid or disable octree with --no-octree")
    if total == 0:
        return None, None, None

    oct_soa_tuple = alloc_soa(total)

    err_flag = np.zeros(total, dtype=np.int8)
    pass2_build_forest(
        oct_soa_tuple,
        root_centers_xyz, half_tables, cand_atom,
        offsets, counts,
        max_level, min_half_size,
        pos, rad_nm, box,  probe_nm,
        err_flag
    )
    bad = np.where(err_flag != 0)[0]
    if bad.size:
        r = int(bad[0])
        raise ValueError(f"pass2 overflow at root {r}: nmax={counts[r]}")

    return oct_soa_tuple, offsets, counts


@njit(parallel=True, fastmath=True, cache=True)
def octree_volume(oct_soa_tuple, offsets, counts, root_size):
    x, y, z, d, nnid, parent, child, level, occ = oct_soa_tuple
    n = x.shape[0]
    node_volume = [(root_size/(2**l))**3 for l in range(16)]

    v_total = 0.0
    v_occ = 0.0
    v_void = 0.0
    n_leafs = 0
    for i in prange(n):
        oc = occ[i]
        if oc != OCC_SPLIT:
            n_leafs += 1
            vv = node_volume[level[i]]
            v_total += vv
            if oc == OCC_SOLID:
                v_occ += vv
            else:  # OCC_VOID
                v_void += vv
    return v_total, v_occ, v_void, n_leafs
