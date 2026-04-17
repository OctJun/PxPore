

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from .connectivity_multicore import percolation_masks
from .geometry import pbc_delta

MODE_NONE = 0
MODE_CONTAINED = 1
MODE_OVERLAP   = 2

@njit(parallel=True,cache=True)
def local_maxima_26_mask(D_nm, acc_u8, rmin_nm=0.0, strict_plateau=True):
    gx, gy, gz = D_nm.shape
    out = np.zeros((gx, gy, gz), dtype=np.uint8)

    for x in prange(gx):
        for y in range(gy):
            for z in range(gz):
                if acc_u8[x,y,z] == 0:
                    continue
                v = float(D_nm[x,y,z])
                if v < rmin_nm:
                    continue
                is_max = 1
                has_smaller = 0
                for dx in (-1,0,1):
                    xx = x+dx
                    if xx < 0 or xx >= gx: 
                        continue
                    for dy in (-1,0,1):
                        yy = y+dy
                        if yy < 0 or yy >= gy:
                            continue
                        for dz in (-1,0,1):
                            zz = z+dz
                            if zz < 0 or zz >= gz:
                                continue
                            if dx==0 and dy==0 and dz==0:
                                continue
                            if acc_u8[xx,yy,zz] == 0:
                                continue
                            nv = float(D_nm[xx,yy,zz])
                            if nv > v:
                                is_max = 0
                                break
                            if nv < v:
                                has_smaller = 1
                        if is_max == 0:
                            break
                    if is_max == 0:
                        break

                if is_max == 1:
                    if (not strict_plateau) or (has_smaller == 1):
                        out[x,y,z] = 1
    return out


@njit(parallel=True,cache=True)
def _prune_from_candidate_list(nodes, r,box,  order, offsets, nbrs,eps, mode_flag):
    """
    统一内核：
      mode_flag = 0: contained
      mode_flag = 1: overlap
    """
    N = nodes.shape[0]
    alive = np.ones(N, dtype=np.uint8)
    Lx,Ly,Lz=box
    for kk in range(order.shape[0]):
        ii = order[kk]
        if alive[ii] == 0:
            continue

        xi = nodes[ii, 0]
        yi = nodes[ii, 1]
        zi = nodes[ii, 2]
        ri = r[ii]

        start = offsets[ii]
        end   = offsets[ii + 1]

        for p in range(start, end):
            j = nbrs[p]
            if j == ii or alive[j] == 0:
                continue

            rj = r[j]

            # contained 模式下，只可能删更小或近似等大的球
            if mode_flag == MODE_CONTAINED:
                if rj > ri + eps:
                    continue
            dx = pbc_delta(nodes[j, 0] - xi,Lx)
            dy = pbc_delta(nodes[j, 1] - yi,Ly)
            dz = pbc_delta(nodes[j, 2] - zi,Lz)
            # dx = nodes[j, 0] - xi
            # dy = nodes[j, 1] - yi
            # dz = nodes[j, 2] - zi
            d2 = dx * dx + dy * dy + dz * dz

            if mode_flag == MODE_CONTAINED:
                # j 被 i 包含：d <= ri - rj + eps
                rhs = (ri - rj) + eps
                if rhs > 0.0 and d2 <= rhs * rhs:
                    alive[j] = 0

            else:
                # overlap: d < ri + rj - eps
                rhs = (ri + rj) - eps
                if rhs > 0.0 and d2 < rhs * rhs:
                    alive[j] = 0

    return np.nonzero(alive)[0].astype(np.int32)


def prune_balls(nodes_nm, r_nm, box,eps=1e-6, mode_flag=MODE_OVERLAP, leafsize=32, workers=-1):
    nodes = np.asarray(nodes_nm, dtype=np.float32)
    r = np.asarray(r_nm, dtype=np.float32)
    N = nodes.shape[0]
    if N == 0:
        return np.empty((0,), dtype=np.int32)

    # offsets, nbrs = _build_candidate_csr(
    #     nodes, r, box,mode_flag=mode_flag, leafsize=leafsize, workers=workers
    # )

    tree = cKDTree(nodes, leafsize=leafsize, boxsize=box,compact_nodes=True, balanced_tree=True)

    if mode_flag == MODE_CONTAINED:
        # 若 j 被 i 包含，则一定满足 center_dist <= ri
        neigh = tree.query_ball_point(nodes, r, workers=workers)
    elif mode_flag == MODE_OVERLAP:
        # 若 i 与 j 重叠，则 center_dist < ri + rj <= ri + rmax
        rmax = float(r.max())
        neigh = tree.query_ball_point(nodes, r + rmax, workers=workers)
    else:
        raise ValueError(f"Unknown mode_flag: {mode_flag}")
    counts = np.empty(N, dtype=np.int32)
    for i in range(N):
        counts[i] = len(neigh[i])

    offsets = np.empty(N + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    nbrs = np.empty(offsets[-1], dtype=np.int32)
    for i in range(N):
        s = offsets[i]
        e = offsets[i + 1]
        nbrs[s:e] = np.asarray(neigh[i], dtype=np.int32)



    order = np.argsort(-r).astype(np.int32)
    return _prune_from_candidate_list(nodes, r, box,order, offsets, nbrs, eps, mode_flag)


def build_centerline_edges(nodes_nm, r_nm,box, k=12, alpha=1.2, max_dist_nm=None, workers=-1):
    """
    returns edges: (E,2) int32, undirected unique
    """
    nodes_nm = np.asarray(nodes_nm, dtype=np.float32)
    r_nm = np.asarray(r_nm, dtype=np.float32)
    N = nodes_nm.shape[0]
    if N <= 1:
        return np.empty((0,2), dtype=np.int32)

    tree = cKDTree(nodes_nm, leafsize=32,boxsize=box, compact_nodes=True, balanced_tree=True) 

    # k+1 because nearest is itself
    kk = min(k + 1, N)
    dists, idxs = tree.query(nodes_nm, k=kk, workers=workers)

    edges_set = set()
    for i in range(N):
        ri = float(r_nm[i])
        for t in range(1, kk):
            j = int(idxs[i, t])
            if j == i:
                continue
            dij = float(dists[i, t])
            rj = float(r_nm[j])

            if max_dist_nm is not None and dij > float(max_dist_nm):
                continue

            # ball overlap / proximity criterion
            if dij <= alpha * (ri + rj):
                a, b = (i, j) if i < j else (j, i)
                edges_set.add((a, b))

    edges = np.array(list(edges_set), dtype=np.int32)
    return edges

def pore_centerline_from_distance_field(D_nm, acc_u8, grid_info,box,
                                        rmin_center_nm=0.10,
                                        strict_plateau=True,
                                        prune=True,
                                        k=12, alpha=1.2, max_dist_nm=None,
                                        workers=-1):
    """
    Returns:
      nodes_nm: (K,3) center points (nm)
      r_nm: (K,) radii at centers (nm)
      edges: (E,2) node indices
    """
    D_nm = np.asarray(D_nm, dtype=np.float32)
    acc_u8 = np.asarray(acc_u8, dtype=np.uint8)

    # 1) local maxima mask
    maxima_u8 = local_maxima_26_mask(D_nm, acc_u8, rmin_nm=rmin_center_nm, strict_plateau=strict_plateau)
    # 2) coords + radii
    idx = np.argwhere(maxima_u8 != 0).astype(np.int32)   # (N,3) in voxel index
    if idx.shape[0] == 0:
        return np.empty((0,3), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,2), dtype=np.int32)
    # voxel center -> nm
    nodes_nm = (idx.astype(np.float32) + 0.5) * np.array(grid_info[3:])
    r_nm = D_nm[idx[:,0], idx[:,1], idx[:,2]].astype(np.float32)
    mask = r_nm > 0.0
    nodes_nm = nodes_nm[mask]
    r_nm = r_nm[mask]
    
    # keep = prune_balls(nodes_nm, r_nm, box,mode_flag=MODE_CONTAINED)
    # nodes_nm = nodes_nm[keep]
    # r_nm = r_nm[keep]

    keep = prune_balls(nodes_nm, r_nm, box,mode_flag=MODE_OVERLAP)
    nodes_nm = nodes_nm[keep]
    r_nm = r_nm[keep]

    edges = build_centerline_edges(nodes_nm, r_nm, k=k, box=box, alpha=alpha, max_dist_nm=max_dist_nm, workers=workers)

    return nodes_nm, r_nm, edges


def get_psd_from_centerline(nodes_nm, r_nm, bin_size=0.01):
    """
    从孔隙中心线结果计算PSD
    """
    sorted_indices = np.argsort(r_nm)[::-1]
    r_nm = r_nm[sorted_indices]
    nodes_nm = nodes_nm[sorted_indices,:]

    diameters_nm = 2.0 * r_nm

    if r_nm.shape[0]==0:
        return None,None
    bins = np.arange(0.0, diameters_nm.max() + 10 * bin_size, bin_size)
    # 数量分布
    hist, bin_edges = np.histogram(diameters_nm, bins=bins)
    # 体积分布（假设每个中心点对应一个球）
    volumes = (4.0 / 3.0) * np.pi * (r_nm ** 3)
    vol_hist, _ = np.histogram(diameters_nm, bins=bins, weights=volumes)
    # 差分分布：按bin宽度归一化后的体积分布
    vol_sum = vol_hist.sum()
    vol_hist_frac = vol_hist / vol_sum
    vol_hist = vol_hist_frac / bin_size
    cumulative = np.cumsum(vol_hist_frac)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    psd_data = np.column_stack((
        np.arange(1, len(bin_centers) + 1, dtype=np.int32),
        bin_centers,
        hist,
        vol_hist_frac,
        cumulative
    ))
    center_data = np.column_stack((
        np.arange(1, len(nodes_nm) + 1, dtype=np.int32),
        nodes_nm[:, 0],
        nodes_nm[:, 1],
        nodes_nm[:, 2],
        diameters_nm
    ))
    return psd_data,center_data        


@njit(parallel=True,cache=True)
def _fill_void_mask(dmin_nm, r_probe_nm, void_mask):
    nx, ny, nz = dmin_nm.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                void_mask[i, j, k] = 1 if dmin_nm[i, j, k] >= r_probe_nm else 0

def _has_percolation(r_probe_nm,dmin_nm,void_mask):
        _fill_void_mask(dmin_nm, r_probe_nm, void_mask)
        label_mask, _ = percolation_masks(void_mask)
        ok = np.any(label_mask == 2)
        return ok, label_mask


def pld_lcd_by_bisection_from_dmin(
    dmin_nm,
    tol_nm,
    probe_nm,
    rlow_nm=0.0,
    rhigh_nm=None,
):
    """
    根据 dmin 场 + percolation_masks 在唯一值上二分查找 PLD，
    同时返回 total LCD。

    Parameters
    ----------
    dmin_nm : ndarray, shape (gx, gy, gz)
    rlow_nm : float
        二分下界（探针半径），单位 nm
    rhigh_nm : float or None
        二分上界（探针半径），单位 nm；默认取 max(dmin_nm)
    verbose : bool

    Returns
    -------
    pld_nm : float
    r_crit_nm : float
    lcd_nm : float
    """
    _dmin_nm = np.ascontiguousarray(dmin_nm) - probe_nm
    dmax = float(np.max(_dmin_nm))
    lcd_nm = 2.0 * dmax
    tol_nm = tol_nm / 2
    if rhigh_nm is None:
        rhigh_nm = dmax
    else:
        rhigh_nm = min(float(rhigh_nm), dmax)

    if rhigh_nm <= 0.0:
        return 0.0, 0.0, lcd_nm

    void_mask = np.empty(_dmin_nm.shape, dtype=np.uint8)

    ok_low,_ = _has_percolation(rlow_nm,_dmin_nm,void_mask)
    if not ok_low:
        return 0.0, 0.0, lcd_nm

    ok_high,_ = _has_percolation(rhigh_nm,_dmin_nm,void_mask)
    if ok_high:
        r_crit_nm = rhigh_nm
        pld_nm = 2.0 * r_crit_nm
        return pld_nm, r_crit_nm, lcd_nm
    # tol_q_nm = tol_nm/10
    # q = np.round(_dmin_nm / tol_q_nm) * tol_q_nm
    # uniq = np.unique(q[(q >= rlow_nm) & (q <= rhigh_nm)])
    uniq = np.unique(_dmin_nm[(_dmin_nm >= rlow_nm) & (_dmin_nm <= rhigh_nm)])
    uniq.sort()
    if uniq.size == 0:
        return 0.0, 0.0, lcd_nm

    lo = 0
    hi = uniq.size - 1
    best_idx = -1

    while lo <= hi:
        mid = (lo + hi) // 2
        rmid = float(uniq[mid])
        ok_mid,_ = _has_percolation(rmid,_dmin_nm,void_mask)
        if ok_mid:
            best_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
        if uniq[hi] - uniq[lo] < tol_nm:
            best_idx = mid
            break

    if best_idx < 0:
        return 0.0, 0.0, lcd_nm

    r_crit_nm = float(uniq[best_idx])
    pld_nm = 2.0 * r_crit_nm
    return pld_nm, r_crit_nm, lcd_nm

    # bisection method by poreblazer
    # for it in range(128):
    #     rmid = 0.5 * (rlow_nm + rhigh_nm)
    #     ok_mid,_ = _has_percolation(rmid,_dmin_nm,void_mask)
    #     if verbose:
    #         print(
    #             "[PLD] iter {:02d}: rlow={:.6f} rhigh={:.6f} rmid={:.6f} ok={}".format(
    #                 it, rlow_nm, rhigh_nm, rmid, ok_mid
    #             )
    #         )
    #     if ok_mid:
    #         rlow_nm = rmid
    #     else:
    #         rhigh_nm = rmid
    #     if (rhigh_nm - rlow_nm) <= tol_nm:
    #         break
    # r_crit_nm = rlow_nm
    # pld_nm = 2.0 * r_crit_nm
    # return pld_nm, r_crit_nm,lcd_nm



@njit(cache=True,parallel=True)
def filter_dmin_by_maxiaum_ball(dmin,nodes_nm,r_nm,grid_info,box):
    Lx, Ly, Lz = box
    gx, gy, gz, dgx, dgy, dgz = grid_info
    n_pores = nodes_nm.shape[0]
    out = np.zeros_like(dmin,dtype=np.float32)
    r2 = r_nm*r_nm
    for ix in prange(gx):
        x = (ix + 0.5) * dgx
        for iy in range(gy):
            y = (iy + 0.5) * dgy
            for iz in range(gz):
                z = (iz+0.5)*dgz

                val = dmin[ix, iy, iz]
                if val <= 0.0:
                    continue

                keep = False
                kp = 0
                for p in range(n_pores):
                    dx = pbc_delta(x - nodes_nm[p, 0],Lx)
                    dy = pbc_delta(y - nodes_nm[p, 1],Ly)
                    dz = pbc_delta(z - nodes_nm[p, 2],Lz)
                    if dx * dx + dy * dy + dz * dz <= r2[p]:
                        kp = p
                        keep = True
                        break
                if keep:
                    out[ix, iy, iz] = r_nm[kp]/2            
    return out
