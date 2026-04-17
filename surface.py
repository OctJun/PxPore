import numpy as np
from numba import njit, prange


from .geometry import pbc_delta


@njit(parallel=True, cache=True)
def check_sample_point(sample_pos, grid_info, grid_mask, label_mask, oct_soa_tuple):
    gx, gy, gz, dgx, dgy, dgz = grid_info
    use_octree = False
    # if oct_soa_tuple is not None:
    #     use_octree = True
    #     x, y, z, d, parent, child, level, occ = oct_soa_tuple
    #     root_lut = np.full((gx, gy, gz), -1, dtype=np.int32)
    #     for nid in range(parent.shape[0]):
    #         if parent[nid] == -1:
    #             ix0 = int(np.floor(x[nid] / dgx)) % gx
    #             iy0 = int(np.floor(y[nid] / dgy)) % gy
    #             iz0 = int(np.floor(z[nid] / dgz)) % gz
    #             root_lut[ix0, iy0, iz0] = nid

    # sample_pos shape (natoms,nsample,3)
    n_atoms = sample_pos.shape[0]
    n_samples = sample_pos.shape[1]
    n_total = n_atoms*n_samples
    keep_idx = np.zeros((n_atoms, n_samples), dtype=np.bool_)
    for na in prange(n_atoms):
        for ns in range(n_samples):
            px, py, pz = sample_pos[na, ns]
            ix = int(np.floor(px/dgx)) % gx
            iy = int(np.floor(py/dgy)) % gy
            iz = int(np.floor(pz/dgz)) % gz
            # keep_idx[na, ns] = label_mask[ix,iy,iz] == 2 # accessible
            # if (not use_octree) or ((grid_mask[ix, iy, iz] & 128) == 0):
            for ii, jj, kk in [(0,0,0),(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                iix = (ix + ii) % gx
                iiy = (iy + jj) % gy
                iiz = (iz + kk) % gz
                if label_mask[iix, iiy, iiz] == 2:
                    keep_idx[na, ns] = True
                    break
            # else:
            #     leaf_id = root_lut[ix, iy, iz]
            #     if leaf_id == -1:
            #         continue
            #     while True:
            #         if leaf_id < 0 or leaf_id >= child.shape[0]:
            #             break
            #         base = child[leaf_id]
            #         if base == -1:
            #             if (occ[leaf_id] & OCC_ACC) == OCC_ACC:
            #                 keep_idx[na, ns] = True
            #             else:
            #                 pid = parent[leaf_id]
            #                 if pid == -1:
            #                     break
            #                 sib_base = child[pid]
            #                 if sib_base == -1:
            #                     break
            #                 valid_count = 0
            #                 for k in range(8):
            #                     sib_id = sib_base + k
            #                     if (occ[sib_id] & OCC_ACC) == OCC_ACC:
            #                         valid_count += 1
            #                 if valid_count >= 1:
            #                     keep_idx[na, ns] = True
            #             break

            #         offset = 0
            #         if px >= x[leaf_id]:
            #             offset |= 1
            #         if py >= y[leaf_id]:
            #             offset |= 2
            #         if pz >= z[leaf_id]:
            #             offset |= 4

            #         leaf_id = base + offset

    return keep_idx


@njit(parallel=True, fastmath=True, cache=True)
def occlusion_check(sample_pos, pos, rad, probe, box, grid_info, cell_list_obj):
    head, next_atom, nx, ny, nz = cell_list_obj
    gx, gy, gz, dgx, dgy, dgz = grid_info
    Lx, Ly, Lz = box
    n_atoms = sample_pos.shape[0]
    n_samples = sample_pos.shape[1]
    valid = np.zeros((n_atoms, n_samples), dtype=np.bool_)
    # 只在第一次循环时初始化
    atom_cells = np.empty(n_atoms, dtype=np.int32)
    atom_neighbor_indices = []
    for na in range(n_atoms):
        px0, py0, pz0 = pos[na]
        cx0 = int(px0 / Lx * nx) % nx
        cy0 = int(py0 / Ly * ny) % ny
        cz0 = int(pz0 / Lz * nz) % nz
        neighbor_indices = []
        for dx_cell in (-1, 0, 1):
            ncx = (cx0 + dx_cell) % nx
            for dy_cell in (-1, 0, 1):
                ncy = (cy0 + dy_cell) % ny
                for dz_cell in (-1, 0, 1):
                    ncz = (cz0 + dz_cell) % nz
                    c = (ncz * ny + ncy) * nx + ncx
                    j = head[c]
                    while j != -1:
                        if j != na:
                            neighbor_indices.append(j)
                        j = next_atom[j]
        atom_cells[na] = (cz0 * ny + cy0) * nx + cx0
        atom_neighbor_indices.append(
            np.array(neighbor_indices, dtype=np.int32))

    for na in prange(n_atoms):
        for ns in range(n_samples):
            px, py, pz = sample_pos[na, ns]
            _valid = True
            for j in atom_neighbor_indices[na]:
                if j==-1:
                    break
                dx = pbc_delta(px - pos[j, 0], Lx)
                dy = pbc_delta(py - pos[j, 1], Ly)
                dz = pbc_delta(pz - pos[j, 2], Lz)
                dist_sq = dx * dx + dy * dy + dz * dz
                rad_sum = rad[j] + probe
                if dist_sq < rad_sum * rad_sum:
                    _valid = False
                    break
            valid[na, ns] = _valid
    return valid


def fibonacci_sphere_surface_area(
    pos,
    rad,
    box,
    probe,
    grid_info,
    grid_mask,
    label_mask,       # 0=solid,1=trap,2=accessible
    cell_list_obj,
    oct_soa_tuple,
    nsample=1000,
    # seed=114514
):
    """
    Estimate the accessible surface area of a set of spheres using Monte Carlo sampling.

    Parameters:
        pos (np.ndarray): Positions of atoms, shape (N, 3).
        rad (np.ndarray): Radii of atoms, shape (N,).
        box (np.ndarray): Simulation box dimensions, shape (3,).
        probe (float): Probe radius for accessibility.
        grid_info (tuple): Grid parameters (gx, gy, gz, dgx, dgy, dgz).
        grid_mask (np.ndarray): Grid mask for accessibility.
        label_mask (np.ndarray): Mask labeling grid points (0=solid, 1=trap, 2=accessible).
        cell_list_obj (object): Cell list object for neighbor search.
        oct_soa_tuple (tuple): Octree structure for spatial queries.
        nsample (int, optional): Number of Monte Carlo samples per atom (default: 50000).
        seed (int, optional): Random seed for reproducibility (default: 114514).

    Returns:
        float: Estimated total accessible surface area (unit: nm^2).
    """
    # seed=114514
    # np.random.seed(seed)

    natoms = pos.shape[0]
    gx, gy, gz, dgx, dgy, dgz = grid_info
    area_per_atom = np.zeros(natoms, dtype=np.float64)
    area_per_atom = 4*np.pi*((rad+probe)**2)

    # u1 = np.random.random(natoms*nsample).reshape((natoms,nsample))
    # u2 = np.random.random(natoms*nsample).reshape((natoms,nsample))
    # phi = 2.0 * np.pi * u1
    # costheta = 1.0 - 2.0 * u2
    # sintheta = np.sqrt(np.maximum(0.0, 1.0 - costheta * costheta))
    # dirs = np.empty((natoms,nsample, 3), dtype=np.float64)
    # dirs[:,:, 0] = sintheta * np.cos(phi)
    # dirs[:,:, 1] = sintheta * np.sin(phi)
    # dirs[:,:, 2] = costheta

    # Fibonacci sphere sampling
    k = np.arange(nsample, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * k / nsample
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))

    dirs = np.empty((nsample, 3), dtype=np.float64)
    dirs[:, 0] = r * np.cos(phi)
    dirs[:, 1] = r * np.sin(phi)
    dirs[:, 2] = z
    dirs = np.broadcast_to(dirs[None, :, :], (natoms, nsample, 3)).copy()

    sample_pos = pos[:, None, :] + (rad[:, None, None]+probe) * dirs
    sample_pos[:, :, 0] = sample_pos[:, :, 0] % box[0]
    sample_pos[:, :, 1] = sample_pos[:, :, 1] % box[1]
    sample_pos[:, :, 2] = sample_pos[:, :, 2] % box[2]

    valid_accessible = check_sample_point(
        sample_pos, grid_info, grid_mask, label_mask, oct_soa_tuple)

    valid_void = occlusion_check(
        sample_pos, pos, rad, probe, box, grid_info, cell_list_obj)
    
    valid = valid_accessible & valid_void

    area_per_atom_total = area_per_atom * (valid_void.sum(axis=1) / nsample)
    area_per_atom_accessible = area_per_atom * (valid.sum(axis=1) / nsample)


    return area_per_atom_accessible.sum(),area_per_atom_total.sum()  # nm^2

