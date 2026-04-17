import math
import numpy as np
from numba import njit, prange

EPS = 1e-8

GRID_MASK_BOUNDRAY = np.uint8(128)
GRID_MASK_VDW = np.uint8(1)
GRID_MASK_PROBE = np.uint8(2)

VOID_MASK_VOID = np.uint8(1)
VOID_MASK_SOLID = np.uint8(0)

@njit(inline='always')
def pbc_delta(dx, L):
    dx -= np.rint(dx / L) * L
    return dx


@njit(cache=True)
def build_cell_list(pos, box, cell_size):
    Lx, Ly, Lz = box
    nx = max(1, int(Lx / cell_size))
    ny = max(1, int(Ly / cell_size))
    nz = max(1, int(Lz / cell_size))
    ncell = nx * ny * nz

    head = -np.ones(ncell, dtype=np.int64)
    nxt = -np.ones(pos.shape[0], dtype=np.int64) 

    for i in range(pos.shape[0]):
        x, y, z = pos[i, 0], pos[i, 1], pos[i, 2]
        cx = int(x / Lx * nx) % nx
        cy = int(y / Ly * ny) % ny
        cz = int(z / Lz * nz) % nz
        c = (cz * ny + cy) * nx + cx #flatten
        nxt[i] = head[c]             # insert the first atom of a cell
        head[c] = i                  # linknode head update

    return head, nxt, nx, ny, nz

@njit(parallel=True,fastmath=True,cache=True)
def grid_masks(pos, rad_nm, box, probe_nm, grid_info,cell_list):
    """
    Returns:
      void_mask: uint8 (gx,gy,gz) 0=solid,1=vdw,2^N=probe(N-1)
      dmin: float32 signed distance to nearest exclusion surface (nm)
    """
    Lx, Ly, Lz = box
    # gx = int(np.floor(Lx / grid_space))
    # gy = int(np.floor(Ly / grid_space))
    # gz = int(np.floor(Lz / grid_space))
    gx,gy,gz,dgx,dgy,dgz = grid_info 
    head, next_atom, nx, ny, nz = cell_list

    max_grid_distance  = max(dgx,dgy,dgz) * 0.7071068 + 1e-6 # 对角线

    status = np.zeros((gx, gy, gz), dtype=np.uint8)
    dmin = np.empty((gx, gy, gz), dtype=np.float32)

    total = gx * gy * gz

    for ix in prange(gx):
        for iy in range(gy):
            for iz in range(gz):
                # ix = idx // (gy * gz)
                # rem = idx - ix * (gy * gz)
                # iy = rem // gz
                # iz = rem - iy * gz
                # current voxel center postion
                x = (ix + 0.5) * dgx
                y = (iy + 0.5) * dgy
                z = (iz + 0.5) * dgz
                # current cell idx
                cx = int(x / Lx * nx) % nx
                cy = int(y / Ly * ny) % ny
                cz = int(z / Lz * nz) % nz
                # signed distance to nearest surface:
                # d = |r-rj| - (rad+probe)
                min_dist = 1e9  # large positive
                nearest_atom = -1
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
                                # rcut = rad_nm[j] + probe_nm
                                dist_vdw = np.sqrt(dx*dx + dy*dy + dz*dz) - rad_nm[j]
                                # dist_probe = dist_vdw - probe_nm
                                # dist = math.sqrt(dx*dx + dy*dy + dz*dz) - rcut
                                if dist_vdw < min_dist:
                                    min_dist = dist_vdw
                                    nearest_atom = j
                                j = next_atom[j] #next atom

                dmin[ix, iy, iz] = min_dist - probe_nm
                status[ix, iy, iz] |= GRID_MASK_VDW if min_dist > 0.0 else 0  # vdwV
                status[ix, iy, iz] |= GRID_MASK_PROBE if min_dist- probe_nm > 0.0 else 0 #probeV
                # need octree
                if dmin[ix, iy, iz] - max_grid_distance < 0.0 and dmin[ix, iy, iz] + max_grid_distance > 0.0:
                    # 边界可能穿过的格点
                    status[ix, iy, iz] |= GRID_MASK_BOUNDRAY
    return status, dmin





@njit(parallel=True,cache=True)
def extract_void_boundary(void_mask):
    """
    void_mask: uint8 (gx,gy,gz)  0 = solid  1 = void
    boundary voxel: 1 = b 
    """
    gx, gy, gz = void_mask.shape
    boundary = np.zeros((gx, gy, gz), dtype=np.uint8)
    for x in prange(gx):
        for y in range(gy):
            for z in range(gz):
                # 只在 solid 体素上定义边界
                if void_mask[x, y, z] == VOID_MASK_VOID:
                    continue
                if (x==0 or x==gx-1 or y==0 or y==gy-1 or z==0 or z==gz-1):
                    boundary[x,y,z] = 1
                    continue
                if (
                    void_mask[x - 1, y, z] != VOID_MASK_SOLID or
                    void_mask[x + 1, y, z] != VOID_MASK_SOLID or
                    void_mask[x, y - 1, z] != VOID_MASK_SOLID or
                    void_mask[x, y + 1, z] != VOID_MASK_SOLID or
                    void_mask[x, y, z - 1] != VOID_MASK_SOLID or
                    void_mask[x, y, z + 1] != VOID_MASK_SOLID
                ):
                    boundary[x, y, z] = 1
    return boundary,None


@njit(parallel=True, fastmath=True, cache=True)
def dmin_by_all_atoms(
    pos,          # float32/float64, (N,3), nm
    rad_nm,       # float32/float64, (N,), nm
    box,          # float32/float64, (3,), nm
    probe_nm,     # float
    grid_info,    # (gx, gy, gz, dgx, dgy, dgz)
    dmin,         # float32/float64, (gx,gy,gz), in-place overwrite on fill_mask
):
    """
    对 grid_mask cell中没有原子导致 dmin >1e9-probe 的体素，采用全原子遍历方式计算：
        dmin = min_j ( ||r-r_j|| - rad_j - probe_nm )
    Returns
    -------
    dmin : same array object as input, filled in-place
    nfill : int64
        实际填充的格点数
    """
    Lx, Ly, Lz = box
    gx, gy, gz, dgx, dgy, dgz = grid_info
    natom = pos.shape[0]
    nfill = 0
    for ix in prange(gx):
        for iy in range(gy):
            for iz in range(gz):
                if dmin[ix, iy, iz] <=1e7:  #初始值应当是1e9-probe, 取一半证明是没被动过 
                    continue
                x = (ix + 0.5) * dgx
                y = (iy + 0.5) * dgy
                z = (iz + 0.5) * dgz
                min_dist = 1e9
                for j in range(natom):
                    dx = pbc_delta(x - pos[j, 0], Lx)
                    dy = pbc_delta(y - pos[j, 1], Ly)
                    dz = pbc_delta(z - pos[j, 2], Lz)
                    dist = np.sqrt(dx * dx + dy * dy + dz * dz) - rad_nm[j] 

                    if dist < min_dist:
                        min_dist = dist
                dmin[ix, iy, iz] = min_dist - probe_nm 
                nfill += 1

    return dmin, nfill




def downsample(field: np.ndarray, k: int,ds_func=np.mean) -> np.ndarray:
    """
    Downsample 3D array by factor k using block mean (k x k x k).
    Strategy: trim tail so each dimension divisible by k.
    """
    if k <= 1:
        return np.asarray(field, dtype=np.float32)
    a = np.asarray(field, dtype=np.float32)
    nx, ny, nz = a.shape
    nx2 = (nx // k) * k
    ny2 = (ny // k) * k
    nz2 = (nz // k) * k
    if nx2 == 0 or ny2 == 0 or nz2 == 0:
        raise ValueError(f"block_mean_3d: k={k} too large for shape={a.shape}")
    a = a[:nx2, :ny2, :nz2]
    a = a.reshape(nx2 // k, k, ny2 // k, k, nz2 // k, k)
    a = ds_func(a,axis=(1,3,5))
    return a.astype(np.float32, copy=False)



def _tanh(d, sigma):
    # sigma > 0, d in nm
    return 0.5 * (1.0 + np.tanh(d / sigma))


def _fermi(d, sigma):
    # logistic; avoid overflow a bit
    x = -d / sigma
    if x > 60.0:
        return 0.0
    if x < -60.0:
        return 1.0
    return 1.0 / (1.0 + np.exp(x))


def _smoothstep(d, w):
    # map d in [-w, +w] to t in [0,1], then smoothstep
    # outside -> 0 or 1
    if d <= -w:
        return 0.0
    if d >= w:
        return 1.0
    t = (d + w) / (2.0 * w)  # [0,1]
    return t * t * (3.0 - 2.0 * t)  # C1 smoothstep