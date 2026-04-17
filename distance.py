import numpy as np
from scipy.spatial import cKDTree

def distance_to_boundary(void_mask, grid_space):
    """
    Approximate distance to boundary voxels.
    """
    gx, gy, gz = void_mask.shape
    idx = np.argwhere(void_mask == 1)

    boundary = []
    for x, y, z in idx:
        for dx, dy, dz in (
            (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
        ):
            nx, ny, nz = x+dx, y+dy, z+dz
            if not (0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz) \
               or void_mask[nx, ny, nz] == 0:
                boundary.append((x, y, z))
                break

    boundary = np.array(boundary, dtype=np.float32)
    boundary = (boundary + 0.5) * grid_space

    tree = cKDTree(boundary)

    dfield = np.zeros_like(void_mask, dtype=np.float32)
    pts = (idx + 0.5) * grid_space
    dist, _ = tree.query(pts)

    # FIX: 减去半个 voxel 近似真实表面
    for i, (x, y, z) in enumerate(idx):
        dfield[x, y, z] = max(0.0, dist[i] - 0.5 * grid_space)

    return dfield


def dmin_by_kdtree(
    boundary_u8: np.ndarray,   # uint8 (gx,gy,gz) 1=boundary voxel
    void_u8: np.ndarray,          # float32/float64 (gx,gy,gz), nm
    grid_space: float,         # nm per voxel
    box: np.ndarray,
    batch: int = 500_000,      # query batch size
    leafsize: int = 32,
    n_jobs: int = -1,          # scipy>=1.6 supports workers in query
    surface_correction: bool = False,  # if True, subtract ~0.5*grid_space (see note)
):
    # 1) boundary voxel centers in index space
    b_idx = np.argwhere(boundary_u8 != 0).astype(np.int32)  # (M,3) in (x,y,z)
    # 2) target points
    t_idx = np.argwhere(void_u8 != 0).astype(np.int32)
    dmin2 = np.zeros_like(void_u8,dtype=np.float32)  # in-place
    if t_idx.shape[0] == 0:
        return dmin2, 0, b_idx.shape[0]
    # 3) build KDTree in physical space (nm) using voxel centers
    b_pts = (b_idx.astype(np.float32) + 0.5) * grid_space
    if box is not None:
        tree = cKDTree(b_pts, leafsize=leafsize, boxsize=box, compact_nodes=True, balanced_tree=True)
    else:
        tree = cKDTree(b_pts, leafsize=leafsize, compact_nodes=True, balanced_tree=True)
    # 4) query in batches (also in physical space)
    t_pts = (t_idx.astype(np.float32) + 0.5)  * grid_space
    out_remap = np.empty((t_idx.shape[0],), dtype=np.float32)
    for s in range(0, t_pts.shape[0], batch):
        e = min(s + batch, t_pts.shape[0])
        # exact nearest neighbor distance to boundary voxel centers
        dist, _ = tree.query(t_pts[s:e], k=1, workers=n_jobs)
        out_remap[s:e] = dist.astype(np.float32)
    if surface_correction:
        # Optional heuristic: convert "to nearest boundary voxel center" -> approx "to boundary surface"
        # For a voxelized boundary, subtract ~0.5 grid. Clamp at 0.
        out_remap = np.maximum(out_remap - 0.5 * grid_space, 0.0)
    # 5) write
    x = t_idx[:, 0]
    y = t_idx[:, 1]
    z = t_idx[:, 2]
    dmin2[x, y, z] = out_remap
    return dmin2, t_idx.shape[0], b_idx.shape[0]
