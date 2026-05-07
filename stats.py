import json
import os
import platform
import sys
import numpy as np
from numba import get_num_threads, threading_layer

from .octree import OCC_VOID, OCC_ACC, OCC_TRAP


def get_stats_and_envs(
    void_mask,
    grid_mask,
    label_mask,
    oct_soa_tuple,
    surface_area,
    pore_data,
    elem_mass,
    box,
    probe_nm,
    grid_info,
    grid_space_set,
    args,
    settings=None,
    timings=None,
):
    Lx, Ly, Lz = box
    gx, gy, gz, dgx, dgy, dgz = grid_info


    settings = {
        "atoms_table_path": os.path.abspath(args.atoms) if args.atoms else None,
        "grid_target_nm": float(args.grid),
        "probe_nm": float(args.probe),
        "threads_requested": int(args.threads),
        "threads_used": int(get_num_threads()),
        "threading_layer": threading_layer(),
        "sched_affinity_count": int(len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else -1),
        "surface_enabled": bool(not args.no_surface),
        "pore_enabled": bool(args.pore),
        "octree_enabled": bool(not args.no_octree),
        "oct_level": int(args.oct_level),
        "oct_grid_nm": float(args.oct_grid),
        "cube_enabled": bool(args.cube),
        "cube_space_nm": None if args.cube_space is None else float(args.cube_space),
        "smooth_cube": bool(args.smooth),
        "stats_enabled": bool(args.stats),
        "debug_enabled": bool(args.debug),
        "debug_print_enabled": bool(args.debug_print),
        "cube_downsample_factor": int(max(1, int(round((max(0.05, args.grid) if args.cube_space is None else args.cube_space) / args.grid)))) if args.cube else None,
    }

    voxel = dgx * dgy * dgz
    Vcell = Lx * Ly * Lz

    avogadro = 6.02214076e23
    m = elem_mass.sum()
    total_mass = m / avogadro  # g
    density = total_mass / (Lx * Ly * Lz * 1e-21)  # g/cm3

    if surface_area is None:
        surface_area = (0.0, 0.0)

    voidV = (void_mask.sum() * voxel).astype(np.float64)
    accV = ((label_mask == 2).sum() * voxel).astype(np.float64)
    trapV = ((label_mask == 1).sum() * voxel).astype(np.float64)

    octree_info = {
        "octree_used": bool(oct_soa_tuple is not None),
        "octree_root_voxels": 0,
        "octree_nodes": 0,
        "octree_root_nodes": 0,
        "octree_leaf_nodes": 0,
    }

    if oct_soa_tuple:
        root_volume = dgx * dgy * dgz
        node_volume = np.array(
            [root_volume / (2 ** (3 * l)) for l in range(16)],
            dtype=np.float64
        )
        octree_mask = (grid_mask & 128) == 128

        x, y, z, d, parent, child, level, occ = oct_soa_tuple

        root_nodes = parent == -1
        leaf_nodes = child == -1

        octree_info["octree_root_voxels"] = int(octree_mask.sum())
        octree_info["octree_nodes"] = int(len(level))
        octree_info["octree_root_nodes"] = int(root_nodes.sum())
        octree_info["octree_leaf_nodes"] = int(leaf_nodes.sum())

        leaf_void = ((occ[leaf_nodes] & OCC_VOID) == OCC_VOID).astype(np.float32)
        blur_void_V = ((octree_mask & void_mask).sum() * voxel).astype(np.float64)
        voidV -= blur_void_V
        voidV += np.sum(leaf_void * node_volume[level[leaf_nodes]])

        leaf_acc = ((occ[leaf_nodes] & OCC_ACC) == OCC_ACC).astype(np.float64)
        blur_acc_V = ((octree_mask & (label_mask == 2)).sum() * voxel).astype(np.float64)
        accV -= blur_acc_V
        accV += np.sum(leaf_acc * node_volume[level[leaf_nodes]])

        leaf_trap = ((occ[leaf_nodes] & OCC_TRAP) == OCC_TRAP).astype(np.float64)
        blur_trap_V = ((octree_mask & (label_mask == 1)).sum() * voxel).astype(np.float64)
        trapV -= blur_trap_V
        trapV += np.sum(leaf_trap * node_volume[level[leaf_nodes]])

    info = {
        "box_nm": (Lx, Ly, Lz),
        "grid_shape": (gx, gy, gz),
        "voxels": int(gx * gy * gz),
        "grid_space_set_nm": grid_space_set,
        "grid_space_real_nm": (dgx, dgy, dgz),
        "voxel_volume_nm3": voxel,
        "probe_nm": probe_nm,
        **octree_info,
    }

    stats = {
        "atoms": int(elem_mass.shape[0]),
        "mass_g/mol": m,
        "density_g/cm3": density,

        "Vcell_nm3": Vcell,
        "Vvoid_nm3": voidV,
        "Vprobe_nm3": Vcell - voidV,
        "Vacc_nm3": accV,
        "Vtrap_nm3": trapV,

        "Vvoid_frac": voidV / Vcell,
        "Vacc_frac": accV / Vcell,
        "Vtrap_frac": trapV / Vcell,

        "Sacc_nm2": surface_area[0],
        "Sacc_m2/g": surface_area[0] * 1e-18 / total_mass if total_mass > 0 else 0.0,

        "Stotal_nm2": surface_area[1],
        "Stotal_m2/g": surface_area[1] * 1e-18 / total_mass if total_mass > 0 else 0.0,

        "PLD_nm": pore_data[0] if pore_data else -1,
        "LCD_nm": pore_data[1] if pore_data else -1,
        "LCD_global_nm": pore_data[2] if pore_data else -1,
    }

    return {
        "info": info,
        "settings": settings if settings is not None else {},
        "stats": stats,
        "run_envs": get_execution_info(),
    }


def save_stats(path, stats):
    def _json_default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    if path:
        with open(path, "w") as f:
            json.dump(stats, f, indent=2, default=_json_default)


def get_execution_info():
    info = {
        "python_exe": sys.executable,
        "cwd": os.getcwd(),
        "platform": platform.platform(),
    }

    cmd_parts = [sys.executable, sys.argv[0]] + sys.argv[1:]
    info["full_command"] = " ".join(cmd_parts)

    return info