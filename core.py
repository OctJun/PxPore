# Relative imports must be placed after setting the OMP_NUM_THREADS environment variable, otherwise the numba threading layer may not be set correctly.
# DO NOT move these imports to the top of the file.
import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"


import datetime
import logging

import threading
import time
from dataclasses import asdict
from typing import Any
import numpy as np

from numba import set_num_threads, get_num_threads, threading_layer

from .config import AnalyseConfig
from .io_input import read_structure
from .io_cube import write_cube
from .octree_conectivity import percolation_masks_with_octree
from .octree import build_octree_forest
from .surface import fibonacci_sphere_surface_area
from .atoms import build_mass, build_radii_nm, load_atom_info, symbols_to_Z
from .pores import (
    filter_dmin_by_maxiaum_ball,
    get_psd_from_centerline,
    pld_lcd_by_bisection_from_dmin,
    pore_centerline_from_distance_field,
)
from .stats import get_stats_and_envs, save_stats
from .geometry import GRID_MASK_PROBE, _tanh, build_cell_list, dmin_by_all_atoms, downsample, grid_masks
from .connectivity_multicore import percolation_masks, LABEL_MASK_ACC, LABEL_MASK_TRAP
logger = logging.getLogger('PxPore')


def analyse(config: AnalyseConfig) -> dict[str, Any]:
    start_time = datetime.datetime.now()
    timings = {"t0": time.perf_counter()}

    # -------------------- threads --------------------
    if config.threads and config.threads >= 1:
        set_num_threads(config.threads)
    else:
        set_num_threads(max(1, get_num_threads() // 2))  # 默认半数可用线程

    logger.info(
        f"[Numba] threading layer: {threading_layer()}, threads: {get_num_threads()}")
    logger.info(
        f"[Numba] affinity: {list(os.sched_getaffinity(0))[0]}, counts:{len(os.sched_getaffinity(0))}")

    # -------------------- output prefix --------------------
    out_parent_path = os.path.abspath(os.path.dirname(config.input))
    if config.out_prefix:
        out_prefix = config.out_prefix
    else:
        out_prefix = f"{os.path.basename(os.path.abspath(config.input))}_g_{config.grid}_p_{config.probe}"

    # -------------------- atom info --------------------
    n_update = 0
    if config.atoms is not None:
        n_update = load_atom_info(config.atoms, overwrite=True)
        if n_update > 0:
            logger.info(
                f"[ATOMS] loaded/updated {n_update} entries from: {config.atoms}")

    timings["init"] = time.perf_counter()

    # -------------------- read structure --------------------
    pos, elems, box = read_structure(config.input)
    timings["io"] = time.perf_counter()

    rad = build_radii_nm(elems)
    elem_mass = build_mass(elems)

    box_inferred = False
    if box[0] <= 0 or box[1] <= 0 or box[2] <= 0:
        logger.warning(
            "[WARN] invalid box dimensions in input file, will be inferred from atomic positions and radii")
        pad = 0.0
        min_x = np.min(pos[:, 0] - rad) - pad
        max_x = np.max(pos[:, 0] + rad) + pad
        min_y = np.min(pos[:, 1] - rad) - pad
        max_y = np.max(pos[:, 1] + rad) + pad
        min_z = np.min(pos[:, 2] - rad) - pad
        max_z = np.max(pos[:, 2] + rad) + pad
        box = np.array([max_x - min_x, max_y - min_y,
                       max_z - min_z], dtype=np.float64)

        pos[:, 0] -= min_x
        pos[:, 1] -= min_y
        pos[:, 2] -= min_z
        box_inferred = True

    Lx, Ly, Lz = box
    g_target = config.grid

    # -------------------- build grid --------------------
    gx = max(int(round(Lx / g_target)), 1)
    gy = max(int(round(Ly / g_target)), 1)
    gz = max(int(round(Lz / g_target)), 1)

    dgx = Lx / gx
    dgy = Ly / gy
    dgz = Lz / gz
    grid_info = (gx, gy, gz, dgx, dgy, dgz)

    tol = 0.05
    errx = abs(dgx - g_target) / g_target
    erry = abs(dgy - g_target) / g_target
    errz = abs(dgz - g_target) / g_target
    if max(errx, erry, errz) > tol:
        raise ValueError(
            f"Requested grid {g_target:.6f} incompatible with box {box}. "
            f"Closest grid = ({dgx:.6f},{dgy:.6f},{dgz:.6f}), "
            f"relative error = ({errx:.2%},{erry:.2%},{errz:.2%})"
        )

    logger.info(
        f"[Input] atoms={pos.shape[0]}, box(nm)={box}, grid_spacing={config.grid} nm, probe={config.probe} nm")
    logger.info(
        f"[INFO] grid=({gx},{gy},{gz}), voxel={gx * gy * gz:.2e} , Actual grid spacing: {dgx:.6f} {dgy:.6f} {dgz:.6f}")

    # -------------------- cell list --------------------
    rmax = 0.0
    for i in range(rad.shape[0]):
        if rad[i] > rmax:
            rmax = rad[i]
    cutoff = rmax + config.probe
    if config.pore:
        cutoff *= 2  # 乘2，减少后面dmin重算的压力
    cell_size = max(cutoff, config.grid)
    cell_list_obj = build_cell_list(pos, box, cell_size)
    timings["cell"] = time.perf_counter()

    # -------------------- base masks --------------------
    grid_mask, dmin = grid_masks(
        pos, rad, box, config.probe, grid_info, cell_list_obj)
    void = ((grid_mask & GRID_MASK_PROBE) == GRID_MASK_PROBE).astype(np.uint8)
    timings["grid"] = time.perf_counter()

    # -------------------- octree --------------------
    if not config.no_octree:
        logger.info("[INFO] octree enabled")
        oct_soa_tuple, _, _ = build_octree_forest(
            grid_mask, grid_info,
            config.oct_level, config.oct_grid,
            pos, rad, box, config.probe,
            cell_list_obj,
        )
    else:
        logger.info("[INFO] octree disabled")
        oct_soa_tuple = None
    timings["octree"] = time.perf_counter()

    # -------------------- volume analysis --------------------
    logger.info("[INFO] Running volume analysis")
    if oct_soa_tuple:
        label_mask, uf_parent = percolation_masks_with_octree(
            void, grid_mask, grid_info, oct_soa_tuple)
    else:
        label_mask, uf_parent = percolation_masks(void)

    acc = (label_mask == LABEL_MASK_ACC)
    trap = (label_mask == LABEL_MASK_TRAP)
    timings["volume"] = time.perf_counter()

    # -------------------- surface analysis --------------------
    surface_area = None
    if not config.no_surface:
        logger.info("[INFO] Running surface area analysis")
        surface_area = fibonacci_sphere_surface_area(
            pos, rad, box, config.probe, grid_info,
            grid_mask, label_mask, cell_list_obj, oct_soa_tuple,
        )
    timings["surface"] = time.perf_counter()

    # -------------------- pore analysis --------------------
    dmin2 = None
    nodes_nm = None
    r_nm = None
    keep_idx = None
    pore_data = None

    if config.pore:
        logger.info("[INFO] Running pore analysis")
        dmin2, nfill = dmin_by_all_atoms(
            pos, rad, box, config.probe, cell_size, grid_info, dmin)
        logger.info(f"[PORE] Fill {nfill} voxels")

        logger.info("[PORE] Calculating maximum balls")
        nodes_nm, r_nm, edges = pore_centerline_from_distance_field(
            D_nm=dmin2,
            acc_u8=acc,
            grid_info=grid_info,
            box=box,
            rmin_center_nm=0.005,
            strict_plateau=True,
            prune=True,
            k=12,
            alpha=1.2,
            max_dist_nm=None,
            workers=-1,
        )

        logger.info("[PORE] Calculating PLD")
        pld, _, _ = pld_lcd_by_bisection_from_dmin(
            dmin, config.grid, config.probe)
        lcd = 2 * r_nm.max()

        psd_data, center_data = get_psd_from_centerline(
            nodes_nm, r_nm, bin_size=0.01)
        logger.info(
            f"[PORE] Found {nodes_nm.shape[0]} nodes, pore size range: {2*r_nm.min():.3f} - {2*r_nm.max():.3f} nm")
        pore_data = (pld, lcd)

        if config.stats:
            out_psd = f"{out_parent_path}/{out_prefix}_psd.txt"
            out_center = f"{out_parent_path}/{out_prefix}_center.txt"
            np.savetxt(out_psd, psd_data, fmt=["%d", "%.6f", "%d", "%.10e", "%.10e"],
                       header="N diameters_nm count volume cumulative", delimiter="\t", comments="#")
            np.savetxt(out_center, center_data, fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"],
                       header="N X_nm Y_nm Z_nm diameters_nm", delimiter="\t", comments="#")

    timings["pore"] = time.perf_counter()

    # -------------------- cube output --------------------
    if config.cube:
        cube_space = max(
            0.05, config.grid) if config.cube_space is None else config.cube_space
        k = max(1, int(round(cube_space / config.grid)))
        smooth_str = "smooth" if config.smooth else "pristine"

        logger.info(
            f"[Output] space: {cube_space} nm, downsample factor: {k}, prefix: {out_prefix}")

        void_out = void.astype(np.float32)
        acc_out = acc.astype(np.float32)
        trap_out = trap.astype(np.float32)
        dmin_out = dmin2.astype(
            np.float32) if dmin2 is not None else dmin.astype(np.float32)
        pore_vis_out = None

        if config.porevis and config.pore:
            pore_vis_out = filter_dmin_by_maxiaum_ball(
                dmin2, nodes_nm, r_nm, grid_info, box)

        if config.smooth:
            sm = _tanh(dmin, config.grid)
            void_out *= sm
            acc_out *= sm
            trap_out *= sm

        atoms_Z = symbols_to_Z(elems)

        out_void = f"{out_parent_path}/{out_prefix}_void_{smooth_str}.cube"
        out_occ = f"{out_parent_path}/{out_prefix}_occ_{smooth_str}.cube"
        out_acc = f"{out_parent_path}/{out_prefix}_acc_{smooth_str}.cube"
        out_trap = f"{out_parent_path}/{out_prefix}_trap_{smooth_str}.cube"
        out_dmin = f"{out_parent_path}/{out_prefix}_dmin.cube"
        out_pore_vis = f"{out_parent_path}/{out_prefix}_porevis.cube"

        tasks = [
            (out_void, downsample(void_out, k), box, cube_space, pos, atoms_Z),
            (out_occ, downsample((1.0 - void_out).astype(np.float32), k),
             box, cube_space, pos, atoms_Z),
            (out_acc, downsample(acc_out, k), box, cube_space, pos, atoms_Z),
            (out_trap, downsample(trap_out, k), box, cube_space, pos, atoms_Z),
            (out_dmin, downsample(dmin_out, k), box, cube_space, pos, atoms_Z),
        ]
        if pore_vis_out is not None:
            tasks.append((out_pore_vis, downsample(
                pore_vis_out, k), box, cube_space, pos, atoms_Z))

        threads = []
        for task_args in tasks:
            t = threading.Thread(
                target=write_cube, args=task_args, daemon=False)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    timings["cube"] = time.perf_counter()

    # -------------------- stats output --------------------
    logger.info("[STATS]")
    settings = {
        "atom_info_updates": int(n_update),
        "box_inferred": bool(box_inferred),
    }
    stats = get_stats_and_envs(
        void, grid_mask, label_mask, oct_soa_tuple, surface_area, pore_data,
        elem_mass, box, config.probe, grid_info, config.grid, config,
        settings=settings,
    )
    timings["stats"] = time.perf_counter()

    for k, v in stats["stats"].items():
        logger.info(f"{k:18s}: {v:>12.6f}")

    if config.stats:
        stats["run_envs"]["time"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
        stats["run_envs"]["execution_time"] = (
            datetime.datetime.now() - start_time).total_seconds()

        tkeys = list(timings.keys())
        timings_delta = {}
        for i in range(1, len(tkeys)):
            timings_delta[tkeys[i]] = timings[tkeys[i]] - timings[tkeys[i - 1]]
        stats["timings"] = timings_delta

        out_stats = f"{out_parent_path}/{out_prefix}_stats.json"
        save_stats(out_stats, stats)

    # -------------------- debug output --------------------
    if config.debug:
        np.save(f"{out_parent_path}/{out_prefix}_grid_mask.npy", grid_mask)
        np.save(f"{out_parent_path}/{out_prefix}_void.npy", void)
        np.save(f"{out_parent_path}/{out_prefix}_acc.npy", acc)
        np.save(f"{out_parent_path}/{out_prefix}_trap.npy", trap)
        np.save(f"{out_parent_path}/{out_prefix}_label_mask.npy", label_mask)
        np.save(f"{out_parent_path}/{out_prefix}_dmin.npy", dmin)

        if dmin2 is not None:
            np.save(f"{out_parent_path}/{out_prefix}_dmin2.npy", dmin2)

        if oct_soa_tuple is not None:
            np.savez(
                f"{out_parent_path}/{out_prefix}_octree_soa.npz",
                x=oct_soa_tuple[0],
                y=oct_soa_tuple[1],
                z=oct_soa_tuple[2],
                d=oct_soa_tuple[3],
                parent=oct_soa_tuple[4],
                child=oct_soa_tuple[5],
                level=oct_soa_tuple[6],
                occ=oct_soa_tuple[7],
            )

        if config.pore and nodes_nm is not None and r_nm is not None and keep_idx is not None:
            np.save(f"{out_parent_path}/{out_prefix}_pore_nodes_nm.npy", nodes_nm)
            np.save(f"{out_parent_path}/{out_prefix}_pore_r_nm.npy", r_nm)
            np.save(f"{out_parent_path}/{out_prefix}_pore_keep_idx.npy", keep_idx)

    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"[INFO] Finish in {elapsed:.2f} seconds")

    return {
        "stats": stats,
        "timings": timings,
        "out_prefix": out_prefix,
        "out_parent_path": out_parent_path,
        "grid_info": grid_info,
        "box": box,
        "elapsed_seconds": elapsed,
        "config": asdict(config),
    }
