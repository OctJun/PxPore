#!/usr/bin/env python3
"""绘制测试体系的 XZ 中截面距离投影图。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pxpore_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
PXPORE_SRC = REPO_ROOT / "src"
if str(PXPORE_SRC) not in sys.path:
    sys.path.insert(0, str(PXPORE_SRC))

from PxPore.atoms import build_radii_nm
from PxPore.io_input import read_structure


DEFAULT_CONFIG = Path(__file__).with_name("sensitivity_study.json")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "docs" / "sensitivity_analysis" / "figures" / "projections"
)
COLOR_OCCUPIED = "#7B7B7B"
DEFAULT_VMAX_NM = 0.50
LABELS = {
    "single_h": "Single H",
    "h512": "512 H",
    "overlapped_h": "Overlapped H",
    "throat": "Throat",
    "inaccessible_pocket": "Inaccessible pocket",
    "anisotropic_pore_x": "Through pore x",
    "anisotropic_pore_z": "Through pore z",
    "anisotropic_sealed_x": "Sealed x",
    "anisotropic_sealed_z": "Sealed z",
    "equal_overlapping": "Equal overlapping cavities",
    "unequal_overlapping": "Unequal overlapping cavities",
    "equal_separated": "Equal separated cavities",
}


def make_distance_cmap():
    colors = (
        "#FFFFFF",
        "#E0E6EC",
        "#C3D0DE",
        "#BCD1EB",
        "#9EBADB",
        "#7FA2CA",
        "#628DBB",
        "#4874A3",
        "#39638F",
        "#2A4970",
    )
    return mpl.colors.LinearSegmentedColormap.from_list(
        "surface_distance", colors, N=256)


DISTANCE_CMAP = make_distance_cmap()


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    category: str
    path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot XZ signed-distance slices for test systems."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--long-pixels",
        type=int,
        default=900,
        help="pixels along the longer X/Z box dimension",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=DEFAULT_VMAX_NM,
        help="shared maximum distance shown by the color scale in nm",
    )
    return parser.parse_args()


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.unicode_minus": False,
        "hatch.color": "#505050",
        "hatch.linewidth": 0.55,
    })


def load_cases(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cases = []
    for key, spec in config["systems"].items():
        if spec["category"] not in {
            "analytic", "anisotropic", "psd_synthetic"
        }:
            continue
        path = Path(spec["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing test structure: {path}")
        cases.append(CaseSpec(
            key=key,
            label=LABELS.get(key, key.replace("_", " ")),
            category=spec["category"],
            path=path,
        ))
    return cases


def load_structure(case):
    positions, elements, box = read_structure(str(case.path))
    positions = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError(f"Invalid orthorhombic box: {case.path}")
    positions %= box
    radii = build_radii_nm(elements)
    return positions, radii, box


def choose_slice_y(case, positions, box):
    if case.key != "h512":
        return float(box[1] / 2.0)
    unique_y = np.unique(np.round(positions[:, 1], decimals=6))
    center = float(box[1] / 2.0)
    offset = np.abs(unique_y - center)
    offset = np.minimum(offset, box[1] - offset)
    return float(unique_y[int(np.argmin(offset))])


def signed_distance_slice(
    positions,
    radii,
    box,
    slice_y,
    long_pixels,
):
    lx, _, lz = box
    longest = max(lx, lz)
    nx = max(160, int(round(long_pixels * lx / longest)))
    nz = max(160, int(round(long_pixels * lz / longest)))
    x = (np.arange(nx, dtype=np.float64) + 0.5) * lx / nx
    z = (np.arange(nz, dtype=np.float64) + 0.5) * lz / nz
    xx, zz = np.meshgrid(x, z, indexing="xy")
    query = np.column_stack((
        xx.ravel(),
        np.full(xx.size, slice_y, dtype=np.float64),
        zz.ravel(),
    ))
    tree = cKDTree(positions, boxsize=box)
    neighbor_count = min(16, len(positions))
    signed_flat = np.empty(len(query), dtype=np.float64)
    chunk_size = 100000
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        distances, indices = tree.query(
            query[start:stop],
            k=neighbor_count,
            workers=-1,
        )
        if neighbor_count == 1:
            signed_flat[start:stop] = distances - radii[indices]
        else:
            signed_flat[start:stop] = np.min(
                distances - radii[indices], axis=1)
    return x, z, signed_flat.reshape(nz, nx)


def draw_projection(
    ax,
    x,
    z,
    signed,
    box,
    *,
    label,
    vmax,
    compact,
):
    extent = (0.0, float(box[0]), 0.0, float(box[2]))
    image = ax.imshow(
        np.maximum(signed, 0.0),
        origin="lower",
        extent=extent,
        cmap=DISTANCE_CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="none",
        aspect="equal",
    )
    occupied = signed <= 0.0
    if np.any(occupied):
        ax.contourf(
            x,
            z,
            occupied.astype(float),
            levels=(0.5, 1.5),
            colors=(COLOR_OCCUPIED,),
            hatches=("////",),
            antialiased=False,
            zorder=3,
        )
    if float(np.min(signed)) <= 0.0 <= float(np.max(signed)):
        ax.contour(
            x,
            z,
            signed,
            levels=(0.0,),
            colors=("#484848",),
            linewidths=0.45,
            zorder=4,
        )
    ax.set_xlim(0.0, float(box[0]))
    ax.set_ylim(0.0, float(box[2]))
    ax.set_aspect("equal")
    ax.set_title(label)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("x")
        ax.set_ylabel("z", rotation=0, labelpad=7)
    else:
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("z (nm)")
    return image


def save_figure(figure, base_path):
    figure.savefig(base_path.with_suffix(".png"))
    figure.savefig(base_path.with_suffix(".pdf"))
    plt.close(figure)


def individual_projection(dataset, output_dir, vmax):
    case, box, slice_y, x, z, signed = dataset
    aspect = float(box[0] / box[2])
    width = min(max(5.0 * aspect, 4.2), 7.4)
    height = min(max(5.0 / max(aspect, 0.35), 4.2), 8.0)
    figure, ax = plt.subplots(figsize=(width, height))
    draw_projection(
        ax,
        x,
        z,
        signed,
        box,
        label=case.label,
        vmax=vmax,
        compact=False,
    )
    slice_text = (
        f"y = Ly/2 = {slice_y:g} nm"
        if np.isclose(slice_y, box[1] / 2.0)
        else f"y = {slice_y:g} nm (nearest occupied layer)"
    )
    ax.text(
        0.02,
        0.02,
        slice_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 2.0,
        },
    )
    figure.tight_layout()
    save_figure(
        figure, output_dir / f"{case.key}_xz_projection")


def overview_projection(
    datasets,
    output_dir,
    filename,
    title,
    vmax,
    columns,
):
    if not datasets:
        return
    rows = int(np.ceil(len(datasets) / columns))
    width = 3.1 * columns if rows == 1 else 4.0 * columns
    height = 4.2 if rows == 1 else 3.5 * rows
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width, height),
        squeeze=False,
    )
    for ax, dataset in zip(axes.flat, datasets):
        case, box, _, x, z, signed = dataset
        draw_projection(
            ax,
            x,
            z,
            signed,
            box,
            label=case.label,
            vmax=vmax,
            compact=True,
        )
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    figure.suptitle(title, y=0.995)
    figure.subplots_adjust(
        left=0.04, right=0.98, bottom=0.06, top=0.90,
        wspace=0.22, hspace=0.25,
    )
    save_figure(figure, output_dir / filename)


def generate_projections(config_path, output_dir, long_pixels, vmax):
    if long_pixels < 160:
        raise ValueError("--long-pixels must be at least 160")
    if vmax <= 0:
        raise ValueError("--vmax must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for case in load_cases(config_path):
        positions, radii, box = load_structure(case)
        slice_y = choose_slice_y(case, positions, box)
        x, z, signed = signed_distance_slice(
            positions,
            radii,
            box,
            slice_y,
            long_pixels,
        )
        dataset = (case, box, slice_y, x, z, signed)
        datasets.append(dataset)
        individual_projection(dataset, output_dir, vmax)
        print(f"projection: {case.key}", flush=True)

    analytic = [
        dataset for dataset in datasets
        if dataset[0].category == "analytic"
    ]
    anisotropic = [
        dataset for dataset in datasets
        if dataset[0].category == "anisotropic"
    ]
    psd_synthetic = [
        dataset for dataset in datasets
        if dataset[0].category == "psd_synthetic"
    ]
    overview_projection(
        analytic,
        output_dir,
        "analytic_test_systems_xz_projection",
        "Analytic test systems: XZ cross-sections",
        vmax,
        columns=5,
    )
    overview_projection(
        anisotropic,
        output_dir,
        "anisotropic_test_systems_xz_projection",
        "Anisotropic membrane systems: XZ cross-sections",
        vmax,
        columns=4,
    )
    overview_projection(
        psd_synthetic,
        output_dir,
        "psd_double_cavity_systems_xz_projection",
        "PSD double-cavity systems: XZ cross-sections",
        vmax,
        columns=3,
    )
    overview_projection(
        datasets,
        output_dir,
        "all_test_systems_xz_projection",
        "PxPore test systems: XZ cross-sections",
        vmax,
        columns=3,
    )


def main():
    args = parse_args()
    set_style()
    generate_projections(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.long_pixels,
        args.vmax,
    )
    print(f"output={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
