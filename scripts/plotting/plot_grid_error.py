from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "grid_error"
ATOM_RADIUS_NM = 0.12
GRID_MAX_NM = 0.1

CASES = (
    {
        "label": "Single H",
        "folder": REPO_ROOT / "case" / "single_H",
        "atom_count": 1,
        "xlim": None,
        "caption":"Single H \nwithout refinement"
    },
    {
        "label": "Single H octree",
        "folder": REPO_ROOT / "case" / "single_H_octree",
        "atom_count": 1,
        "xlim": None,
        "caption":"Single H \nwith refinement"
    },
    {
        "label": "512 H jitter",
        "folder": REPO_ROOT / "case" / "512_H_jitter",
        "atom_count": 512,
        "xlim": (0.002, 0.1),
        "caption":"512 H \nwithout refinement"
    },
    {
        "label": "512 H jitter octree",
        "folder": REPO_ROOT / "case" / "512_H_jitter_octree",
        "atom_count": 512,
        "xlim": (0.002, 0.1),
        "caption":"512 H \nwith refinement"
    },
)


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (7.25, 4.6),
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "axes.grid": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "mathtext.default": "regular",
        }
    )


def load_case(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(folder / "**" / "*.json"), recursive=True)):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        info = record.get("info", {}) or {}
        stats = record.get("stats", {}) or {}
        rows.append(
            {
                "file": os.path.relpath(path, folder),
                "grid_space_nm": info.get("grid_space_nm", np.nan),
                "Vprobe_nm3": stats.get("Vprobe_nm3", np.nan),
                "execution_time": record.get("run_envs", {}).get("execution_time", np.nan),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No JSON files found under: {folder}")

    df = pd.DataFrame(rows)
    cols = ["grid_space_nm", "Vprobe_nm3", "execution_time"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return (
        df.dropna(subset=["grid_space_nm", "Vprobe_nm3"])
        .sort_values(["grid_space_nm", "Vprobe_nm3", "file"])
        .query("grid_space_nm <= @GRID_MAX_NM")
    )


def slugify(label: str) -> str:
    return label.lower().replace(" ", "_")


def plot_case(case: dict, output_dir: Path, show: bool = False) -> Path:
    error_color = "#3B6EA8"
    time_color = "#C44E52"
    df = load_case(case["folder"])
    if case["xlim"] is not None:
        x_min, x_max = case["xlim"]
        df = df[df["grid_space_nm"].between(x_min, x_max, inclusive="both")]

    ref_volume = 4 / 3 * np.pi * ATOM_RADIUS_NM**3 * case["atom_count"]
    error = (df["Vprobe_nm3"] - ref_volume).abs() / ref_volume

    fig, ax = plt.subplots()
    error_line = ax.plot(
        df["grid_space_nm"],
        error,
        marker="o",
        linestyle="-",
        color=error_color,
        label="Volume error",
        zorder=100,
    )
    ax.set(xlabel="Grid size (nm)", ylabel="$V_{error}$", xscale="log", yscale="log")
    ax.yaxis.label.set_color(error_color)
    ax.tick_params(axis="y", colors=error_color)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, which="major", axis="both", alpha=0.25, zorder=0)

    ax_time = ax.twinx()
    time_line = ax_time.plot(
        df["grid_space_nm"],
        df["execution_time"],
        marker="s",
        linestyle="--",
        color=time_color,
        label="Execution time",
        zorder=10,
    )
    ax_time.set_ylabel("Execution Time (s)")
    ax_time.yaxis.label.set_color(time_color)
    ax_time.tick_params(axis="y", colors=time_color)

    ax_time.text(0.1,0.95,case['caption'],fontsize=16,horizontalalignment='left',verticalalignment='top', transform=ax_time.transAxes)

    lines = error_line + time_line
    # ax.legend(lines, [line.get_label() for line in lines], loc="best")
    # plt.title("")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{slugify(case['label'])}.png"
    fig.savefig(output)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output


def plot_grid_error(output_dir: Path, show: bool = False) -> list[Path]:
    set_paper_style()
    return [plot_case(case, output_dir=output_dir, show=show) for case in CASES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grid-size volume error for four PxPore cases.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output figure directory.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in plot_grid_error(args.output_dir.resolve(), show=args.show):
        print(output)


if __name__ == "__main__":
    main()
