#!/usr/bin/env python3
"""Plot fill-benchmark wall and CPU times for PxPore, PB, and Zeo++."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "results/fill_benchmark_results/results.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "docs/figures/fill_benchmark_performance"

PX_COLOR = "#3380B8"
PB_COLOR = "#E06666"
ZEO_COLOR = "#61D161"
GRADIENT_STRIPS = 100

TOOLS = (
    ("pp", "PxPore", PX_COLOR),
    ("pb", "PoreBlazer", PB_COLOR),
    ("zeo", "Zeo++", ZEO_COLOR),
)
CPU_MARKERS = {"pp": "o", "pb": "s", "zeo": "^"}


def set_paper_style() -> None:
    """Match the established performance_comparison figure style."""
    mpl.rcParams.update(
        {
            "figure.figsize": (7.25, 5.95),
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
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


def _resolve_input(path: Path) -> Path:
    """Accept either the benchmark directory or one of its result CSV files."""
    if not path.is_dir():
        return path
    for name in ("results.csv", "benchmark_performance.csv"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No results.csv or benchmark_performance.csv found in {path}"
    )


def _aggregate_long_results(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Convert the current per-run results.csv layout to the plotting layout."""
    required = {"tool", "name", "fill_percent", "status", "wall_seconds"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    if "cpu_seconds" not in frame.columns:
        cpu_parts = {"user_seconds", "system_seconds"}
        missing_cpu = cpu_parts.difference(frame.columns)
        if missing_cpu:
            raise ValueError(
                f"Cannot derive CPU time from {path}; missing {sorted(missing_cpu)}"
            )
        frame["cpu_seconds"] = (
            pd.to_numeric(frame["user_seconds"], errors="coerce")
            + pd.to_numeric(frame["system_seconds"], errors="coerce")
        )

    frame = frame.copy()
    frame["fill_percent"] = pd.to_numeric(
        frame["fill_percent"], errors="coerce"
    )
    frame["wall_seconds"] = pd.to_numeric(
        frame["wall_seconds"], errors="coerce"
    )
    frame["cpu_seconds"] = pd.to_numeric(
        frame["cpu_seconds"], errors="coerce"
    )
    frame = frame.dropna(
        subset=["tool", "name", "fill_percent", "wall_seconds", "cpu_seconds"]
    )

    rows: list[dict[str, object]] = []
    for case, case_runs in frame.groupby("name", sort=False):
        row: dict[str, object] = {
            "case": case,
            "fill_percent": float(case_runs["fill_percent"].iloc[0]),
        }
        for prefix, _, _ in TOOLS:
            runs = case_runs[case_runs["tool"] == prefix]
            wall = runs["wall_seconds"]
            cpu = runs["cpu_seconds"]
            row[f"{prefix}_successes"] = int(runs["status"].eq("ok").sum())
            row[f"{prefix}_attempts"] = int(len(runs))
            row[f"{prefix}_wall_mean_s"] = wall.mean()
            row[f"{prefix}_wall_sd_s"] = wall.std(ddof=1)
            row[f"{prefix}_cpu_mean_s"] = cpu.mean()
            row[f"{prefix}_cpu_sd_s"] = cpu.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def load_results(path: Path) -> pd.DataFrame:
    path = _resolve_input(path)
    frame = pd.read_csv(path)
    if {"tool", "name", "wall_seconds"}.issubset(frame.columns):
        frame = _aggregate_long_results(frame, path)

    required = {"case"}
    for prefix, _, _ in TOOLS:
        required.update(
            {
                f"{prefix}_successes",
                f"{prefix}_attempts",
                f"{prefix}_wall_mean_s",
                f"{prefix}_wall_sd_s",
                f"{prefix}_cpu_mean_s",
                f"{prefix}_cpu_sd_s",
            }
        )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    numeric = sorted(required.difference({"case"}))
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if "fill_percent" not in frame.columns:
        frame["fill_percent"] = pd.to_numeric(
            frame["case"].str.extract(r"fill_(\d+)", expand=False),
            errors="coerce",
        )
    else:
        frame["fill_percent"] = pd.to_numeric(
            frame["fill_percent"], errors="coerce"
        )
    frame = frame.dropna(subset=["fill_percent", *numeric]).copy()
    if frame.empty:
        raise ValueError(f"No valid benchmark rows found in {path}")
    return frame.sort_values("fill_percent").reset_index(drop=True)


def gradient_bars(
    ax: plt.Axes,
    centers: np.ndarray,
    heights: np.ndarray,
    completed: np.ndarray,
    width: float,
    color: str,
    floor: float,
) -> None:
    r, g, b, _ = to_rgba(color)
    denominator = max(GRADIENT_STRIPS - 1, 1)
    for x, height, ok in zip(centers, heights, completed):
        if not ok or not np.isfinite(height) or height <= floor:
            continue
        stops = np.geomspace(floor, height, GRADIENT_STRIPS + 1)
        for index in range(GRADIENT_STRIPS):
            ax.bar(
                x,
                stops[index + 1] - stops[index],
                width,
                bottom=stops[index],
                color=(r, g, b),
                alpha=index / denominator,
                edgecolor="none",
                zorder=4,
            )


def add_error_bars(
    ax: plt.Axes,
    centers: np.ndarray,
    means: np.ndarray,
    errors: np.ndarray,
    color: str,
) -> None:
    ax.errorbar(
        centers,
        means,
        yerr=errors,
        fmt="none",
        ecolor="0.20",
        elinewidth=1.35,
        capsize=3.5,
        capthick=1.35,
        zorder=12,
    )


def add_incomplete_hatching(
    ax: plt.Axes,
    centers: np.ndarray,
    heights: np.ndarray,
    completed: np.ndarray,
    width: float,
    floor: float,
) -> None:
    for x, height, ok in zip(centers, heights, completed):
        if ok or not np.isfinite(height) or height <= floor:
            continue
        ax.bar(
            x,
            height - floor,
            width,
            bottom=floor,
            facecolor="none",
            edgecolor="0.20",
            linewidth=1.0,
            hatch="///",
            zorder=9,
        )


def plot_results(frame: pd.DataFrame, output_dir: Path, show: bool) -> list[Path]:
    positions = np.arange(len(frame), dtype=float)
    width = 0.25
    floor = 0.8
    offsets = (-width, 0.0, width)
    fig, ax = plt.subplots(figsize=(7.25, 5.95))

    maximum = 0.0
    for metric in ("wall", "cpu"):
        for prefix, _, _ in TOOLS:
            mean = frame[f"{prefix}_{metric}_mean_s"].to_numpy(float)
            sd = frame[f"{prefix}_{metric}_sd_s"].to_numpy(float)
            maximum = max(maximum, float(np.nanmax(mean + sd)))

    for offset, (prefix, _, color) in zip(offsets, TOOLS):
        centers = positions + offset
        wall_means = frame[f"{prefix}_wall_mean_s"].to_numpy(float)
        wall_errors = frame[f"{prefix}_wall_sd_s"].to_numpy(float)
        cpu_means = frame[f"{prefix}_cpu_mean_s"].to_numpy(float)
        cpu_errors = frame[f"{prefix}_cpu_sd_s"].to_numpy(float)
        completed = (
            frame[f"{prefix}_successes"].to_numpy(float)
            == frame[f"{prefix}_attempts"].to_numpy(float)
        ) & (frame[f"{prefix}_attempts"].to_numpy(float) > 0)

        gradient_bars(
            ax,
            centers,
            wall_means,
            np.ones(len(frame), dtype=bool),
            width,
            color,
            floor,
        )
        add_incomplete_hatching(
            ax, centers, wall_means, completed, width, floor
        )
        add_error_bars(ax, centers, wall_means, wall_errors, color)
        ax.errorbar(
            centers,
            cpu_means,
            yerr=cpu_errors,
            color=color,
            linestyle="-",
            linewidth=1.6,
            marker=CPU_MARKERS[prefix],
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2,
            ecolor="0.20",
            elinewidth=1.35,
            capsize=3.5,
            capthick=1.35,
            zorder=10,
        )

    ax.set_yscale("log")
    ax.set_ylim(floor, maximum * 15.0)
    ax.set_xlim(-0.65, len(frame) - 0.35)
    ax.set_xticks(positions)
    ax.set_xticklabels(frame["fill_percent"].astype(int))
    ax.set_xlabel("Lattice Site Occupancy (%)")
    ax.set_ylabel("Time (s)")
    ax.grid(True, which="major", axis="y", alpha=0.25, zorder=0)
    ax.grid(
        True, which="minor", axis="y", alpha=0.10,
        linestyle="--", zorder=0,
    )
    for threshold, label in (
        (60.0, "minutes"),
        (3600.0, "hours"),
        (86400.0, "days"),
    ):
        ax.axhline(
            threshold,
            color="0.35",
            linewidth=1.1,
            linestyle=(0, (5, 4)),
            alpha=0.75,
            zorder=2,
        )
        ax.text(
            0.992,
            threshold,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=15,
            color="0.25",
            zorder=13,
        )

    tool_handles = [
        Patch(facecolor=color, label=label) for _, label, color in TOOLS
    ]
    cpu_legend_handle = ax.errorbar(
        [np.nan],
        [np.nan],
        yerr=[1.0],
        color="0.25",
        ecolor="0.20",
        linestyle="-",
        linewidth=1.6,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor="0.25",
        elinewidth=1.35,
        capsize=3.5,
        capthick=1.35,
        label="CPU time",
    )
    style_handles = [
        Patch(facecolor="0.65", label="Wall time"),
        cpu_legend_handle,
        Patch(
            facecolor="white",
            edgecolor="0.20",
            hatch="///",
            label="Incomplete",
        ),
    ]
    tool_legend = ax.legend(
        handles=tool_handles,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.995),
        ncol=3,
        frameon=False,
        borderaxespad=0.7,
        handlelength=1.35,
        labelspacing=0.35,
        columnspacing=0.8,
        handletextpad=0.45,
    )
    ax.add_artist(tool_legend)
    ax.legend(
        handles=style_handles,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.905),
        ncol=3,
        frameon=False,
        borderaxespad=0.7,
        handlelength=1.6,
        labelspacing=0.35,
        columnspacing=0.8,
        handletextpad=0.45,
    )
    fig.tight_layout(pad=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "fill_benchmark_wall_cpu_time.png",
        output_dir / "fill_benchmark_wall_cpu_time.pdf",
    ]
    for output in outputs:
        fig.savefig(output)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_paper_style()
    outputs = plot_results(
        load_results(args.input.resolve()), args.out_dir.resolve(), args.show
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
