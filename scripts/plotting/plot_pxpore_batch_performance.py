#!/usr/bin/env python3
"""Plot 8-process/8-thread versus 1-process/64-thread batch performance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SYSTEMS = ("APC-DAP", "APC-DAL", "APC-MAP")
CONFIGS = ("8p8t", "1p64t")
SYSTEM_COLORS = {
    "APC-DAP": "#1f77b4",
    "APC-DAL": "#d62728",
    "APC-MAP": "#2ca02c",
}
CONFIG_STYLES = {
    "8p8t": {"alpha": 0.90, "hatch": ""},
    "1p64t": {"alpha": 0.45, "hatch": "///"},
}


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (7.25, 5.95),
            "figure.dpi": 140,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 24,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "axes.linewidth": 1.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "axes.grid": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "mathtext.default": "regular",
        }
    )


def load_rows(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["system"], row["config"])] = {
                "wall_clock_s": float(row["wall_clock_s"]),
                "frames_per_hour": float(row["frames_per_hour"]),
            }
    missing = {(system, config) for system in SYSTEMS for config in CONFIGS} - rows.keys()
    if missing:
        raise ValueError(f"Missing performance rows: {sorted(missing)}")
    return rows


def plot(rows: dict[tuple[str, str], dict[str, float]], output: Path) -> None:
    x = np.arange(len(SYSTEMS), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2)
    for offset, config in zip((-0.5, 0.5), CONFIGS):
        wall = [rows[(system, config)]["wall_clock_s"] / 3600.0 for system in SYSTEMS]
        rate = [rows[(system, config)]["frames_per_hour"] for system in SYSTEMS]
        style = CONFIG_STYLES[config]
        colors = [SYSTEM_COLORS[system] for system in SYSTEMS]
        for ax, values in zip(axes, (wall, rate)):
            ax.bar(
                x + offset * width,
                values,
                width,
                color=colors,
                edgecolor=colors,
                linewidth=1.0,
                alpha=style["alpha"],
                hatch=style["hatch"],
            )
    axes[0].set_ylabel("Batch wall time (h)")
    axes[1].set_ylabel("Throughput (frames/h)")
    config_legend = [
        Patch(
            facecolor="0.45",
            edgecolor="0.25",
            alpha=CONFIG_STYLES[config]["alpha"],
            hatch=CONFIG_STYLES[config]["hatch"],
            label=config,
        )
        for config in CONFIGS
    ]
    for ax in axes:
        ax.set_xticks(x, SYSTEMS)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(handles=config_legend)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    set_paper_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    csv_path = args.csv.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else csv_path.with_suffix(".png")
    plot(load_rows(csv_path), output)
    print(output)


if __name__ == "__main__":
    main()
