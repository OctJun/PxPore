from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "scaling_results" / "scaling_summary.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "scaling"

LEFT_COLOR = "#3B6EA8"
RIGHT_COLOR = "#C44E52"

METRICS = {
    "wall_time": {
        "column": "wall_clock_s",
        "label": "Wall Time (s)",
        "marker": "o",
    },
    "cpu_time": {
        "column": "cpu_time_s",
        "label": "CPU Time (s)",
        "marker": "s",
    },
}

METRIC_PAIRS = (("wall_time", "cpu_time"),)


def set_paper_style() -> None:
    """Apply the plotting style used by plot_grid_error.py."""
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


def load_scaling_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"threads", "wall_clock_s", "user_s", "system_s"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    columns = sorted(required)
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=columns).copy()
    df = df[df["threads"] > 0]
    if df.empty:
        raise ValueError(f"No valid scaling rows found in {csv_path}")
    if df["threads"].duplicated().any():
        duplicates = sorted(df.loc[df["threads"].duplicated(False), "threads"].unique())
        raise ValueError(f"Duplicate thread counts in {csv_path}: {duplicates}")

    df["cpu_time_s"] = df["user_s"] + df["system_s"]
    return df.sort_values("threads")


def plot_metric_pair(
    df: pd.DataFrame,
    left_metric: str,
    right_metric: str,
    output_dir: Path,
    show: bool = False,
) -> Path:
    left = METRICS[left_metric]
    right = METRICS[right_metric]

    fig, ax = plt.subplots()
    left_line = ax.plot(
        df["threads"],
        df[left["column"]],
        marker=left["marker"],
        linestyle="-",
        color=LEFT_COLOR,
        label=left["label"],
        zorder=100,
    )
    ax.set_xlabel("Threads")
    ax.set_ylabel(left["label"], color=LEFT_COLOR)
    ax.tick_params(axis="y", colors=LEFT_COLOR)
    tick_step = 8 if df["threads"].max() >= 32 else 4
    thread_ticks = [1] + list(
        range(tick_step, int(df["threads"].max()) + 1, tick_step)
    )
    ax.set_xticks(thread_ticks)
    ax.grid(True, which="major", axis="both", alpha=0.25, zorder=0)

    ax_right = ax.twinx()
    right_line = ax_right.plot(
        df["threads"],
        df[right["column"]],
        marker=right["marker"],
        linestyle="--",
        color=RIGHT_COLOR,
        label=right["label"],
        zorder=10,
    )
    ax_right.set_ylabel(right["label"], color=RIGHT_COLOR)
    ax_right.tick_params(axis="y", colors=RIGHT_COLOR)

    lines = left_line + right_line
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"threads_{left_metric}_vs_{right_metric}.png"
    fig.savefig(output)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


def plot_speedup_and_efficiency(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = False,
) -> Path:
    """Plot wall-time speedup and strong-scaling parallel efficiency."""
    baseline_rows = df.loc[df["threads"] == 1, "wall_clock_s"]
    if baseline_rows.empty:
        raise ValueError("A 1-thread row is required to calculate parallel efficiency")

    threads = df["threads"]
    speedup = baseline_rows.iloc[0] / df["wall_clock_s"]
    efficiency_pct = 100.0 * speedup / threads

    fig, ax = plt.subplots()
    speedup_line = ax.plot(
        threads,
        speedup,
        linestyle="-",
        color=LEFT_COLOR,
        label="Measured Speedup",
        zorder=100,
    )
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup (compared to single threads)", color=LEFT_COLOR)
    ax.tick_params(axis="y", colors=LEFT_COLOR)
    max_speedup_tick = int(speedup.max()) + 1
    ax.set_yticks(range(1, max_speedup_tick + 1))
    ax.set_ylim(0.8, max_speedup_tick + 0.2)

    tick_step = 8 if threads.max() >= 32 else 4
    thread_ticks = [1] + list(range(tick_step, int(threads.max()) + 1, tick_step))
    ax.set_xticks(thread_ticks)
    ax.grid(True, which="major", axis="both", alpha=0.25, zorder=0)

    ax_right = ax.twinx()
    efficiency_line = ax_right.plot(
        threads,
        efficiency_pct,
        linestyle="--",
        color=RIGHT_COLOR,
        label="Parallel Efficiency",
        zorder=80,
    )
    ax_right.set_ylabel("Parallel Efficiency (%)", color=RIGHT_COLOR)
    ax_right.tick_params(axis="y", colors=RIGHT_COLOR)
    ax_right.set_ylim(0, 105)

    lines = speedup_line + efficiency_line
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "threads_speedup_vs_efficiency.png"
    fig.savefig(output)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


def plot_scaling_results(
    csv_path: Path, output_dir: Path, show: bool = False
) -> list[Path]:
    set_paper_style()
    df = load_scaling_results(csv_path)
    outputs = [
        plot_metric_pair(df, left, right, output_dir=output_dir, show=show)
        for left, right in METRIC_PAIRS
    ]
    outputs.append(plot_speedup_and_efficiency(df, output_dir=output_dir, show=show))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot timing, speedup, and parallel-efficiency scaling metrics."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Scaling summary CSV."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Figure directory."
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in plot_scaling_results(
        args.input.resolve(), args.output_dir.resolve(), show=args.show
    ):
        print(output)


if __name__ == "__main__":
    main()
