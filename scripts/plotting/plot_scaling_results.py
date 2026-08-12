#!/usr/bin/env python3
"""Plot original and repeated-run PxPore strong-scaling results."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEW_RESULT_DIR = REPO_ROOT / "results/scaling_results"
DEFAULT_LEGACY_INPUT = DEFAULT_NEW_RESULT_DIR / "scaling_summary.csv"
DEFAULT_PERIODIC_INPUT = (
    DEFAULT_NEW_RESULT_DIR / "periodic_any/scaling_summary.csv"
)
DEFAULT_LEGACY_MC_INPUT = DEFAULT_NEW_RESULT_DIR / "psd_mc/scaling_summary.csv"
DEFAULT_PERIODIC_MC_INPUT = (
    DEFAULT_NEW_RESULT_DIR / "periodic_any/psd_mc/scaling_summary.csv"
)
DEFAULT_ORIGINAL_INPUT = DEFAULT_NEW_RESULT_DIR / "original/scaling_summary.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/figures/scaling"

WALL_COLOR = "#3B6EA8"
CPU_COLOR = "#4C956C"
SPEEDUP_COLOR = "#C44E52"
ORIGINAL_CPU_COLOR = "#C44E52"
ALGORITHM_STYLES = {"Legacy": "-", "Periodic": "--"}
PSD_STYLES = {"Centers": "-", "Monte Carlo": "--"}
TIME_YMAX_S = 140.0
TIME_TOP_PADDING = 0.08


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
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
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


def _load_numeric(csv_path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Columns missing in {csv_path}: {sorted(missing)}")
    numeric = sorted(required)
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=numeric).copy()
    frame = frame[
        (frame["threads"] > 0)
        & (frame["wall_clock_s"] > 0)
        & (frame["cpu_time_s"] > 0)
    ]
    if frame.empty:
        raise ValueError(f"No valid scaling rows found in {csv_path}")
    if frame["threads"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["threads"].duplicated(False), "threads"].unique()
        )
        raise ValueError(f"Duplicate thread counts in {csv_path}: {duplicates}")
    return frame.sort_values("threads").reset_index(drop=True)


def load_repeated_results(
    csv_path: Path,
    expected_repeats: int = 3,
    allow_partial: bool = False,
) -> pd.DataFrame:
    frame = _load_numeric(
        csv_path,
        {
            "threads",
            "repeats",
            "wall_clock_s",
            "wall_clock_s_sd",
            "cpu_time_s",
            "cpu_time_s_sd",
        },
    )
    if allow_partial:
        return frame
    completed = frame[frame["repeats"] >= expected_repeats].copy()
    if completed.empty:
        raise ValueError(
            f"No thread points with at least {expected_repeats} repeats "
            f"found in {csv_path}"
        )
    omitted = len(frame) - len(completed)
    if omitted:
        print(
            f"Ignoring {omitted} incomplete thread point(s) in {csv_path}; "
            f"required repeats={expected_repeats}"
        )
    return completed.reset_index(drop=True)


def align_common_threads(
    suites: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    common = set.intersection(
        *(set(frame["threads"].astype(int)) for frame in suites.values())
    )
    if not common:
        raise ValueError("Compared scaling suites have no common thread counts")
    if 1 not in common:
        raise ValueError("Compared scaling suites require a common 1-thread row")
    return {
        name: frame[frame["threads"].astype(int).isin(common)]
        .sort_values("threads")
        .reset_index(drop=True)
        for name, frame in suites.items()
    }


def load_original_results(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "cpu_time_s" not in frame:
        required_cpu = {"user_s", "system_s"}
        missing = required_cpu.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Cannot derive CPU time from {csv_path}; missing {sorted(missing)}"
            )
        frame["cpu_time_s"] = frame["user_s"] + frame["system_s"]
    required = {"threads", "wall_clock_s", "cpu_time_s"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Columns missing in {csv_path}: {sorted(missing)}")
    frame[list(required)] = frame[list(required)].apply(
        pd.to_numeric, errors="coerce"
    )
    frame = frame.dropna(subset=list(required))
    return frame.sort_values("threads").reset_index(drop=True)


def calculate_speedup(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    baseline = frame.loc[frame["threads"] == 1]
    if len(baseline) != 1:
        raise ValueError("Exactly one 1-thread row is required for speedup")

    wall = frame["wall_clock_s"].to_numpy(float)
    wall_sd = frame["wall_clock_s_sd"].to_numpy(float)
    baseline_wall = float(baseline.iloc[0]["wall_clock_s"])
    baseline_sd = float(baseline.iloc[0]["wall_clock_s_sd"])
    speedup = baseline_wall / wall

    # First-order propagation for S_p = T_1 / T_p. S_1 is exactly 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        speedup_sd = speedup * np.sqrt(
            (baseline_sd / baseline_wall) ** 2 + (wall_sd / wall) ** 2
        )
    speedup_sd[frame["threads"].to_numpy(int) == 1] = 0.0
    return speedup, speedup_sd


def add_band(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    color: str,
) -> None:
    ax.fill_between(
        x,
        np.maximum(mean - sd, 0.0),
        mean + sd,
        color=color,
        alpha=0.14,
        linewidth=0,
        zorder=2,
    )


def plot_original_time(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Recreate and preserve the wall/CPU plot used before repeat support."""
    threads = frame["threads"].to_numpy(float)
    fig, wall_ax = plt.subplots()
    wall_line = wall_ax.plot(
        threads,
        frame["wall_clock_s"],
        color=WALL_COLOR,
        marker="o",
        linewidth=1.8,
        label="Wall time",
    )[0]
    wall_ax.set_xlabel("Threads")
    wall_ax.set_ylabel("Wall time (s)", color=WALL_COLOR)
    wall_ax.tick_params(axis="y", colors=WALL_COLOR)
    wall_ax.grid(True, alpha=0.25)

    cpu_ax = wall_ax.twinx()
    cpu_line = cpu_ax.plot(
        threads,
        frame["cpu_time_s"],
        color=ORIGINAL_CPU_COLOR,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label="CPU time",
    )[0]
    cpu_ax.set_ylabel("CPU time (s)", color=ORIGINAL_CPU_COLOR)
    cpu_ax.tick_params(axis="y", colors=ORIGINAL_CPU_COLOR)
    wall_ax.legend(handles=[wall_line, cpu_line], loc="upper center")
    fig.tight_layout()

    output = output_dir / "threads_wall_time_vs_cpu_time_original.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_original_speedup(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Recreate and preserve the original speedup/efficiency companion plot."""
    threads = frame["threads"].to_numpy(float)
    baseline = float(frame.loc[frame["threads"] == 1, "wall_clock_s"].iloc[0])
    speedup = baseline / frame["wall_clock_s"].to_numpy(float)
    efficiency = 100.0 * speedup / threads

    fig, speed_ax = plt.subplots()
    speed_line = speed_ax.plot(
        threads,
        speedup,
        color=WALL_COLOR,
        linewidth=1.8,
        label="Speedup",
    )[0]
    speed_ax.set_xlabel("Threads")
    speed_ax.set_ylabel("Speedup", color=WALL_COLOR)
    speed_ax.tick_params(axis="y", colors=WALL_COLOR)
    speed_ax.grid(True, alpha=0.25)

    efficiency_ax = speed_ax.twinx()
    efficiency_line = efficiency_ax.plot(
        threads,
        efficiency,
        color=ORIGINAL_CPU_COLOR,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label="Parallel efficiency",
    )[0]
    efficiency_ax.set_ylabel(
        "Parallel efficiency (%)", color=ORIGINAL_CPU_COLOR
    )
    efficiency_ax.tick_params(axis="y", colors=ORIGINAL_CPU_COLOR)
    speed_ax.legend(
        handles=[speed_line, efficiency_line], loc="upper center"
    )
    fig.tight_layout()

    output = output_dir / "threads_speedup_vs_efficiency_original.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_scaling_suites(
    suites: dict[str, pd.DataFrame],
    output_dir: Path,
    filename: str,
    title: str | None = None,
    suite_styles: dict[str, str] | None = None,
    show: bool = False,
) -> Path:
    """Plot complete time and speedup twin axes for one or more suites."""
    all_threads = np.concatenate(
        [frame["threads"].to_numpy(float) for frame in suites.values()]
    )
    fig, time_ax = plt.subplots()
    speed_ax = time_ax.twinx()

    speed_bounds: list[np.ndarray] = []
    suite_styles = suite_styles or ALGORITHM_STYLES
    for algorithm, frame in suites.items():
        threads = frame["threads"].to_numpy(float)
        style = suite_styles[algorithm]
        wall = frame["wall_clock_s"].to_numpy(float)
        wall_sd = frame["wall_clock_s_sd"].to_numpy(float)
        cpu = frame["cpu_time_s"].to_numpy(float)
        cpu_sd = frame["cpu_time_s_sd"].to_numpy(float)
        speedup, speedup_sd = calculate_speedup(frame)

        time_ax.plot(
            threads,
            wall,
            color=WALL_COLOR,
            linestyle=style,
            linewidth=1.9,
            zorder=10,
        )
        add_band(time_ax, threads, wall, wall_sd, WALL_COLOR)
        time_ax.plot(
            threads,
            cpu,
            color=CPU_COLOR,
            linestyle=style,
            linewidth=1.9,
            zorder=9,
        )
        add_band(time_ax, threads, cpu, cpu_sd, CPU_COLOR)
        speed_ax.plot(
            threads,
            speedup,
            color=SPEEDUP_COLOR,
            linestyle=style,
            linewidth=1.9,
            zorder=8,
        )
        add_band(speed_ax, threads, speedup, speedup_sd, SPEEDUP_COLOR)

        speed_bounds.extend([speedup - speedup_sd, speedup + speedup_sd])

    time_ax.set_xlabel("Threads")
    time_ax.set_ylabel("Time (s)")
    speed_ax.set_ylabel("Speedup", color=SPEEDUP_COLOR)
    speed_ax.yaxis.set_label_position("right")
    speed_ax.yaxis.tick_right()
    speed_ax.tick_params(axis="y", colors=SPEEDUP_COLOR)
    speed_ax.spines["right"].set_color(SPEEDUP_COLOR)
    if title and len(suites) > 1:
        time_ax.set_title(title)
    tick_step = 8 if all_threads.max() >= 32 else 4
    thread_ticks = [1] + list(
        range(tick_step, int(all_threads.max()) + 1, tick_step)
    )
    time_ax.set_xticks(thread_ticks)
    x_span = float(all_threads.max() - all_threads.min())
    x_padding = max(1.0, 0.035 * x_span)
    time_ax.set_xlim(all_threads.min() - x_padding, all_threads.max() + x_padding)

    time_ax.set_ylim(0.0, TIME_YMAX_S * (1.0 + TIME_TOP_PADDING))
    speed_values = np.concatenate(speed_bounds)
    speed_low, speed_high = float(speed_values.min()), float(speed_values.max())
    speed_span = max(
        speed_high - speed_low,
        0.05 * max(abs(speed_high), 1.0),
    )
    speed_pad = 0.15 * speed_span
    speed_ax.set_ylim(max(0.0, speed_low - speed_pad), speed_high + speed_pad)

    time_ax.grid(True, which="major", alpha=0.22, zorder=0)

    metric_handles = [
        Line2D([0], [0], color=WALL_COLOR, linewidth=2.2, label="Wall time"),
        Line2D([0], [0], color=CPU_COLOR, linewidth=2.2, label="CPU time"),
        Line2D(
            [0], [0], color=SPEEDUP_COLOR, linewidth=2.2, label="Speedup"
        ),
    ]
    metric_legend_y = 0.99 if len(suites) > 1 else 1.0
    metric_legend = time_ax.legend(
        handles=metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, metric_legend_y),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.45,
    )
    if len(suites) > 1:
        time_ax.add_artist(metric_legend)
        algorithm_handles = [
            Line2D(
                [0],
                [0],
                color="0.25",
                linewidth=2.2,
                linestyle=suite_styles[algorithm],
                label=algorithm,
            )
            for algorithm in suites
        ]
        time_ax.legend(
            handles=algorithm_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=len(algorithm_handles),
            columnspacing=1.1,
            handlelength=1.7,
            handletextpad=0.5,
        )

    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.12, top=0.91)
    output = output_dir / filename
    fig.savefig(output)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


def plot_algorithm_comparison(
    legacy: pd.DataFrame,
    periodic: pd.DataFrame,
    output_dir: Path,
    show: bool = False,
) -> Path:
    return plot_scaling_suites(
        {"Legacy": legacy, "Periodic": periodic},
        output_dir,
        "threads_wall_time_vs_cpu_time.png",
        show=show,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-input",
        type=Path,
        default=DEFAULT_LEGACY_INPUT,
        help="Repeated-run legacy scaling_summary.csv",
    )
    parser.add_argument(
        "--periodic-input",
        type=Path,
        default=DEFAULT_PERIODIC_INPUT,
        help="Repeated-run periodic-any scaling_summary.csv",
    )
    parser.add_argument(
        "--original-input",
        type=Path,
        default=DEFAULT_ORIGINAL_INPUT,
        help="Original single-run scaling_summary.csv",
    )
    parser.add_argument(
        "--legacy-mc-input",
        type=Path,
        default=DEFAULT_LEGACY_MC_INPUT,
        help="Legacy-connectivity Monte Carlo PSD scaling summary",
    )
    parser.add_argument(
        "--periodic-mc-input",
        type=Path,
        default=DEFAULT_PERIODIC_MC_INPUT,
        help="Periodic-connectivity Monte Carlo PSD scaling summary",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Figure directory",
    )
    parser.add_argument(
        "--expected-repeats",
        type=int,
        default=3,
        help="Only plot points with at least this many repeats (default: 3)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Include thread points that have not reached --expected-repeats",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_repeats < 1:
        raise ValueError("--expected-repeats must be positive")
    set_paper_style()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    original_path = args.original_input.resolve()
    if original_path.is_file():
        original = load_original_results(original_path)
        outputs.extend(
            [
                plot_original_time(original, output_dir),
                plot_original_speedup(original, output_dir),
            ]
        )
    else:
        print(f"Skipping original figures; summary not found: {original_path}")

    suite_paths = {
        ("Legacy", "Centers"): args.legacy_input.resolve(),
        ("Legacy", "Monte Carlo"): args.legacy_mc_input.resolve(),
        ("Periodic", "Centers"): args.periodic_input.resolve(),
        ("Periodic", "Monte Carlo"): args.periodic_mc_input.resolve(),
    }
    suite_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for suite, path in suite_paths.items():
        name = " + ".join(suite)
        if path.is_file():
            try:
                suite_frames[suite] = load_repeated_results(
                    path,
                    expected_repeats=args.expected_repeats,
                    allow_partial=args.allow_partial,
                )
            except ValueError as error:
                print(f"Skipping {name}: {error}")
        else:
            print(f"Skipping {name}; summary not found: {path}")

    individual_names = {
        ("Legacy", "Centers"): "threads_time_speedup_legacy_centers.png",
        ("Legacy", "Monte Carlo"): "threads_time_speedup_legacy_mc.png",
        ("Periodic", "Centers"): "threads_time_speedup_periodic_centers.png",
        ("Periodic", "Monte Carlo"): "threads_time_speedup_periodic_mc.png",
    }
    for suite, frame in suite_frames.items():
        label = " + ".join(suite)
        outputs.append(
            plot_scaling_suites(
                {label: frame},
                output_dir,
                individual_names[suite],
                title=label,
                suite_styles={label: "-"},
            )
        )

    comparison_specs = (
        (
            (("Legacy", "Centers"), ("Periodic", "Centers")),
            ("Legacy", "Periodic"),
            ALGORITHM_STYLES,
            "Connectivity comparison: Centers PSD",
            "threads_time_speedup_connectivity_centers.png",
        ),
        (
            (("Legacy", "Monte Carlo"), ("Periodic", "Monte Carlo")),
            ("Legacy", "Periodic"),
            ALGORITHM_STYLES,
            "Connectivity comparison: Monte Carlo PSD",
            "threads_time_speedup_connectivity_mc.png",
        ),
        (
            (("Legacy", "Centers"), ("Legacy", "Monte Carlo")),
            ("Centers", "Monte Carlo"),
            PSD_STYLES,
            "Legacy connectivity: PSD methods",
            "threads_time_speedup_psd_legacy.png",
        ),
        (
            (("Periodic", "Centers"), ("Periodic", "Monte Carlo")),
            ("Centers", "Monte Carlo"),
            PSD_STYLES,
            "Periodic connectivity: PSD methods",
            "threads_time_speedup_psd_periodic.png",
        ),
    )
    for keys, labels, styles, title, filename in comparison_specs:
        if not all(key in suite_frames for key in keys):
            continue
        aligned = align_common_threads(
            {
                label: suite_frames[key]
                for key, label in zip(keys, labels, strict=True)
            }
        )
        output = plot_scaling_suites(
            aligned,
            output_dir,
            filename,
            title=title,
            suite_styles=styles,
        )
        outputs.append(output)

    aliases = {
        "threads_time_speedup_legacy_centers.png":
            "threads_time_speedup_legacy.png",
        "threads_time_speedup_periodic_centers.png":
            "threads_time_speedup_periodic.png",
        "threads_time_speedup_connectivity_centers.png":
            "threads_wall_time_vs_cpu_time.png",
    }
    by_name = {output.name: output for output in outputs}
    for source_name, alias_name in aliases.items():
        source = by_name.get(source_name)
        if source is None:
            continue
        alias = output_dir / alias_name
        shutil.copy2(source, alias)
        outputs.append(alias)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
