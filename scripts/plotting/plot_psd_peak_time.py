#!/usr/bin/env python3
"""Plot the main PxPore PSD peak position versus MD time."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_ROOT = REPO_ROOT / "pxpore_apc_psd_time_48ps"
SYSTEMS = ("APC-DAP", "APC-MAP")
DISPLAY_LABELS = {
    "APC-DAP": "DAP",
    "APC-MAP": "MAP",
}
COLORS = {
    "APC-DAP": "#1f77b4",
    "APC-MAP": "#2ca02c",
}
FRAME_RE = re.compile(r"frame_(\d+)$")
TIME_RE = re.compile(r"\bt\s*=\s*([0-9.+\-Ee]+)")


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


@dataclass(frozen=True)
class Peak:
    system: str
    frame: int
    time_ps: float
    position_nm: float
    height: float
    vacc_nm3: float
    vacc_frac: float
    source: Path


@dataclass(frozen=True)
class PSDFrame:
    system: str
    frame: int
    time_ps: float
    diameter_nm: np.ndarray
    volume_fraction: np.ndarray


def numeric_rows(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(item) for item in line.split()])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric PSD data in {path}")
    return np.asarray(rows, dtype=float)


def load_psd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = numeric_rows(path)
    if values.shape[1] < 5:
        raise ValueError(f"Expected 5 standard PSD columns in {path}")

    diameter_nm = values[:, 1]
    # Standard PxPore PSD columns: N, diameter_nm, count, volume, cumulative.
    # The constant bin width means volume and volume density have the same peak.
    density = values[:, 3]
    valid = np.isfinite(diameter_nm) & np.isfinite(density) & (density >= 0)
    diameter_nm, density = diameter_nm[valid], density[valid]
    order = np.argsort(diameter_nm)
    diameter_nm, density = diameter_nm[order], density[order]
    if len(diameter_nm) < 3 or float(np.max(density)) <= 0:
        raise ValueError(f"Insufficient positive PSD data in {path}")
    return diameter_nm, density


def main_peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return a continuous peak from a parabola through the highest three bins."""
    peak_index = int(np.argmax(y))
    if peak_index == 0 or peak_index == len(x) - 1:
        return float(x[peak_index]), float(y[peak_index])
    coefficients = np.polyfit(x[peak_index - 1 : peak_index + 2], y[peak_index - 1 : peak_index + 2], 2)
    curvature, slope, _ = coefficients
    if curvature >= 0:
        return float(x[peak_index]), float(y[peak_index])
    position = float(-slope / (2.0 * curvature))
    position = float(np.clip(position, x[peak_index - 1], x[peak_index + 1]))
    return position, float(np.polyval(coefficients, position))


def gro_time(path: Path) -> float:
    with path.open(encoding="utf-8", errors="replace") as handle:
        title = handle.readline()
    match = TIME_RE.search(title)
    if not match:
        raise ValueError(f"No 't=' time in GRO title: {path}")
    return float(match.group(1))


def collect_system(
    work_root: Path, system: str, result_dir_name: str
) -> tuple[list[Peak], list[PSDFrame]]:
    result_root = work_root / system / result_dir_name
    if not result_root.is_dir():
        raise FileNotFoundError(f"PxPore result directory not found: {result_root}")

    peaks: list[Peak] = []
    frames: list[PSDFrame] = []
    for frame_dir in result_root.glob("frame_*"):
        match = FRAME_RE.fullmatch(frame_dir.name)
        if not match:
            continue
        frame = int(match.group(1))
        gro = frame_dir / f"frame_{frame}.gro"
        candidates = sorted(
            path for path in frame_dir.glob("*_psd.txt")
            if not path.name.endswith("_voxel_mc_psd.txt")
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one standard PSD in {frame_dir}, found {len(candidates)}"
            )
        x, y = load_psd(candidates[0])
        stats_candidates = sorted(frame_dir.glob("*_stats.json"))
        if len(stats_candidates) != 1:
            raise RuntimeError(
                f"Expected one stats JSON in {frame_dir}, found {len(stats_candidates)}"
            )
        stats = json.loads(stats_candidates[0].read_text(encoding="utf-8"))["stats"]
        time_ps = gro_time(gro)
        position, height = main_peak(x, y)
        peaks.append(
            Peak(
                system, frame, time_ps, position, height,
                float(stats["Vacc_nm3"]), float(stats["Vacc_frac"]), candidates[0],
            )
        )
        frames.append(PSDFrame(system, frame, time_ps, x, y))
    if not peaks:
        raise FileNotFoundError(f"No completed frame results under {result_root}")
    order = lambda item: (item.time_ps, item.frame)
    return sorted(peaks, key=order), sorted(frames, key=order)


def rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    half = window // 2
    means = np.empty_like(values, dtype=float)
    stds = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        sample = values[max(0, index - half) : min(len(values), index + half + 1)]
        means[index] = float(np.mean(sample))
        stds[index] = float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0
    return means, stds


def write_csv(peaks: list[Peak], output: Path, window: int) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "system", "frame", "time_ps", "peak_nm", "peak_smooth_nm", "peak_std_nm",
            "peak_density", "Vacc_nm3", "Vacc_frac", "Vacc_frac_smooth", "Vacc_frac_std",
            "rolling_window_frames", "source",
        ))
        for system in SYSTEMS:
            series = [peak for peak in peaks if peak.system == system]
            peak_mean, peak_std = rolling_mean_std(
                np.asarray([peak.position_nm for peak in series]), window
            )
            vacc_mean, vacc_std = rolling_mean_std(
                np.asarray([peak.vacc_frac for peak in series]), window
            )
            for index, peak in enumerate(series):
                writer.writerow((
                    peak.system, peak.frame, peak.time_ps, peak.position_nm,
                    peak_mean[index], peak_std[index], peak.height, peak.vacc_nm3,
                    peak.vacc_frac, vacc_mean[index], vacc_std[index], window, peak.source,
                ))


def plot_smoothed_series(
    peaks: list[Peak], output_dir: Path, window: int, *, metric: str
) -> None:
    fig, ax = plt.subplots()
    for system in SYSTEMS:
        series = [peak for peak in peaks if peak.system == system]
        time_ns = np.asarray([peak.time_ps for peak in series]) / 1000.0
        values = np.asarray([
            peak.position_nm if metric == "peak" else peak.vacc_frac for peak in series
        ])
        mean, std = rolling_mean_std(values, window)
        color = COLORS[system]
        ax.fill_between(time_ns, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
        ax.plot(
            time_ns, mean, color=color, linewidth=2.0,
            label=DISPLAY_LABELS[system],
        )

    ax.set_xlabel("Time (ns)")
    if metric == "peak":
        ax.set_ylabel("PSD peak (nm)")
        stem = "psd_peak_vs_time"
    else:
        ax.set_ylabel(r"$F_{V,acc}$")
        stem = "vacc_vs_time"
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.legend(ncol=2, loc="best", columnspacing=1.4)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
    plt.close(fig)


def heatmap_matrix(frames: list[PSDFrame], grid: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(frames), len(grid)), dtype=float)
    lookup = {float(value): index for index, value in enumerate(grid)}
    for row, frame in enumerate(frames):
        for diameter, value in zip(frame.diameter_nm, frame.volume_fraction):
            matrix[row, lookup[float(diameter)]] = value
    return matrix


def plot_heatmaps(frames: list[PSDFrame], output_dir: Path) -> None:
    grid = np.asarray(sorted({float(x) for frame in frames for x in frame.diameter_nm}))
    matrices = {
        system: heatmap_matrix([frame for frame in frames if frame.system == system], grid)
        for system in SYSTEMS
    }
    positive = np.concatenate([matrix[matrix > 0] for matrix in matrices.values()])
    vmax = float(np.percentile(positive, 99.5))
    fig, axes = plt.subplots(
        len(SYSTEMS), 1, sharex=True, sharey=True, layout="constrained",
    )
    image = None
    for ax, system in zip(axes, SYSTEMS):
        series = [frame for frame in frames if frame.system == system]
        time_ns = np.asarray([frame.time_ps for frame in series]) / 1000.0
        image = ax.pcolormesh(
            time_ns,
            grid,
            matrices[system].T,
            shading="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_ylabel("Diameter (nm)")
        ax.set_title(DISPLAY_LABELS[system], loc="left")
    axes[-1].set_xlabel("Time (ns)")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, pad=0.02, shrink=0.92)
    colorbar.set_label("PSD volume fraction per bin")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"psd_time_heatmap.{suffix}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the PxPore main PSD peak position versus time."
    )
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--result-dir-name",
        default="pxpore_8p8t",
        help="Per-system PxPore result directory used for the PSD plot.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=11,
        help="Odd centered rolling window in frames; default 11 = 528 ps for 48 ps frames.",
    )
    return parser.parse_args()


def main() -> None:
    set_paper_style()
    args = parse_args()
    if args.rolling_window < 3 or args.rolling_window % 2 == 0:
        raise ValueError("--rolling-window must be an odd integer >= 3")
    work_root = args.work_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else work_root / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    peaks: list[Peak] = []
    frames: list[PSDFrame] = []
    for system in SYSTEMS:
        system_peaks, system_frames = collect_system(work_root, system, args.result_dir_name)
        peaks.extend(system_peaks)
        frames.extend(system_frames)
    write_csv(peaks, output_dir / "time_series_statistics.csv", args.rolling_window)
    plot_smoothed_series(peaks, output_dir, args.rolling_window, metric="peak")
    plot_smoothed_series(peaks, output_dir, args.rolling_window, metric="vacc")
    plot_heatmaps(frames, output_dir)
    print(f"Wrote peak table and plots to {output_dir}")


if __name__ == "__main__":
    main()
