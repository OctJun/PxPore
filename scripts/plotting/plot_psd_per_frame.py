from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "results" / "apc_psd_frames"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "psd_per_frame"

TOOL_STYLE = {
    "PxPore": {"color": "#1F77B4", "linestyle": "-"},
    "PoreBlazer": {"color": "#D62728", "linestyle": "-"},
    "Zeo++": {"color": "#2CA02C", "linestyle": "-"},
}


@dataclass(frozen=True)
class Series:
    tool: str
    frame_id: int
    x_angstrom: np.ndarray
    y: np.ndarray
    source: Path

    @property
    def normalized_y(self) -> np.ndarray:
        maximum = float(np.max(self.y))
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError(f"Non-positive PSD maximum in {self.source}")
        return self.y / maximum

    @property
    def peak_angstrom(self) -> float:
        return float(self.x_angstrom[int(np.argmax(self.y))])


def set_paper_style() -> None:
    """Use the plotting style defined in PxPorePlot.ipynb."""
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
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "axes.grid": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "mathtext.default": "regular",
        }
    )


def numeric_rows(path: Path, minimum_columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < minimum_columns:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric PSD rows found in {path}")
    return np.asarray(rows, dtype=float)


def clean_series(x: np.ndarray, y: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x = x[valid]
    y = y[valid]
    if not len(x) or np.max(y) <= 0:
        raise ValueError(f"No valid positive PSD data found in {path}")
    order = np.argsort(x)
    return x[order], y[order]


def load_pxpore(root: Path, frame_id: int) -> Series | None:
    candidates = sorted((root / f"px_frame_{frame_id}").glob("*_voxel_mc_psd.txt"))
    if not candidates:
        return None
    path = candidates[0]
    values = numeric_rows(path, minimum_columns=6)
    if values.shape[1] >= 7:
        y = values[:, 5]
    else:
        probability = values[:, 3]
        bin_width_nm = float(np.median(np.diff(values[:, 1])))
        y = (probability + np.append(probability[1:], 0.0)) / (2.0 * bin_width_nm)
    # PxPore voxel MC: use the PoreBlazer-compatible central-difference density.
    x, y = clean_series(values[:, 1] * 10.0, y, path)
    return Series("PxPore", frame_id, x, y, path)


def load_poreblazer(root: Path, frame_id: int) -> Series | None:
    path = root / f"pb_frame_{frame_id}" / "Total_psd.txt"
    if not path.is_file():
        return None
    values = numeric_rows(path, minimum_columns=2)
    # PoreBlazer Total_psd.txt is already diameter (angstrom), -dV(d)/dd.
    x, y = clean_series(values[:, 0], values[:, 1], path)
    return Series("PoreBlazer", frame_id, x, y, path)


def load_zeopp(root: Path, frame_id: int, series_name: str) -> Series | None:
    path = root / f"zeo_frame_{frame_id}" / "psd.out"
    if not path.is_file():
        return None
    values = numeric_rows(path, minimum_columns=4)
    # Zeo++ columns: Bin, Count, Cumulative_dist, Derivative_dist.
    column = 1 if series_name == "count" else 3
    x, y = clean_series(values[:, 0], values[:, column], path)
    return Series("Zeo++", frame_id, x, y, path)


def discover_frame_ids(root: Path) -> list[int]:
    frame_ids: set[int] = set()
    pattern = re.compile(r"(?:px|pb|zeo)_frame_(\d+)$")
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            frame_ids.add(int(match.group(1)))
    if not frame_ids:
        raise FileNotFoundError(f"No per-frame PSD directories found under {root}")
    return sorted(frame_ids)


def read_frame_time(root: Path, frame_id: int) -> float | None:
    path = root / f"frame_{frame_id}.gro"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="ignore") as handle:
        title = handle.readline()
    match = re.search(r"\bt\s*=\s*([0-9.+-Ee]+)", title)
    return float(match.group(1)) if match else None


def load_frame(root: Path, frame_id: int, zeo_series: str) -> list[Series]:
    loaders = (
        lambda: load_pxpore(root, frame_id),
        lambda: load_poreblazer(root, frame_id),
        lambda: load_zeopp(root, frame_id, zeo_series),
    )
    return [series for load in loaders if (series := load()) is not None]


def frame_title(frame_id: int, time_ps: float | None) -> str:
    return f"Frame {frame_id}"


def draw_frame(
    ax: plt.Axes,
    series_list: list[Series],
    *,
    title: str,
    x_max: float,
    compact: bool = False,
) -> None:
    for series in series_list:
        style = TOOL_STYLE[series.tool]
        ax.plot(
            series.x_angstrom,
            series.normalized_y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5 if compact else 2.0,
            label=series.tool,
        )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12 if compact else 18)
    ax.grid(True, alpha=0.25)
    if compact:
        ax.tick_params(labelsize=9)


def plot_individual_frames(
    frames: dict[int, list[Series]],
    times: dict[int, float | None],
    output_dir: Path,
    x_max: float,
) -> list[Path]:
    outputs = []
    for frame_id, series_list in frames.items():
        fig, ax = plt.subplots()
        draw_frame(
            ax,
            series_list,
            title=frame_title(frame_id, times[frame_id]),
            x_max=x_max,
        )
        ax.set_xlabel(r"Pore size ($\AA$)")
        ax.set_ylabel("Normalized PSD")
        ax.legend(loc="best")
        fig.tight_layout()
        output = output_dir / f"frame_{frame_id:02d}_psd.png"
        fig.savefig(output)
        plt.close(fig)
        outputs.append(output)
    return outputs


def plot_overview(
    frames: dict[int, list[Series]],
    times: dict[int, float | None],
    output_dir: Path,
    x_max: float,
) -> Path:
    columns = 4
    rows = int(np.ceil(len(frames) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13.0, 3.15 * rows), squeeze=False)
    for ax, (frame_id, series_list) in zip(axes.flat, frames.items()):
        draw_frame(
            ax,
            series_list,
            title=frame_title(frame_id, times[frame_id]),
            x_max=x_max,
            compact=True,
        )
    for ax in axes.flat[len(frames) :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel(r"Pore size ($\AA$)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized PSD", fontsize=11)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output = output_dir / "all_frames_psd_overview.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def write_peak_summary(
    frames: dict[int, list[Series]],
    times: dict[int, float | None],
    output_dir: Path,
) -> Path:
    output = output_dir / "per_frame_psd_peaks.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame", "time_ps", "tool", "peak_A", "points", "source"))
        for frame_id, series_list in frames.items():
            for series in series_list:
                writer.writerow(
                    (
                        frame_id,
                        "" if times[frame_id] is None else times[frame_id],
                        series.tool,
                        series.peak_angstrom,
                        len(series.x_angstrom),
                        series.source,
                    )
                )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PSD comparisons for every MD frame.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-max", type=float, default=15.0)
    parser.add_argument(
        "--zeo-series",
        choices=("count", "derivative"),
        default="count",
        help="Zeo++ y column. 'count' reproduces PxPorePlot.ipynb.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.x_max <= 0:
        raise ValueError("--x-max must be positive")
    if not root.is_dir():
        raise FileNotFoundError(f"Frame root not found: {root}")

    set_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = discover_frame_ids(root)
    frames = {frame_id: load_frame(root, frame_id, args.zeo_series) for frame_id in frame_ids}
    times = {frame_id: read_frame_time(root, frame_id) for frame_id in frame_ids}

    for frame_id, series_list in frames.items():
        tools = ", ".join(series.tool for series in series_list)
        print(f"frame {frame_id}: {tools}")
    for output in plot_individual_frames(frames, times, output_dir, args.x_max):
        print(output)
    print(plot_overview(frames, times, output_dir, args.x_max))
    print(write_peak_summary(frames, times, output_dir))


if __name__ == "__main__":
    main()
