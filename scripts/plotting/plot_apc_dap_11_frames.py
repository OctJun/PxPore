#!/usr/bin/env python3
"""Plot and summarize the 11-frame APC-DAP pore-analysis comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "results" / "apc_dap_11_frames_data"
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "apc_dap_11_frames"
FRAME_SPACING_PS = 50
COLORS = {"PxPore": "#1F77B4", "PoreBlazer": "#D62728", "Zeo++": "#2CA02C"}
MARKERS = {"PxPore": "o", "PoreBlazer": "s", "Zeo++": "^"}
ZORDERS = {"PxPore": 5, "PoreBlazer": 4, "Zeo++": 3}


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
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "axes.grid": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "mathtext.default": "regular",
        }
    )


def numeric_rows(path: Path, columns: int = 2) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(row) >= columns:
            rows.append(row)
    return np.asarray(rows, dtype=float)


def frame_id(path: Path) -> int:
    return int(re.search(r"frame_(\d+)", path.parent.name).group(1))


def frame_files(data: Path, prefix: str, filename: str) -> list[Path]:
    paths = []
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)$")
    for directory in data.glob(f"{prefix}*"):
        if directory.is_dir() and pattern.fullmatch(directory.name):
            paths.extend(directory.glob(filename))
    return paths


def pxpore_mc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = numeric_rows(path, 6)
    x = values[:, 1] * 10.0
    if values.shape[1] >= 7:
        y = values[:, 5]
    else:
        probability = values[:, 3]
        bin_width_nm = float(np.median(np.diff(values[:, 1])))
        y = (probability + np.append(probability[1:], 0.0)) / (2 * bin_width_nm)
    return x, y


def pxpore_centers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = numeric_rows(path, 4)
    return values[:, 1] * 10.0, values[:, 3]


def poreblazer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = numeric_rows(path)
    return values[:, 0], values[:, 1]


def zeopp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = numeric_rows(path, 4)
    return values[:, 0], values[:, 1]


def mean_psd(
    paths: list[Path], loader
) -> tuple[np.ndarray, np.ndarray, int]:
    series = [loader(path) for path in sorted(paths, key=frame_id)]
    base_x = series[0][0]
    aligned = [
        y if len(x) == len(base_x) and np.allclose(x, base_x)
        else np.interp(base_x, x, y, left=0.0, right=0.0)
        for x, y in series
    ]
    mean = np.mean(np.vstack(aligned), axis=0)
    return base_x, mean / np.max(mean), len(series)


def pxpore_fractions(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data["stats"]
    return float(stats["Vacc_frac"]), float(stats["Vvoid_nm3"]) / float(stats["Vcell_nm3"])


def poreblazer_fractions(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    total = float(re.findall(r"FV_PO:\s*([0-9Ee+\-.]+)", text, re.I)[0])
    block = re.search(r"Network-accessible(.*?)(?:\n\s*\n|\Z)", text, re.I | re.S).group(1)
    accessible = float(re.search(r"FV_PO:\s*([0-9Ee+\-.]+)", block, re.I).group(1))
    return accessible, total


def zeopp_fractions(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    accessible = float(re.search(r"POAV_Volume_fraction:\s*([0-9Ee+\-.]+)", text, re.I).group(1))
    inaccessible = float(re.search(r"PONAV_Volume_fraction:\s*([0-9Ee+\-.]+)", text, re.I).group(1))
    return accessible, accessible + inaccessible


def elapsed_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    return sum(part * 60**power for power, part in enumerate(reversed(parts)))


def performance_record(path: Path, software: str) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    user = float(re.search(r"User time \(seconds\):\s*([0-9.]+)", text).group(1))
    system = float(re.search(r"System time \(seconds\):\s*([0-9.]+)", text).group(1))
    elapsed = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)",
        text,
    ).group(1)
    rss_kb = float(
        re.search(r"Maximum resident set size \(kbytes\):\s*([0-9.]+)", text).group(1)
    )
    return {
        "software": software,
        "frame": frame_id(path),
        "memory_usage_mb": rss_kb / 1024.0,
        "wall_time_s": elapsed_seconds(elapsed),
        "cpu_time_s": user + system,
        "user_time_s": user,
        "system_time_s": system,
        "source": str(path),
    }


def write_performance_tables(data: Path, output: Path) -> None:
    specifications = (
        ("PxPore (MC)", "px_frame_"),
        ("PxPore (Centers)", "px_center_frame_"),
        ("PoreBlazer", "pb_frame_"),
        ("Zeo++", "zeo_frame_"),
    )
    rows = []
    for software, prefix in specifications:
        rows.extend(
            performance_record(path, software)
            for path in frame_files(data, prefix, "run.log")
        )
    rows.sort(key=lambda row: (str(row["software"]), int(row["frame"])))
    csv_path = output / "apc_dap_frame_performance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    groups = {
        software: [row for row in rows if row["software"] == software]
        for software, _ in specifications
    }
    lines = [
        "# APC-DAP 11-frame performance summary",
        "",
        "Values are mean ± sample standard deviation across frames 0–10. "
        "Memory usage is GNU time maximum resident set size converted from "
        "KiB to MB by dividing by 1024. CPU time is user plus system time.",
        "",
        "| Software | Frames | Memory usage (MB) | Wall time (s) | CPU time (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for software, _ in specifications:
        group = groups[software]
        measurements = []
        for key in ("memory_usage_mb", "wall_time_s", "cpu_time_s"):
            values = [float(row[key]) for row in group]
            measurements.append(
                f"{statistics.fmean(values):.2f} ± {statistics.stdev(values):.2f}"
            )
        lines.append(
            f"| {software} | {len(group)} | " + " | ".join(measurements) + " |"
        )
    (output / "apc_dap_performance_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def fraction_series(data: Path) -> dict[str, list[tuple[int, float, float]]]:
    specifications = (
        ("PxPore", "px_frame_", "*_stats.json", pxpore_fractions),
        ("PoreBlazer", "pb_frame_", "summary.dat", poreblazer_fractions),
        ("Zeo++", "zeo_frame_", "volpo.out", zeopp_fractions),
    )
    result = {}
    for tool, directory_pattern, filename, loader in specifications:
        rows = []
        for path in frame_files(data, directory_pattern, filename):
            accessible, total = loader(path)
            rows.append((frame_id(path), accessible, total))
        result[tool] = sorted(rows)
    return result


def plot_fractions(series, output: Path) -> None:
    for index, (suffix, ylabel) in enumerate(
        (("accessible", r"$F_{acc,V}$"), ("total", r"$F_{total,V}$")), start=1
    ):
        fig, ax = plt.subplots()
        for tool, rows in series.items():
            ax.plot(
                [row[0] * FRAME_SPACING_PS for row in rows],
                [row[index] for row in rows],
                color=COLORS[tool], marker=MARKERS[tool], linewidth=2, markersize=5,
                label=tool, zorder=ZORDERS[tool],
            )
        ax.set_ylim(0.3125, 0.3275)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"apc_dap_{suffix}_fraction.{extension}")
        plt.close(fig)


def plot_average_psd(data: Path, output: Path) -> list[dict[str, object]]:
    inputs = (
        ("PxPore (MC)", COLORS["PxPore"], "-", frame_files(data, "px_frame_", "*_voxel_mc_psd.txt"), pxpore_mc),
        ("PxPore (Centers)", COLORS["PxPore"], "--", frame_files(data, "px_center_frame_", "*_psd.txt"), pxpore_centers),
        ("PoreBlazer", COLORS["PoreBlazer"], "-", frame_files(data, "pb_frame_", "Network-accessible_psd.txt"), poreblazer),
        ("Zeo++", COLORS["Zeo++"], "-", frame_files(data, "zeo_frame_", "psd.out"), zeopp),
    )
    fig, ax = plt.subplots()
    table = []
    for label, color, linestyle, paths, loader in inputs:
        x, y, count = mean_psd(paths, loader)
        tool = "PxPore" if label.startswith("PxPore") else label
        ax.plot(
            x, y, color=color, linestyle=linestyle, linewidth=2,
            label=label, zorder=ZORDERS[tool],
        )
        table.extend({"series": label, "frames": count, "pore_size_A": xi, "normalized_mean_psd": yi} for xi, yi in zip(x, y))
    ax.set_xlim(0, 15)
    ax.set_xlabel(r"Pore size ($\AA$)")
    ax.set_ylabel("Normalized PSD")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"apc_dap_average_psd.{extension}")
    plt.close(fig)
    return table


def write_tables(series, psd_table, output: Path) -> None:
    with (output / "apc_dap_frame_fractions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tool", "frame", "time_ps", "accessible_fraction", "total_fraction"))
        for tool, rows in series.items():
            writer.writerows((tool, frame, frame * FRAME_SPACING_PS, accessible, total) for frame, accessible, total in rows)
    with (output / "apc_dap_fraction_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("tool", "frames", "accessible_mean", "accessible_sd", "total_mean", "total_sd"))
        for tool, rows in series.items():
            accessible = np.asarray([row[1] for row in rows])
            total = np.asarray([row[2] for row in rows])
            writer.writerow((tool, len(rows), accessible.mean(), accessible.std(ddof=1), total.mean(), total.std(ddof=1)))
    with (output / "apc_dap_average_psd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=psd_table[0].keys())
        writer.writeheader()
        writer.writerows(psd_table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = args.data_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    set_paper_style()
    series = fraction_series(data)
    plot_fractions(series, output)
    psd_table = plot_average_psd(data, output)
    write_tables(series, psd_table, output)
    write_performance_tables(data, output)
    print(output)


if __name__ == "__main__":
    main()
