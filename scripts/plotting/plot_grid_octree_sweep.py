#!/usr/bin/env python3
"""
Plot supplementary grid/octree tradeoff figures from a sweep manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "case" / "grid_octree_sweep"
DEFAULT_FIG_ROOT = REPO_ROOT / "docs" / "figures" / "grid_octree_sweep"
STATUS_ORDER = [
    "success",
    "predicted_oom_voxels",
    "runtime_oom",
    "skipped_after_oom",
    "timeout",
    "missing_stats",
    "error",
]
STATUS_TO_CODE = {name: idx for idx, name in enumerate(STATUS_ORDER)}
MARKERS = ["o", "s", "^", "D", "P", "X", "v"]
METRIC_LABELS = {
    "stats_Vvoid_frac": "Void fraction",
    "stats_Vacc_frac": "Accessible void fraction",
    "stats_PLD_nm": "PLD",
    "stats_LCD_nm": "LCD",
    "stats_Stotal_nm2": "Total surface area",
    "stats_Sacc_nm2": "Accessible surface area",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PxPore supplementary grid/octree sweep figures."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing results.jsonl from sweep_grid_octree_cli.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_FIG_ROOT,
        help="Directory used to save figures and aggregated CSV tables.",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="polymer,hkust1,irmof1,512_h_jitter_octree,single_h_octree",
        help="Comma-separated systems to plot.",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.02,
        help="Highlight the fastest setting whose cross-system max error is below this threshold.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans Mono", "Liberation Sans"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "axes.grid": False,
            "legend.frameon": False,
            "mathtext.default": "regular",
        }
    )


def group_by_system(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["system"]].append(record)
    return grouped


def success_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("status") == "success"]


def pick_reference(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = success_records(records)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda record: (
            float(record["grid_nm"]),
            -int(record["oct_level"]),
            float(record.get("execution_time_s", record.get("wall_time_s", math.inf))),
        ),
    )


def relative_error(value: float | None, ref: float | None) -> float | None:
    if value is None or ref is None:
        return None
    if ref == 0:
        return None
    return abs(float(value) - float(ref)) / abs(float(ref))


def metric_relative_error(record: dict[str, Any], reference_values: dict[str, float], metric_key: str) -> float | None:
    ref_value = reference_values.get(metric_key)
    value = record.get(metric_key)
    if ref_value is None or value is None:
        return None
    if metric_key in ("stats_PLD_nm", "stats_LCD_nm", "stats_Stotal_nm2", "stats_Sacc_nm2") and float(ref_value) <= 0:
        return None
    return relative_error(float(value), float(ref_value))


def aggregate_metric_keys(system: str) -> tuple[str, ...]:
    return ("stats_Vvoid_frac",)


def octree_sensitive_metric_keys(system: str) -> tuple[str, ...]:
    # PLD/LCD are currently computed from the coarse distance field in core.py,
    # so they should not drive an octree-level tradeoff plot.
    return (
        *tradeoff_volume_metric_keys(system),
        "stats_Sacc_nm2",
    )


def aggregate_error(
    system: str,
    record: dict[str, Any],
    reference_values: dict[str, float],
) -> float | None:
    values = []
    for key in aggregate_metric_keys(system):
        err = metric_relative_error(record, reference_values, key)
        if err is not None:
            values.append(err)
    if not values:
        return None
    return max(values)


def octree_sensitive_error(
    system: str,
    record: dict[str, Any],
    reference_values: dict[str, float],
) -> float | None:
    values = []
    for key in octree_sensitive_metric_keys(system):
        err = metric_relative_error(record, reference_values, key)
        if err is not None:
            values.append(err)
    if not values:
        return None
    return max(values)


def build_reference_values(reference: dict[str, Any] | None) -> dict[str, float]:
    if reference is None:
        return {}
    return {
        key: float(reference[key])
        for key in METRIC_LABELS
        if reference.get(key) is not None
    }


def build_h_theoretical_reference(system: str, records: list[dict[str, Any]], radius_nm: float = 0.12) -> dict[str, float] | None:
    if system not in {"single_h_octree", "512_h_jitter_octree"}:
        return None
    sample = success_records(records)
    if not sample:
        return None
    first = sample[0]
    natoms = int(first.get("stats_atoms", 0))
    try:
        lx = float(first.get("box_x_nm"))
        ly = float(first.get("box_y_nm"))
        lz = float(first.get("box_z_nm"))
    except (TypeError, ValueError):
        return None
    if max(abs(lx - ly), abs(ly - lz), abs(lx - lz)) > 1e-8:
        return None
    box_len = lx
    cell_volume = lx * ly * lz
    sphere_volume = (4.0 / 3.0) * math.pi * radius_nm ** 3
    sphere_area = 4.0 * math.pi * radius_nm ** 2
    n_side = round(natoms ** (1.0 / 3.0))
    if n_side < 1 or n_side ** 3 != natoms:
        n_side = 1
    lattice = box_len / n_side
    void_frac = 1.0 - natoms * sphere_volume / cell_volume
    pld = max(2.0 * (lattice / math.sqrt(2.0) - radius_nm), 0.0)
    lcd = max(2.0 * (math.sqrt(3.0) * lattice / 2.0 - radius_nm), 0.0)
    total_surface = natoms * sphere_area
    void_volume = cell_volume * void_frac
    return {
        "stats_Vvoid_nm3": void_volume,
        "stats_Vacc_nm3": void_volume,
        "stats_Vvoid_frac": void_frac,
        "stats_Vacc_frac": void_frac,
        "stats_PLD_nm": pld,
        "stats_LCD_nm": lcd,
        "stats_Stotal_nm2": total_surface,
        "stats_Sacc_nm2": total_surface,
    }


def resolve_reference_values(
    system: str,
    records: list[dict[str, Any]],
) -> tuple[dict[str, float], str, dict[str, Any] | None]:
    theoretical_reference = build_h_theoretical_reference(system, records)
    if theoretical_reference:
        return theoretical_reference, "theory", None
    reference = pick_reference(records)
    return build_reference_values(reference), "finest successful reference", reference


def tradeoff_volume_metric_keys(system: str) -> tuple[str, str]:
    if system in {"single_h_octree", "512_h_jitter_octree"}:
        return ("stats_Vvoid_nm3", "stats_Vacc_nm3")
    return ("stats_Vvoid_frac", "stats_Vacc_frac")


def grid_and_oct_axes(records: list[dict[str, Any]]) -> tuple[list[float], list[int]]:
    grids = sorted({float(record["grid_nm"]) for record in records}, reverse=True)
    oct_levels = sorted({int(record["oct_level"]) for record in records})
    return grids, oct_levels


def build_lookup(records: list[dict[str, Any]]) -> dict[tuple[float, int], dict[str, Any]]:
    lookup: dict[tuple[float, int], dict[str, Any]] = {}
    for record in records:
        lookup[(float(record["grid_nm"]), int(record["oct_level"]))] = record
    return lookup


def format_grid_labels(grids: list[float]) -> list[str]:
    return [format(grid, ".4f").rstrip("0").rstrip(".") for grid in grids]


def decimal_tick_formatter(_: str | None = None) -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: f"{value:g}" if value > 0 else "0")


def apply_decimal_log_ticks(ax: plt.Axes, values: list[float], axis: str = "x") -> None:
    clean_values = [float(v) for v in values if v is not None and np.isfinite(v) and float(v) > 0]
    if not clean_values:
        return
    unique_values = sorted(set(clean_values))
    labels = [format(v, ".4f").rstrip("0").rstrip(".") if v < 1 else f"{v:g}" for v in unique_values]
    if axis == "x":
        ax.set_xticks(unique_values)
        ax.set_xticklabels(labels)
        ax.xaxis.set_major_formatter(decimal_tick_formatter())
    else:
        ax.set_yticks(unique_values)
        ax.set_yticklabels([f"{v:g}" for v in unique_values])
        ax.yaxis.set_major_formatter(decimal_tick_formatter())


def apply_decimal_decade_ticks(ax: plt.Axes, values: list[float]) -> None:
    clean_values = [float(v) for v in values if v is not None and np.isfinite(v) and float(v) > 0]
    if not clean_values:
        return
    vmin = min(clean_values)
    vmax = max(clean_values)
    pmin = int(math.floor(math.log10(vmin)))
    pmax = int(math.ceil(math.log10(vmax)))
    decades = [10.0 ** power for power in range(pmin, pmax + 1)]
    decades = [value for value in decades if vmin * 0.999 <= value <= vmax * 1.001]
    if not decades:
        decades = [vmin]
    labels = [format(value, ".4f").rstrip("0").rstrip(".") if value < 1 else f"{value:g}" for value in decades]
    ax.xaxis.set_major_locator(FixedLocator(decades))
    ax.set_xticklabels(labels)
    ax.xaxis.set_major_formatter(decimal_tick_formatter())


def padded_log_limits(values: list[float], *, lower_factor: float = 0.85, upper_factor: float = 1.15) -> tuple[float, float]:
    clean_values = [float(v) for v in values if v is not None and np.isfinite(v) and float(v) > 0]
    if not clean_values:
        return 1.0, 10.0
    low = min(clean_values) * lower_factor
    high = max(clean_values) * upper_factor
    return low, high


def plot_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    title: str,
    grids: list[float],
    oct_levels: list[int],
    cmap: Any,
    norm: Any = None,
    cbar_label: str = "",
    nan_color: str = "#efefef",
) -> None:
    cmap_obj = mpl.colormaps.get_cmap(cmap).copy()
    cmap_obj.set_bad(nan_color)
    transposed = data.T
    valid_grid = np.any(np.isfinite(transposed), axis=0)
    valid_oct = np.any(np.isfinite(transposed), axis=1)
    if np.any(valid_grid):
        transposed = transposed[:, valid_grid]
        grids = [grid for grid, keep in zip(grids, valid_grid) if keep]
    if np.any(valid_oct):
        transposed = transposed[valid_oct, :]
        oct_levels = [level for level, keep in zip(oct_levels, valid_oct) if keep]
    masked = np.ma.masked_invalid(transposed)
    image = ax.imshow(masked, aspect="auto", cmap=cmap_obj, norm=norm, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Grid (nm)")
    ax.set_ylabel("Oct level")
    ax.set_xticks(np.arange(len(grids)))
    ax.set_xticklabels(format_grid_labels(grids), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(oct_levels)))
    ax.set_yticklabels([str(v) for v in oct_levels])
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label)


def plot_status_matrix(
    ax: plt.Axes,
    records: list[dict[str, Any]],
    grids: list[float],
    oct_levels: list[int],
) -> None:
    matrix = np.full((len(grids), len(oct_levels)), np.nan)
    lookup = build_lookup(records)
    for i, grid in enumerate(grids):
        for j, oct_level in enumerate(oct_levels):
            record = lookup.get((grid, oct_level))
            if record is None:
                continue
            matrix[i, j] = STATUS_TO_CODE.get(record.get("status", "error"), len(STATUS_ORDER) - 1)
    cmap = mpl.colors.ListedColormap(
        ["#3B6EA8", "#D6D6D6", "#CC4C3B", "#F1C27D", "#7A6AA6", "#7F8C8D", "#222222"]
    )
    bounds = np.arange(len(STATUS_ORDER) + 1) - 0.5
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title("Status")
    ax.set_xlabel("Oct level")
    ax.set_ylabel("Grid (nm)")
    ax.set_xticks(np.arange(len(oct_levels)))
    ax.set_xticklabels([str(v) for v in oct_levels])
    ax.set_yticks(np.arange(len(grids)))
    ax.set_yticklabels(format_grid_labels(grids))
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03, ticks=np.arange(len(STATUS_ORDER)))
    cbar.ax.set_yticklabels(STATUS_ORDER)


def runtime_value(record: dict[str, Any]) -> float | None:
    for key in ("execution_time_s", "timings_total_s", "wall_time_s"):
        value = record.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def oct_level_label(oct_level: int) -> str:
    # if oct_level == 0:
    #     return "no-octree"
    return f"{oct_level}"


def oct_level_style(oct_level: int, all_levels: list[int]) -> dict[str, Any]:
    idx = all_levels.index(oct_level)
    cmap = mpl.colormaps.get_cmap("tab10").resampled(max(len(all_levels), 3))
    return {
        "color": cmap(idx),
        "marker": MARKERS[idx % len(MARKERS)],
    }


def collect_tradeoff_rows(
    system: str,
    records: list[dict[str, Any]],
    reference_values: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    if not reference_values:
        return rows
    for record in success_records(records):
        err = octree_sensitive_error(system, record, reference_values)
        rt = runtime_value(record)
        if err is None or rt is None:
            continue
        rows.append(
            {
                "system": record["system"],
                "grid_nm": float(record["grid_nm"]),
                "oct_level": int(record["oct_level"]),
                "runtime_s": float(rt),
                "error": float(err),
            }
        )
    return rows


def positive_floor(values: list[float], default: float = 1e-6) -> float:
    positive = [float(v) for v in values if v is not None and np.isfinite(v) and float(v) > 0]
    if not positive:
        return default
    return max(min(positive) / 3.0, default)


def plot_tradeoff_panel(
    ax: plt.Axes,
    *,
    rows: list[dict[str, Any]],
    title: str,
    runtime_label: str,
    error_threshold: float | None = None,
) -> None:
    if not rows:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No successful data", ha="center", va="center", transform=ax.transAxes)
        return

    ax_right = ax.twinx()
    all_levels = sorted({int(row["oct_level"]) for row in rows})
    err_floor = positive_floor([row["error"] for row in rows])
    rt_floor = positive_floor([row["runtime_s"] for row in rows], default=1e-3)
    red_cmap = mpl.colormaps.get_cmap("Reds").resampled(max(len(all_levels) + 2, 4))
    blue_cmap = mpl.colormaps.get_cmap("Blues").resampled(max(len(all_levels) + 2, 4))

    legend_handles = []
    for oct_level in all_levels:
        style = oct_level_style(oct_level, all_levels)
        subset = [row for row in rows if int(row["oct_level"]) == oct_level]
        subset_by_grid = sorted(subset, key=lambda row: row["grid_nm"], reverse=True)
        grid_x = [row["grid_nm"] for row in subset_by_grid]
        err_y = [max(row["error"], err_floor) for row in subset_by_grid]
        rt_y = [max(row["runtime_s"], rt_floor) for row in subset_by_grid]
        idx = all_levels.index(oct_level) + 1
        err_color = red_cmap(idx)
        rt_color = blue_cmap(idx)
        ax.plot(
            grid_x,
            err_y,
            color=err_color,
            linestyle="solid",
            linewidth=1.8,
            marker=style["marker"],
            markersize=6.2,
            markerfacecolor="white",
            markeredgewidth=0.9,
            zorder=3,
        )
        ax_right.plot(
            grid_x,
            rt_y,
            color=rt_color,
            linestyle="solid",
            linewidth=1.6,
            marker=style["marker"],
            markersize=5.6,
            markerfacecolor=rt_color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            alpha=0.95,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle="solid",
                marker=style["marker"],
                markersize=5,
                linewidth=1.8,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=oct_level_label(oct_level),
            )
        )

    grid_values = [row["grid_nm"] for row in rows]
    runtime_values = [max(row["runtime_s"], rt_floor) for row in rows]
    y_values = [max(row["error"], err_floor) for row in rows]
    x_low, x_high = padded_log_limits(grid_values)
    err_low, err_high = padded_log_limits(y_values, lower_factor=0.8, upper_factor=1.3)
    rt_low, rt_high = padded_log_limits(runtime_values, lower_factor=0.8, upper_factor=1.25)
    err_high *= 1.8
    rt_high *= 1.8

    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlabel("Grid spacing (nm)")
    ax.set_ylabel("Relative error", color="#B22222")
    ax.set_yscale("log")
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(err_low, err_high)
    ax.tick_params(axis="y", colors="#B22222")
    apply_decimal_decade_ticks(ax, grid_values)
    ax.yaxis.set_major_formatter(decimal_tick_formatter())
    ax.grid(True, which="major", alpha=0.25, zorder=0)

    ax_right.set_ylabel(runtime_label, color="#1F4E79")
    ax_right.set_yscale("log")
    ax_right.set_ylim(rt_low, rt_high)
    ax_right.tick_params(axis="y", colors="#1F4E79")
    ax_right.yaxis.set_major_formatter(decimal_tick_formatter())

    if error_threshold is not None:
        ax.axhline(error_threshold, color="#CC4C3B", linestyle="--", linewidth=1.1, zorder=1)

    encoding_handles = [
        Line2D(
            [0],
            [0],
            color="#B22222",
            linestyle="solid",
            linewidth=1.8,
            label="relative error",
        ),
        Line2D(
            [0],
            [0],
            color="#1F4E79",
            marker="o",
            linestyle="solid",
            linewidth=1.6,
            markersize=5,
            markerfacecolor="#1F4E79",
            label="runtime",
        ),
    ]
    legend_encoding = ax.legend(
        handles=encoding_handles,
        title="Curves",
        loc="upper left",
        frameon=False,
        ncol=1,
        borderaxespad=0.6,
        handlelength=1.8,
        fontsize=8,
        title_fontsize=8,
    )
    ax.add_artist(legend_encoding)
    legend_levels = ax.legend(
        handles=legend_handles,
        title="Oct-level",
        loc="upper right",
        frameon=False,
        ncol=min(3, max(1, len(legend_handles))),
        borderaxespad=0.6,
        columnspacing=0.9,
        handlelength=1.6,
        fontsize=8,
        title_fontsize=8,
    )
    ax.add_artist(legend_levels)


def plot_metric_panel(
    ax: plt.Axes,
    *,
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
) -> None:
    if not rows:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
        return
    all_levels = sorted({int(row["oct_level"]) for row in rows})
    floor = positive_floor([row["error"] for row in rows])
    for oct_level in all_levels:
        style = oct_level_style(oct_level, all_levels)
        subset = [row for row in rows if int(row["oct_level"]) == oct_level]
        subset = sorted(subset, key=lambda row: row["grid_nm"], reverse=True)
        ax.plot(
            [row["grid_nm"] for row in subset],
            [max(row["error"], floor) for row in subset],
            color=style["color"],
            linestyle="solid",
            linewidth=1.6,
            marker=style["marker"],
            markersize=5.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grid spacing (nm)")
    ax.set_ylabel("Relative error")
    grid_values = [row["grid_nm"] for row in rows]
    y_values = [max(row["error"], floor) for row in rows]
    x_low, x_high = padded_log_limits(grid_values)
    y_low, y_high = padded_log_limits(y_values, lower_factor=0.8, upper_factor=1.3)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    apply_decimal_decade_ticks(ax, grid_values)
    ax.yaxis.set_major_formatter(decimal_tick_formatter())
    ax.grid(True, which="major", alpha=0.25)


def collect_metric_rows(
    records: list[dict[str, Any]],
    reference_values: dict[str, float],
    metric_key: str,
) -> list[dict[str, Any]]:
    rows = []
    if not reference_values:
        return rows
    for record in success_records(records):
        err = metric_relative_error(record, reference_values, metric_key)
        rt = runtime_value(record)
        if err is None or rt is None:
            continue
        rows.append(
            {
                "grid_nm": float(record["grid_nm"]),
                "oct_level": int(record["oct_level"]),
                "error": float(err),
                "runtime_s": float(rt),
            }
        )
    return rows


def plot_metric_error_figure(
    system: str,
    records: list[dict[str, Any]],
    output_root: Path,
    reference_values: dict[str, float],
    reference_label: str,
) -> None:
    metric_order = [
        "stats_Vvoid_frac",
        "stats_Vacc_frac",
        "stats_PLD_nm",
        "stats_LCD_nm",
        "stats_Stotal_nm2",
        "stats_Sacc_nm2",
    ]
    rows_by_metric = {
        metric_key: collect_metric_rows(records, reference_values, metric_key)
        for metric_key in metric_order
    }
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2))
    axes_flat = axes.ravel()
    all_levels = sorted({int(record["oct_level"]) for record in success_records(records)})
    for ax, metric_key in zip(axes_flat, metric_order):
        plot_metric_panel(
            ax,
            rows=rows_by_metric[metric_key],
            metric_key=metric_key,
            title=METRIC_LABELS[metric_key],
        )
    legend_handles = []
    for oct_level in all_levels:
        style = oct_level_style(oct_level, all_levels)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="solid",
                marker=style["marker"],
                markersize=5,
                linewidth=1.6,
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=oct_level_label(oct_level),
            )
        )
    fig.legend(
        handles=legend_handles,
        title="Oct-level",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=min(4, max(1, len(legend_handles))),
    )
    fig.suptitle(f"{system}: metric-wise relative errors ({reference_label})", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_root / f"{system}_metric_relative_errors.png")
    plt.close(fig)


def collect_octree_rows(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    rows = []
    for record in success_records(records):
        if int(record["oct_level"]) == 0:
            continue
        value = record.get(field)
        if value is None:
            continue
        value = float(value)
        if value <= 0:
            continue
        rows.append(
            {
                "grid_nm": float(record["grid_nm"]),
                "oct_level": int(record["oct_level"]),
                "value": value,
            }
        )
    return rows


def plot_octree_metric_panel(ax: plt.Axes, rows: list[dict[str, Any]], title: str, ylabel: str) -> None:
    if not rows:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
        return
    all_levels = sorted({int(row["oct_level"]) for row in rows})
    for oct_level in all_levels:
        style = oct_level_style(oct_level, all_levels)
        subset = [row for row in rows if int(row["oct_level"]) == oct_level]
        subset = sorted(subset, key=lambda row: row["grid_nm"], reverse=True)
        ax.plot(
            [row["grid_nm"] for row in subset],
            [row["value"] for row in subset],
            color=style["color"],
            linestyle="solid",
            linewidth=1.6,
            marker=style["marker"],
            markersize=5.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grid spacing (nm)")
    ax.set_ylabel(ylabel)
    grid_values = [row["grid_nm"] for row in rows]
    y_values = [row["value"] for row in rows]
    x_low, x_high = padded_log_limits(grid_values)
    y_low, y_high = padded_log_limits(y_values, lower_factor=0.8, upper_factor=1.3)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    apply_decimal_decade_ticks(ax, grid_values)
    ax.yaxis.set_major_formatter(decimal_tick_formatter())
    ax.grid(True, which="major", alpha=0.25)


def plot_octree_figure(system: str, records: list[dict[str, Any]], output_root: Path) -> None:
    nodes_rows = collect_octree_rows(records, "info_octree_nodes")
    leaf_rows = collect_octree_rows(records, "info_octree_leaf_nodes")
    roots_rows = collect_octree_rows(records, "info_octree_root_voxels")
    ratio_rows = []
    for record in success_records(records):
        if int(record["oct_level"]) == 0:
            continue
        roots = float(record.get("info_octree_root_voxels", 0) or 0)
        nodes = float(record.get("info_octree_nodes", 0) or 0)
        if roots <= 0 or nodes <= 0:
            continue
        ratio_rows.append(
            {
                "grid_nm": float(record["grid_nm"]),
                "oct_level": int(record["oct_level"]),
                "value": nodes / roots,
            }
        )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    plot_octree_metric_panel(axes[0, 0], roots_rows, "Octree root voxels", "Count")
    plot_octree_metric_panel(axes[0, 1], nodes_rows, "Octree total nodes", "Count")
    plot_octree_metric_panel(axes[1, 0], leaf_rows, "Octree leaf nodes", "Count")
    plot_octree_metric_panel(axes[1, 1], ratio_rows, "Refinement factor", "Nodes / root voxel")
    all_levels = sorted({int(record["oct_level"]) for record in success_records(records) if int(record["oct_level"]) > 0})
    legend_handles = []
    for oct_level in all_levels:
        style = oct_level_style(oct_level, all_levels)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="solid",
                marker=style["marker"],
                markersize=5,
                linewidth=1.6,
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=oct_level_label(oct_level),
            )
        )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=min(4, len(legend_handles)),
        )
    fig.suptitle(f"{system}: octree complexity metrics", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_root / f"{system}_octree_metrics.png")
    plt.close(fig)


def plot_system_figure(system: str, records: list[dict[str, Any]], output_root: Path) -> None:
    reference_values, reference_label, reference = resolve_reference_values(system, records)
    grids, oct_levels = grid_and_oct_axes(records)
    lookup = build_lookup(records)

    runtime = np.full((len(grids), len(oct_levels)), np.nan)
    error = np.full((len(grids), len(oct_levels)), np.nan)
    for i, grid in enumerate(grids):
        for j, oct_level in enumerate(oct_levels):
            record = lookup.get((grid, oct_level))
            if record is None or record.get("status") != "success":
                continue
            runtime[i, j] = runtime_value(record)
            if reference_values:
                err = aggregate_error(system, record, reference_values)
                if err is not None:
                    error[i, j] = err

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    positive_runtime = runtime[np.isfinite(runtime) & (runtime > 0)]
    if positive_runtime.size:
        norm_runtime = mpl.colors.LogNorm(vmin=np.nanmin(positive_runtime), vmax=np.nanmax(positive_runtime))
    else:
        norm_runtime = None
    plot_heatmap(
        axes[0],
        runtime,
        title="Runtime",
        grids=grids,
        oct_levels=oct_levels,
        cmap="viridis",
        norm=norm_runtime,
        cbar_label="Seconds",
    )

    positive_error = error[np.isfinite(error) & (error > 0)]
    if positive_error.size:
        norm_error = mpl.colors.LogNorm(vmin=max(np.nanmin(positive_error), 1e-5), vmax=np.nanmax(positive_error))
    else:
        norm_error = None
    plot_heatmap(
        axes[1],
        error,
        title="Void-fraction error vs reference",
        grids=grids,
        oct_levels=oct_levels,
        cmap="magma",
        norm=norm_error,
        cbar_label="Relative error",
    )

    ref_text = "reference: unavailable"
    if reference is not None:
        ref_text = (
            f"reference = g {float(reference['grid_nm']):.4f} nm, "
            f"oct {int(reference['oct_level'])}"
        )
    if reference_label == "theory":
        ref_text = "reference = analytic theory (r = 0.12 nm)"
    fig.suptitle(f"{system}: grid/octree sweep ({ref_text})", y=1.02)
    fig.tight_layout()
    fig.savefig(output_root / f"{system}_grid_octree_heatmaps.png")
    plt.close(fig)

    tradeoff_rows = collect_tradeoff_rows(system, records, reference_values)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    plot_tradeoff_panel(
        ax,
        rows=tradeoff_rows,
        title=f"{system}: octree-sensitive error vs grid / runtime",
        runtime_label="Runtime (s)",
    )
    fig.tight_layout()
    fig.savefig(output_root / f"{system}_grid_runtime_tradeoff.png")
    plt.close(fig)
    plot_metric_error_figure(
        system,
        records,
        output_root,
        reference_values=reference_values,
        reference_label=reference_label,
    )
    plot_octree_figure(system, records, output_root)


def cross_system_summary(
    systems: list[str],
    grouped: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    reference_values_by_system = {}
    for system in systems:
        records = grouped.get(system, [])
        reference_values, _, _ = resolve_reference_values(system, records)
        reference_values_by_system[system] = reference_values
    all_pairs = sorted(
        {
            (float(record["grid_nm"]), int(record["oct_level"]))
            for system in systems
            for record in grouped.get(system, [])
        },
        key=lambda item: (-item[0], item[1]),
    )
    rows = []
    for grid_nm, oct_level in all_pairs:
        per_system = []
        for system in systems:
            lookup = build_lookup(grouped.get(system, []))
            record = lookup.get((grid_nm, oct_level))
            if record is None or record.get("status") != "success":
                per_system = []
                break
            reference_values = reference_values_by_system.get(system)
            if not reference_values:
                per_system = []
                break
            err = octree_sensitive_error(system, record, reference_values)
            rt = runtime_value(record)
            if err is None or rt is None:
                per_system = []
                break
            per_system.append((system, rt, err))
        if not per_system:
            continue
        rows.append(
            {
                "grid_nm": grid_nm,
                "oct_level": oct_level,
                "runtime_sum_s": sum(item[1] for item in per_system),
                "runtime_max_s": max(item[1] for item in per_system),
                "error_mean": float(np.mean([item[2] for item in per_system])),
                "error_max": max(item[2] for item in per_system),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_cross_system_pareto(
    rows: list[dict[str, Any]],
    output_root: Path,
    error_threshold: float,
) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    grids = np.array([row["grid_nm"] for row in rows], dtype=float)
    oct_levels = np.array([row["oct_level"] for row in rows], dtype=int)
    runtime = np.array([row["runtime_sum_s"] for row in rows], dtype=float)
    error = np.array([row["error_max"] for row in rows], dtype=float)

    scatter = ax.scatter(
        runtime,
        error,
        c=grids,
        s=60 + 20 * oct_levels,
        cmap="viridis_r",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cross-system runtime sum (s)")
    ax.set_ylabel("Cross-system max relative error")
    ax.axhline(error_threshold, color="#CC4C3B", linestyle="--", linewidth=1.2)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Grid (nm)")

    feasible = [row for row in rows if row["error_max"] <= error_threshold]
    if feasible:
        best = min(feasible, key=lambda row: row["runtime_sum_s"])
        ax.scatter(
            [best["runtime_sum_s"]],
            [best["error_max"]],
            marker="*",
            s=220,
            color="#CC4C3B",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        ax.annotate(
            f"g={best['grid_nm']:.4f}, oct={best['oct_level']}",
            (best["runtime_sum_s"], best["error_max"]),
            xytext=(10, 10),
            textcoords="offset points",
        )

    ax.grid(True, which="major", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "cross_system_pareto.png")
    plt.close(fig)


def plot_cross_system_grid_runtime_tradeoff(
    rows: list[dict[str, Any]],
    output_root: Path,
    error_threshold: float,
) -> None:
    if not rows:
        return
    tradeoff_rows = [
        {
            "grid_nm": float(row["grid_nm"]),
            "oct_level": int(row["oct_level"]),
            "runtime_s": float(row["runtime_sum_s"]),
            "error": float(row["error_max"]),
        }
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    plot_tradeoff_panel(
        ax,
        rows=tradeoff_rows,
        title="Cross-system tradeoff: octree-sensitive max error vs grid / runtime sum",
        runtime_label="Runtime sum across systems (s)",
        error_threshold=error_threshold,
    )
    fig.tight_layout()
    fig.savefig(output_root / "cross_system_grid_runtime_tradeoff.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest_path = args.input_root.resolve() / "results.jsonl"
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    systems = [item.strip().lower() for item in args.systems.split(",") if item.strip()]
    records = load_records(manifest_path)
    grouped = group_by_system(records)

    set_paper_style()
    for system in systems:
        if system in grouped:
            plot_system_figure(system, grouped[system], output_root)

    rows = cross_system_summary(systems, grouped)
    write_csv(output_root / "cross_system_summary.csv", rows)
    plot_cross_system_pareto(rows, output_root, args.error_threshold)
    plot_cross_system_grid_runtime_tradeoff(rows, output_root, args.error_threshold)


if __name__ == "__main__":
    main()
