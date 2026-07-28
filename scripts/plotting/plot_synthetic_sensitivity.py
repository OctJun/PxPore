#!/usr/bin/env python3
"""Evaluate and plot the five synthetic grid/octree sensitivity cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "synthetic_sensitivity_final"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs" / "figures" / "synthetic_sensitivity"
SYSTEMS = ("single_h", "512_h", "case1", "case2", "case3")
SYSTEM_LABELS = {
    "single_h": "Single H",
    "512_h": "512 H",
    "case1": "Center-overlapping H",
    "case2": "Throat",
    "case3": "Disconnected Region",
}
PARETO_PANEL_LABELS = {
    "single_h": "Single H",
    "512_h": "512 H",
    "case1": "Overlapping H",
    "case2": "Throat",
    "case3": "Trapped cavity",
}
PARETO_METRICS = {
    "single_h": ("error_Vprobe_nm3", r"Probe volume error, $V_{\mathrm{probe}}$"),
    "512_h": ("error_Vprobe_nm3", r"Probe volume error, $V_{\mathrm{probe}}$"),
    "case1": ("error_Vprobe_nm3", r"Probe volume error, $V_{\mathrm{probe}}$"),
    "case2": ("error_PLD_nm", "PLD error"),
    "case3": ("error_Vtrap_nm3", r"Inaccessible volume error, $V_{\mathrm{trap}}$"),
}
COLORS = ("#3B6EA8", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD", "#8C8C8C")
CASE_MARKERS = ("o", "s", "^", "D", "v")
METRICS = {
    "atoms": ("stats_atoms", "Atom count", "#4C4C4C"),
    "mass_g_per_mol": ("stats_mass_g/mol", "Molar mass", "#4C4C4C"),
    "density_g_per_cm3": ("stats_density_g/cm3", "Density", "#4C4C4C"),
    "Vcell_nm3": ("stats_Vcell_nm3", "Total volume", "#4C4C4C"),
    "Vprobe_nm3": ("stats_Vprobe_nm3", "Solid volume", "#3B6EA8"),
    "Vvoid_nm3": ("stats_Vvoid_nm3", "Void volume", "#55A868"),
    "Vacc_nm3": ("stats_Vacc_nm3", "Accessible volume", "#2A9D8F"),
    "Vtrap_nm3": ("stats_Vtrap_nm3", "Disconnected volume", "#C44E52"),
    "Vvoid_frac": ("stats_Vvoid_frac", "Void fraction", "#55A868"),
    "Vacc_frac": ("stats_Vacc_frac", "Accessible fraction", "#2A9D8F"),
    "Vtrap_frac": ("stats_Vtrap_frac", "Disconnected fraction", "#C44E52"),
    "Stotal_nm2": ("stats_Stotal_nm2", "Total surface", "#8172B2"),
    "Sacc_nm2": ("stats_Sacc_nm2", "Accessible surface", "#CCB974"),
    "Stotal_m2_per_g": ("stats_Stotal_m2/g", "Total specific surface", "#8172B2"),
    "Sacc_m2_per_g": ("stats_Sacc_m2/g", "Accessible specific surface", "#CCB974"),
    "PLD_nm": ("stats_PLD_nm", "PLD", "#E17C05"),
    "LCD_nm": ("stats_LCD_nm", "LCD", "#64B5CD"),
    "LCD_global_nm": ("stats_LCD_global_nm", "Global LCD", "#64B5CD"),
}
METRIC_OUTPUT_NAMES = {
    "atoms": "atom_count",
    "mass_g_per_mol": "molar_mass",
    "density_g_per_cm3": "density",
    "Vcell_nm3": "total_volume",
    "Vprobe_nm3": "solid_volume",
    "Vvoid_nm3": "void_volume",
    "Vacc_nm3": "accessible_volume",
    "Vtrap_nm3": "disconnected_volume",
    "Vvoid_frac": "void_fraction",
    "Vacc_frac": "accessible_fraction",
    "Vtrap_frac": "disconnected_fraction",
    "Stotal_nm2": "total_surface",
    "Sacc_nm2": "accessible_surface",
    "Stotal_m2_per_g": "total_specific_surface",
    "Sacc_m2_per_g": "accessible_specific_surface",
    "PLD_nm": "pld",
    "LCD_nm": "lcd",
    "LCD_global_nm": "global_lcd",
}
SYSTEM_OUTPUT_NAMES = {
    "single_h": "single_h",
    "512_h": "512_h",
    "case1": "center_overlapping_h",
    "case2": "throat",
    "case3": "disconnected_region",
}
METRIC_UNITS = {
    "atoms": "",
    "mass_g_per_mol": "g mol⁻¹",
    "density_g_per_cm3": "g cm⁻³",
    "Vcell_nm3": "nm³",
    "Vprobe_nm3": "nm³",
    "Vvoid_nm3": "nm³",
    "Vacc_nm3": "nm³",
    "Vtrap_nm3": "nm³",
    "Vvoid_frac": "",
    "Vacc_frac": "",
    "Vtrap_frac": "",
    "Stotal_nm2": "nm²",
    "Sacc_nm2": "nm²",
    "Stotal_m2_per_g": "m² g⁻¹",
    "Sacc_m2_per_g": "m² g⁻¹",
    "PLD_nm": "nm",
    "LCD_nm": "nm",
    "LCD_global_nm": "nm",
}
R_H_NM = 0.12
ERROR_FLOOR = 1e-7
AVOGADRO = 6.02214076e23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and tabulate synthetic PxPore sensitivity results."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "axes.grid": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def finite_float(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def max_available(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    return max(clean) if clean else None


def runtime_s(record: dict[str, Any]) -> float | None:
    for key in ("wall_time_s", "execution_time_s"):
        value = finite_float(record, key)
        if value is not None:
            return value
    return None


def theoretical_references() -> dict[str, dict[str, float]]:
    sphere_volume = 4.0 * math.pi * R_H_NM**3 / 3.0
    sphere_area = 4.0 * math.pi * R_H_NM**2
    case1_distance = R_H_NM
    cap_height = R_H_NM - case1_distance / 2.0
    cap_volume = math.pi * cap_height**2 * (R_H_NM - cap_height / 3.0)
    cap_area = 2.0 * math.pi * R_H_NM * cap_height
    case1_volume = 2.0 * sphere_volume - 2.0 * cap_volume
    case1_area = 2.0 * sphere_area - 2.0 * cap_area

    single_lattice = 0.4
    lattice_512 = 20.0 / 8.0
    single_solid = sphere_volume
    solid_512 = 512.0 * sphere_volume
    case3_inner_side = 1.2 - 2.0 * R_H_NM
    case3_volume_edge = (
        12.0 * case3_inner_side * (1.0 - math.pi / 4.0) * R_H_NM**2
    )
    case3_volume_corner = 8.0 * (1.0 - math.pi / 6.0) * R_H_NM**3
    case3_area_edge = (
        12.0
        * case3_inner_side
        * (1.0 - math.pi / 4.0)
        * (2.0 * R_H_NM)
    )
    case3_area_corner = (
        8.0 * (1.0 - math.pi / 6.0) * (3.0 * R_H_NM**2)
    )
    case3_void = (
        case3_inner_side**3 - case3_volume_edge - case3_volume_corner
    )
    case3_surface = (
        6.0 * case3_inner_side**2 - case3_area_edge - case3_area_corner
    )
    references = {
        "single_h": {
            "stats_Vprobe_nm3": single_solid,
            "stats_Vvoid_nm3": 0.4**3 - single_solid,
            "stats_Vacc_nm3": 0.4**3 - single_solid,
            "stats_Vtrap_nm3": 0.0,
            "stats_Stotal_nm2": sphere_area,
            "stats_Sacc_nm2": sphere_area,
            "stats_PLD_nm": math.sqrt(2.0) * single_lattice - 2.0 * R_H_NM,
            "stats_LCD_nm": math.sqrt(3.0) * single_lattice - 2.0 * R_H_NM,
        },
        "512_h": {
            "stats_Vprobe_nm3": solid_512,
            "stats_Vvoid_nm3": 20.0**3 - solid_512,
            "stats_Vacc_nm3": 20.0**3 - solid_512,
            "stats_Vtrap_nm3": 0.0,
            "stats_Stotal_nm2": 512.0 * sphere_area,
            "stats_Sacc_nm2": 512.0 * sphere_area,
            "stats_PLD_nm": math.sqrt(2.0) * lattice_512 - 2.0 * R_H_NM,
            "stats_LCD_nm": math.sqrt(3.0) * lattice_512 - 2.0 * R_H_NM,
        },
        "case1": {
            "stats_Vprobe_nm3": case1_volume,
            "stats_Vvoid_nm3": 1.0 - case1_volume,
            "stats_Vacc_nm3": 1.0 - case1_volume,
            "stats_Vtrap_nm3": 0.0,
            "stats_Stotal_nm2": case1_area,
            "stats_Sacc_nm2": case1_area,
        },
        "case2": {
            "stats_Vtrap_nm3": 0.0,
            "stats_PLD_nm": 0.46,
            "stats_LCD_nm": 1.46,
        },
        "case3": {
            "stats_Vprobe_nm3": 2.0**3 - case3_void,
            "stats_Vvoid_nm3": case3_void,
            "stats_Vacc_nm3": 0.0,
            "stats_Vtrap_nm3": case3_void,
            "stats_Stotal_nm2": case3_surface,
            "stats_Sacc_nm2": 0.0,
        },
    }
    system_properties = {
        "single_h": (1, 0.4**3),
        "512_h": (512, 20.0**3),
        "case1": (2, 1.0),
        "case2": (23010, 3.0 * 3.0 * 6.0),
        "case3": (4139, 2.0**3),
    }
    for system, (atom_count, cell_volume) in system_properties.items():
        system_reference = references[system]
        molar_mass = atom_count * 1.008
        system_reference["stats_atoms"] = float(atom_count)
        system_reference["stats_mass_g/mol"] = molar_mass
        system_reference["stats_Vcell_nm3"] = cell_volume
        system_reference["stats_density_g/cm3"] = (
            molar_mass / AVOGADRO / (cell_volume * 1e-21)
        )
        for volume_key, fraction_key in (
            ("stats_Vvoid_nm3", "stats_Vvoid_frac"),
            ("stats_Vacc_nm3", "stats_Vacc_frac"),
            ("stats_Vtrap_nm3", "stats_Vtrap_frac"),
        ):
            if volume_key in system_reference:
                system_reference[fraction_key] = (
                    system_reference[volume_key] / cell_volume
                )
        for surface_key, specific_key in (
            ("stats_Stotal_nm2", "stats_Stotal_m2/g"),
            ("stats_Sacc_nm2", "stats_Sacc_m2/g"),
        ):
            if surface_key in system_reference:
                system_reference[specific_key] = (
                    system_reference[surface_key]
                    * 1e-18
                    * AVOGADRO
                    / molar_mass
                )
        if "stats_LCD_nm" in system_reference:
            system_reference["stats_LCD_global_nm"] = system_reference["stats_LCD_nm"]
    return references


def select_numerical_reference(records: list[dict[str, Any]], system: str) -> dict[str, Any] | None:
    success = [
        record
        for record in records
        if record.get("system") == system and record.get("status") == "success"
    ]
    if not success:
        return None
    return min(
        success,
        key=lambda record: (
            float(record["grid_nm"]),
            -int(record["oct_level"]),
            runtime_s(record) or math.inf,
        ),
    )


def zero_reference_scale(
    metric: str,
    references: dict[str, dict[str, Any]],
) -> float:
    if metric.endswith("_frac"):
        fraction_reference = references.get("stats_Vvoid_frac", {}).get("value")
        if fraction_reference is not None and float(fraction_reference) > 0:
            return float(fraction_reference)
    if metric.startswith("stats_V"):
        void_reference = references.get("stats_Vvoid_nm3", {}).get("value")
        if void_reference is not None and float(void_reference) > 0:
            return float(void_reference)
    if metric.endswith("_m2/g"):
        specific_reference = references.get("stats_Stotal_m2/g", {}).get("value")
        if specific_reference is not None and float(specific_reference) > 0:
            return float(specific_reference)
    if metric.startswith("stats_S"):
        surface_reference = references.get("stats_Stotal_nm2", {}).get("value")
        if surface_reference is not None and float(surface_reference) > 0:
            return float(surface_reference)
    return 1.0


def build_references(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    theory = theoretical_references()
    numerical = {system: select_numerical_reference(records, system) for system in SYSTEMS}
    references: dict[str, dict[str, dict[str, Any]]] = {}
    reference_rows: list[dict[str, Any]] = []

    for system in SYSTEMS:
        system_references: dict[str, dict[str, Any]] = {}
        ref_record = numerical[system]
        for _short_name, (metric, _label, _color) in METRICS.items():
            if metric in theory.get(system, {}):
                reference_type = (
                    "theory (continuous geometry)"
                    if system in {"case2", "case3"}
                    else "analytic theory"
                )
                system_references[metric] = {
                    "value": float(theory[system][metric]),
                    "type": reference_type,
                }
                continue
            if ref_record is None:
                continue
            value = finite_float(ref_record, metric)
            if value is None or (
                metric in {"stats_PLD_nm", "stats_LCD_nm", "stats_LCD_global_nm"}
                and value < 0
            ):
                continue
            system_references[metric] = {
                "value": value,
                "type": "finest successful result",
            }

        for metric, info in system_references.items():
            info["scale"] = zero_reference_scale(metric, system_references)
            reference_rows.append(
                {
                    "system": system,
                    "metric": metric,
                    "reference": info["value"],
                    "type": info["type"],
                    "zero_reference_scale": info["scale"] if info["value"] == 0 else "",
                }
            )
        if ref_record is not None:
            reference_rows.append(
                {
                    "system": system,
                    "metric": "numerical_reference_setting",
                    "reference": f"grid={float(ref_record['grid_nm']):g}, oct={int(ref_record['oct_level'])}",
                    "type": "finest successful",
                }
            )
        references[system] = system_references

    return references, reference_rows


def metric_error(
    value: float | None,
    reference: dict[str, Any] | None,
) -> float | None:
    if value is None or reference is None:
        return None
    reference_value = float(reference["value"])
    if reference_value == 0:
        scale = float(reference.get("scale", 1.0))
        return abs(value) / scale if scale > 0 else None
    return abs(value - reference_value) / abs(reference_value)


def evaluate_records(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    references, reference_rows = build_references(records)
    rows: list[dict[str, Any]] = []

    for record in records:
        system = record.get("system")
        if system not in SYSTEMS:
            continue
        row = {
            "system": system,
            "status": record.get("status"),
            "grid_nm": float(record["grid_nm"]),
            "oct_level": int(record["oct_level"]),
            "estimated_voxels": record.get("estimated_voxels"),
            "runtime_s": runtime_s(record),
            "octree_nodes": record.get("info_octree_nodes"),
        }
        for short_name, (metric, _label, _color) in METRICS.items():
            row[short_name] = finite_float(record, metric)
        if record.get("status") != "success":
            rows.append(row)
            continue

        errors: list[float | None] = []
        for short_name, (metric, _label, _color) in METRICS.items():
            error = metric_error(
                finite_float(record, metric),
                references[system].get(metric),
            )
            row[f"error_{short_name}"] = error
            errors.append(error)
        row["max_metric_error"] = max_available(errors)
        row["primary_error"] = row["max_metric_error"]
        rows.append(row)

    return pd.DataFrame(rows), reference_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    success = df[df["status"] == "success"].copy()
    for system in SYSTEMS:
        subset = success[success["system"] == system].dropna(subset=["primary_error", "runtime_s"])
        for threshold in (0.01, 0.02, 0.05):
            feasible = subset[subset["primary_error"] <= threshold]
            if feasible.empty:
                rows.append({"system": system, "threshold": threshold, "status": "no feasible setting"})
                continue
            best = feasible.sort_values(["runtime_s", "primary_error"]).iloc[0]
            rows.append(
                {
                    "system": system,
                    "threshold": threshold,
                    "status": "success",
                    "grid_nm": best["grid_nm"],
                    "oct_level": int(best["oct_level"]),
                    "primary_error": best["primary_error"],
                    "runtime_s": best["runtime_s"],
                }
            )
    return pd.DataFrame(rows)


def style_grid_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grid spacing (nm)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25)


def plot_metric_errors(ax: plt.Axes, subset: pd.DataFrame) -> None:
    for short_name, (_metric, label, color) in METRICS.items():
        column = f"error_{short_name}"
        if column not in subset:
            continue
        part = subset.dropna(subset=[column]).sort_values("grid_nm")
        if part.empty:
            continue
        ax.plot(
            part["grid_nm"],
            np.maximum(part[column].to_numpy(float), ERROR_FLOOR),
            color=color,
            linewidth=1.7,
            label=label,
        )
    style_grid_axis(ax, "Relative / normalized error")


def plot_wall_time(ax: plt.Axes, subset: pd.DataFrame, color: str = "#333333") -> None:
    part = subset.dropna(subset=["runtime_s"]).sort_values("grid_nm")
    if not part.empty:
        ax.plot(part["grid_nm"], part["runtime_s"], color=color, linewidth=1.9)
    style_grid_axis(ax, "Wall time (s)")


def metric_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=color, linewidth=2.0, label=label)
        for _short_name, (_metric, label, color) in METRICS.items()
    ]


def save_oct_level_figure(
    df: pd.DataFrame,
    output_root: Path,
    oct_level: int,
) -> Path:
    fig, axes = plt.subplots(len(SYSTEMS), 2, figsize=(12.6, 16.5))
    for row_index, system in enumerate(SYSTEMS):
        subset = df[
            (df["system"] == system)
            & (df["status"] == "success")
            & (df["oct_level"] == oct_level)
        ]
        error_ax, time_ax = axes[row_index]
        plot_metric_errors(error_ax, subset)
        plot_wall_time(time_ax, subset)
        error_ax.set_title(f"{SYSTEM_LABELS[system]} — errors", loc="left")
        time_ax.set_title(f"{SYSTEM_LABELS[system]} — wall time", loc="left")

    fig.suptitle(f"Grid sensitivity at octree level {oct_level}", y=0.995)
    fig.legend(
        handles=metric_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=4,
    )
    fig.text(
        0.5,
        0.004,
        f"Exact zero errors are displayed at {ERROR_FLOOR:g} on the logarithmic axis.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94), h_pad=1.2, w_pad=1.0)
    output = output_root / f"oct_level_{oct_level}_metrics_walltime.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def save_combined_figure(df: pd.DataFrame, output_root: Path) -> Path:
    fig, axes = plt.subplots(len(SYSTEMS), 2, figsize=(12.6, 16.5))
    levels = sorted(int(value) for value in df["oct_level"].dropna().unique())
    for row_index, system in enumerate(SYSTEMS):
        system_rows = df[(df["system"] == system) & (df["status"] == "success")]
        error_ax, time_ax = axes[row_index]
        for color_index, oct_level in enumerate(levels):
            color = COLORS[color_index % len(COLORS)]
            subset = system_rows[system_rows["oct_level"] == oct_level]
            error_part = subset.dropna(subset=["max_metric_error"]).sort_values("grid_nm")
            if not error_part.empty:
                error_ax.plot(
                    error_part["grid_nm"],
                    np.maximum(error_part["max_metric_error"], ERROR_FLOOR),
                    color=color,
                    linewidth=1.7,
                    label=f"Oct {oct_level}",
                )
            time_part = subset.dropna(subset=["runtime_s"]).sort_values("grid_nm")
            if not time_part.empty:
                time_ax.plot(
                    time_part["grid_nm"],
                    time_part["runtime_s"],
                    color=color,
                    linewidth=1.7,
                )
        style_grid_axis(error_ax, "Maximum metric error")
        style_grid_axis(time_ax, "Wall time (s)")
        error_ax.set_title(f"{SYSTEM_LABELS[system]} — max error", loc="left")
        time_ax.set_title(f"{SYSTEM_LABELS[system]} — wall time", loc="left")

    level_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[index % len(COLORS)],
            linewidth=2.0,
            label=f"Oct {level}",
        )
        for index, level in enumerate(levels)
    ]
    fig.suptitle("Combined grid/octree sensitivity", y=0.995)
    fig.legend(
        handles=level_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=len(level_handles),
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.94), h_pad=1.2, w_pad=1.0)
    output = output_root / "all_oct_levels_combined.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def pareto_marker_size(oct_level: int) -> float:
    """Scatter marker area in points^2; deliberately separated for readability."""
    return 55.0 + 40.0 * float(oct_level)


def save_accuracy_runtime_pareto(df: pd.DataFrame, output_root: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=False)
    axes_flat = axes.ravel()
    success = df[df["status"] == "success"].copy()
    grid_values = success["grid_nm"].dropna().to_numpy(float)
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(grid_values)), vmax=float(np.nanmax(grid_values)))
    cmap = mpl.colormaps["viridis_r"]

    for index, system in enumerate(SYSTEMS):
        ax = axes_flat[index]
        error_column, metric_label = PARETO_METRICS[system]
        subset = success.dropna(subset=["runtime_s", "grid_nm", "oct_level", error_column])
        subset = subset[subset["system"] == system].copy()
        if subset.empty:
            ax.axis("off")
            continue

        y = np.maximum(subset[error_column].to_numpy(float), ERROR_FLOOR)
        sizes = [pareto_marker_size(int(level)) for level in subset["oct_level"]]
        sc = ax.scatter(
            subset["runtime_s"],
            y,
            c=subset["grid_nm"],
            s=sizes,
            cmap=cmap,
            norm=norm,
            edgecolors="#333333",
            linewidths=0.45,
            alpha=0.94,
            zorder=10,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Relative error")
        ax.set_title(f"{PARETO_PANEL_LABELS[system]}\n{metric_label}", y=1.02, pad=8)
        ax.grid(True, which="major", alpha=0.25, zorder=0)
        cb = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.055)
        cb.set_label("Grid (nm)")

    axes_flat[-1].axis("off")

    oct_levels = sorted(int(value) for value in success["oct_level"].dropna().unique())
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#BDBDBD",
            markeredgecolor="#333333",
            markersize=math.sqrt(pareto_marker_size(level)),
            label=f"Oct {level}",
        )
        for level in oct_levels
    ]
    fig.legend(
        handles=size_handles,
        title="Marker size: octree level",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(size_handles),
        columnspacing=1.1,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.1, w_pad=1.2)
    output = output_root / "synthetic_accuracy_runtime_pareto.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def save_metric_oct_level_figure(
    df: pd.DataFrame,
    output_root: Path,
    short_name: str,
    oct_level: int,
) -> Path:
    _metric, metric_label, _metric_color = METRICS[short_name]
    error_column = f"error_{short_name}"
    fig, ax_error = plt.subplots(figsize=(7.25, 5.95))
    ax_time = ax_error.twinx()
    plotted_systems: list[str] = []

    for index, system in enumerate(SYSTEMS):
        subset = df[
            (df["system"] == system)
            & (df["status"] == "success")
            & (df["oct_level"] == oct_level)
        ].sort_values("grid_nm")
        error_rows = subset.dropna(subset=[error_column])
        time_rows = subset.dropna(subset=["runtime_s"])
        if error_rows.empty:
            continue

        color = COLORS[index]
        marker = CASE_MARKERS[index]
        ax_error.plot(
            error_rows["grid_nm"],
            np.maximum(error_rows[error_column].to_numpy(float), ERROR_FLOOR),
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.8,
            markersize=5.0,
            zorder=100,
        )
        if not time_rows.empty:
            ax_time.plot(
                time_rows["grid_nm"],
                time_rows["runtime_s"],
                color=color,
                marker=marker,
                markerfacecolor="white",
                linestyle="--",
                linewidth=1.5,
                markersize=4.5,
                alpha=0.8,
                zorder=20,
            )
        plotted_systems.append(system)

    ax_error.set_xscale("log")
    ax_error.set_yscale("log")
    ax_time.set_yscale("log")
    ax_error.set_xlabel("Grid size (nm)")
    ax_error.set_ylabel(f"{metric_label} relative error", color="#3B6EA8")
    ax_time.set_ylabel("Wall Time (s)", color="#C44E52")
    ax_error.tick_params(axis="y", colors="#3B6EA8")
    ax_time.tick_params(axis="y", colors="#C44E52")
    ax_error.grid(True, which="major", axis="both", alpha=0.25, zorder=0)

    case_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[SYSTEMS.index(system)],
            marker=CASE_MARKERS[SYSTEMS.index(system)],
            linewidth=1.8,
            markersize=4.8,
            label=SYSTEM_LABELS[system],
        )
        for system in plotted_systems
    ]
    style_handles = [
        Line2D([0], [0], color="#333333", linewidth=1.8, linestyle="-", label="Error"),
        Line2D([0], [0], color="#333333", linewidth=1.5, linestyle="--", label="Wall time"),
    ]
    ax_error.legend(
        handles=case_handles + style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=9,
        columnspacing=1.0,
        handlelength=2.0,
    )
    ax_error.text(
        0.02,
        0.02,
        f"Octree level {oct_level}",
        transform=ax_error.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    metric_root = output_root / METRIC_OUTPUT_NAMES[short_name]
    metric_root.mkdir(parents=True, exist_ok=True)
    output = metric_root / f"oct_level_{oct_level}.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def save_case_metric_oct_level_figure(
    df: pd.DataFrame,
    output_root: Path,
    system: str,
    short_name: str,
    oct_level: int,
) -> Path | None:
    _metric, metric_label, _metric_color = METRICS[short_name]
    error_column = f"error_{short_name}"
    subset = df[
        (df["system"] == system)
        & (df["status"] == "success")
        & (df["oct_level"] == oct_level)
    ].sort_values("grid_nm")
    error_rows = subset.dropna(subset=[error_column])
    time_rows = subset.dropna(subset=["runtime_s"])
    if error_rows.empty or time_rows.empty:
        return None

    fig, ax_error = plt.subplots(figsize=(7.25, 5.95))
    ax_error.plot(
        error_rows["grid_nm"],
        np.maximum(error_rows[error_column].to_numpy(float), ERROR_FLOOR),
        marker="o",
        linestyle="-",
        color="#3B6EA8",
        label=f"{metric_label} error",
        zorder=100,
    )
    ax_error.set_xscale("log")
    ax_error.set_yscale("log")
    ax_error.set_xlabel("Grid size (nm)")
    ax_error.set_ylabel(f"{metric_label} relative error", color="#3B6EA8")
    ax_error.tick_params(axis="y", colors="#3B6EA8")
    ax_error.grid(True, which="major", axis="both", alpha=0.25, zorder=0)

    ax_time = ax_error.twinx()
    ax_time.plot(
        time_rows["grid_nm"],
        time_rows["runtime_s"],
        marker="s",
        linestyle="--",
        color="#C44E52",
        label="Wall time",
        zorder=20,
    )
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Wall Time (s)", color="#C44E52")
    ax_time.tick_params(axis="y", colors="#C44E52")

    ax_error.text(
        0.05,
        0.95,
        f"{SYSTEM_LABELS[system]}\nOctree level {oct_level}",
        transform=ax_error.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )
    fig.tight_layout()

    figure_root = (
        output_root
        / SYSTEM_OUTPUT_NAMES[system]
        / METRIC_OUTPUT_NAMES[short_name]
    )
    figure_root.mkdir(parents=True, exist_ok=True)
    output = figure_root / f"oct_level_{oct_level}.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def format_reference_annotation(
    reference: dict[str, Any] | None,
    short_name: str,
    numerical_reference: dict[str, Any] | None,
) -> str:
    if reference is None:
        return "N/A"
    value = float(reference["value"])
    unit = METRIC_UNITS[short_name]
    value_text = f"{value:.5g}" + (f" {unit}" if unit else "")
    if str(reference["type"]).startswith("theory") or reference["type"] == "analytic theory":
        return f"Theory: {value_text}"
    if numerical_reference is None:
        return f"Best: {value_text}"
    return (
        f"Best: {value_text} "
        f"(g={float(numerical_reference['grid_nm']):g}, "
        f"oct={int(numerical_reference['oct_level'])})"
    )


def save_case_metric_table_figure(
    df: pd.DataFrame,
    output_root: Path,
    system: str,
    references: dict[str, dict[str, Any]],
    numerical_reference: dict[str, Any] | None,
) -> Path:
    levels = (0, 1, 2, 3)
    fig, axes = plt.subplots(6, 3, figsize=(15.0, 22.0))
    axes_flat = axes.ravel()

    for metric_index, (short_name, (metric, metric_label, _metric_color)) in enumerate(METRICS.items()):
        ax_error = axes_flat[metric_index]
        reference = references.get(metric)
        annotation = format_reference_annotation(
            reference,
            short_name,
            numerical_reference,
        )
        if reference is None:
            ax_error.set_title(f"{metric_label}\n{annotation}", fontsize=10)
            ax_error.axis("off")
            continue

        ax_time = ax_error.twinx()
        error_column = f"error_{short_name}"
        for level in levels:
            subset = df[
                (df["system"] == system)
                & (df["status"] == "success")
                & (df["oct_level"] == level)
            ].sort_values("grid_nm")
            error_rows = subset.dropna(subset=[error_column])
            time_rows = subset.dropna(subset=["runtime_s"])
            color = COLORS[level]
            if not error_rows.empty:
                ax_error.plot(
                    error_rows["grid_nm"],
                    np.maximum(error_rows[error_column].to_numpy(float), ERROR_FLOOR),
                    color=color,
                    linestyle="-",
                    linewidth=1.4,
                )
            if not time_rows.empty:
                ax_time.plot(
                    time_rows["grid_nm"],
                    time_rows["runtime_s"],
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                )

        ax_error.set_xscale("log")
        ax_error.set_yscale("log")
        ax_time.set_yscale("log")
        ax_error.set_title(f"{metric_label}\n{annotation}", fontsize=10)
        ax_error.grid(True, which="major", alpha=0.22, zorder=0)
        ax_error.tick_params(axis="y", colors="#3B6EA8", labelsize=8)
        ax_time.tick_params(axis="y", colors="#C44E52", labelsize=8)
        ax_error.tick_params(axis="x", labelsize=8)
        if metric_index % 3 == 0:
            ax_error.set_ylabel("Relative error", color="#3B6EA8", fontsize=9)
        if metric_index % 3 == 2:
            ax_time.set_ylabel("Wall time (s)", color="#C44E52", fontsize=9)
        if metric_index >= 15:
            ax_error.set_xlabel("Grid size (nm)", fontsize=9)

    level_handles = [
        Line2D([0], [0], color=COLORS[level], linewidth=2.0, label=f"Oct {level}")
        for level in levels
    ]
    style_handles = [
        Line2D([0], [0], color="#333333", linewidth=1.6, linestyle="-", label="Error"),
        Line2D([0], [0], color="#333333", linewidth=1.2, linestyle="--", label="Wall time"),
    ]
    fig.suptitle(SYSTEM_LABELS[system], fontsize=18, y=0.995)
    fig.legend(
        handles=level_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.977),
        ncol=6,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.004,
        f"Octree levels 0–3; exact zero errors are shown at {ERROR_FLOOR:g}.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.015, 1, 0.955), h_pad=1.5, w_pad=1.1)

    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{SYSTEM_OUTPUT_NAMES[system]}_oct_0_to_3_metrics.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = load_records(input_root / "results.jsonl")
    evaluated, reference_rows = evaluate_records(records)
    references_by_system, _ = build_references(records)
    evaluated.to_csv(output_root / "synthetic_all_results.csv", index=False)
    write_csv(output_root / "synthetic_references.csv", reference_rows)
    recommendations = build_recommendations(evaluated)
    recommendations.to_csv(output_root / "synthetic_recommendations.csv", index=False)

    set_paper_style()
    save_accuracy_runtime_pareto(evaluated, output_root)
    by_case_root = output_root / "by_case_metric_oct_level"
    by_case_root.mkdir(parents=True, exist_ok=True)
    oct_levels = sorted(int(value) for value in evaluated["oct_level"].dropna().unique())
    for system in SYSTEMS:
        for short_name in METRICS:
            for oct_level in oct_levels:
                save_case_metric_oct_level_figure(
                    evaluated,
                    by_case_root,
                    system,
                    short_name,
                    oct_level,
                )
    case_table_root = output_root / "case_metric_tables"
    for system in SYSTEMS:
        save_case_metric_table_figure(
            evaluated,
            case_table_root,
            system,
            references_by_system[system],
            select_numerical_reference(records, system),
        )

    status_counts = evaluated.groupby(["system", "status"], dropna=False).size().reset_index(name="count")
    status_counts.to_csv(output_root / "synthetic_status_counts.csv", index=False)
    print(output_root)


if __name__ == "__main__":
    main()
