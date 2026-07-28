#!/usr/bin/env python3
"""汇总敏感性扫描，输出表格以及 PNG/PDF 图片。"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

MPL_CACHE = Path(
    os.environ.get("MPLCONFIGDIR", "/tmp/pxpore_matplotlib")
)
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("sensitivity_study.json")
PSD_SCANS = (
    "psd_method",
    "psd_local_max",
    "psd_min_radius",
    "psd_overlap",
    "psd_weighting",
    "psd_mc_samples",
)
ALL_SCANS = (
    "grid", "octree", "surface", "probe",
    *PSD_SCANS, "connectivity",
)
PORE_METRICS = {
    "PSD", "center_count", "PLD_nm", "LCD_nm", "LCD_global_nm"
}
SCAN_LABELS = {
    "grid": ("Grid spacing (nm)", False),
    "octree": ("Octree level", False),
    "surface": ("Fibonacci samples per atom", True),
    "probe": ("Probe radius (nm)", False),
    "connectivity": ("Transport direction", False),
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_scans(value):
    if value == "all":
        return ALL_SCANS
    items = [item.strip() for item in value.split(",") if item.strip()]
    if "psd" in items:
        items = [
            scan
            for item in items
            for scan in (PSD_SCANS if item == "psd" else (item,))
        ]
    scans = tuple(dict.fromkeys(items))
    unknown = sorted(set(scans) - set(ALL_SCANS))
    if unknown:
        raise ValueError(f"Unknown scans: {', '.join(unknown)}")
    return scans


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_value(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def write_markdown(path, rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                markdown_value(row.get(column, ""))
                for column in columns
            ) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def numeric(value):
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def valid_output(row, metric):
    value = numeric(row.get(metric))
    if value is None:
        return None
    if metric in {"PLD_nm", "LCD_nm", "LCD_global_nm"} and value < 0:
        return None
    return value


def study_design_rows(config):
    ranges = config["parameter_ranges"]
    rows = []
    for system, spec in config["systems"].items():
        rows.append({
            "system": system,
            "category": spec["category"],
            "description": spec["description"],
            "structure": spec["path"],
            "connectivity": spec["connectivity"],
            "transport_direction": spec["transport_direction"],
            "baseline_grid_nm": spec["baseline_grid_nm"],
            "octree_grid_nm": spec.get(
                "octree_grid_nm", spec["baseline_grid_nm"]),
            "grid_range_nm": ", ".join(
                str(value)
                for value in ranges["grid_nm"][spec["grid_class"]]
            ),
            "octree_levels": ", ".join(
                str(value) for value in spec.get(
                    "octree_levels", ranges["octree_level"])),
            "surface_samples": ", ".join(
                str(value) for value in ranges["surface_samples"]),
            "probe_values_nm": (
                ", ".join(str(value) for value in ranges["probe_nm"])
                if spec.get("probe_scan", False)
                else "not applicable"
            ),
            "psd_methods": (
                "centers, mc (3 seeds)"
                if spec["pore_applicable"] else "not applicable"
            ),
            "psd_local_max_modes": (
                ", ".join(ranges["psd_local_max_mode"])
                if spec["pore_applicable"] else "not applicable"
            ),
            "psd_min_center_radius_nm": (
                ", ".join(
                    str(value)
                    for value in ranges["psd_min_center_radius_nm"]
                )
                if spec["pore_applicable"] else "not applicable"
            ),
            "psd_overlap_thresholds": (
                ", ".join(
                    str(value)
                    for value in ranges["psd_overlap_threshold"]
                ) + ", pruning disabled"
                if spec["pore_applicable"] else "not applicable"
            ),
            "psd_hist_weightings": (
                ", ".join(ranges["psd_hist_weighting"])
                if spec["pore_applicable"] else "not applicable"
            ),
            "psd_mc_samples": (
                ", ".join(
                    str(value) for value in ranges["psd_mc_samples"]
                )
                if spec["pore_applicable"] else "not applicable"
            ),
            "grid_metrics": ", ".join(spec["metrics"]["grid"]),
            "octree_metrics": ", ".join(spec["metrics"]["octree"]),
            "surface_metrics": ", ".join(spec["metrics"]["surface"]),
            "psd_metrics": ", ".join(spec["metrics"]["psd_method"]),
            "connectivity_directions": (
                ", ".join(ranges["transport_direction"])
                if spec.get("connectivity_scan", False)
                else "not applicable"
            ),
            "connectivity_metrics": ", ".join(
                spec["metrics"].get("connectivity", ())),
        })
    return rows


def reference_for(
    config, raw_rows, system, scan, metric, parameter_value
):
    spec = config["systems"][system]
    if scan == "connectivity":
        expected = spec.get(
            "connectivity_reference", {}).get(
                str(parameter_value), {}).get(metric)
        if expected is not None:
            return (
                float(expected),
                f"directional expected value: {parameter_value}",
                "expected",
            )
    if scan == "probe":
        model = spec.get("probe_reference", {}).get(metric)
        if model is not None:
            value = (
                float(model["intercept"])
                + float(model["slope"]) * float(parameter_value)
            )
            value = max(float(model.get("minimum", -math.inf)), value)
            return value, model["source"], "theoretical"
    reference = spec.get(
        "reference", {}).get(metric)
    if reference is not None:
        return (
            float(reference["value"]),
            reference["source"],
            "theoretical",
        )
    candidates = [
        row for row in raw_rows
        if row["system"] == system
        and row["scan"] == scan
        and row["status"] in {"PASS", "CACHED"}
        and valid_output(row, metric) is not None
    ]
    if not candidates:
        return None, "", ""
    if scan == "grid":
        selected = min(
            candidates, key=lambda row: float(row["parameter_value"]))
        source = f"finest grid {selected['parameter_value']} nm"
    elif scan == "octree":
        selected = max(
            candidates, key=lambda row: int(float(row["parameter_value"])))
        source = f"deepest octree level {selected['parameter_value']}"
    elif scan == "surface":
        selected = max(
            candidates, key=lambda row: int(float(row["parameter_value"])))
        source = (
            f"highest surface sampling {selected['parameter_value']}"
        )
    else:
        selected = next(
            (
                row for row in candidates
                if row["parameter_value"] == "centers"
            ),
            candidates[0],
        )
        source = "centers method"
    return valid_output(selected, metric), source, "numerical"


def metric_sensitivity_rows(config, raw_rows):
    rows = []
    for row in raw_rows:
        if row["status"] not in {"PASS", "CACHED"}:
            continue
        system = row["system"]
        scan = row["scan"]
        for metric in config["systems"][system]["metrics"].get(scan, ()):
            if metric == "PSD":
                continue
            output = valid_output(row, metric)
            if output is None:
                continue
            reference, source, reference_type = reference_for(
                config,
                raw_rows,
                system,
                scan,
                metric,
                row["parameter_value"],
            )
            if reference is None:
                continue
            signed_error = output - reference
            relative_error = (
                100.0 * signed_error / abs(reference)
                if reference != 0 else None
            )
            rows.append({
                "run_id": row["run_id"],
                "system": system,
                "category": row["category"],
                "scan": scan,
                "parameter_value": row["parameter_value"],
                "metric": metric,
                "output": output,
                "reference": reference,
                "reference_type": reference_type,
                "reference_source": source,
                "signed_error": signed_error,
                "absolute_error": abs(signed_error),
                "relative_error_percent": (
                    "" if relative_error is None else relative_error
                ),
                "elapsed_s": numeric(row.get("elapsed_s")) or 0.0,
            })
    return rows


def convergence_summary_rows(metric_rows):
    grouped = defaultdict(list)
    for row in metric_rows:
        grouped[
            row["system"], row["scan"], row["metric"]
        ].append(row)
    summary = []
    for (system, scan, metric), values in sorted(grouped.items()):
        outputs = [row["output"] for row in values]
        relative = [
            numeric(row["relative_error_percent"]) for row in values
        ]
        relative = [value for value in relative if value is not None]
        directional = scan == "connectivity"
        summary.append({
            "system": system,
            "scan": scan,
            "metric": metric,
            "minimum": min(outputs),
            "maximum": max(outputs),
            "range": max(outputs) - min(outputs),
            "max_absolute_error": max(
                row["absolute_error"] for row in values),
            "max_absolute_relative_error_percent": (
                max(abs(value) for value in relative)
                if relative else ""
            ),
            "reference": "" if directional else values[0]["reference"],
            "reference_type": (
                "expected" if directional
                else values[0]["reference_type"]
            ),
            "reference_source": (
                "direction-specific expected values"
                if directional else values[0]["reference_source"]
            ),
        })
    return summary


def probe_result_rows(config, raw_rows):
    rows = []
    for row in raw_rows:
        if (
            row["scan"] != "probe"
            or row["status"] not in {"PASS", "CACHED"}
        ):
            continue
        result = {
            "system": row["system"],
            "probe_nm": float(row["parameter_value"]),
            "Vacc_nm3": valid_output(row, "Vacc_nm3"),
            "Vtrap_nm3": valid_output(row, "Vtrap_nm3"),
            "elapsed_s": numeric(row.get("elapsed_s")) or 0.0,
        }
        for metric in ("PLD_nm", "LCD_nm"):
            calculated = valid_output(row, metric)
            expected, source, _ = reference_for(
                config,
                raw_rows,
                row["system"],
                "probe",
                metric,
                row["parameter_value"],
            )
            result[f"calculated_{metric}"] = calculated
            result[f"expected_{metric}"] = expected
            result[f"absolute_error_{metric}"] = (
                abs(calculated - expected)
                if calculated is not None and expected is not None
                else ""
            )
            result[f"reference_{metric}"] = source
        rows.append(result)
    return sorted(rows, key=lambda row: (row["system"], row["probe_nm"]))


def read_psd_curve(path, method):
    if not path.is_file():
        return np.array([]), np.array([])
    diameters = []
    fractions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        diameters.append(float(fields[1]))
        fractions.append(float(fields[3]))
    x = np.asarray(diameters, dtype=float)
    p = np.asarray(fractions, dtype=float)
    valid = np.isfinite(x) & np.isfinite(p) & (p >= 0)
    x, p = x[valid], p[valid]
    if not len(x) or p.sum() <= 0:
        return np.array([]), np.array([])
    order = np.argsort(x)
    return x[order], p[order] / p.sum()


def quantile(x, p, probability):
    if not len(x):
        return float("nan")
    index = np.searchsorted(
        np.cumsum(p), probability, side="left")
    return float(x[min(index, len(x) - 1)])


def curve_descriptors(x, p):
    return {
        "peak_diameter_nm": float(x[np.argmax(p)]),
        "mean_diameter_nm": float(np.sum(x * p)),
        "D10_nm": quantile(x, p, 0.10),
        "D50_nm": quantile(x, p, 0.50),
        "D90_nm": quantile(x, p, 0.90),
    }


def curve_distances(x, p, reference_x, reference_p):
    support = np.unique(np.concatenate((x, reference_x)))
    cdf = np.array([p[x <= value].sum() for value in support])
    reference_cdf = np.array([
        reference_p[reference_x <= value].sum()
        for value in support
    ])
    wasserstein = float(np.trapezoid(
        np.abs(cdf - reference_cdf), support))
    rounded_support = sorted(
        set(np.round(x, 8)) | set(np.round(reference_x, 8)))
    left = {round(value, 8): weight for value, weight in zip(x, p)}
    right = {
        round(value, 8): weight
        for value, weight in zip(reference_x, reference_p)
    }
    pv = np.array([left.get(value, 0.0) for value in rounded_support])
    qv = np.array([right.get(value, 0.0) for value in rounded_support])
    mean = 0.5 * (pv + qv)
    left_mask = pv > 0
    right_mask = qv > 0
    js = 0.5 * np.sum(
        pv[left_mask] * np.log2(pv[left_mask] / mean[left_mask])
    ) + 0.5 * np.sum(
        qv[right_mask] * np.log2(qv[right_mask] / mean[right_mask])
    )
    return wasserstein, float(js)


def collect_psd(config, raw_rows, work_dir):
    curves = {}
    descriptors = []
    for row in raw_rows:
        if (
            row["scan"] not in PSD_SCANS
            or row["status"] not in {"PASS", "CACHED"}
        ):
            continue
        method = row.get("argument_psd_method", "")
        if method not in {"centers", "mc"}:
            method = (
                "mc" if row["scan"] == "psd_mc_samples"
                or row["parameter_value"] == "mc" else "centers"
            )
        filename = (
            "result_psd.txt"
            if method == "centers" else "result_voxel_mc_psd.txt"
        )
        path = work_dir / "runs" / row["run_id"] / filename
        x, p = read_psd_curve(path, method)
        if not len(x):
            continue
        curves[row["run_id"]] = (x, p)
        descriptor = {
            "run_id": row["run_id"],
            "system": row["system"],
            "scan": row["scan"],
            "parameter_value": row["parameter_value"],
            "method": method,
            "label": row["label"],
            **curve_descriptors(x, p),
        }
        if method == "centers":
            descriptor["center_count"] = len(read_center_radii(
                work_dir / "runs" / row["run_id"] / "result_center.txt"
            ))
        else:
            descriptor["center_count"] = ""
        descriptors.append(descriptor)

    centers = {
        row["system"]: curves[row["run_id"]]
        for row in raw_rows
        if row["scan"] == "psd_method"
        and row["parameter_value"] == "centers"
        and row["run_id"] in curves
    }
    for row in descriptors:
        reference = centers.get(row["system"])
        curve = curves[row["run_id"]]
        if reference is None:
            row["wasserstein_to_centers_nm"] = ""
            row["JS_to_centers"] = ""
        else:
            wasserstein, js = curve_distances(
                curve[0], curve[1], reference[0], reference[1])
            row["wasserstein_to_centers_nm"] = wasserstein
            row["JS_to_centers"] = js
    return curves, descriptors


def read_center_radii(path):
    if not path.is_file():
        return np.array([])
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return np.array([])
    try:
        count = int(lines[0])
    except ValueError:
        return np.array([])
    return np.empty(count, dtype=float)


def psd_comparison_rows(descriptors):
    grouped = defaultdict(list)
    centers = {}
    for row in descriptors:
        if row["scan"] != "psd_method":
            continue
        if row["method"] == "centers":
            centers[row["system"]] = row
        else:
            grouped[row["system"]].append(row)
    metrics = (
        "peak_diameter_nm", "mean_diameter_nm",
        "D10_nm", "D50_nm", "D90_nm",
        "wasserstein_to_centers_nm", "JS_to_centers",
    )
    rows = []
    for system, mc_rows in sorted(grouped.items()):
        result = {
            "system": system,
            "mc_repeats": len(mc_rows),
        }
        center = centers.get(system, {})
        for metric in metrics[:5]:
            result[f"centers_{metric}"] = center.get(metric, "")
        for metric in metrics:
            values = [
                numeric(row.get(metric)) for row in mc_rows
            ]
            values = [value for value in values if value is not None]
            result[f"mc_mean_{metric}"] = (
                float(np.mean(values)) if values else "")
            result[f"mc_std_{metric}"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1 else 0.0 if values else ""
            )
        rows.append(result)
    return rows


def psd_parameter_rows(descriptors):
    columns = (
        "center_count", "peak_diameter_nm", "mean_diameter_nm",
        "D10_nm", "D50_nm", "D90_nm",
        "wasserstein_to_centers_nm", "JS_to_centers",
    )
    rows = []
    for descriptor in descriptors:
        row = {
            key: descriptor.get(key, "")
            for key in (
                "run_id", "system", "scan", "parameter_value",
                "label", "method",
            )
        }
        row.update({
            key: descriptor.get(key, "")
            for key in columns
        })
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["system"], row["scan"], row["label"], row["run_id"]
        ),
    )


def save_figure(figure, base_path):
    figure.savefig(
        base_path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        base_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_metric_panels(system, scan, rows, figure_dir):
    selected = [
        row for row in rows
        if row["system"] == system and row["scan"] == scan
    ]
    metrics = sorted({row["metric"] for row in selected})
    if not metrics:
        return
    columns = min(3, len(metrics))
    row_count = math.ceil(len(metrics) / columns)
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(4.3 * columns, 3.4 * row_count),
        squeeze=False,
    )
    xlabel, log_x = SCAN_LABELS[scan]
    for ax, metric in zip(axes.flat, metrics):
        values = [
            row for row in selected if row["metric"] == metric
        ]
        if scan == "connectivity":
            order = {"x": 0, "y": 1, "z": 2, "any": 3}
            values.sort(
                key=lambda row: order[row["parameter_value"]])
            x = np.arange(len(values))
        else:
            values.sort(key=lambda row: float(row["parameter_value"]))
            x = [float(row["parameter_value"]) for row in values]
        y = [row["output"] for row in values]
        ax.plot(
            x, y, "o-", linewidth=1.4, color="#2f6f9f",
            label="calculated value",
        )
        if scan in {"connectivity", "probe"}:
            ax.plot(
                x,
                [row["reference"] for row in values],
                "s--",
                color="black",
                linewidth=0.9,
                markersize=4,
                label="expected value",
            )
        else:
            reference = values[0]["reference"]
            ax.axhline(
                reference, color="black", linestyle="--",
                linewidth=0.9, label=values[0]["reference_type"])
        if log_x:
            ax.set_xscale("log")
        if scan == "connectivity":
            ax.set_xticks(
                x, [row["parameter_value"] for row in values])
        ax.set_title(metric)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Calculated value")
        ax.grid(alpha=0.25)
        axes_for_legend = [ax]
        relative_values = [
            numeric(row["relative_error_percent"]) for row in values
        ]
        if metric not in PORE_METRICS and any(
            value is not None for value in relative_values
        ):
            error_ax = ax.twinx()
            valid = [
                (x_value, error)
                for x_value, error in zip(x, relative_values)
                if error is not None
            ]
            error_ax.plot(
                [value[0] for value in valid],
                [value[1] for value in valid],
                "s:", linewidth=1.2, markersize=4,
                color="#b3453f", label="relative error",
            )
            error_ax.axhline(
                0.0, color="#b3453f", linewidth=0.7, alpha=0.35)
            error_ax.set_ylabel(
                "Relative error (%)", color="#b3453f")
            error_ax.tick_params(axis="y", colors="#b3453f")
            axes_for_legend.append(error_ax)
        handles = []
        labels = []
        for legend_ax in axes_for_legend:
            axis_handles, axis_labels = (
                legend_ax.get_legend_handles_labels())
            handles.extend(axis_handles)
            labels.extend(axis_labels)
        ax.legend(
            handles, labels, fontsize=7, frameon=False,
            loc="best",
        )
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    figure.suptitle(
        f"{system.replace('_', ' ')}: {scan} sensitivity")
    figure.tight_layout()
    save_figure(
        figure, figure_dir / f"{system}_{scan}_sensitivity")


def plot_psd_system(system, raw_rows, curves, figure_dir):
    selected = [
        row for row in raw_rows
        if row["system"] == system
        and row["scan"] == "psd_method"
        and row["run_id"] in curves
    ]
    center_rows = [
        row for row in selected
        if row["parameter_value"] == "centers"
    ]
    mc_rows = [
        row for row in selected
        if row["parameter_value"] == "mc"
    ]
    if not center_rows or not mc_rows:
        return
    figure, ax = plt.subplots(figsize=(6.5, 4.2))
    center_x, center_p = curves[center_rows[0]["run_id"]]
    ax.plot(
        center_x, center_p, linewidth=1.8,
        color="black", label="centers")

    supports = sorted(set().union(*[
        set(curves[row["run_id"]][0]) for row in mc_rows
    ]))
    support = np.asarray(supports, dtype=float)
    matrix = []
    for row in mc_rows:
        x, p = curves[row["run_id"]]
        mapping = {value: weight for value, weight in zip(x, p)}
        matrix.append([mapping.get(value, 0.0) for value in support])
    matrix = np.asarray(matrix)
    mean = matrix.mean(axis=0)
    ax.plot(
        support, mean, linewidth=1.6,
        color="#1f77b4", label="MC mean")
    if len(matrix) > 1:
        ax.fill_between(
            support, matrix.min(axis=0), matrix.max(axis=0),
            color="#1f77b4", alpha=0.2, label="MC seed range")
    ax.set_title(f"{system.replace('_', ' ')}: PSD method")
    ax.set_xlabel("Pore diameter (nm)")
    ax.set_ylabel("Bin fraction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    figure.tight_layout()
    save_figure(
        figure, figure_dir / f"{system}_psd_method_sensitivity")


def plot_psd_parameter_system(
    system, scan, raw_rows, curves, figure_dir
):
    selected = [
        row for row in raw_rows
        if row["system"] == system
        and row["scan"] == scan
        and row["run_id"] in curves
    ]
    if not selected:
        return
    figure, ax = plt.subplots(figsize=(7.0, 4.5))
    for row in selected:
        x, p = curves[row["run_id"]]
        ax.plot(x, p, linewidth=1.25, label=row["label"])
    ax.set_title(
        f"{system.replace('_', ' ')}: "
        f"{scan.removeprefix('psd_').replace('_', ' ')}"
    )
    ax.set_xlabel("Pore diameter (nm)")
    ax.set_ylabel("Bin fraction")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    figure.tight_layout()
    save_figure(figure, figure_dir / f"{system}_{scan}")


def plot_combined_metric_summary(summary, scan, figure_dir):
    selected = [
        row for row in summary
        if row["scan"] == scan
        and row["metric"] not in PORE_METRICS
        and numeric(
            row["max_absolute_relative_error_percent"]) is not None
    ]
    if not selected:
        return
    systems = sorted({row["system"] for row in selected})
    values = []
    for system in systems:
        errors = [
            float(row["max_absolute_relative_error_percent"])
            for row in selected if row["system"] == system
        ]
        values.append(max(errors))
    figure, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(np.arange(len(systems)), values, color="#4472a8")
    ax.set_xticks(
        np.arange(len(systems)),
        [system.replace("_", "\n") for system in systems],
    )
    ax.set_ylabel("Maximum absolute relative deviation (%)")
    ax.set_title(f"All systems: {scan} sensitivity")
    ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(
        figure, figure_dir / f"all_systems_{scan}_sensitivity")


def plot_combined_psd(comparison, figure_dir):
    if not comparison:
        return
    systems = [row["system"] for row in comparison]
    means = [
        numeric(row["mc_mean_wasserstein_to_centers_nm"]) or 0.0
        for row in comparison
    ]
    deviations = [
        numeric(row["mc_std_wasserstein_to_centers_nm"]) or 0.0
        for row in comparison
    ]
    figure, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(
        np.arange(len(systems)), means, yerr=deviations,
        capsize=3, color="#548c5a")
    ax.set_xticks(
        np.arange(len(systems)),
        [system.replace("_", "\n") for system in systems],
    )
    ax.set_ylabel("Wasserstein distance to centers PSD (nm)")
    ax.set_title("Centers versus Monte Carlo PSD")
    ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(
        figure, figure_dir / "all_systems_psd_method_sensitivity")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze unified PxPore sensitivity results."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--docs-dir", type=Path)
    parser.add_argument("--scans", default="all")
    return parser


def main():
    args = build_parser().parse_args()
    config = load_json(args.config.resolve())
    selected_scans = parse_scans(args.scans)
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir
        else (REPO_ROOT / config["work_dir"]).resolve()
    )
    docs_dir = (
        args.docs_dir.resolve()
        if args.docs_dir
        else (REPO_ROOT / config["docs_dir"]).resolve()
    )
    table_dir = docs_dir / "tables"
    figure_dir = docs_dir / "figures"
    psd_dir = docs_dir / "psd_method"
    psd_table_dir = psd_dir / "tables"
    psd_figure_dir = psd_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    psd_table_dir.mkdir(parents=True, exist_ok=True)
    psd_figure_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_csv(work_dir / "raw_results.csv")
    selected_rows = [
        row for row in raw_rows if row["scan"] in selected_scans
    ]
    design = study_design_rows(config)
    metric_rows = metric_sensitivity_rows(config, selected_rows)
    summary = convergence_summary_rows(metric_rows)
    probe_results = probe_result_rows(config, selected_rows)
    curves, descriptors = collect_psd(
        config, selected_rows, work_dir)
    comparison = psd_comparison_rows(descriptors)
    psd_parameters = psd_parameter_rows(descriptors)

    write_csv(table_dir / "study_design.csv", design)
    write_markdown(
        table_dir / "study_design.md",
        design,
        (
            "system", "description", "grid_range_nm",
            "octree_levels", "surface_samples", "probe_values_nm",
            "psd_methods",
            "psd_local_max_modes", "psd_min_center_radius_nm",
            "psd_overlap_thresholds", "psd_hist_weightings",
            "psd_mc_samples",
            "connectivity_directions",
        ),
    )
    write_csv(table_dir / "raw_results.csv", selected_rows)
    write_csv(table_dir / "metric_sensitivity.csv", metric_rows)
    write_csv(table_dir / "convergence_summary.csv", summary)
    write_markdown(
        table_dir / "convergence_summary.md",
        summary,
        (
            "system", "scan", "metric", "minimum", "maximum",
            "range", "max_absolute_relative_error_percent",
            "reference_type", "reference_source",
        ),
    )
    write_csv(table_dir / "probe_sensitivity.csv", probe_results)
    write_markdown(
        table_dir / "probe_sensitivity.md",
        probe_results,
        (
            "system", "probe_nm",
            "calculated_PLD_nm", "expected_PLD_nm",
            "absolute_error_PLD_nm",
            "calculated_LCD_nm", "expected_LCD_nm",
            "absolute_error_LCD_nm",
            "Vacc_nm3", "Vtrap_nm3",
        ),
    )
    write_csv(psd_table_dir / "psd_descriptors.csv", descriptors)
    write_csv(
        psd_table_dir / "psd_parameter_sensitivity.csv",
        psd_parameters,
    )
    write_csv(
        psd_table_dir / "psd_method_comparison.csv", comparison)
    write_markdown(
        psd_table_dir / "psd_method_comparison.md",
        comparison,
        (
            "system", "centers_peak_diameter_nm",
            "mc_mean_peak_diameter_nm", "mc_std_peak_diameter_nm",
            "mc_mean_wasserstein_to_centers_nm",
            "mc_std_wasserstein_to_centers_nm",
            "mc_mean_JS_to_centers", "mc_std_JS_to_centers",
        ),
    )
    (docs_dir / "study_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    systems = list(config["systems"])
    for system in systems:
        has_results = any(
            row["system"] == system for row in selected_rows
        )
        if not has_results:
            continue
        for scan in (
            "grid", "octree", "surface", "probe", "connectivity"
        ):
            plot_metric_panels(
                system, scan, metric_rows, figure_dir)
        if "psd_method" in selected_scans:
            plot_psd_system(
                system, selected_rows, curves, psd_figure_dir)
        for scan in PSD_SCANS[1:]:
            if scan in selected_scans:
                plot_psd_parameter_system(
                    system, scan, selected_rows, curves,
                    psd_figure_dir,
                )
        print(f"tables/figures: {system}", flush=True)
    for scan in ("grid", "octree", "surface"):
        plot_combined_metric_summary(summary, scan, figure_dir)
    if "psd_method" in selected_scans:
        plot_combined_psd(comparison, psd_figure_dir)
    print(f"docs={docs_dir}", flush=True)


if __name__ == "__main__":
    main()
