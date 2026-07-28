#!/usr/bin/env python3
"""Map the TMC-DAP FFV curve onto the Experimental MWCO axis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_DIR = REPO_ROOT / "results" / "tmc_dap" / "probe_stats"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "tmc_dap_mwco_ffv"
PROBE_PATTERN = re.compile(r"_p_([0-9]+(?:\.[0-9]+)?)")

EXPERIMENTAL_MWCO = np.asarray([200.0, 400.0, 600.0, 1000.0])
EXPERIMENTAL_REJECTION_PERCENT = np.asarray(
    [33.67591, 76.62338,98.48128, 97.90261,]
)

LEFT_COLOR = "#3B6EA8"
RIGHT_COLOR = "#C44E52"

@dataclass(frozen=True)
class FfvCurve:
    probe_nm: np.ndarray
    fraction: np.ndarray
    rejection_proxy: np.ndarray


@dataclass(frozen=True)
class MwcoMapping:
    equivalent_probe_nm: np.ndarray
    prefactor: float
    exponent: float
    fitted_probe_nm: np.ndarray
    ffv_rejection_percent_at_mwco: np.ndarray
    r_squared: float
    pearson_r: float
    rmse_percentage_points: float
    mae_percentage_points: float


def set_plot_style() -> None:
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


def probe_from_filename(path: Path) -> float | None:
    match = PROBE_PATTERN.search(path.name)
    return float(match.group(1)) if match else None


def load_ffv_curve(
    stats_dir: Path,
    fraction_kind: str,
    max_probe_nm: float,
) -> FfvCurve:
    rows: dict[float, float] = {}
    fraction_key = "Vvoid_frac" if fraction_kind == "void" else "Vacc_frac"

    for path in sorted(stats_dir.glob("*_stats.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        probe = data.get("settings", {}).get("probe_nm", probe_from_filename(path))
        fraction = data.get("stats", {}).get(fraction_key)
        if probe is None or fraction is None:
            continue

        probe = float(probe)
        if probe <= max_probe_nm:
            rows[probe] = float(fraction)

    if len(rows) < 3:
        raise ValueError(f"Need at least three FFV points in {stats_dir}")

    probe_nm = np.asarray(sorted(rows), dtype=float)
    fraction = np.asarray([rows[probe] for probe in probe_nm], dtype=float)
    if not np.isclose(probe_nm[0], 0.0):
        raise ValueError("The FFV curve must include probe radius 0 nm")
    if not np.isfinite(fraction[0]) or fraction[0] <= 0:
        raise ValueError("FFV at probe radius 0 nm must be positive")

    rejection_proxy = np.clip(1.0 - fraction / fraction[0], 0.0, 1.0)
    rejection_proxy = np.maximum.accumulate(rejection_proxy)
    return FfvCurve(probe_nm, fraction, rejection_proxy)


def inverse_ffv_rejection(
    rejection_fraction: np.ndarray,
    ffv: FfvCurve,
) -> np.ndarray:
    unique_rejection, first_indices = np.unique(
        ffv.rejection_proxy, return_index=True
    )
    unique_probe = ffv.probe_nm[first_indices]
    targets = np.clip(rejection_fraction, unique_rejection[0], unique_rejection[-1])
    return np.interp(targets, unique_rejection, unique_probe)


def fit_power_law(
    mwco: np.ndarray,
    probe_nm: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    if np.any(mwco <= 0) or np.any(probe_nm <= 0):
        raise ValueError("Power-law fit requires positive MWCO and probe radii")

    exponent, log_prefactor = np.polyfit(np.log(mwco), np.log(probe_nm), deg=1)
    prefactor = float(np.exp(log_prefactor))
    fitted_probe = prefactor * mwco**exponent
    return prefactor, float(exponent), fitted_probe


def coefficient_of_determination(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual_sum = float(np.sum((observed - predicted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    return float("nan") if total_sum <= 0 else 1.0 - residual_sum / total_sum


def build_mapping(ffv: FfvCurve) -> MwcoMapping:
    experimental_rejection = EXPERIMENTAL_REJECTION_PERCENT / 100.0
    equivalent_probe = inverse_ffv_rejection(experimental_rejection, ffv)
    prefactor, exponent, fitted_probe = fit_power_law(
        EXPERIMENTAL_MWCO, equivalent_probe
    )
    ffv_rejection = np.interp(fitted_probe, ffv.probe_nm, ffv.rejection_proxy)
    ffv_rejection_percent = 100.0 * ffv_rejection

    observed = EXPERIMENTAL_REJECTION_PERCENT
    predicted = ffv_rejection_percent
    return MwcoMapping(
        equivalent_probe_nm=equivalent_probe,
        prefactor=prefactor,
        exponent=exponent,
        fitted_probe_nm=fitted_probe,
        ffv_rejection_percent_at_mwco=ffv_rejection_percent,
        r_squared=coefficient_of_determination(observed, predicted),
        pearson_r=float(np.corrcoef(observed, predicted)[0, 1]),
        rmse_percentage_points=float(np.sqrt(np.mean((observed - predicted) ** 2))),
        mae_percentage_points=float(np.mean(np.abs(observed - predicted))),
    )


def mwco_to_probe(mwco: np.ndarray, mapping: MwcoMapping) -> np.ndarray:
    return mapping.prefactor * mwco**mapping.exponent


def power_law_text(mapping: MwcoMapping) -> str:
    prefactor = f"{mapping.prefactor:.4e}".rstrip("0").rstrip(".")
    exponent = f"{mapping.exponent:.4f}".rstrip("0").rstrip(".")
    return rf"$r={prefactor}M^{{{exponent}}}$"


def plot_experimental_mwco_vs_ffv_curve(
    ffv: FfvCurve,
    mapping: MwcoMapping,
    output_dir: Path,
    plot_max_mwco: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mwco_grid = np.linspace(0.0, plot_max_mwco, 1000)
    probe_grid = mwco_to_probe(mwco_grid, mapping)
    ffv_rejection_percent = 100.0 * np.interp(
        probe_grid, ffv.probe_nm, ffv.rejection_proxy
    )

    fig, ax = plt.subplots()
    ax.plot(
        mwco_grid,
        ffv_rejection_percent,
        color=LEFT_COLOR,
        linewidth=2.4,
        label="Mapped $F_{v,void}$",
    )
    ax.plot(
        EXPERIMENTAL_MWCO,
        EXPERIMENTAL_REJECTION_PERCENT,
        color=RIGHT_COLOR,
        marker="s",
        markersize=7,
        linewidth=2.2,
        label="Experimental",
        zorder=5,
    )
    # ax.plot(
    #     EXPERIMENTAL_MWCO,
    #     mapping.ffv_rejection_percent_at_mwco,
    #     color="#222222",
    #     marker="o",
    #     linestyle="none",
    #     markersize=5.5,
    #     label="FFV at MWCO points",
    #     zorder=4,
    # )

    # ax.set_xlim(0.0, plot_max_mwco)
    # ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("MWCO (g mol$^{-1}$)")
    ax.set_ylabel("Rejection (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    ax.text(
        0.04,
        0.96,
        # power_law_text(mapping)
        # + "\n"
        rf"$R^2={mapping.r_squared:.3f}$",
        # + "\n"
        # + rf"$RMSE={mapping.rmse_percentage_points:.2f}$ pp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
    )
    fig.tight_layout()

    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"tmc_dap_experimental_mwco_ffv_curve.{suffix}"
        fig.savefig(path)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_mapping_csv(output_dir: Path, mapping: MwcoMapping) -> Path:
    path = output_dir / "tmc_dap_experimental_mwco_ffv_curve.csv"
    fieldnames = [
        "mwco_g_mol",
        "experimental_rejection_percent",
        "equivalent_probe_nm",
        "power_fit_probe_nm",
        "ffv_curve_rejection_percent",
        "residual_percentage_points",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, mwco in enumerate(EXPERIMENTAL_MWCO):
            experimental = EXPERIMENTAL_REJECTION_PERCENT[index]
            ffv_value = mapping.ffv_rejection_percent_at_mwco[index]
            writer.writerow(
                {
                    "mwco_g_mol": mwco,
                    "experimental_rejection_percent": experimental,
                    "equivalent_probe_nm": mapping.equivalent_probe_nm[index],
                    "power_fit_probe_nm": mapping.fitted_probe_nm[index],
                    "ffv_curve_rejection_percent": ffv_value,
                    "residual_percentage_points": experimental - ffv_value,
                }
            )
    return path


def write_summary_csv(output_dir: Path, mapping: MwcoMapping) -> Path:
    path = output_dir / "tmc_dap_experimental_mwco_ffv_curve_summary.csv"
    fieldnames = ["metric", "value"]
    rows = [
        ("power_prefactor", f"{mapping.prefactor:.10f}"),
        ("power_exponent", f"{mapping.exponent:.10f}"),
        ("rejection_r_squared", f"{mapping.r_squared:.10f}"),
        ("pearson_r", f"{mapping.pearson_r:.10f}"),
        ("rmse_percentage_points", f"{mapping.rmse_percentage_points:.10f}"),
        ("mae_percentage_points", f"{mapping.mae_percentage_points:.10f}"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map the TMC-DAP FFV curve onto the Experimental MWCO axis."
    )
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fraction-kind",
        choices=("void", "accessible"),
        default="void",
        help="FFV fraction used to build the rejection proxy",
    )
    parser.add_argument("--max-probe-nm", type=float, default=0.20)
    parser.add_argument("--plot-max-mwco", type=float, default=1000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stats_dir.is_dir():
        raise FileNotFoundError(f"Stats directory not found: {args.stats_dir}")

    set_plot_style()
    ffv = load_ffv_curve(args.stats_dir, args.fraction_kind, args.max_probe_nm)
    mapping = build_mapping(ffv)

    outputs = []
    outputs.extend(
        plot_experimental_mwco_vs_ffv_curve(
            ffv, mapping, args.output_dir, args.plot_max_mwco
        )
    )
    outputs.append(write_mapping_csv(args.output_dir, mapping))
    outputs.append(write_summary_csv(args.output_dir, mapping))

    print(f"FFV points: {len(ffv.probe_nm)} ({args.fraction_kind})")
    print(f"power law: r = {mapping.prefactor:.10f} * M^{mapping.exponent:.5f}")
    print(f"R2: {mapping.r_squared:.6f}")
    print(f"Pearson r: {mapping.pearson_r:.6f}")
    print(f"RMSE: {mapping.rmse_percentage_points:.6f} percentage points")
    print(f"MAE: {mapping.mae_percentage_points:.6f} percentage points")
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
