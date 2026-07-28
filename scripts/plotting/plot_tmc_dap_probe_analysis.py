#!/usr/bin/env python3
"""Plot TMC-DAP probe-size fractions, numerical derivatives, and PSD curves."""

from __future__ import annotations

import argparse
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
DEFAULT_PSD_DIR = REPO_ROOT / "results" / "tmc_dap" / "psd"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "tmc_dap_probe_analysis"

PROBE_PATTERN = re.compile(r"_p_([0-9]+(?:\.[0-9]+)?)")
COLORS = {
    "void": "#1F77B4",
    "accessible": "#2CA02C",
    "trapped": "#D62728",
}
PSD_COLORS = ("#1F77B4", "#2CA02C", "#D62728", "#9467BD")


@dataclass(frozen=True)
class FractionSeries:
    probe_nm: np.ndarray
    void: np.ndarray
    accessible: np.ndarray
    trapped: np.ndarray


@dataclass(frozen=True)
class PsdSeries:
    probe_nm: float
    diameter_nm: np.ndarray
    density_per_nm: np.ndarray
    source: Path


def set_paper_style() -> None:
    """Use the paper plotting style shared by the project notebooks."""
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


def load_fraction_series(stats_dir: Path, max_probe_nm: float | None) -> FractionSeries:
    rows: dict[float, tuple[float, float, float]] = {}
    for path in sorted(stats_dir.glob("*_stats.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        settings = data.get("settings", {})
        stats = data.get("stats", {})
        probe = settings.get("probe_nm", probe_from_filename(path))
        values = (
            stats.get("Vvoid_frac"),
            stats.get("Vacc_frac"),
            stats.get("Vtrap_frac"),
        )
        if probe is None or any(value is None for value in values):
            print(f"[skip] incomplete fraction data: {path}")
            continue

        probe = float(probe)
        if max_probe_nm is not None and probe > max_probe_nm:
            continue
        if probe in rows:
            raise ValueError(f"Duplicate probe radius {probe:g} nm in {stats_dir}")
        rows[probe] = tuple(float(value) for value in values)

    if len(rows) < 3:
        raise ValueError(f"Need at least three valid stats files under {stats_dir}")

    probes = np.asarray(sorted(rows), dtype=float)
    values = np.asarray([rows[probe] for probe in probes], dtype=float)
    if np.any(np.diff(probes) <= 0):
        raise ValueError("Probe radii must be strictly increasing")
    return FractionSeries(probes, values[:, 0], values[:, 1], values[:, 2])


def numeric_psd_rows(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            try:
                rows.append([float(value) for value in parts[:5]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric PSD rows found in {path}")
    return np.asarray(rows, dtype=float)


def load_psd_series(psd_dir: Path) -> list[PsdSeries]:
    series: list[PsdSeries] = []
    seen_probes: set[float] = set()
    for path in sorted(psd_dir.glob("*_psd.txt")):
        # This plot targets the centerline PSD format, not voxel_mc_psd files.
        if path.name.endswith("_voxel_mc_psd.txt"):
            continue
        probe = probe_from_filename(path)
        if probe is None:
            print(f"[skip] cannot parse probe radius: {path}")
            continue
        if probe in seen_probes:
            raise ValueError(f"Duplicate PSD for probe radius {probe:g} nm in {psd_dir}")

        values = numeric_psd_rows(path)
        diameter = values[:, 1]
        probability = values[:, 3]
        valid = np.isfinite(diameter) & np.isfinite(probability) & (probability >= 0)
        diameter = diameter[valid]
        probability = probability[valid]
        if len(diameter) < 2 or float(np.sum(probability)) <= 0:
            raise ValueError(f"Insufficient PSD data in {path}")

        order = np.argsort(diameter)
        diameter = diameter[order]
        probability = probability[order]
        bin_width = float(np.median(np.diff(diameter)))
        if not np.isfinite(bin_width) or bin_width <= 0:
            raise ValueError(f"Invalid PSD bin width in {path}")

        # Column 4 is normalized volume probability per bin. Divide by bin width
        # so the plotted curve integrates to one and retains physical units.
        density = probability / (float(np.sum(probability)) * bin_width)
        series.append(PsdSeries(probe, diameter, density, path))
        seen_probes.add(probe)

    if not series:
        raise FileNotFoundError(f"No probe-resolved *_psd.txt files under {psd_dir}")
    return sorted(series, key=lambda item: item.probe_nm)


def gaussian_smooth_density(
    density: np.ndarray,
    bin_width: float,
    sigma_bins: float,
) -> np.ndarray:
    """Smooth a histogram density while preserving non-negativity and area."""
    if sigma_bins <= 0:
        return density.copy()

    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)

    padded = np.pad(density, radius, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="same")[radius:-radius]
    smoothed = np.maximum(smoothed, 0.0)
    area = float(np.sum(smoothed) * bin_width)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("PSD smoothing produced a non-positive integral")
    return smoothed / area


def peak_normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Cannot normalize a PSD with a non-positive maximum")
    return values / maximum


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    outputs = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_fraction_curves(data: FractionSeries, output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots()
    ax.plot(data.probe_nm, data.void, color=COLORS["void"], lw=2, label=r"$F_{v,void}$")
    ax.plot(
        data.probe_nm,
        data.accessible,
        color=COLORS["accessible"],
        lw=2,
        label=r"$F_{v,acc}$",
    )
    ax.plot(
        data.probe_nm,
        data.trapped,
        color=COLORS["trapped"],
        lw=2,
        label=r"$F_{v,trap}$",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Probe radius (nm)")
    ax.set_ylabel("Fraction")
    ax.set_xlim(max(-0.01, data.probe_nm.min() - 0.01), data.probe_nm.max() + 0.01)
    ax.legend(ncol=1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "tmc_dap_probe_fractions")


def plot_fraction_derivative(
    data: FractionSeries,
    output_dir: Path,
    order: int,
) -> list[Path]:
    if order not in (1, 2):
        raise ValueError("Derivative order must be 1 or 2")

    curves = {
        "void": data.void,
        "accessible": data.accessible,
        "trapped": data.trapped,
    }
    labels = {
        "void": "void",
        "accessible": "acc",
        "trapped": "trap",
    }

    fig, ax = plt.subplots()
    for name, values in curves.items():
        derivative = values
        for _ in range(order):
            derivative = np.gradient(derivative, data.probe_nm, edge_order=2)
        if order == 1:
            label = rf"$dF_{{v,{labels[name]}}}/dp$"
        else:
            label = rf"$d^2F_{{v,{labels[name]}}}/dp^2$"
        ax.plot(data.probe_nm, derivative, color=COLORS[name], lw=2, label=label)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Probe radius (nm)")
    power = -order
    ax.set_ylabel(rf"Fraction derivative (nm$^{{{power}}}$)")
    ax.set_xlim(max(-0.01, data.probe_nm.min() - 0.01), data.probe_nm.max() + 0.01)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    ordinal = "first" if order == 1 else "second"
    return save_figure(fig, output_dir, f"tmc_dap_probe_fraction_{ordinal}_derivative")


def plot_psd_by_probe(
    series: list[PsdSeries],
    output_dir: Path,
    sigma_bins: float,
) -> list[Path]:
    fig, ax = plt.subplots()
    if len(series) > len(PSD_COLORS):
        raise ValueError(
            f"Found {len(series)} PSD curves but only {len(PSD_COLORS)} fixed colors"
        )
    for item, color in zip(series, PSD_COLORS):
        bin_width = float(np.median(np.diff(item.diameter_nm)))
        density = gaussian_smooth_density(
            item.density_per_nm,
            bin_width=bin_width,
            sigma_bins=sigma_bins,
        )
        normalized_psd = peak_normalize(density)
        ax.plot(
            item.diameter_nm,
            normalized_psd,
            color=color,
            lw=2,
            label=rf"$r_p={item.probe_nm:g}$ nm",
        )
    ax.set_xlim(left=0)
    # ax.set_ylim(0, 1)
    ax.set_xlabel("Pore radius (nm)")
    ax.set_ylabel("Normalized PSD")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "tmc_dap_psd_by_probe")


def plot_zero_probe_psd(
    series: list[PsdSeries],
    output_dir: Path,
    sigma_bins: float,
) -> list[Path]:
    matches = [item for item in series if np.isclose(item.probe_nm, 0.0)]
    if len(matches) != 1:
        raise ValueError(f"Expected one r_p=0 PSD curve, found {len(matches)}")

    item = matches[0]
    bin_width = float(np.median(np.diff(item.diameter_nm)))
    density = gaussian_smooth_density(
        item.density_per_nm,
        bin_width=bin_width,
        sigma_bins=sigma_bins,
    )
    normalized_psd = peak_normalize(density)

    fig, ax = plt.subplots()
    ax.plot(
        item.diameter_nm,
        normalized_psd,
        color=PSD_COLORS[0],
        lw=2,
        label=r"$r_p = 0.00$ nm",
    )
    ax.set_xlim(left=0)
    # ax.set_ylim(0, 1)
    ax.set_xlabel("Pore radius (nm)")
    ax.set_ylabel("Normalized PSD")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "tmc_dap_psd_probe_0nm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot TMC-DAP void fractions and numerical derivatives across probe "
            "radii, plus PSD curves resolved by probe radius."
        )
    )
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--psd-dir", type=Path, default=DEFAULT_PSD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-probe-nm",
        type=float,
        default=0.15,
        help="Largest probe radius included in fraction/derivative plots (default: 0.15)",
    )
    parser.add_argument(
        "--psd-sigma-bins",
        type=float,
        default=1.5,
        help=(
            "Gaussian smoothing width for PSD curves in histogram bins; "
            "use 0 to disable smoothing (default: 1.5)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stats_dir.is_dir():
        raise FileNotFoundError(f"Stats directory not found: {args.stats_dir}")
    if not args.psd_dir.is_dir():
        raise FileNotFoundError(f"PSD directory not found: {args.psd_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    fractions = load_fraction_series(args.stats_dir, args.max_probe_nm)
    psd_series = load_psd_series(args.psd_dir)

    outputs: list[Path] = []
    outputs.extend(plot_fraction_curves(fractions, args.output_dir))
    outputs.extend(plot_fraction_derivative(fractions, args.output_dir, order=1))
    outputs.extend(plot_fraction_derivative(fractions, args.output_dir, order=2))
    outputs.extend(
        plot_psd_by_probe(
            psd_series,
            args.output_dir,
            sigma_bins=args.psd_sigma_bins,
        )
    )
    outputs.extend(
        plot_zero_probe_psd(
            psd_series,
            args.output_dir,
            sigma_bins=args.psd_sigma_bins,
        )
    )

    print(
        f"Loaded {len(fractions.probe_nm)} probe-fraction points from "
        f"{args.stats_dir}"
    )
    print(
        "Loaded PSD probes (nm): "
        + ", ".join(f"{item.probe_nm:g}" for item in psd_series)
    )
    print(f"PSD Gaussian smoothing sigma: {args.psd_sigma_bins:g} bins")
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
