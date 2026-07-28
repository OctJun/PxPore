from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "results" / "hmof_plotting_data"
DEFAULT_PB_CSV = DEFAULT_DATA_DIR / "hmof_existing_results.csv"
DEFAULT_PP_1P64T_CSV = DEFAULT_DATA_DIR / "hmof_pp_results.csv"
DEFAULT_PP_8P8T_CSV = DEFAULT_DATA_DIR / "hmof_pp_results_8p8t.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "figures" / "hmof_timing_pb_vs_pxpore"


@dataclass(frozen=True)
class PxPoreRun:
    tag: str
    label: str
    csv_path: Path
    reference_threads: int
    parallel_label: str


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


def make_soft_blue_cmap() -> LinearSegmentedColormap:
    colors = [
        "#e0e6ec",
        "#c3d0de",
        "#bcd1eb",
        "#9ebadb",
        "#7fa2ca",
        "#628dbb",
        "#4874a3",
        "#39638f",
        "#2A4970",
    ]
    return LinearSegmentedColormap.from_list("soft_blue", colors, N=256)


def extract_hmof_id(name: object) -> float:
    match = re.search(r"hMOF-(\d+)", str(name))
    return float(match.group(1)) if match else np.nan


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_pb(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame(
        {
            "hmof_id": df["name"].map(extract_hmof_id),
            "name": df["name"],
            "pb_wall_s": numeric(df["time_elapsed_seconds"]),
            "pb_cpu_s": numeric(df["time_user_seconds"])
            + numeric(df["time_sys_seconds"]),
        }
    )
    return out.dropna(subset=["hmof_id", "pb_wall_s", "pb_cpu_s"])


def load_pxpore(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).eq("ok")].copy()

    out = pd.DataFrame(
        {
            "hmof_id": df["name"].map(extract_hmof_id),
            "name_raw": df["name"],
            "voxels": numeric(df["voxels"]),
            "pp_wall_s": numeric(df["time_elapsed_seconds"]),
            "pp_cpu_s": numeric(df["time_user_seconds"])
            + numeric(df["time_sys_seconds"]),
        }
    )
    return out.dropna(subset=["hmof_id", "voxels", "pp_wall_s", "pp_cpu_s"])


def merge_timing(pb: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    df = pb.merge(pp, on="hmof_id", how="inner")
    for col in ["pb_wall_s", "pb_cpu_s", "pp_wall_s", "pp_cpu_s", "voxels"]:
        df = df[np.isfinite(df[col]) & (df[col] > 0)].copy()
    df["wall_speedup_pb_over_pp"] = df["pb_wall_s"] / df["pp_wall_s"]
    df["cpu_speedup_pb_over_pp"] = df["pb_cpu_s"] / df["pp_cpu_s"]
    return df.sort_values("hmof_id")


def style_colorbar(cb, label: str = "count") -> None:
    cb.set_label(label)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(labelsize=20, width=0.8, length=3.5)


def power10_formatter(value: float, _pos) -> str:
    return rf"$10^{{{int(round(value))}}}$"


def latex_scientific_formatter(value, _pos):
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / (10**exp)
    mant = round(mant, 1)
    if mant >= 10:
        mant /= 10
        exp += 1
    if float(mant).is_integer():
        mant_str = str(int(mant))
    else:
        mant_str = f"{mant:.1f}".rstrip("0").rstrip(".")
    return f"${mant_str}\\times10^{{{exp}}}$"


def clean_positive_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[mask], y[mask]


def add_parallel_label(ax, text: str) -> None:
    ax.text(
        0.05,
        0.05,
        text,
        fontsize=16,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        zorder=30,
    )


def compute_log_ratio_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) >= 2:
        lx = np.log10(x)
        ly = np.log10(y)
        r = np.corrcoef(lx, ly)[0, 1]
        mae_ratio = np.mean(np.abs(np.log10(y / x)))
        rmse_ratio = np.sqrt(np.mean((np.log10(y / x)) ** 2))
    else:
        r = np.nan
        mae_ratio = np.nan
        rmse_ratio = np.nan
    return float(r), float(mae_ratio), float(rmse_ratio)


def fit_log_log(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept = np.polyfit(lx, ly, deg=1)
    predicted = slope * lx + intercept
    ss_res = np.sum((ly - predicted) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def log_axis_ticks(ax, lx: np.ndarray, ly: np.ndarray) -> None:
    xticks = np.arange(int(np.ceil(lx.min())), int(np.floor(lx.max())) + 1, 1)
    yticks = np.arange(int(np.ceil(ly.min())), int(np.floor(ly.max())) + 1, 1)
    ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))
    ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(power10_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(power10_formatter))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def plot_log_comparison(
    x,
    y,
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: Path,
    bins: int,
    gridsize: int,
    parallel_label: str,
) -> None:
    x, y = clean_positive_xy(x, y)
    if len(x) == 0:
        raise ValueError(f"{title}: no positive data")

    lx = np.log10(x)
    ly = np.log10(y)
    vlo = min(lx.min(), ly.min())
    vhi = max(lx.max(), ly.max())
    if vlo == vhi:
        vlo -= 0.5
        vhi += 0.5
    else:
        pad = 0.03 * (vhi - vlo)
        vlo -= pad
        vhi += pad

    fig = plt.figure(figsize=(7.25, 5.95))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=(5.45, 0.42, 0.18),
        height_ratios=(0.42, 5.45),
        wspace=0.02,
        hspace=0.02,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    ax_cbar = fig.add_subplot(gs[1, 2])

    hb = ax_scatter.hexbin(
        lx,
        ly,
        gridsize=gridsize,
        mincnt=1,
        extent=(vlo, vhi, vlo, vhi),
        linewidths=0.0,
        cmap=make_soft_blue_cmap(),
        zorder=2,
    )
    cb = plt.colorbar(hb, cax=ax_cbar)
    style_colorbar(cb, label="count")

    ax_scatter.plot(
        [vlo, vhi],
        [vlo, vhi],
        linestyle="--",
        linewidth=1.5,
        color="#666666",
        zorder=10,
    )

    ax_scatter.set_xlim(vlo, vhi)
    ax_scatter.set_ylim(vlo, vhi)
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)

    ticks = np.arange(int(np.ceil(vlo)), int(np.floor(vhi)) + 1, 1)
    ax_scatter.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax_scatter.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax_scatter.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, p: rf"$10^{{{int(round(v))}}}$")
    )
    ax_scatter.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, p: rf"$10^{{{int(round(v))}}}$")
    )
    ax_scatter.xaxis.set_minor_locator(mticker.NullLocator())
    ax_scatter.yaxis.set_minor_locator(mticker.NullLocator())
    add_parallel_label(ax_scatter, parallel_label)

    ax_histx.hist(lx, bins=bins, color="#bed0e6", edgecolor="none", alpha=0.98)
    ax_histx.set_xlim(vlo, vhi)
    ax_histx.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_histx.tick_params(axis="y", labelleft=False, left=False)

    ax_histy.hist(
        ly,
        bins=bins,
        orientation="horizontal",
        color="#bed0e6",
        edgecolor="none",
        alpha=0.98,
    )
    ax_histy.set_ylim(vlo, vhi)
    ax_histy.tick_params(axis="y", labelleft=False, left=False)
    ax_histy.tick_params(axis="x", labelbottom=False, bottom=False)
    for ax in (ax_histx, ax_histy):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("none")

    if title:
        fig.suptitle(title, y=0.995, fontsize=24)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_wall_speedup_vs_voxels_trend(
    voxels,
    speedup,
    title: str,
    out_png: Path,
    bins: int,
    gridsize: int,
    reference_threads: int,
    parallel_label: str,
) -> tuple[float, float, float]:
    x, y = clean_positive_xy(voxels, speedup)
    if len(x) == 0:
        raise ValueError(f"{title}: no positive data")

    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept, r2 = fit_log_log(x, y)

    fig = plt.figure(figsize=(7.25, 5.95))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=(5.45, 0.42, 0.18),
        height_ratios=(0.42, 5.45),
        wspace=0.02,
        hspace=0.02,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    ax_cbar = fig.add_subplot(gs[1, 2])

    hb = ax_scatter.hexbin(
        lx,
        ly,
        gridsize=gridsize,
        mincnt=1,
        extent=(lx.min(), lx.max(), ly.min(), ly.max()),
        linewidths=0.0,
        cmap=make_soft_blue_cmap(),
        zorder=2,
    )
    cb = plt.colorbar(hb, cax=ax_cbar)
    style_colorbar(cb)

    fit_lx = np.linspace(lx.min(), lx.max(), 256)
    fit_ly = slope * fit_lx + intercept
    ax_scatter.plot(
        fit_lx,
        fit_ly,
        color="red",
        linestyle="-",
        linewidth=1.2,
        alpha=0.90,
        label=rf"trend: $y=10^{{{intercept:.2f}}}x^{{{slope:.2f}}}$",
        zorder=11,
    )
    ax_scatter.axhline(
        np.log10(reference_threads),
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"{reference_threads} threads",
        zorder=10,
    )

    if abs(slope) > 1e-12:
        lx_intersect = (np.log10(reference_threads) - intercept) / slope
        intersect_voxels = 10.0**lx_intersect
        ly_intersect = np.log10(reference_threads)
        if lx.min() <= lx_intersect <= lx.max():
            ax_scatter.scatter(
                [lx_intersect],
                [ly_intersect],
                marker="o",
                s=80,
                facecolors="none",
                edgecolors="#4874a3",
                linewidths=1.5,
                zorder=20,
            )
            ax_scatter.annotate(
                "voxels = " + latex_scientific_formatter(intersect_voxels, None),
                xy=(lx_intersect, ly_intersect),
                xytext=(lx_intersect - 0.2, ly_intersect - 0.5),
                fontsize=14,
                color="#4874a3",
                arrowprops=dict(arrowstyle="->", color="#4874a3", lw=1.0),
                zorder=20,
            )

    ax_scatter.set_xlabel("Voxels")
    ax_scatter.set_ylabel("Wall time speedup (PxPore / PB)")
    ax_scatter.set_ylim(-1.0, 2.0)
    ax_scatter.legend(loc="upper left")
    log_axis_ticks(ax_scatter, lx, np.array([-1.0, 2.0]))
    add_parallel_label(ax_scatter, parallel_label)

    ax_histx.hist(lx, bins=bins, color="#bed0e6", edgecolor="none", alpha=0.98)
    ax_histx.set_xlim(lx.min(), lx.max())
    ax_histx.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_histx.tick_params(axis="y", labelleft=False, left=False)
    ax_histy.hist(
        ly,
        bins=bins,
        orientation="horizontal",
        color="#bed0e6",
        edgecolor="none",
        alpha=0.98,
    )
    ax_histy.set_ylim(-1.0, 2.0)
    ax_histy.tick_params(axis="y", labelleft=False, left=False)
    ax_histy.tick_params(axis="x", labelbottom=False, bottom=False)
    for ax in (ax_histx, ax_histy):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("none")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    return slope, intercept, r2


def plot_wall_speedup_histogram(
    speedup,
    out_png: Path,
    bins: int,
    reference_threads: int,
    parallel_label: str,
) -> None:
    values = np.asarray(speedup, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        raise ValueError("wall speedup histogram: no positive data")

    fig, ax = plt.subplots(figsize=(7.25, 5.95))
    hist_bins = np.logspace(np.log10(values.min()), np.log10(values.max()), bins)
    ax.hist(
        values,
        bins=hist_bins,
        color="#bed0e6",
        edgecolor="#4874a3",
        linewidth=0.7,
        alpha=0.98,
    )

    median = float(np.median(values))
    mean = float(np.mean(values))
    ax.axvline(
        median,
        color="red",
        linestyle="-",
        linewidth=1.2,
        label=f"median = {median:.2g}",
        zorder=10,
    )
    ax.axvline(
        reference_threads,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"{reference_threads} threads",
        zorder=10,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Wall time speedup (PB / PxPore)")
    ax.set_ylabel("Count")
    ax.grid(False)
    ax.legend(loc="upper right")
    ax.text(
        0.05,
        0.95,
        f"N = {len(values)}\nmean = {mean:.2g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.78),
    )
    add_parallel_label(ax, parallel_label)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_one_run(
    pb: pd.DataFrame,
    run: PxPoreRun,
    out_dir: Path,
    bins: int,
    gridsize: int,
    write_merged_csv: bool,
) -> pd.DataFrame:
    pp = load_pxpore(run.csv_path)
    df = merge_timing(pb, pp)
    out_dir.mkdir(parents=True, exist_ok=True)
    if write_merged_csv:
        merged_csv = out_dir / f"hmof_pb_pxpore_{run.tag}_merged_timing.csv"
        df.to_csv(merged_csv, index=False)
        print(f"{run.tag}: merged CSV = {merged_csv}")

    prefix = out_dir / run.tag
    plot_log_comparison(
        x=df["pp_wall_s"],
        y=df["pb_wall_s"],
        xlabel=f"PxPore {run.label} wall time (s)",
        ylabel="PoreBlazer wall time (s)",
        title=None,
        out_png=prefix.with_name(f"{run.tag}_wall_time_pb_vs_pxpore.png"),
        bins=bins,
        gridsize=gridsize,
        parallel_label=run.parallel_label,
    )
    plot_log_comparison(
        x=df["pp_cpu_s"],
        y=df["pb_cpu_s"],
        xlabel=f"PxPore {run.label} CPU time (s)",
        ylabel="PoreBlazer CPU time (s)",
        title=None,
        out_png=prefix.with_name(f"{run.tag}_cpu_time_pb_vs_pxpore.png"),
        bins=bins,
        gridsize=gridsize,
        parallel_label=run.parallel_label,
    )
    plot_wall_speedup_histogram(
        speedup=df["wall_speedup_pb_over_pp"],
        out_png=prefix.with_name(f"{run.tag}_wall_time_speedup_hist.png"),
        bins=bins,
        reference_threads=run.reference_threads,
        parallel_label=run.parallel_label,
    )
    slope, intercept, r2 = plot_wall_speedup_vs_voxels_trend(
        voxels=df["voxels"],
        speedup=df["wall_speedup_pb_over_pp"],
        title=f"Wall-Time Speedup vs Voxels ({run.label})",
        out_png=prefix.with_name(f"{run.tag}_wall_time_speedup_vs_voxels_trend.png"),
        bins=bins,
        gridsize=gridsize,
        reference_threads=run.reference_threads,
        parallel_label=run.parallel_label,
    )

    print(f"{run.tag}: merged rows = {len(df)}")
    print(
        f"{run.tag}: log10(wall speedup) = {slope:.6g} * "
        f"log10(voxels) + {intercept:.6g}; R^2 = {r2:.6g}"
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PB-vs-PxPore timing comparison figures for hMOF "
            "using 1p64t and 8p8t PxPore result CSVs."
        )
    )
    parser.add_argument("--pb-csv", type=Path, default=DEFAULT_PB_CSV)
    parser.add_argument("--pp-1p64t-csv", type=Path, default=DEFAULT_PP_1P64T_CSV)
    parser.add_argument("--pp-8p8t-csv", type=Path, default=DEFAULT_PP_8P8T_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--gridsize", type=int, default=50)
    parser.add_argument(
        "--write-merged-csv",
        action="store_true",
        help="Write per-run merged timing CSVs next to the figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_paper_style()

    pb = load_pb(args.pb_csv.resolve())
    runs = [
        PxPoreRun(
            "1p64t",
            "1p64t",
            args.pp_1p64t_csv.resolve(),
            64,
            "1 process × 64 threads",
        ),
        PxPoreRun(
            "8p8t",
            "8p8t",
            args.pp_8p8t_csv.resolve(),
            8,
            "8 processes × 8 threads",
        ),
    ]
    for run in runs:
        plot_one_run(
            pb=pb,
            run=run,
            out_dir=args.out_dir.resolve(),
            bins=args.bins,
            gridsize=args.gridsize,
            write_merged_csv=args.write_merged_csv,
        )

    print(f"figures written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
