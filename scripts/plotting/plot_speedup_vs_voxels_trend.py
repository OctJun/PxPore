from __future__ import annotations

import argparse
import os
from pathlib import Path
import math
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "compare_pp_pb_blue_style" / "merged_time_compare.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "figures" / "consistent" / "speedup_vs_voxels_trend.png"
)


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
            "legend.fontsize": 16,
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


def _style_colorbar(cb, label: str = "count") -> None:
    cb.set_label(label)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(labelsize=14, width=0.8, length=3.5)


def load_speedup_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = {"size_voxels", "speedup_wall"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    x = pd.to_numeric(df["size_voxels"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["speedup_wall"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if not np.any(mask):
        raise ValueError(f"No valid positive size/speedup rows found in {csv_path}")
    return x[mask], y[mask]

def latex_scientific_formatter(value, pos):
    """返回形如 '1.2×10^3' 的 LaTeX 字符串，保留两位有效数字"""
    if value == 0:
        return '0'
    exp = int(math.floor(math.log10(abs(value))))
    mant = value / (10 ** exp)
    mant = round(mant, 2 - 1)  # 两位有效数字
    # 处理进位（如 9.99 -> 1.00×10^3）
    if mant >= 10:
        mant /= 10
        exp += 1
    # 去除尾数多余的小数点和零
    if mant.is_integer():
        mant_str = str(int(mant))
    else:
        mant_str = f"{mant:.1f}".rstrip('0').rstrip('.')
    return f"${mant_str}\\times10^{{{exp}}}$"

def fit_log_log_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept = np.polyfit(lx, ly, deg=1)
    predicted = slope * lx + intercept
    ss_res = np.sum((ly - predicted) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def plot_speedup_vs_voxels_with_trend(
    x: np.ndarray,
    y: np.ndarray,
    out_png: Path,
    bins: int = 50,
    gridsize: int = 50,
    point_mode: str = "hexbin",
    show: bool = False,
) -> tuple[Path, tuple[float, float, float], float | None]:
    cmap = make_soft_blue_cmap()
    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept, r2 = fit_log_log_trend(x, y)

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

    if point_mode == "hexbin":
        hb = ax_scatter.hexbin(
            lx,
            ly,
            gridsize=gridsize,
            mincnt=1,
            extent=(lx.min(), lx.max(), ly.min(), ly.max()),
            linewidths=0.0,
            cmap=cmap,
            zorder=2,
        )
        cb = plt.colorbar(hb, cax=ax_cbar)
        _style_colorbar(cb, label="count")
    else:
        ax_cbar.axis("off")
        ax_scatter.scatter(
            lx,
            ly,
            s=16,
            alpha=0.68,
            c="#5f8fbd",
            edgecolors="none",
            zorder=2,
        )

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
        np.log10(64),
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="64 threads",
        zorder=10,
    )

    # --- intersection: trend line × 64-thread line ---
    if abs(slope) > 1e-12:
        lx_intersect = (np.log10(64) - intercept) / slope
        intersect_voxels = 10.0**lx_intersect
        ly_intersect = np.log10(64)  # by definition
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
                "voxels = "+latex_scientific_formatter(intersect_voxels,None),
                xy=(lx_intersect, ly_intersect),
                xytext=(lx_intersect - 0.2, ly_intersect - 0.5),
                fontsize=14,
                color="#4874a3",
                # bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkred", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="#4874a3", lw=1.0),
                zorder=20,
            )
    else:
        lx_intersect = np.nan
        intersect_voxels = np.nan

    ax_scatter.set_xlabel("Voxels")
    ax_scatter.set_ylabel("Wall time speedup (PB / PxPore)")
    ax_scatter.legend(loc="upper left", frameon=False)

    xticks = np.arange(int(np.ceil(lx.min())), int(np.floor(lx.max())) + 1, 1)
    yticks = np.arange(int(np.ceil(ly.min())), int(np.floor(ly.max())) + 1, 1)
    ax_scatter.xaxis.set_major_locator(mticker.FixedLocator(xticks))
    ax_scatter.yaxis.set_major_locator(mticker.FixedLocator(yticks))
    ax_scatter.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, p: rf"$10^{{{int(round(v))}}}$")
    )
    ax_scatter.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, p: rf"$10^{{{int(round(v))}}}$")
    )
    ax_scatter.xaxis.set_minor_locator(mticker.NullLocator())
    ax_scatter.yaxis.set_minor_locator(mticker.NullLocator())

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
    ax_histy.set_ylim(ly.min(), ly.max())
    ax_histy.tick_params(axis="y", labelleft=False, left=False)
    ax_histy.tick_params(axis="x", labelbottom=False, bottom=False)

    for ax in (ax_histx, ax_histy):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("none")

    fig.suptitle("Speedup vs System Size", y=0.995, fontsize=12)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_png, (slope, intercept, r2), intersect_voxels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot speedup versus voxels with one log-log trend line fitted "
            "across all valid data points."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Input CSV containing size_voxels and speedup_wall columns.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Histogram bin count for the marginal distributions.",
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=50,
        help="Hexbin grid size.",
    )
    parser.add_argument(
        "--point-mode",
        choices=("hexbin", "scatter"),
        default="hexbin",
        help="Point rendering mode.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_paper_style()
    x, y = load_speedup_data(args.csv.resolve())
    fig_path, (slope, intercept, r2), intersect_vx = plot_speedup_vs_voxels_with_trend(
        x=x,
        y=y,
        out_png=args.out_png.resolve(),
        bins=args.bins,
        gridsize=args.gridsize,
        point_mode=args.point_mode,
        show=args.show,
    )
    print(fig_path)
    print(f"log10(speedup) = {slope:.6g} * log10(voxels) + {intercept:.6g}")
    print(f"R^2 = {r2:.6g}")
    print(f"N = {len(x)}")
    if intersect_vx is not None and np.isfinite(intersect_vx):
        print(f"intersection: trend × 64-threads at voxels = {intersect_vx:.6g}")
    else:
        print("intersection: trend does not cross 64-thread line in data range")


if __name__ == "__main__":
    main()
