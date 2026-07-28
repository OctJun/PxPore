from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "performance"

# ---------------------------------------------------------------------------
# Original colour palette
# ---------------------------------------------------------------------------
PX_COLOR = "#3380B8"
PB_COLOR = "#E06666"
ZEO_COLOR = "#61D161"


# c_px  = "#1F77B4"   # PxPore
# c_pb  = "#D62728"   # PoreBlazer
# c_zeo = "#2CA02C"   # Zeo++



GRADIENT_STRIPS = 100

# ---------------------------------------------------------------------------
GROUP_LABELS = (
    "H atom",
    "512 H atoms",
    "HKUST-1",
    "TaPa-1",
    "Crosslinked\npolymer",
    "Protein \n(PDB: 7UZE)",
)

# Dummy wall-clock timings in seconds (illustrative only).
PXPORE_TIMES = np.array([0.20, 7.83, 3.48, 2.95, 7.03, 49.98])
POREBLAZER_TIMES = np.array([0.18, 52.21, 82.39, 88.34, 10269.6, 18090.37])
ZEO_TIMES = np.array([0.76, 22.69, 657.83, 655.51, 10191.3, 10666.88])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_time(val: float) -> str:
    """Compact time formatting — at most 3 characters."""
    if val < 10:
        return f"{val:.2f}"       # "0.2", "7.8"
    if val < 1000:
        return f"{val:.1f}"       # "52", "658"
    return f"{val / 1000:.1f}k"   # "10k", "18k"


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (12.4, 5.5),          # 2× width of compare_pp_pb_blue_style square plots
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


def _add_value_labels(
    ax: plt.Axes,
    x_centers: np.ndarray,
    heights: np.ndarray,
    width: float,
    color: str,
) -> None:
    """Rotated value labels above each bar."""
    for x, val in zip(x_centers, heights):
        if val <= 0:
            continue
        label_y = max(val * 1.06, 0.22)
        ax.text(
            x,
            label_y,
            _fmt_time(val),
            ha="center",
            va="bottom",
            fontsize=12,
            rotation=0,
            color=color,
            alpha=1,
        )


def _add_speedup_annotations(
    ax: plt.Axes,
    x_centers: np.ndarray,
    heights: np.ndarray,
    speedups: np.ndarray,
    width: float,
    y_floor: float,
    color: str,
) -> None:
    """Annotate ×N speedup inside bars, left side, vertical, near x-axis."""
    import matplotlib.patheffects as pe

    for x, h, sp in zip(x_centers, heights, speedups):
        if sp >= 2.0 and x>0.85:
            label = f"×{sp:.1f}" if sp >= 10 else f"×{sp:.2f}"
            ax.text(
                x - width / 4,
                y_floor * 1.5,
                label,
                ha="left",
                va="bottom",
                # weight='bold',
                style='italic',
                fontsize=14,
                rotation=90,
                # color=color,
                color='black',
                alpha=1,
                # path_effects=[pe.withStroke(linewidth=0.8, foreground="gray")],
            )


# ---------------------------------------------------------------------------
# 2D grouped bar chart
# ---------------------------------------------------------------------------


def _gradient_bars(
    ax: plt.Axes,
    x_centers: np.ndarray,
    heights: np.ndarray,
    width: float,
    color: str,
    y_floor: float,
    n_strips: int = GRADIENT_STRIPS,
) -> None:
    """Draw bars with vertical transparency gradient: 0 (bottom) → 1 (top)."""
    from matplotlib.colors import to_rgba

    r, g, b, _ = to_rgba(color)
    denom = max(n_strips - 1, 1)

    for x, h in zip(x_centers, heights):
        if h <= y_floor:
            continue
        y_stops = np.geomspace(y_floor, h, n_strips + 1)
        for j in range(n_strips):
            alpha = j / denom        # 0 → 1 over full bar height
            y0 = y_stops[j]
            dy = y_stops[j + 1] - y0
            ax.bar(
                x, dy, width, bottom=y0,
                color=(r, g, b), alpha=alpha,
                edgecolor="none",
                zorder=5,
            )


def plot_2d(output_dir: Path, show: bool = False) -> Path:
    positions = np.arange(len(GROUP_LABELS))
    width = 0.25
    y_floor = 0.08     # matches ylim lower bound

    # width = 2× compare_pp_pb_blue_style square plots
    W, H = 12.4, 5.5
    fig, ax = plt.subplots(figsize=(W, H))

    # ---- gradient bars -------------------------------------------------------
    x_px = positions - width
    x_pb = positions
    x_zeo = positions + width

    _gradient_bars(ax, x_px, PXPORE_TIMES, width, PX_COLOR, y_floor)
    _gradient_bars(ax, x_pb, POREBLAZER_TIMES, width, PB_COLOR, y_floor)
    _gradient_bars(ax, x_zeo, ZEO_TIMES, width, ZEO_COLOR, y_floor)

    # ---- value labels --------------------------------------------------------
    _add_value_labels(ax, x_px, PXPORE_TIMES, width, PX_COLOR)
    _add_value_labels(ax, x_pb, POREBLAZER_TIMES, width, PB_COLOR)
    _add_value_labels(ax, x_zeo, ZEO_TIMES, width, ZEO_COLOR)

    # ---- speedup annotations -------------------------------------------------
    pb_speedup = POREBLAZER_TIMES / PXPORE_TIMES
    zeo_speedup = ZEO_TIMES / PXPORE_TIMES
    _add_speedup_annotations(ax, x_pb, POREBLAZER_TIMES, pb_speedup, width, y_floor, PB_COLOR)
    _add_speedup_annotations(ax, x_zeo, ZEO_TIMES, zeo_speedup, width, y_floor, ZEO_COLOR)

    # ---- axes & grid ---------------------------------------------------------
    ax.set_yscale("log")
    ax.set_ylabel("Wall Time (s)")
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_LABELS, rotation=0, ha="center")
    ax.set_xlim(-0.65, len(GROUP_LABELS) - 0.35)
    ax.set_ylim(0.08, 60_000)   # headroom for speedup annotations

    # y-ticks: 10⁰ … 10⁴ only
    ax.set_yticks([1, 10, 100, 1_000, 10_000])
    ax.set_yticklabels(["$10^0$", "$10^1$", "$10^2$", "$10^3$", "$10^4$"])

    ax.grid(True, which="major", axis="y", alpha=0.25, zorder=0)
    ax.grid(True, which="minor", axis="y", alpha=0.10, linestyle="--", zorder=0)

    # proxy legend with mid-tone colours (gradient strips don't represent well)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=PX_COLOR, label="PxPore"),
        Patch(facecolor=PB_COLOR, label="PoreBlazer"),
        Patch(facecolor=ZEO_COLOR, label="Zeo++"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, ncol=3)

    # ---- final layout --------------------------------------------------------
    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.13, top=0.96)

    fig_path = output_dir / "performance_comparison.png"
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(fig_path, pad_inches=0.03)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grouped runtime comparison: PxPore vs PoreBlazer vs Zeo++."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save the generated figure(s).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    p = plot_2d(output_dir=output_dir, show=args.show)
    print(p)


if __name__ == "__main__":
    main()
