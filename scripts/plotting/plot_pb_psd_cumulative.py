from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
POREBLAZER_DIR = REPO_ROOT / "poreblazer"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "psd"

PB_COLOR = "#D62728"
PP_COLOR = "#1F77B4"
MATERIALS = ("HKUST1", "IRMOF1", "ZIF8")
PXPore_PATHS = {
    "HKUST1": (
        REPO_ROOT / "poreblazer" / "HKUST1.gro_g_0.02_p_0.0_psd.txt",
    ),
    "IRMOF1": (
        REPO_ROOT / "poreblazer" / "IRMOF1.gro_g_0.02_p_0.0_psd.txt",
    ),
    "ZIF8": (
        REPO_ROOT / "poreblazer" / "ZIF8AP.gro_g_0.02_p_0.0_psd.txt",
    )
}


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


def load_pb_curve(material: str) -> tuple[np.ndarray, np.ndarray]:
    path = POREBLAZER_DIR / f"{material}_Network-accessible_psd.txt"
    data = np.loadtxt(path, comments="#")
    diameters = data[:, 0]
    fractions = data[:, 1]
    return diameters, fractions


def find_pxpore_path(material: str) -> Path | None:
    for path in PXPore_PATHS.get(material, ()):
        if path.exists():
            return path
    return None


def load_pxpore_curve(material: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = find_pxpore_path(material)
    if path is None:
        return None

    data = np.loadtxt(path, comments="#")
    diameters_angstrom = data[:, 1] * 10.0
    volume_fraction = data[:, 3]
    # bin_width = float(np.median(np.diff(diameters_angstrom)))
    # Convert per-bin volume fraction into a density curve comparable to PB's dV/dd.
    # density = volume_fraction / bin_width
    return diameters_angstrom, volume_fraction


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    # if np.isclose(vmax, vmin):
    #     return np.zeros_like(values)
    # return (values - vmin) / (vmax - vmin)
    return values/vmax


def plot_material(material: str, output_dir: Path, show: bool = False) -> Path:
    diameters, fractions = load_pb_curve(material)
    fractions = min_max_normalize(fractions)
    pxpore_curve = load_pxpore_curve(material)

    fig, ax = plt.subplots()

    ax.plot(
        diameters,
        fractions,
        color=PB_COLOR,
        lw=2.2,
        # marker="s",
        # ms=4.5,
        # markevery=max(len(diameters) // 24, 1),
        label="PoreBlazer",
        zorder=4,
    )

    if pxpore_curve is not None:
        px_diameters, px_fractions = pxpore_curve
        px_fractions = min_max_normalize(px_fractions)
        ax.plot(
            px_diameters,
            px_fractions,
            color=PP_COLOR,
            lw=2.2,
            # ls="--",
            label="PxPore",
            zorder=5,
        )

    # ax.set_title(material)
    ax.set_xlabel("Probe diameter ($\\AA$)")
    ax.set_ylabel("Normalized Pore Size Distribution")
    xmin = float(np.min(diameters))
    xmax = float(np.max(diameters))
    if pxpore_curve is not None:
        xmin = min(xmin, float(np.min(px_diameters)))
        xmax = max(xmax, float(np.max(px_diameters)))
    ax.set_xlim(left=max(-0.1, xmin - 0.25), right=xmax + 0.25)
    ax.set_ylim(bottom=-0.1, top=1.05)
    ax.grid(True, alpha=0.25)
    if pxpore_curve is not None:
        ax.legend(loc="best")
    else:
        placeholder_handles = [
            Line2D([0], [0], color=PB_COLOR, lw=2.2, marker="s", ms=5, label="PoreBlazer"),
            Line2D(
                [0],
                [0],
                color=PP_COLOR,
                lw=2.2,
                ls="--",
                marker="o",
                ms=5,
                alpha=0.9,
                label="PxPore (pending)",
            ),
        ]
        ax.legend(handles=placeholder_handles, loc="best")
    plt.tight_layout()
    fig_path = output_dir / f"{material}_network_accessible_psd.png"
    fig.savefig(fig_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accessible PSD curves for PoreBlazer and available PxPore results."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save generated figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    for material in MATERIALS:
        fig_path = plot_material(material, output_dir=output_dir, show=args.show)
        print(fig_path)


if __name__ == "__main__":
    main()
