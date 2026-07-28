#!/usr/bin/env python3
"""Summarize PxPore metrics for TMC-DAP, APC-DAP, and APC-MAP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TMC_STATS_DIR = REPO_ROOT / "results" / "tmc_dap" / "probe_stats"
DEFAULT_APC_ROOT = REPO_ROOT / "results" / "apc_psd_time_48ps"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "figures" / "polymer_system_metrics.csv"

TIME_PATTERNS = {
    "user": re.compile(r"User time \(seconds\):\s*([0-9.]+)"),
    "sys": re.compile(r"System time \(seconds\):\s*([0-9.]+)"),
    "elapsed": re.compile(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)"
    ),
}


@dataclass(frozen=True)
class Sample:
    system: str
    source: Path
    atoms: float
    box_x_nm: float
    box_y_nm: float
    box_z_nm: float
    volume_nm3: float
    accessible_volume_nm3: float
    void_volume_nm3: float
    accessible_volume_fraction: float
    ffv: float
    accessible_surface_area_nm2: float
    total_surface_area_nm2: float
    density_g_cm3: float
    pld_nm: float
    lcd_nm: float
    wall_time_s: float | None
    cpu_time_s: float | None


def as_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def elapsed_to_seconds(text: str) -> float:
    parts = text.strip().split(":")
    if len(parts) == 3:
        return 3600.0 * float(parts[0]) + 60.0 * float(parts[1]) + float(parts[2])
    if len(parts) == 2:
        return 60.0 * float(parts[0]) + float(parts[1])
    return float(text)


def parse_time_log(path: Path) -> tuple[float | None, float | None]:
    if not path.is_file():
        return None, None

    text = path.read_text(encoding="utf-8", errors="replace")
    user_match = TIME_PATTERNS["user"].search(text)
    sys_match = TIME_PATTERNS["sys"].search(text)
    elapsed_match = TIME_PATTERNS["elapsed"].search(text)

    wall = elapsed_to_seconds(elapsed_match.group(1)) if elapsed_match else None
    cpu = None
    if user_match and sys_match:
        cpu = float(user_match.group(1)) + float(sys_match.group(1))
    return wall, cpu


def load_sample(system: str, stats_path: Path) -> Sample:
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = data["stats"]
    info = data.get("info", {})
    run_envs = data.get("run_envs", {})

    box = info.get("box_nm")
    if not box or len(box) < 3:
        volume = as_float(stats.get("Vcell_nm3"))
        edge = volume ** (1.0 / 3.0)
        box = [edge, edge, edge]

    wall, cpu = parse_time_log(stats_path.with_name("time.log"))
    if wall is None:
        wall = run_envs.get("execution_time")
        wall = float(wall) if wall is not None else None

    return Sample(
        system=system,
        source=stats_path,
        atoms=as_float(stats.get("atoms")),
        box_x_nm=as_float(box[0]),
        box_y_nm=as_float(box[1]),
        box_z_nm=as_float(box[2]),
        volume_nm3=as_float(stats.get("Vcell_nm3")),
        accessible_volume_nm3=as_float(stats.get("Vacc_nm3")),
        void_volume_nm3=as_float(stats.get("Vvoid_nm3")),
        accessible_volume_fraction=as_float(stats.get("Vacc_frac")),
        ffv=as_float(stats.get("Vvoid_frac")),
        accessible_surface_area_nm2=as_float(stats.get("Sacc_nm2")),
        total_surface_area_nm2=as_float(stats.get("Stotal_nm2")),
        density_g_cm3=as_float(stats.get("density_g/cm3")),
        pld_nm=as_float(stats.get("PLD_nm")),
        lcd_nm=as_float(stats.get("LCD_nm")),
        wall_time_s=wall,
        cpu_time_s=cpu,
    )


def mean_optional(values: list[float | None]) -> float | str:
    numeric = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return mean(numeric) if numeric else ""


def mean_attr(samples: list[Sample], attr: str) -> float:
    return mean(float(getattr(sample, attr)) for sample in samples)


def summarize(samples: list[Sample]) -> dict[str, float | int | str]:
    first = samples[0]
    return {
        "system": first.system,
        "n_samples": len(samples),
        "atom_count": int(round(first.atoms)),
        "avg_box_x_nm": mean_attr(samples, "box_x_nm"),
        "avg_box_y_nm": mean_attr(samples, "box_y_nm"),
        "avg_box_z_nm": mean_attr(samples, "box_z_nm"),
        "Volume (nm3)": mean_attr(samples, "volume_nm3"),
        "Accessible volume (nm3)": mean_attr(samples, "accessible_volume_nm3"),
        "Void volume (nm3)": mean_attr(samples, "void_volume_nm3"),
        "Accessible volume fraction": mean_attr(
            samples, "accessible_volume_fraction"
        ),
        "FFV": mean_attr(samples, "ffv"),
        "Accessible surface area (nm2)": mean_attr(
            samples, "accessible_surface_area_nm2"
        ),
        "Total surface area (nm2)": mean_attr(samples, "total_surface_area_nm2"),
        "Density (g/cm3)": mean_attr(samples, "density_g_cm3"),
        "PLD (nm)": mean_attr(samples, "pld_nm"),
        "LCD (nm)": mean_attr(samples, "lcd_nm"),
        "Wall time (s)": mean_optional([sample.wall_time_s for sample in samples]),
        "CPU time (s)": mean_optional([sample.cpu_time_s for sample in samples]),
    }


def tmc_samples(stats_dir: Path) -> list[Sample]:
    candidates = sorted(stats_dir.glob("*_p_0.0_stats.json"))
    if not candidates:
        candidates = sorted(stats_dir.glob("*_p_0_stats.json"))
    if not candidates:
        raise FileNotFoundError(f"No probe=0 stats JSON found in {stats_dir}")
    return [load_sample("TMC-DAP", candidates[0])]


def apc_samples(apc_root: Path, system: str, result_dir: str) -> list[Sample]:
    result_root = apc_root / system / result_dir
    if not result_root.is_dir():
        raise FileNotFoundError(f"Missing result directory: {result_root}")

    paths = sorted(
        result_root.glob("frame_*/frame_*.gro_g_0.02_p_0.0_stats.json"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"No frame stats found in {result_root}")
    return [load_sample(system, path) for path in paths]


def write_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TMC-DAP, APC-DAP, and APC-MAP PxPore metrics."
    )
    parser.add_argument("--tmc-stats-dir", type=Path, default=DEFAULT_TMC_STATS_DIR)
    parser.add_argument("--apc-root", type=Path, default=DEFAULT_APC_ROOT)
    parser.add_argument("--apc-result-dir", default="pxpore_8p8t")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        summarize(tmc_samples(args.tmc_stats_dir)),
        summarize(apc_samples(args.apc_root, "APC-DAP", args.apc_result_dir)),
        summarize(apc_samples(args.apc_root, "APC-MAP", args.apc_result_dir)),
    ]
    write_csv(rows, args.output)
    print(f"Wrote: {args.output}")
    for row in rows:
        print(
            f"{row['system']}: n={row['n_samples']}, "
            f"atoms={row['atom_count']}, "
            f"V={float(row['Volume (nm3)']):.3f} nm3, "
            f"FFV={float(row['FFV']):.6f}"
        )


if __name__ == "__main__":
    main()
