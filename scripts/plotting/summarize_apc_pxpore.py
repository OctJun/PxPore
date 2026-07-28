#!/usr/bin/env python3
"""Collect per-frame pore properties and dataset-level PxPore performance."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


SYSTEMS = ("APC-DAP", "APC-DAL", "APC-MAP")
CONFIGS = {"8p8t": (8, 8), "1p64t": (1, 64)}
FRAME_RE = re.compile(r"frame_(\d+)$")
TIME_RE = re.compile(r"\bt\s*=\s*([0-9.+\-Ee]+)")
TIME_PATTERNS = {
    "user_time_s": re.compile(r"^\s*User time \(seconds\):\s*([0-9.]+)\s*$", re.M),
    "system_time_s": re.compile(r"^\s*System time \(seconds\):\s*([0-9.]+)\s*$", re.M),
    "cpu_percent": re.compile(r"^\s*Percent of CPU this job got:\s*([0-9.]+)%\s*$", re.M),
    "elapsed": re.compile(
        r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)\s*$", re.M
    ),
    "max_rss_kb": re.compile(
        r"^\s*Maximum resident set size \(kbytes\):\s*([0-9]+)\s*$", re.M
    ),
}
PP_KEYS = (
    "density_g/cm3", "Vcell_nm3", "Vvoid_nm3", "Vprobe_nm3", "Vacc_nm3",
    "Vtrap_nm3", "Vvoid_frac", "Vacc_frac", "Vtrap_frac", "Sacc_nm2",
    "Sacc_m2/g", "Stotal_nm2", "Stotal_m2/g", "PLD_nm", "LCD_nm",
    "LCD_global_nm",
)


def elapsed_seconds(value: str) -> float:
    numbers = [float(part) for part in value.split(":")]
    if not 1 <= len(numbers) <= 3:
        raise ValueError(f"Unsupported elapsed time: {value}")
    return sum(number * 60**power for power, number in enumerate(reversed(numbers)))


def parse_time(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = {}
    for key, pattern in TIME_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Missing {key} in {path}")
        matches[key] = match.group(1)
    return {
        "wall_clock_s": elapsed_seconds(matches["elapsed"]),
        "user_time_s": float(matches["user_time_s"]),
        "system_time_s": float(matches["system_time_s"]),
        "cpu_percent": float(matches["cpu_percent"]),
        "max_rss_kb": float(matches["max_rss_kb"]),
    }


def frame_time(path: Path) -> float:
    match = TIME_RE.search(path.read_text(encoding="utf-8", errors="replace").splitlines()[0])
    if not match:
        raise ValueError(f"No frame time in {path}")
    return float(match.group(1))


def collect_config(
    work_root: Path, system: str, config: str, suffix: str, allow_partial: bool
) -> tuple[list[dict[str, object]], dict[str, object]]:
    processes, threads = CONFIGS[config]
    result_root = work_root / system / f"pxpore_{config}{suffix}"
    expected = len(list((work_root / system / "frames").glob("frame_*.gro")))
    complete = (result_root / ".complete").is_file()
    if not complete and not allow_partial:
        raise RuntimeError(f"Incomplete result set: {result_root}")

    rows: list[dict[str, object]] = []
    for frame_dir in sorted(result_root.glob("frame_*"), key=lambda p: int(p.name.split("_")[-1])):
        match = FRAME_RE.fullmatch(frame_dir.name)
        if not match:
            continue
        stats_files = list(frame_dir.glob("*_stats.json"))
        gro = frame_dir / f"{frame_dir.name}.gro"
        time_log = frame_dir / "time.log"
        if len(stats_files) != 1 or not gro.is_file() or not time_log.is_file():
            continue
        document = json.loads(stats_files[0].read_text(encoding="utf-8"))
        stats = document["stats"]
        timing = parse_time(time_log)
        row: dict[str, object] = {
            "system": system,
            "config": config,
            "frame": int(match.group(1)),
            "time_ps": frame_time(gro),
            "atoms": int(stats["atoms"]),
            "execution_time_s": float(document["run_envs"]["execution_time"]),
            **{key: float(stats[key]) for key in PP_KEYS},
            **timing,
        }
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No complete frame records in {result_root}")

    batch_file = result_root / "performance.csv"
    batch: dict[str, str] = {}
    if batch_file.is_file():
        with batch_file.open(encoding="utf-8", newline="") as handle:
            batch = next(csv.DictReader(handle))

    def mean(key: str) -> float:
        return statistics.fmean(float(row[key]) for row in rows)

    def std(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return statistics.stdev(values) if len(values) > 1 else 0.0

    atoms = int(rows[0]["atoms"])
    batch_wall = float(batch["wall_clock_s"]) if batch else float("nan")
    summary: dict[str, object] = {
        "system": system,
        "config": config,
        "complete": complete,
        "expected_frames": expected,
        "completed_frames": len(rows),
        "processes": processes,
        "threads_per_process": threads,
        "total_cores": processes * threads,
        "atoms_per_frame": atoms,
        "batch_wall_clock_s": batch_wall,
        "dataset_throughput_frames_per_hour": (
            float(batch["frames_per_hour"]) if batch else float("nan")
        ),
        "atom_frames_per_second": (
            atoms * len(rows) / batch_wall if batch_wall > 0 else float("nan")
        ),
        "frame_wall_clock_s_mean": mean("wall_clock_s"),
        "frame_wall_clock_s_std": std("wall_clock_s"),
        "frame_user_time_s_mean": mean("user_time_s"),
        "frame_system_time_s_mean": mean("system_time_s"),
        "frame_cpu_percent_mean": mean("cpu_percent"),
        "frame_max_rss_gb_mean": mean("max_rss_kb") / 1024**2,
        "frame_max_rss_gb_max": max(float(row["max_rss_kb"]) for row in rows) / 1024**2,
        "estimated_batch_peak_rss_gb": processes * mean("max_rss_kb") / 1024**2,
    }
    for key in PP_KEYS:
        summary[f"{key}_mean"] = mean(key)
        summary[f"{key}_std"] = std(key)
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-suffix", default="_uff")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    work_root = args.work_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for system in SYSTEMS:
        for config in CONFIGS:
            config_frames, summary = collect_config(
                work_root, system, config, args.result_suffix, args.allow_partial
            )
            frames.extend(config_frames)
            summaries.append(summary)
    write_csv(output_dir / "pxpore_frame_statistics.csv", frames)
    write_csv(output_dir / "pxpore_dataset_statistics.csv", summaries)
    (output_dir / "pxpore_dataset_statistics.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(frames)} frame rows and {len(summaries)} dataset rows to {output_dir}")


if __name__ == "__main__":
    main()
