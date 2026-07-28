#!/usr/bin/env python3
"""执行 PxPore 统一参数扫描并生成最终文档结果。"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "sensitivity_study.json"


def execute(command):
    print("+", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, check=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run and analyze the PxPore parameter sensitivity study."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--docs-dir", type=Path)
    parser.add_argument("--systems", default="all")
    parser.add_argument("--scans", default="all")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--threads-per-process", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser


def main():
    args = build_parser().parse_args()
    python = Path(sys.executable).absolute()
    scan = [
        python,
        SCRIPT_DIR / "run_parameter_sensitivity.py",
        "--config", args.config.resolve(),
        "--systems", args.systems,
        "--scans", args.scans,
        "--jobs", str(args.jobs),
        "--threads-per-process", str(args.threads_per_process),
    ]
    if args.work_dir:
        scan.extend(("--work-dir", args.work_dir.resolve()))
    if args.dry_run:
        scan.append("--dry-run")
    if not args.resume:
        scan.append("--no-resume")
    execute(scan)
    if args.dry_run:
        return

    analyze = [
        python,
        SCRIPT_DIR / "analyze_parameter_sensitivity.py",
        "--config", args.config.resolve(),
        "--scans", args.scans,
    ]
    if args.work_dir:
        analyze.extend(("--work-dir", args.work_dir.resolve()))
    if args.docs_dir:
        analyze.extend(("--docs-dir", args.docs_dir.resolve()))
    execute(analyze)

    requested_scans = {
        value.strip() for value in args.scans.split(",")
    }
    if args.scans != "all" and not requested_scans.intersection({
        "grid", "octree", "surface", "connectivity",
    }):
        return

    projection = [
        python,
        SCRIPT_DIR / "plot_case_midplane_slices.py",
        "--config", args.config.resolve(),
    ]
    if args.docs_dir:
        projection.extend((
            "--output-dir",
            args.docs_dir.resolve() / "figures" / "projections",
        ))
    execute(projection)


if __name__ == "__main__":
    main()
