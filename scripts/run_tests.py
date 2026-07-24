#!/usr/bin/env python3
"""逐项运行单元测试和 CLI 结构回归测试。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def available_cpu_count():
    """返回当前进程实际可用的 CPU 核心数。"""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def test_environment():
    """为所有测试子进程设置统一的 OMP/Numba 环境。"""
    threads = available_cpu_count()
    environment = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_path}{os.pathsep}{old_pythonpath}"
        if old_pythonpath else source_path
    )
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["NUMBA_NUM_THREADS"] = str(threads)
    environment["NUMBA_THREADING_LAYER"] = "omp"
    return environment


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run verbose unit tests and CLI structure regressions."
    )
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--strict-unset", action="store_true")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument("--regression-json", type=Path, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    environment = test_environment()

    print("MISCELLANEOUS UNIT TESTS", flush=True)
    unit_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )

    print("\nCLI STRUCTURE REGRESSION TESTS", flush=True)
    regression_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_regression_suite.py"),
        "--color",
        args.color,
    ]
    if args.include_slow:
        regression_command.append("--include-slow")
    if args.strict_unset:
        regression_command.append("--strict-unset")
    if args.regression_json is not None:
        regression_command.extend((
            "--json",
            str(args.regression_json.resolve()),
        ))
    regression_result = subprocess.run(
        regression_command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )

    if unit_result.returncode or regression_result.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
