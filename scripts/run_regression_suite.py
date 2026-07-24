#!/usr/bin/env python3
"""按清单运行结构文件，并比较指定的 PxPore 数值指标。"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from numbers import Real
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_COLORS = {
    "PASS": "\033[32m",
    "FAIL": "\033[31m",
    "ERROR": "\033[91m",
    "SKIP": "\033[33m",
    "UNSET": "\033[35m",
}
COLOR_RESET = "\033[0m"
TABLE_HEADERS = (
    "RESULT",
    "TEST NAME",
    "OUTPUT VALUE",
    "EXPECTED VALUE",
    "METRIC",
)


def load_manifest(path):
    """读取并检查回归清单的基本结构。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("regression manifest schema_version must be 1")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("regression manifest cases must be a list")
    return cases


def resolve_input(path_text):
    """相对输入路径统一以 PxPore 仓库根目录为基准。"""
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Regression input does not exist: {path}")
    return path


def get_metric(payload, dotted_path):
    """从嵌套字典中读取以点分隔的指标路径。"""
    value = payload
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Metric path not found: {dotted_path}")
        value = value[key]
    return value


def compare_values(observed, reference, absolute_tolerance, relative_tolerance):
    """数值使用 isclose，其它标量使用严格相等。"""
    if (
        isinstance(observed, Real)
        and not isinstance(observed, bool)
        and isinstance(reference, Real)
        and not isinstance(reference, bool)
    ):
        return math.isclose(
            float(observed),
            float(reference),
            abs_tol=float(absolute_tolerance),
            rel_tol=float(relative_tolerance),
        )
    return observed == reference


def available_cpu_count():
    """返回当前进程实际可用的 CPU 核心数。"""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def arguments_to_cli(arguments):
    """将 API 风格参数转换为 PxPore CLI 参数。"""
    command = []
    for key, value in arguments.items():
        if key in ("input", "out_prefix", "stats", "threads"):
            continue
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None:
            command.extend((option, str(value)))
    return command


def case_status_records(case, status, message):
    """为跳过的用例保留每一个配置指标及其预期值。"""
    metrics = case.get("metrics", {})
    records = []
    if metrics:
        records.extend({
            "case": case.get("name", "<unnamed>"),
            "metric": metric_path,
            "status": status,
            "observed": None,
            "reference": spec.get("reference"),
            "message": message,
        } for metric_path, spec in metrics.items())
    else:
        records.append({
            "case": case.get("name", "<unnamed>"),
            "metric": "",
            "status": status,
            "observed": None,
            "reference": None,
            "message": message,
        })
    return records


def run_case(case, include_slow=False, threads=None):
    """在临时目录中运行一个结构，并返回逐指标比较记录。"""
    name = case.get("name", "<unnamed>")
    threads = available_cpu_count() if threads is None else threads
    if not case.get("enabled", True):
        return case_status_records(case, "SKIP", "disabled")
    if case.get("slow", False) and not include_slow:
        return case_status_records(case, "SKIP", "slow")

    source = resolve_input(case["input"])
    arguments = dict(case.get("arguments", {}))
    if "input" in arguments:
        raise ValueError(f"Case {name}: arguments must not contain input")
    metrics = case.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Case {name}: metrics must be an object")

    with tempfile.TemporaryDirectory(prefix="pxpore_regression_") as tmp:
        tmp_path = Path(tmp)
        staged_input = tmp_path / source.name
        shutil.copy2(source, staged_input)
        command = [
            sys.executable,
            "-m",
            "PxPore",
            str(staged_input),
            *arguments_to_cli(arguments),
            "--threads",
            str(threads),
            "--stats",
            "--out_prefix",
            "regression",
        ]
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
        completed = subprocess.run(
            command,
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"PxPore CLI exited with {completed.returncode}: {detail}"
            )
        stats_path = tmp_path / "regression_stats.json"
        if not stats_path.is_file():
            raise RuntimeError(
                f"PxPore CLI did not create stats output: {stats_path}"
            )
        stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
        threads_used = get_metric(
            stats_payload, "settings.threads_used")
        threading_layer = get_metric(
            stats_payload, "settings.threading_layer")
        if threads_used != threads or threading_layer != "omp":
            raise RuntimeError(
                "unexpected parallel configuration: "
                f"threads={threads_used}/{threads}, "
                f"threading_layer={threading_layer}/omp"
            )

        if not metrics:
            records = [{
                "case": name,
                "metric": "",
                "status": "UNSET",
                "observed": None,
                "reference": None,
                "message": "no metrics configured",
            }]
        else:
            records = []

        for metric_path, spec in metrics.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Case {name}, metric {metric_path}: spec must be an object"
                )
            observed = get_metric(stats_payload, metric_path)
            if hasattr(observed, "item"):
                observed = observed.item()
            reference = spec.get("reference")
            record = {
                "case": name,
                "metric": metric_path,
                "observed": observed,
                "reference": reference,
            }
            if reference is None:
                record["status"] = "UNSET"
                record["message"] = "reference is not configured"
            else:
                absolute_tolerance = spec.get("absolute_tolerance", 0.0)
                relative_tolerance = spec.get("relative_tolerance", 0.0)
                passed = compare_values(
                    observed,
                    reference,
                    absolute_tolerance,
                    relative_tolerance,
                )
                record["status"] = "PASS" if passed else "FAIL"
                record["absolute_tolerance"] = absolute_tolerance
                record["relative_tolerance"] = relative_tolerance
            records.append(record)
        return records


def record_row(record):
    """将一条比较记录转换为可打印文本。"""
    observed = record.get("observed")
    reference = record.get("reference")
    return (
        str(record["status"]),
        str(record["case"]),
        "N/A" if observed is None else str(observed),
        "N/A" if reference is None else str(reference),
        str(record.get("metric", "")),
    )


def table_widths(rows):
    """根据已知内容计算稳定列宽。"""
    widths = [
        max(len(TABLE_HEADERS[index]), *(len(row[index]) for row in rows))
        for index in range(len(TABLE_HEADERS))
    ]
    widths[2] = max(widths[2], 22)
    return widths


def manifest_table_widths(cases):
    """运行前从清单计算列宽，保证流式输出时表格不跳动。"""
    rows = []
    for case in cases:
        name = str(case.get("name", "<unnamed>"))
        metrics = case.get("metrics", {})
        if metrics:
            for metric_path, spec in metrics.items():
                reference = (
                    spec.get("reference", "N/A")
                    if isinstance(spec, dict) else "N/A"
                )
                rows.append((
                    "UNSET",
                    name,
                    "N/A",
                    str(reference),
                    str(metric_path),
                ))
        else:
            rows.append(("UNSET", name, "N/A", "N/A", ""))
    return table_widths(rows)


def print_table_header(widths):
    """打印一次表头。"""
    row_format = " | ".join(
        f"{{:<{width}}}" for width in widths
    )
    print(row_format.format(*TABLE_HEADERS), flush=True)
    print("-+-".join("-" * width for width in widths), flush=True)


def print_record(record, widths, use_color):
    """立即打印一条运行状态或比较结果。"""
    row = record_row(record)
    cells = [
        f"{value:<{widths[index]}}"
        for index, value in enumerate(row)
    ]
    if use_color and row[0] in STATUS_COLORS:
        cells[0] = f"{STATUS_COLORS[row[0]]}{cells[0]}{COLOR_RESET}"
    print(" | ".join(cells), flush=True)


def print_summary(records, use_color):
    """打印最终状态汇总。"""
    counts = {}
    for record in records:
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1
    summary_items = []
    for status in sorted(counts):
        item = f"{status}={counts[status]}"
        if use_color and status in STATUS_COLORS:
            item = f"{STATUS_COLORS[status]}{item}{COLOR_RESET}"
        summary_items.append(item)
    print(
        "\nSUMMARY: "
        + (" ".join(summary_items) if summary_items else "no cases"),
        flush=True,
    )


def print_records(records, use_color=None):
    """一次性打印记录，供独立调用和测试使用。"""
    if use_color is None:
        use_color = sys.stdout.isatty()
    rows = [record_row(record) for record in records]
    widths = table_widths(rows)
    print_table_header(widths)
    for record in records:
        print_record(record, widths, use_color)
    print_summary(records, use_color)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run manifest-driven PxPore numerical regressions."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "tests" / "regression_cases.json",
    )
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--strict-unset", action="store_true")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    cases = load_manifest(args.manifest.resolve())
    threads = available_cpu_count()
    records = []
    use_color = (
        sys.stdout.isatty()
        if args.color == "auto"
        else args.color == "always"
    )
    widths = manifest_table_widths(cases)
    print_table_header(widths)
    for case in cases:
        try:
            case_records = run_case(case, args.include_slow, threads)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            case_records = [{
                "case": case.get("name", "<unnamed>"),
                "metric": "",
                "status": "ERROR",
                "observed": message,
                "reference": "successful CLI execution",
                "message": message,
            }]
        records.extend(case_records)
        for record in case_records:
            print_record(record, widths, use_color)
    print_summary(records, use_color)
    if args.json is not None:
        args.json.write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8")

    failed = any(
        record["status"] in ("FAIL", "ERROR") for record in records
    )
    unset = any(record["status"] == "UNSET" for record in records)
    if failed or (args.strict_unset and unset):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
