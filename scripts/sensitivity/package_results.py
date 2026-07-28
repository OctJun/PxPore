#!/usr/bin/env python3
"""打包可公开发布的敏感性分析原始结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "sensitivity_study.json"
DEFAULT_WORK_DIR = REPO_ROOT / "sensitivity_work"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "sensitivity_analysis_raw_results.zip"
)
RAW_EXCLUDED_COLUMNS = {
    "source",
    "command",
    "error",
    "settings_atoms_table_path",
    "settings_sched_affinity_count",
}
RESULT_FILENAMES = (
    "result_stats.json",
    "result_center.txt",
    "result_psd.txt",
    "result_voxel_mc_psd.txt",
)
ARCHIVE_ROOT = "sensitivity_analysis_raw"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def sanitize_raw_results(rows):
    if not rows:
        return b""
    columns = [
        column for column in rows[0]
        if column not in RAW_EXCLUDED_COLUMNS
    ]
    sanitized = [
        {column: row.get(column, "") for column in columns}
        for row in rows
    ]
    return csv_bytes(sanitized, columns)


def sanitize_matrix(matrix_rows, config):
    source_by_system = {
        system: spec["path"]
        for system, spec in config["systems"].items()
    }
    columns = (
        "run_id",
        "system",
        "category",
        "scan",
        "parameter_value",
        "metrics",
        "source",
        "arguments",
    )
    sanitized = []
    for row in matrix_rows:
        result = {column: row.get(column, "") for column in columns}
        result["source"] = source_by_system[row["system"]]
        sanitized.append(result)
    return csv_bytes(sanitized, columns)


def sanitize_stats(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("run_envs", None)
    settings = payload.get("settings", {})
    settings.pop("sched_affinity_count", None)
    atoms_path = settings.get("atoms_table_path")
    if atoms_path:
        settings["atoms_table_path"] = Path(atoms_path).name
    return (
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def archive_readme(matrix_rows, result_rows, missing_rows):
    statuses = Counter(row["status"] for row in result_rows)
    lines = [
        "# PxPore sensitivity analysis raw results",
        "",
        "This archive contains sanitized numerical outputs from the "
        "PxPore sensitivity study.",
        "",
        f"- Planned runs: {len(matrix_rows)}",
        f"- Recorded runs: {len(result_rows)}",
        f"- Missing runs: {len(missing_rows)}",
        "- Status counts: "
        + ", ".join(
            f"{status}={count}"
            for status, count in sorted(statuses.items())
        ),
        "",
        "Included per-run files: result_stats.json, result_center.txt, "
        "result_psd.txt, and result_voxel_mc_psd.txt when produced.",
        "",
        "Excluded as nonessential or environment-specific: staged input "
        "structures, command lines, run records, stdout/stderr, Python "
        "paths, working directories, timestamps, and host information.",
        "",
        "The missing runs are listed in missing_runs.csv. They are not "
        "silently treated as successful.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Package sanitized sensitivity raw results."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main():
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    work_dir = args.work_dir.resolve()
    output = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    matrix_rows = read_csv(work_dir / "run_matrix.csv")
    result_rows = read_csv(work_dir / "raw_results.csv")
    result_ids = {row["run_id"] for row in result_rows}
    missing_rows = [
        row for row in matrix_rows if row["run_id"] not in result_ids
    ]

    entries = {
        f"{ARCHIVE_ROOT}/README.md": archive_readme(
            matrix_rows, result_rows, missing_rows),
        f"{ARCHIVE_ROOT}/study_config.json": (
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
        f"{ARCHIVE_ROOT}/run_matrix.csv": sanitize_matrix(
            matrix_rows, config),
        f"{ARCHIVE_ROOT}/raw_results.csv": sanitize_raw_results(
            result_rows),
        f"{ARCHIVE_ROOT}/missing_runs.csv": sanitize_matrix(
            missing_rows, config),
    }

    for row in result_rows:
        run_id = row["run_id"]
        run_dir = work_dir / "runs" / run_id
        for filename in RESULT_FILENAMES:
            path = run_dir / filename
            if not path.is_file():
                continue
            archive_path = f"{ARCHIVE_ROOT}/runs/{run_id}/{filename}"
            entries[archive_path] = (
                sanitize_stats(path)
                if filename == "result_stats.json"
                else path.read_bytes()
            )

    manifest_lines = []
    for path, content in sorted(entries.items()):
        digest = hashlib.sha256(content).hexdigest()
        relative = path.removeprefix(f"{ARCHIVE_ROOT}/")
        manifest_lines.append(f"{digest}  {relative}")
    entries[f"{ARCHIVE_ROOT}/MANIFEST.sha256"] = (
        "\n".join(manifest_lines) + "\n"
    ).encode("ascii")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path, content in sorted(entries.items()):
            info = zipfile.ZipInfo(
                filename=path,
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compresslevel=9)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(
        f"{digest}  {output.name}\n", encoding="ascii")
    print(
        f"archive={output} runs={len(result_rows)}/{len(matrix_rows)} "
        f"missing={len(missing_rows)} files={len(entries)}",
        flush=True,
    )
    print(f"sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
