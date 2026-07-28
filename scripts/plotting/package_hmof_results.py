#!/usr/bin/env python3
"""Sanitize and package the HMOF CSV files used by the plotting scripts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "hmof_plotting_data.zip"
ARCHIVE_ROOT = "hmof_plotting_data"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

PB_PRIVATE_COLUMNS = {"run_dir", "summary_dat", "time_log"}
PXP_PRIVATE_COLUMNS = {
    "run_dir",
    "gro_file",
    "json_file",
    "time_log",
    "python_exe",
    "cwd",
    "platform",
    "full_command",
    "run_time_str",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_csv(source: Path, destination: Path, removed: set[str]) -> dict:
    with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        columns = [column for column in reader.fieldnames if column not in removed]
        removed_present = [
            column for column in reader.fieldnames if column in removed
        ]

        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=columns,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            row_count = 0
            for row in reader:
                writer.writerow(row)
                row_count += 1

    return {
        "rows": row_count,
        "columns": columns,
        "removed_columns": removed_present,
        "sha256": sha256_file(destination),
    }


def write_zip_member(archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
    info = zipfile.ZipInfo(arcname, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with path.open("rb") as source, archive.open(info, "w") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def package_results(args: argparse.Namespace) -> None:
    sources = {
        "hmof_existing_results.csv": (args.pb_csv, PB_PRIVATE_COLUMNS),
        "hmof_pp_results.csv": (args.pp_1p64t_csv, PXP_PRIVATE_COLUMNS),
        "hmof_pp_results_8p8t.csv": (args.pp_8p8t_csv, PXP_PRIVATE_COLUMNS),
        "hmof_pb_pxpore_1p64t_merged_timing.csv": (
            args.merged_1p64t_csv,
            set(),
        ),
        "hmof_pb_pxpore_8p8t_merged_timing.csv": (
            args.merged_8p8t_csv,
            set(),
        ),
    }
    missing = [str(path) for path, _ in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source CSV files:\n" + "\n".join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pxpore_hmof_") as temporary:
        staging = Path(temporary)
        manifest: dict[str, object] = {
            "format_version": 1,
            "description": "Sanitized HMOF numerical results for plotting.",
            "files": {},
        }
        for name, (source, removed) in sources.items():
            metadata = sanitize_csv(source, staging / name, removed)
            manifest["files"][name] = metadata
            print(f"{name}: {metadata['rows']} rows, {len(metadata['columns'])} columns")

        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        readme_path = staging / "README.md"
        readme_path.write_text(
            "# HMOF plotting data\n\n"
            "This directory contains the sanitized numerical CSV files used by "
            "the HMOF comparison plots. Runtime paths, input-file paths, Python "
            "environment details, platform strings, and full commands were "
            "removed. `manifest.json` records retained columns, removed columns, "
            "row counts, and SHA-256 checksums.\n",
            encoding="utf-8",
        )

        with zipfile.ZipFile(
            args.output,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(staging.iterdir(), key=lambda item: item.name):
                write_zip_member(
                    archive,
                    path,
                    f"{ARCHIVE_ROOT}/{path.name}",
                )

    checksum = sha256_file(args.output)
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    checksum_path.write_text(
        f"{checksum}  {args.output.name}\n",
        encoding="ascii",
    )
    print(f"archive: {args.output}")
    print(f"sha256: {checksum}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pb-csv", type=Path, required=True)
    parser.add_argument("--pp-1p64t-csv", type=Path, required=True)
    parser.add_argument("--pp-8p8t-csv", type=Path, required=True)
    parser.add_argument("--merged-1p64t-csv", type=Path, required=True)
    parser.add_argument("--merged-8p8t-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    package_results(parse_args())
