#!/usr/bin/env python3
"""通过 PxPore CLI 执行统一的单因素敏感性扫描。"""

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("sensitivity_study.json")
PSD_SCANS = (
    "psd_method",
    "psd_local_max",
    "psd_min_radius",
    "psd_overlap",
    "psd_weighting",
    "psd_mc_samples",
)
SCANS = (
    "grid", "octree", "surface", "probe",
    *PSD_SCANS, "connectivity",
)
COLORS = {
    "PASS": "\033[32m",
    "ERROR": "\033[31m",
    "CACHED": "\033[36m",
}
COLOR_RESET = "\033[0m"


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError("sensitivity config schema_version must be 2")
    if not isinstance(config.get("systems"), dict):
        raise ValueError("sensitivity config systems must be an object")
    return config


def resolve_workspace_path(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_structure(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def cli_arguments(arguments):
    values = []
    for key, value in arguments.items():
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                values.append(option)
        elif value is not None:
            values.extend((option, str(value)))
    return values


def common_arguments(config, spec, grid):
    fixed = config["fixed_parameters"]
    return {
        "grid": grid,
        "probe": fixed["probe_nm"],
        "connectivity": spec["connectivity"],
        "transport_direction": spec["transport_direction"],
    }


def add_run(runs, system, spec, scan, label, value, arguments):
    run_id = f"{system}__{scan}__{label}"
    runs.append({
        "run_id": run_id,
        "system": system,
        "category": spec["category"],
        "description": spec["description"],
        "source": str(resolve_structure(spec["path"])),
        "scan": scan,
        "parameter_value": value,
        "label": label,
        "metrics": spec["metrics"].get(scan, ("PSD", "center_count")),
        "arguments": arguments,
    })


def build_grid_runs(config, system, spec, runs):
    values = config["parameter_ranges"]["grid_nm"][spec["grid_class"]]
    fixed = config["fixed_parameters"]
    for grid in values:
        arguments = common_arguments(config, spec, grid)
        arguments.update({
            "no_octree": True,
            "surface_samples": fixed["surface_samples"],
            "pore": bool(spec["pore_applicable"]),
            "psd_method": fixed["psd_method"],
        })
        add_run(
            runs, system, spec, "grid", f"{grid:.4f}", grid, arguments)


def build_octree_runs(config, system, spec, runs):
    fixed = config["fixed_parameters"]
    periodic_directional = spec["connectivity"] == "periodic"
    octree_grid = spec.get(
        "octree_grid_nm", spec["baseline_grid_nm"])
    octree_levels = spec.get(
        "octree_levels",
        config["parameter_ranges"]["octree_level"],
    )
    for level in octree_levels:
        arguments = common_arguments(
            config, spec, octree_grid)
        arguments.update({
            "oct_level": level,
            "oct_grid": fixed["oct_grid_nm"],
            "surface_samples": fixed["surface_samples"],
            "pore": bool(
                spec["pore_applicable"] and not periodic_directional),
            "psd_method": fixed["psd_method"],
        })
        if periodic_directional:
            arguments["connectivity"] = spec.get(
                "octree_connectivity", "legacy")
            arguments["transport_direction"] = "any"
        add_run(
            runs, system, spec, "octree", str(level), level, arguments)


def build_surface_runs(config, system, spec, runs):
    for samples in config["parameter_ranges"]["surface_samples"]:
        arguments = common_arguments(
            config, spec, spec["baseline_grid_nm"])
        arguments.update({
            "no_octree": True,
            "surface_samples": samples,
        })
        add_run(
            runs, system, spec, "surface",
            str(samples), samples, arguments)


def build_probe_runs(config, system, spec, runs):
    if not spec.get("probe_scan", False):
        return
    fixed = config["fixed_parameters"]
    settings = config["scan_defaults"]["probe"]
    for probe in config["parameter_ranges"]["probe_nm"]:
        arguments = common_arguments(
            config, spec, settings["grid_nm"])
        arguments.update({
            "probe": probe,
            "no_octree": True,
            "no_surface": True,
            "pore": True,
            "psd_method": fixed["psd_method"],
        })
        add_run(
            runs, system, spec, "probe",
            f"{probe:.3f}", probe, arguments)


def center_psd_arguments(config, spec):
    fixed = config["fixed_parameters"]
    arguments = common_arguments(
        config, spec, spec["baseline_grid_nm"])
    arguments.update({
        "no_octree": True,
        "no_surface": True,
        "pore": True,
        "psd_method": "centers",
        "psd_local_max_mode": fixed["psd_local_max_mode"],
        "psd_min_center_radius": fixed["psd_min_center_radius_nm"],
        "psd_overlap_threshold": fixed["psd_overlap_threshold"],
        "psd_hist_weighting": fixed["psd_hist_weighting"],
    })
    return arguments


def mc_psd_arguments(config, spec, samples, seed):
    arguments = common_arguments(
        config, spec, spec["baseline_grid_nm"])
    arguments.update({
        "no_octree": True,
        "no_surface": True,
        "pore": True,
        "psd_method": "mc",
        "psd_mc_samples": samples,
        "psd_mc_seed": seed,
    })
    return arguments


def build_psd_method_runs(config, system, spec, runs):
    if not spec["pore_applicable"]:
        return
    fixed = config["fixed_parameters"]
    arguments = center_psd_arguments(config, spec)
    add_run(
        runs, system, spec, "psd_method",
        "centers", "centers", arguments)
    for seed in fixed["psd_mc_seeds"]:
        arguments = mc_psd_arguments(
            config, spec, fixed["psd_mc_samples"], seed)
        add_run(
            runs, system, spec, "psd_method",
            f"mc_seed_{seed}", "mc", arguments)


def build_psd_local_max_runs(config, system, spec, runs):
    if not spec["pore_applicable"]:
        return
    for mode in config["parameter_ranges"]["psd_local_max_mode"]:
        arguments = center_psd_arguments(config, spec)
        arguments["psd_local_max_mode"] = mode
        add_run(
            runs, system, spec, "psd_local_max",
            mode, mode, arguments)


def build_psd_min_radius_runs(config, system, spec, runs):
    if not spec["pore_applicable"]:
        return
    for radius in config["parameter_ranges"]["psd_min_center_radius_nm"]:
        arguments = center_psd_arguments(config, spec)
        arguments["psd_min_center_radius"] = radius
        add_run(
            runs, system, spec, "psd_min_radius",
            f"{radius:.4f}", radius, arguments)


def build_psd_overlap_runs(config, system, spec, runs):
    if not spec["pore_applicable"]:
        return
    for threshold in config["parameter_ranges"]["psd_overlap_threshold"]:
        arguments = center_psd_arguments(config, spec)
        arguments["psd_overlap_threshold"] = threshold
        add_run(
            runs, system, spec, "psd_overlap",
            f"threshold_{threshold:.2f}", threshold, arguments)
    arguments = center_psd_arguments(config, spec)
    arguments["no_psd_overlap_prune"] = True
    add_run(
        runs, system, spec, "psd_overlap",
        "prune_off", "off", arguments)


def build_psd_weighting_runs(config, system, spec, runs):
    if not spec["pore_applicable"]:
        return
    for weighting in config["parameter_ranges"]["psd_hist_weighting"]:
        arguments = center_psd_arguments(config, spec)
        arguments["psd_hist_weighting"] = weighting
        add_run(
            runs, system, spec, "psd_weighting",
            weighting, weighting, arguments)


def build_psd_mc_sample_runs(config, system, spec, runs):
    if not spec["pore_applicable"]:
        return
    ranges = config["parameter_ranges"]
    for samples in ranges["psd_mc_samples"]:
        for seed in config["fixed_parameters"]["psd_mc_seeds"]:
            arguments = mc_psd_arguments(
                config, spec, samples, seed)
            add_run(
                runs, system, spec, "psd_mc_samples",
                f"{samples}_seed_{seed}", samples, arguments)


def build_connectivity_runs(config, system, spec, runs):
    if not spec.get("connectivity_scan", False):
        return
    settings = config["scan_defaults"]["connectivity"]
    for direction in config["parameter_ranges"]["transport_direction"]:
        arguments = common_arguments(
            config, spec, settings["grid_nm"])
        arguments.update({
            "connectivity": settings["connectivity"],
            "transport_direction": direction,
            "no_octree": True,
            "no_surface": True,
        })
        add_run(
            runs,
            system,
            spec,
            "connectivity",
            direction,
            direction,
            arguments,
        )


def build_matrix(config, selected_systems, selected_scans):
    builders = {
        "grid": build_grid_runs,
        "octree": build_octree_runs,
        "surface": build_surface_runs,
        "probe": build_probe_runs,
        "psd_method": build_psd_method_runs,
        "psd_local_max": build_psd_local_max_runs,
        "psd_min_radius": build_psd_min_radius_runs,
        "psd_overlap": build_psd_overlap_runs,
        "psd_weighting": build_psd_weighting_runs,
        "psd_mc_samples": build_psd_mc_sample_runs,
        "connectivity": build_connectivity_runs,
    }
    runs = []
    for system, spec in config["systems"].items():
        if system not in selected_systems:
            continue
        for scan in SCANS:
            if scan in selected_scans:
                builders[scan](config, system, spec, runs)
    return runs


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matrix_csv_rows(runs):
    rows = []
    for run in runs:
        rows.append({
            "run_id": run["run_id"],
            "system": run["system"],
            "category": run["category"],
            "scan": run["scan"],
            "parameter_value": run["parameter_value"],
            "metrics": ", ".join(run["metrics"]),
            "source": run["source"],
            "arguments": json.dumps(
                run["arguments"], ensure_ascii=False, sort_keys=True),
        })
    return rows


def flatten_payload(record, payload):
    for key, value in payload.get("stats", {}).items():
        record[key] = value
    for section in ("settings", "info", "timings"):
        for key, value in payload.get(section, {}).items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            record[f"{section}_{key}"] = value


def result_record(run, status, elapsed, command, payload=None, error=""):
    record = {
        "status": status,
        "run_id": run["run_id"],
        "system": run["system"],
        "category": run["category"],
        "scan": run["scan"],
        "parameter_value": run["parameter_value"],
        "label": run["label"],
        "metrics": ", ".join(run["metrics"]),
        "elapsed_s": elapsed,
        "source": run["source"],
        "command": shlex.join(str(value) for value in command),
        "error": error,
    }
    for key, value in run["arguments"].items():
        record[f"argument_{key}"] = value
    if payload is not None:
        flatten_payload(record, payload)
    return record


def run_one(run, python, work_dir, threads, resume):
    run_dir = work_dir / "runs" / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    staged_input = run_dir / Path(run["source"]).name
    stats_path = run_dir / "result_stats.json"
    command = [
        str(python),
        "-m",
        "PxPore",
        str(staged_input),
        *cli_arguments(run["arguments"]),
        "--threads",
        str(threads),
        "--stats",
        "--out_prefix",
        "result",
    ]
    command_text = shlex.join(command) + "\n"
    command_path = run_dir / "command.txt"
    record_path = run_dir / "run_record.json"
    cache_matches = (
        resume
        and stats_path.is_file()
        and command_path.is_file()
        and command_path.read_text(encoding="utf-8") == command_text
    )
    if cache_matches:
        try:
            if record_path.is_file():
                record = json.loads(
                    record_path.read_text(encoding="utf-8"))
                record["status"] = "CACHED"
                return record
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            return result_record(
                run, "CACHED", 0.0, command, payload=payload)
        except (OSError, ValueError, TypeError):
            pass

    shutil.copy2(run["source"], staged_input)
    command_path.write_text(command_text, encoding="utf-8")
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
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=run_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    (run_dir / "stdout.log").write_text(
        completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(
        completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        record = result_record(
            run, "ERROR", elapsed, command, error=error)
    elif not stats_path.is_file():
        record = result_record(
            run, "ERROR", elapsed, command,
            error="PxPore did not create result_stats.json")
    else:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        record = result_record(
            run, "PASS", elapsed, command, payload=payload)
    record["returncode"] = completed.returncode
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record


def print_record(index, total, record):
    status = record["status"]
    color = COLORS.get(status, "")
    print(
        f"{color}{status:6s}{COLOR_RESET} "
        f"[{index:3d}/{total}] "
        f"{record['system']:<24s} "
        f"{record['scan']:<10s} "
        f"{record['label']}",
        flush=True,
    )


def parse_selection(value, choices, label):
    if value == "all":
        return tuple(choices)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if label == "scans" and "psd" in items:
        items = [
            scan
            for item in items
            for scan in (PSD_SCANS if item == "psd" else (item,))
        ]
    selected = tuple(dict.fromkeys(items))
    unknown = sorted(set(selected) - set(choices))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return selected


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run unified PxPore parameter sensitivity scans."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--systems", default="all")
    parser.add_argument("--scans", default="all")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--threads-per-process", type=int)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser


def main():
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    systems = parse_selection(
        args.systems, config["systems"], "systems")
    scans = parse_selection(args.scans, SCANS, "scans")
    for system in systems:
        source = resolve_structure(config["systems"][system]["path"])
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing structure {system}: {source}")

    work_dir = (
        args.work_dir.resolve()
        if args.work_dir else resolve_workspace_path(config["work_dir"])
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, work_dir / "study_config.json")
    runs = build_matrix(config, systems, scans)
    all_runs = build_matrix(
        config, tuple(config["systems"]), SCANS)
    (work_dir / "run_matrix.json").write_text(
        json.dumps(all_runs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(
        work_dir / "run_matrix.csv", matrix_csv_rows(all_runs))
    (work_dir / "selected_run_matrix.json").write_text(
        json.dumps(runs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(
        work_dir / "selected_run_matrix.csv",
        matrix_csv_rows(runs),
    )
    print(
        f"selected={len(runs)} matrix={len(all_runs)} "
        f"work_dir={work_dir}",
        flush=True,
    )
    if args.dry_run:
        return

    execution = config["execution"]
    jobs = max(1, args.jobs or execution["jobs"])
    threads = max(
        1, args.threads_per_process or execution["threads_per_process"])
    python = (
        args.python.absolute()
        if args.python else Path(sys.executable).absolute()
    )
    print(
        f"parallel={jobs} processes x {threads} threads",
        flush=True,
    )
    summary_path = work_dir / "raw_results.csv"
    current_ids = {run["run_id"] for run in runs}
    valid_ids = {run["run_id"] for run in all_runs}
    record_map = {
        row["run_id"]: row for row in read_csv(summary_path)
        if (
            row.get("run_id") in valid_ids
            and row.get("run_id") not in current_ids
        )
    }
    current_records = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                run_one,
                run,
                python,
                work_dir,
                threads,
                args.resume,
            ): run
            for run in runs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            run = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = result_record(
                    run, "ERROR", 0.0, (),
                    error=f"{type(exc).__name__}: {exc}")
            current_records.append(record)
            record_map[record["run_id"]] = record
            write_csv(
                summary_path,
                sorted(
                    record_map.values(),
                    key=lambda item: item["run_id"],
                ),
            )
            print_record(index, len(runs), record)

    failures = [
        row for row in current_records if row["status"] == "ERROR"
    ]
    print(
        f"complete={len(current_records) - len(failures)} "
        f"failed={len(failures)}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
