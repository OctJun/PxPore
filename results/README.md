# Packaged numerical results

## Sensitivity analysis

`sensitivity_analysis_raw_results.zip` contains the sanitized raw numerical
outputs used by the sensitivity-analysis workflow.

Contents:

- Study configuration and the complete 637-run matrix.
- Sanitized summary records for 589 completed or cached runs.
- Per-run stats, pore-center, center-PSD, and Monte Carlo PSD files.
- `missing_runs.csv`, listing the 48 runs not present in this snapshot.
- `MANIFEST.sha256`, covering every file inside the archive.

The omitted 48 runs are the grid, octree, and surface scans for the three
double-cavity PSD systems. Their PSD parameter scans are included.

The archive deliberately excludes duplicated input structures, commands,
working paths, Python paths, host metadata, timestamps, run records, and
stdout/stderr. The adjacent `.sha256` file verifies the ZIP itself.

Rebuild the package from a local `sensitivity_work/` directory with:

```bash
python scripts/sensitivity/package_results.py
```

## HMOF plotting data

`hmof_plotting_data.zip` contains the sanitized numerical tables used by the
HMOF property and performance comparison plots:

- the PoreBlazer summary table;
- PxPore results for 1 process x 64 threads;
- PxPore results for 8 processes x 8 threads;
- merged timing tables for both PxPore configurations.

Runtime paths, input-file paths, Python environment details, platform strings,
and full commands are excluded. The archive contains a manifest with retained
columns, removed columns, row counts, and per-file SHA-256 checksums. The
adjacent `.sha256` file verifies the ZIP itself.

Usage and rebuild instructions are documented in
[`scripts/plotting/README.md`](../scripts/plotting/README.md).

## APC-DAP plotting data

`apc_dap_11_frames_data.zip` contains the compact numerical outputs used for
the 11-frame APC-DAP comparison:

- PxPore Monte Carlo and center-based PSD results;
- PoreBlazer network-accessible PSD and summary files;
- Zeo++ PSD and pore-volume files;
- runtime logs used to summarize wall time, CPU time, and peak memory.

Large structure and temporary files are excluded. Absolute input paths and
runtime-environment metadata are removed from the retained PxPore statistics.
The adjacent `.sha256` file verifies the ZIP archive.

## Strong-scaling benchmark

The benchmark is split into two archives:

- `scaling_input.zip` contains the single polymer GRO structure shared by all
  runs;
- `scaling_results.zip` contains the legacy/periodic connectivity and
  center/Monte Carlo PSD results for 1--64 threads and three repeats, including
  summaries, raw timing logs, commands, metadata, statistics, and PSD outputs.

The duplicated copy of the input structure in each run directory is omitted.
Per-run `*_center.txt` coordinate tables are also omitted because they do not
vary with thread count and are not used by the scaling or PSD plots.
Absolute local paths in the retained records are replaced by `<REPO_ROOT>` and
`<WORK_DIR>`; complete commands and arguments are retained.

## Fill benchmark

The 10 nm synthetic fill benchmark is also split into two archives:

- `fill_benchmark_input.zip` contains the 0%, 10%, 30%, 50%, 70%, 90%, and
  100% fill structures in GRO, XYZ, and CIF formats, together with UFF and
  PoreBlazer input files;
- `fill_benchmark_results.zip` contains all available PxPore, PoreBlazer, and
  Zeo++ numerical outputs, timing logs, commands, failure logs, and summary
  tables for three repeats.

Structure and force-field files duplicated inside individual run directories
are omitted. Absolute local paths are replaced by the same placeholders while
the commands remain intact. Each archive has an adjacent `.sha256` checksum.
