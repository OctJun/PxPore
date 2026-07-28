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
