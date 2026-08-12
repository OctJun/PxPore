# Plotting workflow

This directory contains the plotting and result-summarization scripts used for
the project figures. Install the optional plotting dependency before running
them:

```bash
python -m pip install -e '.[plotting]'
```

Every script exposes its input and output paths through command-line options.
Run a script with `--help` to inspect the expected files. Defaults use only
repository-relative paths under `results/` and `docs/figures/`.

## HMOF figures

The sanitized HMOF result tables are stored in
`results/hmof_plotting_data.zip`. Extract them before running the HMOF scripts:

```bash
python -m zipfile -e \
  results/hmof_plotting_data.zip \
  results/hmof_plotting_data
python scripts/plotting/plot_hmof_timing_pb_vs_pxpore.py
```

`consistent.ipynb` uses the same extracted directory and writes figures to
`docs/figures/hmof_consistency/`. The archive contains the original PoreBlazer
summary, PxPore results for 1 process x 64 threads and 8 processes x 8 threads,
and the two merged timing tables. Local paths and runtime-environment fields
have been removed.

To rebuild the archive from private source tables, provide every source path
explicitly:

```bash
python scripts/plotting/package_hmof_results.py \
  --pb-csv PB.csv \
  --pp-1p64t-csv PXP_1P64T.csv \
  --pp-8p8t-csv PXP_8P8T.csv \
  --merged-1p64t-csv MERGED_1P64T.csv \
  --merged-8p8t-csv MERGED_8P8T.csv
```

## APC-DAP 11-frame figures

The compact APC-DAP outputs are stored in
`results/apc_dap_11_frames_data.zip`. Extract the archive into `results/`, then
run the comparison script:

```bash
python -m zipfile -e results/apc_dap_11_frames_data.zip results
python scripts/plotting/plot_apc_dap_11_frames.py
```

The script compares free-volume fraction and pore-size distributions over 11
frames. PxPore Monte Carlo PSD is shown as a solid line and the center-based
PSD as a dashed line. It also writes per-frame runtime and memory records plus
an across-frame summary to `docs/figures/apc_dap_11_frames/`.

## Scaling and fill benchmark figures

Extract the packaged results before plotting:

```bash
python -m zipfile -e results/scaling_results.zip results
python -m zipfile -e results/fill_benchmark_results.zip results
python scripts/plotting/plot_scaling_results.py
python scripts/plotting/plot_fill_benchmark_performance.py
```

The corresponding structures and third-party input files are independently
available in `results/scaling_input.zip` and
`results/fill_benchmark_input.zip`. Benchmark execution and input-generation
scripts are intentionally not included in this plotting workflow.

## Script groups

- `plot_hmof_timing_pb_vs_pxpore.py`, `consistent.ipynb`, and
  `plot_speedup_vs_voxels_trend.py`: HMOF performance comparison.
- `plot_grid_error.py`, `plot_grid_octree_sweep.py`, and
  `plot_synthetic_sensitivity.py`: numerical sensitivity figures.
- `plot_psd_per_frame.py`, `plot_psd_peak_time.py`, and
  `plot_pb_psd_cumulative.py`: PSD comparison figures.
- `plot_apc_dap_11_frames.py`: APC-DAP multi-frame property, PSD, and
  performance comparison.
- `plot_tmc_dap_mwco_ffv.py` and `plot_tmc_dap_probe_analysis.py`: TMC-DAP
  probe-size and MWCO analyses.
- `plot_performance_grouped.py`, `plot_pxpore_batch_performance.py`,
  `plot_scaling_results.py`, `plot_fill_benchmark_performance.py`, and
  `plot_speedup_vs_voxels_trend.py`: runtime and scaling figures.
- `summarize_apc_pxpore.py` and `summarize_polymer_system_metrics.py`: input
  table preparation.
