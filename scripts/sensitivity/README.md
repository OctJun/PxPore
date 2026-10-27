# PxPore parameter sensitivity study

This workflow runs one-factor parameter scans through the PxPore CLI using
structures supplied in the repository. Scripts and configuration are in
`scripts/sensitivity/`, intermediate files are in `sensitivity_work/`, and
final tables and figures are written to `docs/sensitivity_analysis/`.

Install the optional plotting dependencies:

```bash
python -m pip install -e '.[sensitivity]'
```

## Study design

| Scan | Variable | Fixed settings | Main outputs |
|---|---|---|---|
| Grid | `--grid` | `--no-octree`, surface samples=1000, PSD=centers | Volumes, accessibility, surface areas, PLD/LCD, PSD |
| Octree | `--oct-level` | Baseline grid, oct-grid=0.001 nm, surface samples=1000 | Volumes, surface areas, PLD/LCD |
| Surface | `--surface-samples` | Baseline grid, `--no-octree`, pore analysis disabled | Sacc, Stotal |
| Probe | `--probe` | Throat system only; grid=0.02 nm; octree and surface disabled | PLD, LCD, Vacc/Vtrap in raw stats |
| PSD method | `--psd-method` | Baseline grid, `--no-octree --no-surface --pore` | PSD descriptors, Wasserstein and JS distances |
| PSD local maximum | `--psd-local-max-mode` | Centers; other PSD parameters at defaults | Center count, PSD descriptors, Wasserstein and JS distances |
| PSD minimum radius | `--psd-min-center-radius` | Centers; other PSD parameters at defaults | Center count, PSD descriptors, Wasserstein and JS distances |
| PSD overlap pruning | `--psd-overlap-threshold` / `--no-psd-overlap-prune` | Centers; other PSD parameters at defaults | Center count, PSD descriptors, Wasserstein and JS distances |
| PSD weighting | `--psd-hist-weighting` | Centers; other PSD parameters at defaults | PSD descriptors, Wasserstein and JS distances |
| PSD MC samples | `--psd-mc-samples` | MC; three seeds per sample count | PSD descriptors, Wasserstein and JS distances |
| Connectivity | `--transport-direction` | anisotropic_pore_x and anisotropic_sealed_x; periodic, grid=0.04 nm; octree, surface and pore analysis disabled | Vacc, Vtrap |

The PSD method scan uses one centers calculation and three MC calculations.
MC uses 50,000 samples and seeds `11451466/11451467/11451468` to measure
variation between seeds.

| PSD parameter | Values |
|---|---|
| Local maximum | strict, plateau |
| Minimum center radius (nm) | 0, 0.005, 0.01, 0.02 |
| Overlap threshold | 0.6, 0.75, 0.9, 1.0, 1.1, 1.25; pruning disabled |
| Histogram weighting | number, volume |
| MC samples | 10000, 50000, 200000; three seeds each |

## Structures and parameter ranges

| System | Grid (nm) | Octree depth | Surface samples | PSD |
|---|---|---|---|---|
| single_h | 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 | 1–5 | 100–10000 | centers/MC |
| h512 | 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 | 1–4 | 100–10000 | centers/MC |
| overlapped_h | 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 | 1–5 | 100–10000 | centers/MC |
| throat | 0.01, 0.015, 0.02, 0.03, 0.04 | 1–4 | 100–10000 | centers/MC |
| inaccessible_pocket | 0.01, 0.015, 0.02, 0.03, 0.04 | 1–5 | 100–10000 | N/A |
| anisotropic_pore_x/z | 0.01, 0.015, 0.02, 0.03, 0.04 | 1–4 | 100–10000 | centers/MC |
| anisotropic_sealed_x/z | 0.01, 0.015, 0.02, 0.03, 0.04 | 1–4 | 100–10000 | N/A |
| HKUST1/IRMOF1 | 0.01, 0.015, 0.02, 0.03, 0.04 | 1–4 | 100–10000 | centers/MC |
| ZIF8AP | 0.01, 0.015, 0.02, 0.03, 0.04 | 1–3 at grid 0.02 nm | 100–10000 | centers/MC |
| Equal/unequal overlapping, equal separated cavities | 0.01, 0.015, 0.02, 0.03, 0.04 | 1–4 | 100–10000 | centers/MC and all PSD controls |
| TMC-DAP | 0.02, 0.025, 0.03, 0.035, 0.04 | 1–3 at grid 0.04 nm | 100–10000 | centers/MC |

Complete definitions are in `scripts/sensitivity/sensitivity_study.json`.
Single-H scans use `tests/data/synthetic_systems/single_H.gro`; all three MOFs
use GRO inputs. The three double-cavity PSD structures are supplied under
`tests/data/synthetic_systems/` and are read without modification.

Grid, surface and PSD scans of anisotropic systems use periodic through-plane
connectivity. Their octree scans use legacy/any and summarize Vvoid, Vprobe
and Stotal. These settings define the scan design in
`run_parameter_sensitivity.py`. Depth limits and fixed grids for larger
systems control the number of octree nodes.

## Commands

Generate and check the task matrix:

```bash
python scripts/sensitivity/run_sensitivity_study.py --dry-run
```

Run all scans and generate results:

```bash
python scripts/sensitivity/run_sensitivity_study.py
```

The default allocation is eight CLI processes with eight OMP/Numba threads
each. Resume is enabled by default; cached results are reused when the complete
command matches the recorded command.

Run a subset:

```bash
python scripts/sensitivity/run_sensitivity_study.py \
  --systems single_h,throat \
  --scans grid,surface
```

Run the nonzero-probe throat scan:

```bash
python scripts/sensitivity/run_sensitivity_study.py \
  --systems throat --scans probe
```

Probe radii are `0/0.05/0.10/0.14/0.20 nm`. The ideal geometric references are
`PLD=max(0, 0.46-2*probe)` and `LCD=max(0, 1.46-2*probe)`. The scan maximum is
below the throat-closing radius of 0.23 nm.

Run and summarize all PSD scans:

```bash
python scripts/sensitivity/run_sensitivity_study.py --scans psd
```

Add `--no-resume` to recalculate matching cached runs.

## Working directory

`sensitivity_work/` contains:

- `study_config.json`
- `run_matrix.json`
- `run_matrix.csv`
- `raw_results.csv`
- `runs/<run_id>/command.txt`
- `runs/<run_id>/stdout.log`
- `runs/<run_id>/stderr.log`
- Per-run PxPore stats, center and PSD outputs.

This local working directory includes runtime paths, complete CLI commands
and logs. It is excluded by `.gitignore`.

## Outputs

`docs/sensitivity_analysis/tables/` contains the general scan results:

- `study_design.csv/.md`
- `raw_results.csv`
- `metric_sensitivity.csv`
- `convergence_summary.csv/.md`
- `probe_sensitivity.csv/.md`

`docs/sensitivity_analysis/figures/` contains grid, octree and surface plots
for each system, plus summaries across systems.

`docs/sensitivity_analysis/psd_method/` contains:

- `tables/psd_descriptors.csv`
- `tables/psd_parameter_sensitivity.csv`
- `tables/psd_method_comparison.csv/.md`
- `figures/<system>_<psd_scan>.png/.pdf`
- `figures/all_systems_psd_method_sensitivity.png/.pdf`

Figures are exported as 300 dpi PNGs and vector PDFs.

`docs/sensitivity_analysis/figures/projections/` contains XZ midplane
projections of 12 analytical, anisotropic and double-cavity test systems.
Gray hatching marks atomic regions; blue shading shows the distance to the
nearest atomic surface. Combined layouts include a 1×5 analytical-system
panel, a 1×4 anisotropic-system panel, an overview of all systems, and a 1×3
double-cavity panel, each exported as PNG and PDF.

Analytical systems use theoretical reference values. Other systems use the
finest grid, deepest octree or highest surface-sampling count as a numerical
reference. Centers–MC PSD comparisons report differences between estimators.

For volume, volume fraction and surface-area plots, the left axis shows
absolute values and the right axis shows signed relative error against the
reference, in percent. PLD, LCD, PSD and center-count plots show absolute
values or distributions. When the reference is zero, only absolute values
and the reference are plotted.

Generated outputs in `docs/sensitivity_analysis/` are excluded by `.gitignore`.
The numerical snapshot is in `results/sensitivity_analysis_raw_results.zip`;
[results/README.md](../../results/README.md) describes its coverage and checksums.
