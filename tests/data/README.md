# Test and example structures

This directory contains small input structures used for examples, smoke tests,
and reproducibility checks. It intentionally stores structure files only; derived
statistics, plots, logs, and other calculation results are not included.

## Contents

- `single_H.gro`: minimal single-atom structure for quick command-line checks.
- `benchmark_structures/`: benchmark crystalline structures in several common
  formats.
- `synthetic_systems/`: synthetic geometries used to test pore-analysis
  behavior. The `anisotropic_*` structures are deterministic sealed and
  through-pore membranes with x- and z-directed surface normals. The workspace
  case generator is `../scripts/generate_anisotropic_membranes.py`.
- `tmc_dap/`: a single TMC-DAP frame for example analysis.
