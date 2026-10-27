# Supplementary R4 benchmark inputs

These two files complete the five-structure performance benchmark inputs.
Both are byte-identical to the inputs
used in `performance_benchmark_v11_candidate_20260918` (all three repeats).

| File | Benchmark case | Size | Original workspace path, relative to tmp_work_zone |
|---|---|---:|---|
| single_H_10A.gro | single_h | 129 bytes | performance_benchmark_v11_candidate_20260918/runs/single_h/pp/r1/single_H_10A.gro |
| TaPa-1.gro | tapa1 | 18,827 bytes | performance_benchmark_v11_candidate_20260918/runs/tapa1/pp/r1/TaPa-1.gro |

`single_H_10A.gro` contains one hydrogen atom at (0.5, 0.5, 0.5) nm in a
1 nm cubic box. Do not substitute `tests/data/single_H.gro`, whose box is
0.4 nm. The remaining inputs are already in the repository:

- `tests/data/benchmark_structures/HKUST1.gro`
- `tests/data/synthetic_systems/H512.gro`
- `tests/data/tmc_dap/TMC-DAP_prod.gro`

Verify these supplementary inputs from this directory:

```bash
sha256sum -c MANIFEST.sha256
```

The structure files are supplied here; the results-only archive contains the
corresponding numerical outputs.
