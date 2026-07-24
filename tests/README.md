# PxPore tests

## Fast algorithm tests

Run the dependency-free unittest suite from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
```

These tests cover connectivity, PSD, surface sampling, default parameters, and
the regression runner itself. They use small synthetic arrays and do not write
to existing data directories.

## Structure regression manifest

`regression_cases.json` is intentionally empty. Add current and future
structures after their reference values and tolerances have been reviewed.

Each case has this shape:

```json
{
  "name": "descriptive-name",
  "input": "tests/data/path/to/input.gro",
  "arguments": {
    "grid": 0.02,
    "probe": 0.0,
    "stats": true
  },
  "metrics": {
    "stats.Vacc_nm3": {
      "reference": null,
      "absolute_tolerance": 1e-6,
      "relative_tolerance": 1e-4
    }
  },
  "enabled": true,
  "slow": false,
  "notes": ""
}
```

Metric paths start from the stats payload returned by `analyse()`. Common
prefixes are `stats.`, `info.`, and `settings.`. A `null` reference is reported
as `UNSET` and is never silently accepted as a passing comparison.

Run configured structures with:

```bash
python scripts/run_regression_suite.py
python scripts/run_regression_suite.py --include-slow --strict-unset
python scripts/run_regression_suite.py --color always
```

Run both suites with every miscellaneous test name printed:

```bash
python scripts/run_tests.py
python scripts/run_tests.py --include-slow
```

The runner does not import PxPore. Every case is executed through
`python -m PxPore` in an independent subprocess, with `OMP_NUM_THREADS`,
`NUMBA_NUM_THREADS`, and `--threads` set to the CPU cores available through
the current process affinity. The reported thread count and `omp` threading
layer are checked for every executed case.

Every input is copied to a temporary directory before analysis. The runner
never writes generated results beside the original structure. Each configured
metric prints the result first, followed by the test name, output value,
expected value, and metric name. Status values use terminal colors by default;
`--color always` and `--color never` override automatic terminal detection.
The table is streamed: each case's metric results are flushed immediately
after that case finishes.
