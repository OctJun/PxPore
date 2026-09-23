# PxPore

PxPore is a Python toolkit for post-processing molecular structures and
molecular-dynamics snapshots. It focuses on grid-based pore analysis, free-volume
calculation, accessible/trapped volume classification, surface-area estimation,
and pore-size descriptors.

## Features

- Reads orthogonal `.gro`, `.xyz`, `.pdb`, and `.cif` structure files.
- Computes cell volume, void volume, accessible volume, trapped volume, and
  corresponding fractions.
- Estimates accessible and total surface areas.
- Computes pore descriptors such as PLD and LCD.
- Supports optional octree refinement near molecular boundaries.
- Uses Numba to accelerate grid, connectivity, and pore-analysis kernels.
- Can write statistics, Gaussian cube files, and pore-visualization outputs.

## Requirements

Python 3.10 or newer. The source version is `1.1.0`; `pyproject.toml` defines
runtime dependencies and version constraints (NumPy, SciPy, Numba, llvmlite,
pandas, psutil, and scikit-learn). Install the package and its dependencies
from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Project layout

```text
src/PxPore/        Python package source code
src/PxPore/data/   Package data, including atom-parameter tables
tests/data/        Input structures used for examples and checks
```

## Run from source

PxPore can be run directly from the source tree. From the repository root:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
mkdir -p docs/example_run
cp tests/data/single_H.gro docs/example_run/input.gro
python -m PxPore docs/example_run/input.gro \
  --grid 0.02 \
  --probe 0.0 \
  --threads 8 \
  --atoms src/PxPore/data/UFF.atoms \
  --pore \
  --stats
```

If you are working from an extracted source archive, set `PYTHONPATH` to the
archive's `src` directory, for example:

```bash
export PYTHONPATH="/path/to/source_tree/src:$PYTHONPATH"
```

## Command-line usage

PxPore uses Numba `njit` kernels for the compute-heavy parts of the analysis.
The first run in a fresh Python environment may spend extra time compiling
these kernels. To remove this one-time compilation cost from a benchmark or
production run, either run the warmup command first:

```bash
python -m PxPore.warmup
```

or run one preliminary analysis on any representative structure.

```bash
python -m PxPore input.gro \
  --grid 0.02 \
  --probe 0.0 \
  --threads 8 \
  --atoms UFF.atoms \
  --pore \
  --cube \
  --stats
```

This command analyzes `input.gro` with a 0.02 nm grid spacing and zero probe
radius, enables pore analysis, and writes statistics. The optional `--cube`
flag writes volumetric cube files.

## Python API

```python
from PxPore import analyse

result = analyse(
    input="structure.gro",
    grid=0.02,
    probe=0.0,
    atoms="UFF.atoms",
    threads=8,
    pore=True,
    stats=True,
)
```

## Parameters

- `input`: input structure file. Supported formats are `.gro`, `.xyz`, `.pdb`,
  and `.cif` for orthogonal simulation cells.
- `--grid`, `-g`: target grid spacing in nm; default is `0.01`.
- `--probe`, `-p`: probe radius in nm; default is `0.0`.
- `--connectivity`: `legacy` (default, nonperiodic boundaries) or `periodic`
  (periodic winding criterion).
- `--transport-direction`: accessibility direction, `any` (default), `x`, `y`, or `z`.
- `--atoms`: atom parameter file used to override default radii and masses.
  Expected format: `symbol Z mass(g/mol) LJsigma(nm) epsilon(K)`.
- `--threads`: number of Numba threads; `0` uses half of available threads.
- `--out_prefix`: output file prefix.
- `--no-surface`: disable surface-area analysis.
- `--surface-samples`: Fibonacci surface samples per atom; default is `1000`.
- `--pore`: enable pore analysis.
- `--porevis`: write pore-visualization output.
- `--psd-method`: PSD method, `mc` (default), `centers`, or `both`.
- `--psd-mc-samples`: Monte Carlo PSD sample count; default is `50000`.
- `--psd-mc-seed`: Monte Carlo PSD random seed; default is `11451466`.
- `--psd-mc-bin-size`: Monte Carlo PSD bin size in nm; default uses `--grid`.
- `--psd-mc-search`: largest-containing-ball search, `pyramid` (default) or
  `offsets` (the original search implementation).
- `--psd-mc-grid`: MC sampling grid, `uniform` (default) or `octree`.
  `octree` requires octree refinement and `pyramid` search.
- `--psd-center-bin-size`: center PSD bin size in nm; defaults to `--grid`.
- `--psd-local-max-mode`: center local-maximum criterion, `strict` (default) or `plateau`.
- `--psd-min-center-radius`: minimum center-ball radius; default `0.005` nm.
- `--no-psd-overlap-prune`: disable center-ball overlap pruning (enabled by default).
- `--psd-overlap-threshold`: center-ball overlap pruning factor; default `1.0`.
- `--psd-hist-weighting`: center histogram weighting, `volume` (default) or `number`.
- `--no-octree`: disable octree refinement.
- `--oct-level`: maximum octree refinement level; default is `2`.
- `--oct-grid`: minimum octree leaf size in nm; default is `0.001`.
- `--cube`: write Gaussian cube files.
- `--cube-space`: cube-file spatial resolution.
- `--smooth`: smooth output fields.
- `--stats`: write statistics JSON.
- `--debug`: save intermediate arrays.
- `--debug-print`: print extra debug information.

## PSD and outputs

Pore calculations require `--pore`; setting PSD options alone does not enable
them. Default MC sampling uses accessible uniform-grid voxels and searches for
the largest ball containing each sample. Octree refinement remains enabled for
volume/connectivity analysis: `--psd-mc-grid uniform` does not mean `--no-octree`.
Octree connectivity supports periodic boundaries and directional accessibility.
Center coordinates and center PSD require `--psd-method centers` or `both`.

Outputs are written beside the input file. The default prefix is
`<input_filename>_g_<grid>_p_<probe>`; `--out_prefix` changes the filename prefix.
The source example above stages its input under `docs/example_run/`.

| File suffix | Required options | Contents and units |
|---|---|---|
| `_stats.json` | `--stats` | Statistics, settings, and runtime environment |
| `_voxel_mc_psd.txt` | `--pore --stats`, MC/both | Seven columns: index, diameter nm, count, probability, density nm⁻¹, PB central-difference density nm⁻¹, cumulative probability |
| `_Network-accessible_psd.txt` | Same | PB-format diameter Å and differential density Å⁻¹ |
| `_Network-accessible_psd_cumulative.txt` | Same | Probe diameter Å and remaining accessible volume fraction, decreasing with diameter |
| `_psd.txt` | `--pore --stats`, centers/both | Center PSD: index, diameter nm, count, weighted fraction, cumulative fraction |
| `_center.txt` | Same | Extended XYZ; coordinates in Å, ball diameter in nm |
| `*.cube` | `--cube` | Void, occupied, accessible, trapped, and distance fields |
| `_porevis.cube` | `--cube --pore --porevis`, centers/both | Center-ball visualization |
| `*.npy` / `*.npz` | `--debug` | Intermediate arrays |

PB-format files convert the PxPore MC results; they do not run PoreBlazer.
Default `mc` produces no center balls; use `both` for center-ball visualization.
The Python API returns statistics, configuration, grid, timing, and output-path
information; numeric metrics are in `result["stats"]["stats"]`. With no accessible
pores, PLD/LCD are `-1` and the main PSD tables are empty.

See [docs/reproducibility.md](docs/reproducibility.md) for the release materials
and their relationship to the current source.

## Sensitivity study

The reproducible parameter-sensitivity and plotting workflow is documented in
[scripts/sensitivity/README.md](scripts/sensitivity/README.md). A sanitized
snapshot of the raw numerical results is described in
[results/README.md](results/README.md).

## Plotting

The sanitized publication plotting scripts, HMOF comparison notebook, and data
package instructions are documented in
[scripts/plotting/README.md](scripts/plotting/README.md).

## Citation

If you use PxPore, please cite the associated manuscript or repository record.

## License

PxPore is released under the MIT License. See [LICENSE](LICENSE).
