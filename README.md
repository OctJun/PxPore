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

- Python 3.10 or newer
- NumPy
- SciPy
- Numba

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy numba
```

If a `requirements.txt` file is provided with your copy of the project, you can
install from it instead:

```bash
python -m pip install -r requirements.txt
```

For local development, the project can also be installed in editable mode:

```bash
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
python -m PxPore tests/data/single_H.gro \
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
- `--atoms`: atom parameter file used to override default radii and masses.
  Expected format: `symbol Z mass(g/mol) LJsigma(nm) epsilon(K)`.
- `--threads`: number of Numba threads; `0` uses half of available threads.
- `--out_prefix`: output file prefix.
- `--no-surface`: disable surface-area analysis.
- `--pore`: enable pore analysis.
- `--porevis`: write pore-visualization output.
- `--no-octree`: disable octree refinement.
- `--oct-level`: maximum octree refinement level; default is `4`.
- `--oct-grid`: minimum octree leaf size in nm; default is `0.001`.
- `--cube`: write Gaussian cube files.
- `--cube-space`: cube-file spatial resolution.
- `--smooth`: smooth output fields.
- `--stats`: write statistics JSON.
- `--debug`: save intermediate arrays.
- `--debug-print`: print extra debug information.

## Outputs

Depending on the selected options, PxPore writes:

- statistics JSON files containing geometric and pore descriptors;
- optional cube files for volumetric fields;
- optional pore-visualization outputs;
- optional diagnostic arrays for verification.

## Citation

If you use PxPore, please cite the associated manuscript or repository record.

## License

PxPore is released under the MIT License. See [LICENSE](LICENSE).
