# PxPore

A Python toolkit for post-processing molecular dynamics simulations, focusing on pore structure analysis, free volume calculation, and pore size distribution.

## Installation

Assuming published on PyPI:

```bash
pip install pxpore
```

## Requirements

- Python >= 3.8
- NumPy
- SciPy
- Numba (for parallel acceleration)
- Other dependencies: see `requirements.txt`

## Usage Examples

### Basic Command Line Usage

```bash
python -m PxPore input.gro --grid 0.01 --probe 0.0 --pore --cube --stats
```

This analyzes the `input.gro` file with grid spacing 0.01 nm, no probe, enables pore analysis, outputs .cube files and statistics.

### Python API Usage

```python
from PxPore import analyse

config = {
    'input': 'structure.gro',
    'grid': 0.01,
    'probe': 0.1,
    'pore': True,
    'cube': True
}
result = analyse(**config)
```

## Parameter Description

- `--input` / `-i`: Input structure file (supports .gro, .xyz, .pdb, .cif)
- `--grid` / `-g`: Grid spacing (nm), default 0.01
- `--probe` / `-p`: Probe radius (nm), default 0.0
- `--atoms`: Atom info file to override default radii
- `--threads`: Numba threads, default half of available
- `--out_prefix`: Output file prefix
- `--no-surface`: Disable surface area analysis
- `--pore`: Enable pore analysis
- `--porevis`: Enable pore visualization output
- `--no-octree`: Disable octree refinement
- `--oct-level`: Max octree levels, default 4
- `--oct-grid`: Min octree leaf size (nm), default 0.001
- `--cube`: Output .cube files
- `--cube-space`: .cube file spatial resolution
- `--smooth`: Smooth output
- `--stats`: Save statistics
- `--debug`: Save intermediate arrays
- `--debug-print`: Print extra debug info

## Citation

If you use PxPore, please cite appropriately:

[Provide relevant paper or GitHub link]

## License

[Specify license]