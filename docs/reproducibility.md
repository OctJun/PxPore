# Reproducibility materials

The repository contains the PxPore source, example structures, benchmark inputs,
plotting scripts and numerical result archives. Experiment-specific result
packages provide their own input mappings, commands and file manifests.

## Source and calculation settings

`src/PxPore/config.py`, `api.py` and `cli.py` define the parameters;
`core.py` and `pores.py` implement the analysis and outputs. The English and
Chinese READMEs document the options and units.

- The default PSD estimator is `mc`, with 50,000 samples, seed `11451466`,
  `pyramid` search and a `uniform` sampling grid.
- Center-based PSDs and center-sphere visualization use `centers` or `both`.
- Octree refinement is enabled by default, with maximum depth 2 and minimum
  leaf size 0.001 nm. MC sampling-grid selection is a separate option.
- `--psd-mc-grid octree` requires octree refinement and `pyramid` search.
- Default connectivity is `legacy/any`. Periodic directional connectivity is
  also supported, including the octree representation.
- MC tables use nm and nm⁻¹. PoreBlazer-compatible outputs use Å and Å⁻¹;
  their cumulative curve gives the remaining accessible-volume fraction.

For an existing experiment, use its recorded source revision and parameters:
PSD estimator, MC search and sampling grid, sample count, seed, histogram bin
width, octree configuration, atomic radii and connectivity.

## Repository data

[results/README.md](../results/README.md) describes the numerical archives and
their checksums. [tests/data/README.md](../tests/data/README.md) describes the
example structures. The supplementary five-structure benchmark inputs are
listed in [reproducibility_inputs/README.md](reproducibility_inputs/README.md).
Verify those inputs with:

```bash
cd docs/reproducibility_inputs
sha256sum -c MANIFEST.sha256
```

The hMOF tables supply inputs for plotting. Calculations from atomic coordinates
use the structure dataset and preprocessing settings identified by the selected
experiment. Each result archive represents its own calculation configuration;
select the corresponding data and plotting workflow together.

## Trajectory extraction

APC-DAP and APC-MAP trajectories are distributed with the experiment data.
Use each system's matching `prod.xtc` and `prod.tpr` in a separate working
directory.

For the 209-frame series, extract 0–9984 ps at 48 ps intervals for each system:

```bash
mkdir frames
printf 'System\n' | gmx trjconv -s prod.tpr -f prod.xtc -o frames/frame_.gro -sep -dt 48 -b 0 -e 9984
```

For the APC-DAP 11-frame comparison, extract 1000–2000 ps at 100 ps intervals:

```bash
mkdir frames_11
printf 'System\n' | gmx trjconv -s prod.tpr -f prod.xtc -o frames_11/frame_.gro -sep -b 1000 -e 2000 -dt 100
```

The 11-frame comparison displays elapsed time 0–1000 ps relative to the first
extracted frame.
