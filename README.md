## SBUP3-Ultrafast

SBUP3-Ultrafast is a research workflow for modeling femtosecond light interaction with bulk GaAs by coupling microscopic carrier dynamics to macroscopic pulse propagation. The repository combines transition inputs, a multi-transition Semiconductor Bloch Equation solver, a UPPE propagation stage, and validation plus plotting utilities into one reproducible pipeline. It is designed for practical collaboration: colleagues can run, inspect, and compare results with minimal setup while still keeping access to advanced calibration, dispersion, and DFT regeneration paths. Outputs include time-domain fields, optical-response spectra, convergence metrics, publication-ready figures, and machine-readable reports for downstream analysis, benchmarking, and traceability across experiments in shared collaborative development cycles.

Use one command from the repo root:

```bash
./run_all.sh
```

This runs end to end:
- dependency setup/check
- SBE
- UPPE
- plots
- validation report

## Requirements

- Python 3.10+
- `bash`
- For optional DFT regeneration only: GPAW in a conda environment

## Installation

Conda:

```bash
conda env create -f environment.yml
conda activate sbup3
```

Local venv:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib
```

## Quickstart

Default portable run:

```bash
./run_all.sh
```

Common variants:

```bash
# Coupled loop mode
./run_all.sh --mode coupling -- --iters 3

# Full mode (sequential + coupling)
./run_all.sh --mode full

# Use an existing Python env without creating/updating .venv
./run_all.sh --no-setup --python /path/to/python
```

Optional DFT regeneration (requires GPAW env):

```bash
./run_all.sh --run-dft --dft-env gpaw --mode sequential
```

## Direct Pipeline

If you prefer direct Python entry points:

```bash
# Sequential SBE -> UPPE + plots + validation
python analysis/run_pipeline.py --mode sequential --plots --validate

# Coupling + plots + validation
python analysis/run_pipeline.py --mode coupling --plots --validate
```

## Validation and Tests

Validation only:

```bash
python analysis/validate_sbup3.py --ref-csv analysis/data/gaas_dielectric_reference.csv
```

Unit tests:

```bash
python -m unittest discover -s tests
```

## Outputs

- `sbe/*.npy` and `sbe/transition_model.json`
- `uppe/outputs/*.npy`
- `analysis/figures/*.png`
- `analysis/reports/*.json`

## Repository Layout

- `analysis/` pipeline, plotting, validation, calibration utilities
- `sbe/` SBE solver
- `uppe/` UPPE propagation and dispersion
- `coupling/` iterative SBE/UPPE loop
- `dft/` GPAW-based ground-state and response scripts
- `tests/` unit tests
- `docs/` manuscript and extended notes

## Advanced Usage

Detailed knobs (calibration, probe settings, multi-transition controls, dispersion switches, manual DFT/LR-TDDFT commands) are documented in [docs/advanced.md](/Users/denniswayo/sbup3-ultrafast/docs/advanced.md).
