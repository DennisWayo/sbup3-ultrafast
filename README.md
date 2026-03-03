![Status](https://img.shields.io/badge/status-active--development-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Model](https://img.shields.io/badge/model-SBE%20%2B%20UPPE-purple)
![Physics](https://img.shields.io/badge/physics-ultrafast--optics-red)
![Scale](https://img.shields.io/badge/scale-multiscale-brightgreen)
![DFT](https://img.shields.io/badge/DFT-GPAW-lightgrey)
![FDTD](https://img.shields.io/badge/FDTD-MEEP-lightgrey)


## SBUP³-Ultrafast

**SBUP³** — Semiconductor Bloch–Unidirectional Pulse Propagation Platform

A coupled Semiconductor Bloch–UPPE platform for multiscale ultrafast laser–matter simulations.

SBUP³ is a modular computational framework for modeling femtosecond laser excitation
in semiconductors by self-consistently coupling:

1. Electronic structure inputs (band energies and transition dipole moments),
2. Semiconductor Bloch Equations (SBE) for nonequilibrium carrier and polarization dynamics,
3. Unidirectional Pulse Propagation Equation (UPPE) for ultrafast electromagnetic pulse evolution.

The framework bridges microscopic carrier dynamics and macroscopic pulse propagation,
enabling physically interpretable simulations beyond phenomenological nonlinear optics models.


### Scientific Scope

SBUP³ is a methods and modeling platform rather than a discovery engine.
Its primary goal is to develop, validate, and reuse a multiscale workflow for
ultrafast laser–matter interaction.

Current focus:
- Bulk GaAs (two-band reduction)
- Femtosecond pulse excitation (20–100 fs)
- Carrier excitation, polarization dynamics, and energy deposition

Planned extensions:
- Defects and emitters
- Nanostructures and heterostructures
- Coupling to FDTD (e.g. MEEP)
- Quantum photonic materials

### Repository Structure

- `dft/` — electronic structure parametrizations
- `sbe/` — Semiconductor Bloch Equation solver (RK4)
- `uppe/` — 2D UPPE propagation model
- `coupling/` — self-consistent SBUP³ coupling loop
- `analysis/` — visualization scripts and figures

### Installation

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate sbup3
```

### Quickstart

From the repo root:

```bash
# 1) SBE (writes outputs into sbe/)
cd sbe
python sbe_rk4.py
cd ..

# 2) UPPE (reads sbe outputs, writes uppe/outputs/)
python -m uppe.uppe_2d

# 3) Coupled loop (experimental)
python -m coupling.sbup3_loop
```

### Pipeline

```bash
# Coupling run + figures + validation report
python analysis/run_pipeline.py --mode coupling --plots --validate

# Sequential run (SBE -> UPPE) without coupling
python analysis/run_pipeline.py --mode sequential
```

### Validation

```bash
# Metrics from current outputs
python analysis/validate_sbup3.py

# Simple convergence sweeps
python analysis/validate_sbup3.py --sweep-dt-as 5,2.5 --sweep-dz 1e-6,5e-7
```

### Feedback Field

If `sbe/E_t_feedback.npy` exists, `sbe/sbe_rk4.py` mixes it with the internal Gaussian pulse
(see `MIXING` in `sbe/sbe_rk4.py`). Delete or rename the file to force the internal drive.

### Visualization

```bash
python analysis/visualize_sbup3.py
```

To view plots interactively:

```bash
python analysis/visualize_sbup3.py --show
```

### Optional DFT

DFT inputs are precomputed in `dft/*.npy`. Regenerating them requires GPAW and its dependencies;
see `dft/gaas_bulk_gpaw.py` and `dft/gaas_lrtddft.py`.

To run the DFT stage through the pipeline:

```bash
python analysis/run_pipeline.py --run-dft --dft-env <gpaw-env> --mode sequential
```
