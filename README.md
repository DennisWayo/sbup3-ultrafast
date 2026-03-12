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

Baseline comparison (optional):

```bash
# CSV with energy_eV and one or more columns: chi_real, chi_imag, alpha, n, osc
python analysis/validate_sbup3.py --ref-csv analysis/data/gaas_tddft_reference.csv
```

If no CSV is provided, validation falls back to parsing `dft/gaas_lrtddft.log`
(or `dft/lrtddft.log`) and uses |me| as an oscillator proxy.

To export a CSV from the TDDFT log:

```bash
python analysis/export_tddft_csv.py --log dft/gaas_lrtddft.log --out analysis/data/gaas_tddft_reference.csv
```

Linear‑regime check:

```bash
python analysis/validate_sbup3.py --linear-check
```

Response-support extraction controls (to avoid divide-by-small-field artifacts):

```bash
python analysis/run_pipeline.py --mode sequential --chi-eps 1e-3 --response-power-cutoff 0.999 --chi-max-ev 3.0
```

Tuned analysis profile (current best tradeoff for TDDFT overlay support):

```bash
export SBUP3_POLARIZATION_SCALE=7.375504e8
export SBUP3_IGNORE_FEEDBACK=1
python analysis/run_pipeline.py --mode sequential --plots --validate \
  --ref-csv analysis/data/gaas_tddft_reference.csv \
  --t-fs 50 --dt-as 2.5 \
  --chi-eps 1e-5 --response-power-cutoff 1.0 --chi-max-ev 3.0 \
  --overlay-window 0,1.2 --overlay-scale fit --overlay-abs
```

This setting expanded the validated overlap from 10 to 16 support points (0.91 eV -> 1.24 eV)
while reducing shape error in the TDDFT oscillator overlay.

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
Ground‑state provenance is recorded in `dft/gaas_fd.txt`.

To run the DFT stage through the pipeline:

```bash
python analysis/run_pipeline.py --run-dft --dft-env <gpaw-env> --mode sequential
```

### Physical Scaling

By default, SBE polarization uses unit density. For absolute scaling, set:

```bash
export SBUP3_DENSITY_M3=<carrier_density>
export SBUP3_POLARIZATION_SCALE=<scale_factor>
```

These affect the macroscopic polarization that feeds UPPE and should be recorded for publication.

SBE now writes physically interpretable optics outputs:
- `sbe/alpha_m_inv.npy` : absorption coefficient $\alpha(\omega)$ in 1/m
- `sbe/n_real.npy` : refractive index $n(\omega)$
- `sbe/kappa.npy` : extinction coefficient $\kappa(\omega)$
- `sbe/chi_support_mask.npy` : frequencies where $\chi(\omega)$ extraction is trusted

Based on the current TDDFT log baseline, a provisional polarization scale is:

```bash
export SBUP3_POLARIZATION_SCALE=7.375504e8
```
