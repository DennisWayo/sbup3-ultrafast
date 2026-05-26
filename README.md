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

Local venv (minimal):

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib
```

### Quickstart

From the repo root:

```bash
# End-to-end portable run for colleagues:
# setup/check deps -> SBE -> UPPE -> plots -> validation
./run_all.sh
```

The default `run_all.sh` path does not require GPAW/DFT. It uses existing
transition arrays when present, otherwise it enables the CSV/single-transition
fallback so a fresh clone can still run end to end. For coupling or DFT runs:

```bash
./run_all.sh --mode coupling -- --iters 3
./run_all.sh --run-dft --dft-env gpaw --mode sequential
```

```bash
# For quick local smoke tests without regenerated DFT transitions:
# export SBUP3_REQUIRE_GROUND_TRUTH_TRANSITIONS=0

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

`--mode sequential` now defaults to internal-drive baselines (`SBUP3_IGNORE_FEEDBACK=1`).
To explicitly reuse `sbe/E_t_feedback.npy` in sequential mode:

```bash
python analysis/run_pipeline.py --mode sequential --allow-feedback-in-sequential
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
# CSV with energy_eV and one or more columns: chi_real, chi_imag, alpha, n
python analysis/validate_sbup3.py --ref-csv analysis/data/gaas_dielectric_reference.csv
```

`alpha` may be provided as either `alpha` or `alpha_m_inv`.

Published GaAs experimental references are included:

- `analysis/data/gaas_exp_aspnes_1986.csv`
- `analysis/data/gaas_exp_jellison_1992.csv`

Example:

```bash
python analysis/validate_sbup3.py --ref-csv analysis/data/gaas_exp_aspnes_1986.csv
```

`validate_sbup3.py` now prefers dedicated probe-response outputs (`sbe/probe_*`)
for baseline metrics when present, and falls back to drive-response outputs otherwise.

Probe-mode controls (SBE):

```bash
export SBUP3_PROBE_ENABLE=1
export SBUP3_PROBE_E_FIELD_AMPL=5e4
export SBUP3_PROBE_TAU_FS=0.35
export SBUP3_PROBE_CHI_MIN_EV=1.5
export SBUP3_PROBE_CHI_MAX_EV=3.0
```

Absolute-scale calibration (recommended before publication claims):

```bash
python analysis/calibrate_polarization_scale.py \
  --ref-csv analysis/data/gaas_dielectric_reference.csv \
  --fit-channels alpha,n \
  --write-env analysis/reports/polarization_recommendation.env
```

Calibration defaults to `--fit-mode physical` (constrained `density_m3` +
`osc_norm_factor` with uncertainty bands). Use `--fit-mode free` only for
legacy scalar-ratio diagnostics.

For experimental overlays in the 1.5–3.0 eV window, keep spectral support open:

```bash
python analysis/run_pipeline.py --mode sequential --validate \
  --ref-csv analysis/data/gaas_exp_aspnes_1986.csv \
  --chi-eps 1e-12 --response-power-cutoff 1.0 --chi-max-ev 3.0
```

Or run it directly from the pipeline:

```bash
python analysis/run_pipeline.py --mode sequential --validate \
  --ref-csv analysis/data/gaas_dielectric_reference.csv \
  --calibrate-scale --calibration-fit-channels alpha,n \
  --calibration-write-env analysis/reports/polarization_recommendation.env
```

Proxy/oscillator-only references are not accepted in this validation path.

Linear‑regime check:

```bash
python analysis/validate_sbup3.py --linear-check
```

Response-support extraction controls (to avoid divide-by-small-field artifacts):

```bash
python analysis/run_pipeline.py --mode sequential --chi-eps 1e-3 --response-power-cutoff 0.999 --chi-max-ev 3.0
```

Tuned analysis profile:

```bash
export SBUP3_POLARIZATION_SCALE=7.375504e8
export SBUP3_IGNORE_FEEDBACK=1
python analysis/run_pipeline.py --mode sequential --plots --validate \
  --ref-csv analysis/data/gaas_dielectric_reference.csv \
  --t-fs 50 --dt-as 2.5 \
  --chi-eps 1e-5 --response-power-cutoff 1.0 --chi-max-ev 3.0 \
  --overlay-window 0,1.2 --overlay-scale fit --overlay-abs
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

DFT inputs are regenerated with periodic k-point sampling and dielectric response.
The default pipeline path now uses a PW ground state (response-compatible in GPAW)
followed by dense-k dielectric extraction.
See `dft/gaas_bulk_gpaw.py` and `dft/gaas_dielectric_response.py`.

To run the DFT stage through the pipeline:

```bash
python analysis/run_pipeline.py --run-dft --dft-env <gpaw-env> --mode sequential \
  --dft-gs-mode pw --dft-pw-ecut-ev 200 \
  --dft-kpts 8,8,8 --dft-response-kpts 10,10,10 \
  --dft-force-ground-state
```

Output optical constants CSV:

```bash
analysis/data/gaas_dielectric_reference.csv
```

This stage also writes:
- `dft/gaas_ground_state_meta.json`
- `dft/gaas_dielectric_meta.json`
- `dft/transitions_E_eV.npy`
- `dft/transitions_f.npy`
- `dft/transitions_meta.json`

`dft/gaas_dielectric_response.py` now validates `gaas_ground_state_meta.json`
and refuses publication runs if the ground-state provenance is missing or
gamma-only.

Transition extraction defaults are tuned for richer reduced models:
- `--dft-transition-min-rel-height 0.01`
- `--dft-transition-min-sep-ev 0.02`
- `--dft-transition-min-count 16`
- `--dft-transition-max-count 24`

Legacy LR-TDDFT transition extraction is still available for compatibility:

```bash
python analysis/run_pipeline.py --run-dft --dft-gs-mode fd --run-lrtddft-stage --mode sequential
```

### Physical Scaling

By default, SBE polarization uses unit density. For absolute scaling, set:

```bash
export SBUP3_DENSITY_M3=<carrier_density>
export SBUP3_OSC_NORM_FACTOR=<oscillator_normalization>
export SBUP3_POLARIZATION_SCALE=<scale_factor>
```

To use a physically derived GaAs pair density (from lattice constant) when
`SBUP3_DENSITY_M3` is not set:

```bash
export SBUP3_USE_PHYSICAL_DENSITY=1
```

These affect the macroscopic polarization that feeds UPPE and should be recorded
for publication.
The calibration utility reports constrained recommendations for density,
oscillator normalization, and scale (plus target-density-adjusted scale via
`--target-density-m3`).

SBE writes physically interpretable optics outputs:
- `sbe/alpha_m_inv.npy` : absorption coefficient $\alpha(\omega)$ in 1/m
- `sbe/n_real.npy` : refractive index $n(\omega)$
- `sbe/kappa.npy` : extinction coefficient $\kappa(\omega)$
- `sbe/chi_support_mask.npy` : frequencies where $\chi(\omega)$ extraction is trusted

Use `analysis/calibrate_polarization_scale.py` against real optical references
to set publication values for `SBUP3_DENSITY_M3`, `SBUP3_OSC_NORM_FACTOR`, and
`SBUP3_POLARIZATION_SCALE`.

### Multi-Transition SBE

SBE now supports a reduced multi-transition model (enabled by default):

```bash
export SBUP3_MULTI_ENABLE=1
export SBUP3_MULTI_OSC_MIN=1e-6
export SBUP3_MULTI_MAX_EV=3.0
export SBUP3_MULTI_MAX_TRANSITIONS=32
export SBUP3_REQUIRE_GROUND_TRUTH_TRANSITIONS=1
# optional rigid shift so the first bright transition matches a target gap
export SBUP3_MULTI_TARGET_GAP_EV=1.42
```

With strict mode (`SBUP3_REQUIRE_GROUND_TRUTH_TRANSITIONS=1`), SBE refuses
legacy scalar fallbacks and requires `dft/transitions_E_eV.npy` and
`dft/transitions_f.npy`.

Used transition metadata is written to:
- `sbe/transition_model.json`
- `sbe/transition_*_used.npy`

### UPPE Dispersion

UPPE now supports configurable dispersion:

```bash
# default: wavelength-table GaAs model
export SBUP3_DISPERSION_MODEL=table

# legacy behavior
export SBUP3_DISPERSION_MODEL=constant
export SBUP3_GAAS_N0=3.6
```

Optional custom table:

```bash
export SBUP3_DISPERSION_TABLE_CSV=/abs/path/to/gaas_n_table.csv
```
