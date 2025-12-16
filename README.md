![Status](https://img.shields.io/badge/status-active--development-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Model](https://img.shields.io/badge/model-SBE%20%2B%20UPPE-purple)
![Physics](https://img.shields.io/badge/physics-ultrafast--optics-red)
![Scale](https://img.shields.io/badge/scale-multiscale-brightgreen)
![DFT](https://img.shields.io/badge/DFT-GPAW-lightgrey)
![FDTD](https://img.shields.io/badge/FDTD-MEEP-lightgrey)


# SBUP³-Ultrafast

**SBUP³** — *Semiconductor Bloch–Unidirectional Pulse Propagation Platform*

A coupled Semiconductor Bloch–UPPE platform for multiscale ultrafast laser–matter simulations.

---

SBUP³ is a modular computational framework for modeling femtosecond laser excitation
in semiconductors by self-consistently coupling:

1. **Electronic structure inputs** (band energies and transition dipole moments),
2. **Semiconductor Bloch Equations (SBE)** for nonequilibrium carrier and polarization dynamics,
3. **Unidirectional Pulse Propagation Equation (UPPE)** for ultrafast electromagnetic pulse evolution.

The framework bridges microscopic carrier dynamics and macroscopic pulse propagation,
enabling physically interpretable simulations beyond phenomenological nonlinear optics models.

---

### Scientific Scope

SBUP³ is designed as a **methods and modeling platform**, not a discovery engine.
Its primary goal is to teach, validate, and reuse a multiscale workflow for
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

---

### Repository Structure

- `sbup3/dft/` — electronic structure parametrizations
- `sbup3/sbe/` — Semiconductor Bloch Equation solvers (RK4)
- `sbup3/uppe/` — 1D UPPE propagation models
- `sbup3/coupling/` — self-consistent SBUP³ coupling loop
- `sbup3/analysis/` — energy deposition and diagnostics
- `notebooks/` — reproducible Jupyter demonstrations

---

### Installation

```bash
conda env create -f environment.yml
conda activate sbup3
