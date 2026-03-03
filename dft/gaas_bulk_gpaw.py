"""
SBUP³ — TDDFT-based parameter extraction for Semiconductor Bloch Equations (GaAs)
=================================================================================

Method:
  1) LCAO ground-state DFT (Γ-only)
  2) Finite-difference (FD) restart for accurate band energies
  3) Linear-response TDDFT (Casida / delta-kick, Γ-only)
  4) Extract effective interband dipole from oscillator strength

Notes:
  • Linear-response TDDFT in GPAW is restricted to Γ-only calculations.
  • The extracted dipole is an effective, k-averaged quantity suitable
    for two-band Semiconductor Bloch Equations (SBE).
  • k-resolved dipoles are NOT computed in this workflow.

Outputs:
  - omega0.npy : fundamental transition frequency ω₀ [rad/s]
  - d_eff.npy  : effective interband dipole |d| [C·m]

Author: Dennis Wayo
"""

from __future__ import annotations

import os
import numpy as np
from ase.build import bulk
from gpaw import GPAW, FermiDirac

# -------------------------------------------------
# Paths
# -------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

LCAO_GPW = os.path.join(HERE, "gaas_lcao_gamma.gpw")
FD_GPW   = os.path.join(HERE, "gaas_fd_gamma.gpw")

# -------------------------------------------------
# Physical constants
# -------------------------------------------------
hbar = 1.054571817e-34  # J·s
e0   = 1.602176634e-19  # C
me   = 9.10938356e-31   # kg

# -------------------------------------------------
# User parameters (Γ-only is REQUIRED)
# -------------------------------------------------
XC = "PBE"
KPTS = (1, 1, 1)        # 🔴 REQUIRED for LR-TDDFT
NBANDS = 16
SMEARING = 0.01         # eV

# -------------------------------------------------
# Build bulk GaAs
# -------------------------------------------------
def build_gaas(a: float = 5.653):
    atoms = bulk("GaAs", "zincblende", a=a)
    atoms.pbc = True
    return atoms

# -------------------------------------------------
# LCAO ground state (Γ-only)
# -------------------------------------------------
def run_lcao(atoms):
    if os.path.exists(LCAO_GPW):
        print(f"[SKIP] LCAO → {LCAO_GPW}")
        return GPAW(LCAO_GPW)

    print("[RUN ] LCAO ground state (Γ-only)")
    calc = GPAW(
        mode="lcao",
        basis="dzp",
        xc=XC,
        kpts=KPTS,
        nbands=NBANDS,
        occupations=FermiDirac(SMEARING),
        symmetry="off",
        txt=os.path.join(HERE, "gaas_lcao.txt"),
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(LCAO_GPW, mode="all")
    return calc

# -------------------------------------------------
# FD restart (Γ-only)
# -------------------------------------------------
def run_fd():
    if os.path.exists(FD_GPW):
        print(f"[SKIP] FD → {FD_GPW}")
        return GPAW(FD_GPW)

    print("[RUN ] FD restart (Γ-only)")
    calc = GPAW(
        LCAO_GPW,
        mode="fd",
        h=0.25,
        occupations=FermiDirac(SMEARING),
        txt=os.path.join(HERE, "gaas_fd.txt"),
    )
    calc.get_potential_energy()
    calc.write(FD_GPW, mode="all")
    return calc

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":

    print("SBUP³ TDDFT-based DFT extraction: GaAs")
    print("--------------------------------------")

    atoms = build_gaas()

    run_lcao(atoms)
    run_fd()

    print("Ground-state preparation complete.")
    print("Proceed with LR-TDDFT (Casida) for dipole extraction.")