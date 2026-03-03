#!/usr/bin/env python3
"""
SBUP³ — LR-TDDFT (Casida) dipole extraction for GaAs (Γ-only)
============================================================

Outputs:
  - omega0_rad_s.npy : lowest bright excitation angular frequency [rad/s]
  - E0_eV.npy        : excitation energy [eV]
  - f0.npy           : oscillator strength
  - d_eff_Cm.npy     : effective dipole magnitude [C·m]

Author: Dennis Wayo
"""

from __future__ import annotations
import os
import argparse
import csv
import numpy as np

from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT

# -------------------------------------------------
# Paths
# -------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
GPW_FD = os.path.join(HERE, "gaas_fd_gamma.gpw")

# Outputs
OUT_OMEGA0 = os.path.join(HERE, "omega0_rad_s.npy")
OUT_E0EV   = os.path.join(HERE, "E0_eV.npy")
OUT_F0     = os.path.join(HERE, "f0.npy")
OUT_DEFF   = os.path.join(HERE, "d_eff_Cm.npy")

# -------------------------------------------------
# Constants (SI)
# -------------------------------------------------
hbar = 1.054571817e-34
e0   = 1.602176634e-19
me   = 9.1093837015e-31

OSC_MIN = 1e-6   # brightness threshold


def dipole_from_f(omega, f):
    """Convert oscillator strength to dipole magnitude."""
    return np.sqrt((3 * hbar * e0**2 / (2 * me * omega)) * f)


def extract_lowest_bright_excitation(gpw):
    print(f"[LOAD] {gpw}")
    calc = GPAW(gpw, txt=os.path.join(HERE, "lrtddft.log"))

    nk = len(calc.wfs.kd.bzk_kc)
    if nk != 1:
        raise RuntimeError("LR-TDDFT requires Γ-only ground state")

    print("[RUN ] LrTDDFT diagonalization")
    lr = LrTDDFT(calc, txt=os.path.join(HERE, "lrtddft.log"))

    # LEGACY-CORRECT CALL
    lr.diagonalize()

    best = None

    for exc in lr:
        E_eV = exc.get_energy() * 27.2114
        fvec = np.array(exc.get_oscillator_strength())
        fmag = np.linalg.norm(fvec)

        if fmag > OSC_MIN:
            if best is None or E_eV < best[0]:
                best = (E_eV, fmag)

    if best is None:
        raise RuntimeError("No bright excitation found")

    E0_eV, f0 = best
    omega0 = (E0_eV * e0) / hbar
    d_eff = dipole_from_f(omega0, f0)

    return omega0, d_eff, E0_eV, f0, lr


def export_excitation_csv(lr: LrTDDFT, path: str, osc_min: float) -> None:
    rows = []
    for exc in lr:
        E_eV = exc.get_energy() * 27.2114
        fvec = np.array(exc.get_oscillator_strength())
        fmag = np.linalg.norm(fvec)
        if fmag >= osc_min:
            rows.append((E_eV, fmag))

    rows.sort(key=lambda x: x[0])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["energy_eV", "osc"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="SBUP3 LR-TDDFT extraction: GaAs")
    parser.add_argument("--export-csv", default=None, help="Export excitations to CSV")
    parser.add_argument("--osc-min", type=float, default=OSC_MIN, help="Oscillator strength cutoff")
    args = parser.parse_args()

    print("SBUP³ LR-TDDFT extraction: GaAs")
    print("--------------------------------")

    omega0, d_eff, E0_eV, f0, lr = extract_lowest_bright_excitation(GPW_FD)

    np.save(OUT_OMEGA0, omega0)
    np.save(OUT_E0EV, E0_eV)
    np.save(OUT_F0, f0)
    np.save(OUT_DEFF, d_eff)

    print("\nSaved:")
    print("  ω0 (rad/s):", omega0)
    print("  E0 (eV)   :", E0_eV)
    print("  f0        :", f0)
    print("  |d| (C·m) :", d_eff)
    print("  |d| (Debye):", d_eff / 3.33564e-30)

    if args.export_csv:
        export_excitation_csv(lr, args.export_csv, args.osc_min)
        print(f"  export CSV: {args.export_csv}")


if __name__ == "__main__":
    main()
