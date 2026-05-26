#!/usr/bin/env python3
"""
SBUP3 — legacy LR-TDDFT (Casida) dipole extraction for GaAs (gamma-only).

Note: This stage is retained for compatibility, but publication-grade optical
validation should use `gaas_dielectric_response.py` with converged k-point
sampling.

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
import json
import numpy as np

from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT

# -------------------------------------------------
# Paths
# -------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
GPW_FD_DEFAULT = os.path.join(HERE, "gaas_fd.gpw")

# Outputs
OUT_OMEGA0 = os.path.join(HERE, "omega0_rad_s.npy")
OUT_E0EV   = os.path.join(HERE, "E0_eV.npy")
OUT_F0     = os.path.join(HERE, "f0.npy")
OUT_DEFF   = os.path.join(HERE, "d_eff_Cm.npy")
OUT_TRANS_E = os.path.join(HERE, "transitions_E_eV.npy")
OUT_TRANS_F = os.path.join(HERE, "transitions_f.npy")
OUT_TRANS_JSON = os.path.join(HERE, "transitions_meta.json")

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


def collect_excitations(
    lr: LrTDDFT,
    osc_min: float,
    scissor_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for exc in lr:
        E_eV = exc.get_energy() * 27.2114 + scissor_ev
        fvec = np.array(exc.get_oscillator_strength())
        fmag = np.linalg.norm(fvec)
        if fmag >= osc_min:
            rows.append((E_eV, fmag))

    rows.sort(key=lambda x: x[0])
    if not rows:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    e = np.asarray([r[0] for r in rows], dtype=float)
    f = np.asarray([r[1] for r in rows], dtype=float)
    return e, f


def extract_lowest_bright_excitation(gpw: str, osc_min: float, scissor_ev: float):
    print(f"[LOAD] {gpw}")
    calc = GPAW(gpw, txt=os.path.join(HERE, "gaas_lrtddft.log"))

    nk = len(calc.wfs.kd.bzk_kc)
    if nk != 1:
        raise RuntimeError(
            "LR-TDDFT Casida stage requires a gamma-only ground state. "
            "Current input has nk != 1."
        )

    print("[RUN ] LrTDDFT diagonalization")
    lr = LrTDDFT(calc, txt=os.path.join(HERE, "gaas_lrtddft.log"))

    # LEGACY-CORRECT CALL
    lr.diagonalize()

    trans_e, trans_f = collect_excitations(lr, osc_min, scissor_ev)
    if len(trans_e) == 0:
        raise RuntimeError("No bright excitation found")

    E0_eV = float(trans_e[0])
    f0 = float(trans_f[0])
    omega0 = (E0_eV * e0) / hbar
    d_eff = dipole_from_f(omega0, f0)

    return omega0, d_eff, E0_eV, f0, lr, trans_e, trans_f


def export_excitation_csv(lr: LrTDDFT, path: str, osc_min: float, scissor_ev: float) -> None:
    trans_e, trans_f = collect_excitations(lr, osc_min, scissor_ev)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["energy_eV", "osc"])
        writer.writerows(zip(trans_e.tolist(), trans_f.tolist()))


def main():
    parser = argparse.ArgumentParser(description="SBUP3 LR-TDDFT extraction: GaAs")
    parser.add_argument(
        "--gpw",
        default=GPW_FD_DEFAULT,
        help="Ground-state .gpw file (must be gamma-only for this legacy stage).",
    )
    parser.add_argument("--export-csv", default=None, help="Export excitations to CSV")
    parser.add_argument("--osc-min", type=float, default=OSC_MIN, help="Oscillator strength cutoff")
    parser.add_argument(
        "--scissor-ev",
        type=float,
        default=0.0,
        help="Rigid shift [eV] applied to all excitation energies (for gap correction).",
    )
    args = parser.parse_args()

    print("SBUP³ LR-TDDFT extraction: GaAs")
    print("--------------------------------")

    omega0, d_eff, E0_eV, f0, lr, trans_e, trans_f = extract_lowest_bright_excitation(
        args.gpw,
        args.osc_min,
        args.scissor_ev,
    )

    np.save(OUT_OMEGA0, omega0)
    np.save(OUT_E0EV, E0_eV)
    np.save(OUT_F0, f0)
    np.save(OUT_DEFF, d_eff)
    np.save(OUT_TRANS_E, trans_e)
    np.save(OUT_TRANS_F, trans_f)

    meta = {
        "osc_min": float(args.osc_min),
        "scissor_ev": float(args.scissor_ev),
        "n_transitions": int(len(trans_e)),
        "energy_min_eV": float(np.min(trans_e)) if len(trans_e) else None,
        "energy_max_eV": float(np.max(trans_e)) if len(trans_e) else None,
        "source_gpw": args.gpw,
    }
    with open(OUT_TRANS_JSON, "w") as fp:
        json.dump(meta, fp, indent=2)

    print("\nSaved:")
    print("  ω0 (rad/s):", omega0)
    print("  E0 (eV)   :", E0_eV)
    print("  f0        :", f0)
    print("  |d| (C·m) :", d_eff)
    print("  |d| (Debye):", d_eff / 3.33564e-30)
    print("  transitions:", len(trans_e))

    if args.export_csv:
        export_excitation_csv(lr, args.export_csv, args.osc_min, args.scissor_ev)
        print(f"  export CSV: {args.export_csv}")


if __name__ == "__main__":
    main()
