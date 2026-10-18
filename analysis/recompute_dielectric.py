#!/usr/bin/env python3
"""Recompute optical constants from a frozen GPAW response wavefunction file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gpaw.response.df import DielectricFunction

from dft.gaas_dielectric_response import _optical_constants_from_eps, _write_csv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-gpw", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--ecut", type=float, required=True)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--domega0", type=float, default=0.02)
    parser.add_argument("--omega2", type=float, default=5.0)
    parser.add_argument("--omegamax", type=float, default=8.0)
    args = parser.parse_args()

    response_gpw = args.response_gpw.resolve()
    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = out_csv.with_suffix(".raw.csv")
    df = DielectricFunction(
        calc=str(response_gpw), eta=args.eta, ecut=args.ecut, intraband=False,
        frequencies={"type": "nonlinear", "domega0": args.domega0,
                     "omega2": args.omega2, "omegamax": args.omegamax},
        integrationmode="point integration",
    )
    df.get_dielectric_function(filename=str(raw_csv))
    raw = np.loadtxt(raw_csv, delimiter=",")
    raw_csv.unlink()
    energy = np.asarray(raw[:, 0], dtype=float)
    eps_nlf = np.asarray(raw[:, 1] + 1j * raw[:, 2], dtype=complex)
    eps_lf = np.asarray(raw[:, 3] + 1j * raw[:, 4], dtype=complex)
    n, k, alpha = _optical_constants_from_eps(energy, eps_lf)
    _write_csv(out_csv, energy, eps_nlf, eps_lf, n, k, alpha)
    meta = {
        "response_gpw": str(response_gpw),
        "response_gpw_sha256": sha256(response_gpw),
        "out_csv": str(out_csv), "out_csv_sha256": sha256(out_csv),
        "ecut_eV": args.ecut, "eta_eV": args.eta,
        "domega0_eV": args.domega0, "omega2_eV": args.omega2,
        "omegamax_eV": args.omegamax, "n_rows": len(energy),
    }
    out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[OK] {out_csv}")


if __name__ == "__main__":
    main()
