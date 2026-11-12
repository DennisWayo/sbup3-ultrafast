#!/usr/bin/env python3
"""Export optical curves from an existing GPAW response restart at a new eta."""

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
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--ecut", type=float, default=150.0)
    parser.add_argument("--domega0", type=float, default=0.02)
    parser.add_argument("--omega2", type=float, default=5.0)
    parser.add_argument("--omegamax", type=float, default=8.0)
    args = parser.parse_args()
    gpw = args.response_gpw.resolve()
    if not gpw.is_file():
        raise FileNotFoundError(gpw)
    if args.eta <= 0:
        raise ValueError("Broadening eta must be positive")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_csv.with_suffix(".raw.csv")
    df = DielectricFunction(
        calc=str(gpw), eta=args.eta, ecut=args.ecut, intraband=False,
        frequencies={"type": "nonlinear", "domega0": args.domega0,
                     "omega2": args.omega2, "omegamax": args.omegamax},
        integrationmode="point integration",
    )
    df.get_dielectric_function(filename=str(raw_path))
    raw = np.loadtxt(raw_path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] < 5:
        raise RuntimeError(f"Unexpected GPAW response format: {raw_path}")
    energy = np.asarray(raw[:, 0], dtype=float)
    eps_nlf = np.asarray(raw[:, 1] + 1j * raw[:, 2], dtype=complex)
    eps_lf = np.asarray(raw[:, 3] + 1j * raw[:, 4], dtype=complex)
    n, k, alpha = _optical_constants_from_eps(energy, eps_lf)
    _write_csv(args.out_csv, energy, eps_nlf, eps_lf, n, k, alpha)
    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.meta_json.write_text(json.dumps({
        "response_gpw": str(gpw), "response_gpw_sha256": sha256(gpw),
        "integrationmode": "point integration",
        "eta_eV": args.eta, "ecut_eV": args.ecut,
        "frequencies": {"type": "nonlinear", "domega0_eV": args.domega0,
                        "omega2_eV": args.omega2, "omegamax_eV": args.omegamax},
        "intraband": False, "rows": len(energy),
        "out_csv": str(args.out_csv),
    }, indent=2) + "\n")
    raw_path.unlink()
    print(f"[OK] {args.out_csv}")


if __name__ == "__main__":
    main()
