#!/usr/bin/env python3
"""Test the archived GaAs Gaussian beam against a wider periodic FFT box."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from uppe.uppe_2d import propagate_uppe_2d


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbe-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    sbe = args.sbe_dir.resolve()
    paths = {key: sbe / filename for key, filename in {
        "time_s": "time_s.npy", "E_t": "E_t.npy", "polarization": "polarization.npy"
    }.items()}
    time_s = np.asarray(np.load(paths["time_s"]), dtype=float)
    E_t = np.asarray(np.load(paths["E_t"]), dtype=float)
    P_t = np.asarray(np.load(paths["polarization"]), dtype=complex).real
    if len(time_s) != 10000 or E_t.shape != time_s.shape or P_t.shape != time_s.shape:
        raise ValueError("Expected the archived 10,000-sample GaAs SBE grid")

    fields = {}
    edge_fraction = {}
    for label, width, nx, steps in (("narrow", 10e-6, 256, 100),
                                    ("step_refined", 10e-6, 256, 200),
                                    ("wide", 20e-6, 512, 100)):
        x_m = np.linspace(-width / 2, width / 2, nx, endpoint=False)
        profile = np.exp(-(x_m[:, None] ** 2) / (2e-6) ** 2)
        fields[label] = propagate_uppe_2d(
            x_m, time_s, profile * E_t[None, :], profile * P_t[None, :],
            100e-6 / steps, steps, background_index=1.0,
            polarization_kind="full",
        )
        mask = np.abs(x_m) >= 0.4 * width
        edge_fraction[label] = float(np.sum(fields[label][mask] ** 2) /
                                     np.sum(fields[label] ** 2))

    central_wide = fields["wide"][128:384]
    metrics = {
        "step_100_to_200_relative_rms": float(np.linalg.norm(fields["narrow"] - fields["step_refined"]) /
                                              np.linalg.norm(fields["narrow"])),
        "window_10_to_20um_central_relative_rms": float(np.linalg.norm(fields["narrow"] - central_wide) /
                                                       np.linalg.norm(central_wide)),
        "outer_20_percent_energy_fraction_10um": edge_fraction["narrow"],
        "outer_20_percent_energy_fraction_20um": edge_fraction["wide"],
    }
    report = {
        "model": "100 micrometre sequential GaAs propagation with a 2 micrometre Gaussian transverse waist",
        "grid": "10 micrometre/256 and 20 micrometre/512 periodic FFT windows; same spatial step",
        "interpretation": "Step refinement agrees, but box-size sensitivity prevents a finite-beam field claim",
        "inputs_sha256": {key: sha256(path) for key, path in paths.items()},
        "solver_sha256": sha256(ROOT / "uppe/uppe_2d.py"),
        "metrics": metrics,
    }
    out_json = args.out_json.resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
