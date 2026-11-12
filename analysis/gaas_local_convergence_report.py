#!/usr/bin/env python3
"""Compare four finite-beam local-SBE propagation checks on common grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(prefix: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    npz = prefix.with_suffix(".npz")
    metadata = prefix.with_suffix(".json")
    with np.load(npz) as arrays:
        x = arrays["x_m"]
        time = arrays["time_s"]
        field = arrays["field_xt"]
    report = json.loads(metadata.read_text())
    if report["output_npz_sha256"] != sha256(npz):
        raise ValueError(f"Saved field does not match its report: {prefix}")
    files = {name: {"path": str(path), "sha256": sha256(path)}
             for name, path in (("field", npz), ("report", metadata))}
    return x, time, field, report, files


def relative_rms(first: np.ndarray, reference: np.ndarray) -> float:
    if first.shape != reference.shape:
        raise ValueError("Comparison arrays have different shapes")
    return float(np.linalg.norm(first - reference) / np.linalg.norm(reference))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "half_dz", "half_dt", "double_width"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    records = {name: load(getattr(args, name)) for name in
               ("base", "half_dz", "half_dt", "double_width")}
    x, t, field, base, _ = records["base"]
    xd, td, refined_z, zmeta, _ = records["half_dz"]
    xt, tt, refined_t, tmeta, _ = records["half_dt"]
    xw, tw, wide, wmeta, _ = records["double_width"]
    if (not np.allclose(x, xd, rtol=1e-12, atol=1e-24)
            or not np.allclose(t, td, rtol=1e-12, atol=1e-27)
            or not np.allclose(x, xt, rtol=1e-12, atol=1e-24)
            or not np.allclose(t, tt[::2], rtol=1e-12, atol=1e-27)
            or not np.allclose(t, tw, rtol=1e-12, atol=1e-27)
            or not np.allclose(x, xw[len(xw)//2-len(x)//2:
                                     len(xw)//2+len(x)//2], rtol=1e-12, atol=1e-24)):
        raise ValueError("The four runs do not share the stated comparison grids")
    if (not np.isclose(zmeta["dz_um"] * 2, base["dz_um"])
            or tmeta["time_stride"] * 2 != base["time_stride"]
            or not np.isclose(wmeta["width_um"], base["width_um"] * 2)):
        raise ValueError("Step and box changes do not match the intended checks")
    centered_wide = wide[len(xw)//2-len(x)//2:len(xw)//2+len(x)//2]
    result = {
        "model": base["model"],
        "comparison_region_um": [float(x[0] * 1e6), float(x[-1] * 1e6)],
        "propagation_um": base["z_um"],
        "relative_rms": {
            "dz_0p25_vs_0p125_um": relative_rms(field, refined_z),
            "dt_0p05_vs_0p025_fs": relative_rms(field, refined_t[:, ::2]),
            "box_10_vs_20_um": relative_rms(field, centered_wide),
        },
        "peak_population": {name: row[3]["peak_population"] for name, row in records.items()},
        "source_files": {name: row[4] for name, row in records.items()},
        "interpretation": "Numerical sensitivity of a reduced local-SBE 2 um run; not a converged first-principles GaAs transmission prediction",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["relative_rms"], indent=2))


if __name__ == "__main__":
    main()
