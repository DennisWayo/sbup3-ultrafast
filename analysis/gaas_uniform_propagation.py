#!/usr/bin/env python3
"""Archive a reproducible uniform-transverse GaAs SBE-to-UPPE calculation.

This is a frozen-polarization sequential run. The uniform transverse profile
removes the finite-box diffraction seen with the 2 micrometre Gaussian input.
"""

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


def record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbe-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    sbe = args.sbe_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    files = {key: sbe / filename for key, filename in {
        "time_s": "time_s.npy", "E_t": "E_t.npy", "polarization": "polarization.npy"
    }.items()}
    time_s = np.asarray(np.load(files["time_s"]), dtype=float)
    E_t = np.asarray(np.load(files["E_t"]), dtype=float)
    P_t = np.asarray(np.load(files["polarization"]), dtype=complex)
    if len(time_s) != 10000 or E_t.shape != time_s.shape or P_t.shape != time_s.shape:
        raise ValueError("Expected the 10,000-sample GaAs SBE run")
    if not np.isfinite(time_s).all() or not np.isfinite(E_t).all() or not np.isfinite(P_t).all():
        raise ValueError("Nonfinite SBE input")

    nx = 16
    E_xt = np.broadcast_to(E_t, (nx, len(time_s))).copy()
    P_xt = np.broadcast_to(P_t.real, (nx, len(time_s))).copy()
    outputs: dict[str, np.ndarray] = {}
    for label, width, steps in (("primary", 10e-6, 100), ("step2", 10e-6, 200),
                                ("one_step", 10e-6, 1), ("wide", 80e-6, 100)):
        x_m = np.linspace(-width / 2, width / 2, nx, endpoint=False)
        outputs[label] = propagate_uppe_2d(
            x_m=x_m, time_s=time_s, E_xt0=E_xt, P_xt=P_xt,
            dz=100e-6 / steps, z_steps=steps,
            background_index=1.0, polarization_kind="full",
        )
        if not np.isfinite(outputs[label]).all():
            raise RuntimeError(f"Nonfinite propagated field: {label}")

    primary = outputs["primary"]
    transverse_spread = float(np.max(np.abs(primary - primary[0:1])))
    metrics = {
        "step_100_to_200_relative_rms": relative_rms(primary, outputs["step2"]),
        "step_100_to_1_relative_rms": relative_rms(primary, outputs["one_step"]),
        "window_10_to_80um_relative_rms": relative_rms(primary, outputs["wide"]),
        "transverse_spread_V_m": transverse_spread,
        "input_rms_V_m": float(np.sqrt(np.mean(E_t**2))),
        "output_rms_V_m": float(np.sqrt(np.mean(primary**2))),
        "output_peak_abs_V_m": float(np.max(np.abs(primary))),
    }
    output_path = out / "gaas_uniform_100um.npz"
    np.savez_compressed(output_path, time_s=time_s,
                        x_m=np.linspace(-5e-6, 5e-6, nx, endpoint=False),
                        E_xt0=E_xt, P_xt=P_xt, E_xt_out=primary)
    report = {
        "model": "GaAs SBE drive, frozen full polarization, uniform transverse field, vacuum-background real-field UPPE",
        "scope": "Corrected sequential propagation; not self-consistent intra-step field-material coupling",
        "length_m": 100e-6, "nx": nx, "time_samples": len(time_s),
        "inputs": {key: record(path) for key, path in files.items()},
        "code": {"sbe": record(ROOT / "sbe" / "sbe_rk4.py"),
                 "uppe": record(ROOT / "uppe" / "uppe_2d.py"),
                 "study": record(Path(__file__))},
        "metrics": metrics,
        "output": record(output_path),
    }
    report_path = out / "gaas_uniform_propagation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if max(metrics["step_100_to_200_relative_rms"],
           metrics["step_100_to_1_relative_rms"],
           metrics["window_10_to_80um_relative_rms"]) > 1e-10:
        raise RuntimeError("Uniform-propagation invariance check failed")
    print(json.dumps(metrics, indent=2))
    print(f"[OK] {report_path}")


if __name__ == "__main__":
    main()
