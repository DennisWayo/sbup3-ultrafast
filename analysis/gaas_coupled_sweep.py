#!/usr/bin/env python3
"""Run a bounded, provenance-checked local SBE--UPPE refinement sweep.

The four cases isolate propagation step, temporal sampling, and transverse
box sensitivity. They test the calibrated reduced model, not ab-initio GaAs
transmission or convergence of the upstream dielectric calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


INPUTS = {
    "time": "time_s.npy",
    "field": "E_t.npy",
    "omega": "transition_omega_rad_s_used.npy",
    "dipole": "transition_d_eff_Cm_used.npy",
    "weights": "transition_weight_used.npy",
    "model": "transition_model.json",
}
DEPENDENCIES = (
    "analysis/gaas_local_sbe_uppe.py",
    "analysis/gaas_linear_material_window.py",
    "sbe/local_response.py",
    "uppe/dispersive_residual.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def check_output(prefix: Path, *, width_um: float, dz_um: float,
                 z_um: float, stride: int, dx_nm: float,
                 inputs: dict[str, str]) -> dict:
    report_path = prefix.with_suffix(".json")
    field_path = prefix.with_suffix(".npz")
    report = json.loads(report_path.read_text())
    for key, expected in (("width_um", width_um), ("dz_um", dz_um),
                          ("z_um", z_um), ("dx_nm", dx_nm)):
        if not math.isclose(report[key], expected, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError(f"{prefix}: {key} does not match the plan")
    if report["time_stride"] != stride or report["input_sha256"] != inputs:
        raise ValueError(f"{prefix}: temporal sampling or input hash mismatch")
    if report["output_npz_sha256"] != sha256(field_path):
        raise ValueError(f"{prefix}: field checksum mismatch")
    with np.load(field_path) as saved:
        x, t, field = (saved[key] for key in ("x_m", "time_s", "field_xt"))
        if (field.shape != (len(x), len(t))
                or len(x) != round(width_um * 1000 / dx_nm)
                or not np.isfinite(field).all()):
            raise ValueError(f"{prefix}: invalid field dimensions or values")
    if not 0 <= report["peak_population"] <= 1:
        raise ValueError(f"{prefix}: invalid excited population")
    return {"report_sha256": sha256(report_path), "field_sha256": sha256(field_path),
            "peak_population": report["peak_population"],
            "peak_field_vpm": report["peak_field_vpm"]}


def compare(out: Path, z_um: float) -> dict:
    arrays = {}
    for case in ("base", "half_dz", "half_dt", "double_width"):
        with np.load(out / f"{case}.npz") as saved:
            arrays[case] = tuple(saved[key] for key in ("x_m", "time_s", "field_xt"))
    x, t, coarse = arrays["base"]
    xd, td, refined = arrays["half_dz"]
    xt, tt, fine_t = arrays["half_dt"]
    xw, tw, wide = arrays["double_width"]
    centered_wide = wide[len(xw)//2-len(x)//2:len(xw)//2+len(x)//2]
    if (not np.allclose(x, xd, rtol=1e-12, atol=1e-24)
            or not np.allclose(x, xt, rtol=1e-12, atol=1e-24)
            or not np.allclose(t, td, rtol=1e-12, atol=1e-27)
            or not np.allclose(t, tt[::2], rtol=1e-12, atol=1e-27)
            or not np.allclose(t, tw, rtol=1e-12, atol=1e-27)
            or not np.allclose(x, xw[len(xw)//2-len(x)//2:
                                     len(xw)//2+len(x)//2], rtol=1e-12, atol=1e-24)):
        raise ValueError("Comparison grids do not match the declared refinements")
    center = np.abs(x) <= 5e-6
    outer = np.abs(xw) >= 0.4 * (xw[-1] - xw[0])

    def error(candidate: np.ndarray, reference: np.ndarray) -> float:
        return float(np.linalg.norm(candidate - reference)
                     / max(np.linalg.norm(reference), 1e-300))

    return {
        "scope": "Numerical sensitivity of the calibrated local SBE--UPPE model",
        "z_um": z_um,
        "central_region_um": [-5, 5],
        "relative_rms": {
            "propagation_step": error(coarse[center], refined[center]),
            "temporal_sampling": error(refined[center], fine_t[center, ::2]),
            "transverse_box": error(refined[center], centered_wide[center]),
        },
        "outer_fifth_energy_fraction": float(
            np.sum(wide[outer] ** 2) / np.sum(wide ** 2)),
        "interpretation": "No claim of physical GaAs validation or dielectric convergence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--z-um", type=float, default=10.0)
    parser.add_argument("--width-um", type=float, default=40.0)
    parser.add_argument("--dx-nm", type=float, default=156.25)
    parser.add_argument("--dz-um", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--case-timeout-hours", type=float, default=12.0)
    parser.add_argument("--wait-for-status", type=Path)
    parser.add_argument("--max-wait-hours", type=float, default=30.0)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if (args.z_um <= 0 or args.width_um <= 0 or args.dx_nm <= 0
            or args.dz_um <= 0 or args.stride < 2 or args.stride % 2
            or not 0 < args.case_timeout_hours <= 24
            or not 0 < args.max_wait_hours <= 48
            or round(args.z_um / (args.dz_um / 2)) * (args.dz_um / 2) != args.z_um
            or round(args.width_um * 1000 / args.dx_nm) % 2):
        parser.error("Invalid refinement grid, stride, or time limit")
    root, out = args.root.resolve(), args.out_dir.resolve()
    input_dir = root / "archive/2026-10-19-local-coupling/inputs"
    paths = {key: input_dir / name for key, name in INPUTS.items()}
    dependencies = {name: root / name for name in DEPENDENCIES}
    for path in (args.python, *paths.values(), *dependencies.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {key: sha256(path) for key, path in paths.items()}
    cases = (
        ("base", args.width_um, args.dz_um, args.stride),
        ("half_dz", args.width_um, args.dz_um / 2, args.stride),
        ("half_dt", args.width_um, args.dz_um / 2, args.stride // 2),
        ("double_width", args.width_um * 2, args.dz_um / 2, args.stride),
    )
    entries = []
    for label, width, dz, stride in cases:
        command = [str(args.python), "-m", "analysis.gaas_local_sbe_uppe",
                   "--sbe-dir", str(input_dir), "--out-prefix", str(out / label),
                   "--width-um", str(width), "--dx-nm", str(args.dx_nm),
                   "--waist-um", "2", "--z-um", str(args.z_um),
                   "--dz-um", str(dz), "--stride", str(stride),
                   "--tail-fs", "150", "--max-field-vpm", "1e9"]
        entries.append({"case": label, "state": "pending", "command": command,
                        "log": str(out / f"{label}.log")})
    record = {
        "purpose": "Bounded local material--field refinement, not physical validation",
        "state": "planned", "started_utc": now(), "pid": os.getpid(),
        "source_sha256": {name: sha256(path) for name, path in dependencies.items()},
        "runner_sha256": sha256(Path(__file__)),
        "input_sha256": input_hashes,
        "wait_for_status": str(args.wait_for_status) if args.wait_for_status else None,
        "case_timeout_hours": args.case_timeout_hours,
        "cases": entries,
    }
    if args.plan:
        print(json.dumps(record, indent=2))
        return 0
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    status = out / "coupled_status.json"
    save(status, record)
    if args.wait_for_status:
        until = time.monotonic() + args.max_wait_hours * 3600
        while True:
            upstream = json.loads(args.wait_for_status.read_text())
            if upstream.get("state") != "running":
                if upstream.get("state") != "completed":
                    record["state"] = "blocked"
                    record["error"] = "The upstream response sweep did not complete"
                    save(status, record)
                    return 1
                break
            if time.monotonic() >= until:
                record["state"] = "blocked"
                record["error"] = "Timed out waiting for the response sweep"
                save(status, record)
                return 1
            record["state"] = "waiting_for_response_sweep"
            save(status, record)
            time.sleep(120)
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    for entry, (label, width, dz, stride) in zip(entries, cases):
        entry["state"] = "running"
        entry["started_utc"] = now()
        record["state"] = "running"
        save(status, record)
        try:
            with (out / f"{label}.log").open("w") as stream:
                completed = subprocess.run(entry["command"], cwd=root, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=args.case_timeout_hours * 3600)
            if completed.returncode != 0:
                raise RuntimeError(f"Case exited {completed.returncode}")
            entry["output"] = check_output(
                out / label, width_um=width, dz_um=dz, z_um=args.z_um,
                stride=stride, dx_nm=args.dx_nm, inputs=input_hashes)
        except Exception as exc:
            entry["state"] = "failed"
            entry["error"] = str(exc)
            record["state"] = "failed"
            save(status, record)
            return 1
        entry["state"] = "completed"
        entry["finished_utc"] = now()
        save(status, record)
    try:
        report = compare(out, args.z_um)
        report_path = out / "coupled_sensitivity.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        record["sensitivity_sha256"] = sha256(report_path)
    except Exception as exc:
        record["state"] = "failed"
        record["error"] = f"Comparison failed: {exc}"
        save(status, record)
        return 2
    record["state"] = "completed"
    record["finished_utc"] = now()
    save(status, record)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
