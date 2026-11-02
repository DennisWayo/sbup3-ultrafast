#!/usr/bin/env python3
"""Run bounded 100 µm reduced-SBE field/material sensitivity checks.

The run tests numerical stability of the effective, calibrated model. It is
not a first-principles GaAs transmission prediction or an optical-data fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CASES = (
    ("w160_dz050_dt050", 160.0, 0.5, 10),
    ("w160_dz025_dt050", 160.0, 0.25, 10),
    ("w160_dz025_dt025", 160.0, 0.25, 5),
    ("w320_dz025_dt050", 320.0, 0.25, 10),
)
INPUTS = ("time_s.npy", "E_t.npy", "transition_omega_rad_s_used.npy",
          "transition_d_eff_Cm_used.npy", "transition_weight_used.npy",
          "transition_model.json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def validate(prefix: Path, expected_width: float, expected_dz: float,
             expected_stride: int, input_hashes: dict) -> dict:
    report_path = prefix.with_suffix(".json")
    field_path = prefix.with_suffix(".npz")
    report = json.loads(report_path.read_text())
    expected = {"z_um": 100.0, "width_um": expected_width,
                "dz_um": expected_dz, "time_stride": expected_stride,
                "dx_nm": 156.25, "input_sha256": input_hashes}
    for key, value in expected.items():
        observed = report.get(key)
        if key in {"z_um", "width_um", "dz_um", "dx_nm"}:
            valid = isinstance(observed, (float, int)) and math.isclose(
                observed, value, rel_tol=1e-12, abs_tol=1e-10)
        else:
            valid = observed == value
        if not valid:
            raise ValueError(f"{prefix}: unexpected {key}: {observed!r}")
    if report["output_npz_sha256"] != digest(field_path):
        raise ValueError(f"{prefix}: field checksum mismatch")
    with np.load(field_path) as arrays:
        x, time, field = (arrays[key] for key in ("x_m", "time_s", "field_xt"))
        if (field.shape != (len(x), len(time)) or
                len(x) != round(expected_width * 1e3 / 156.25) or
                not np.isfinite(field).all()):
            raise ValueError(f"{prefix}: invalid field array")
    return {"report": str(report_path), "report_sha256": digest(report_path),
            "field": str(field_path), "field_sha256": digest(field_path),
            "peak_population": report["peak_population"],
            "peak_field_vpm": report["peak_field_vpm"]}


def relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Compared arrays have different shapes")
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def report_sensitivity(out_dir: Path) -> dict:
    arrays = {}
    for label, _, _, _ in CASES:
        with np.load(out_dir / f"{label}.npz") as saved:
            arrays[label] = tuple(saved[key] for key in
                                  ("x_m", "time_s", "field_xt"))
    x, t, baseline = arrays["w160_dz025_dt050"]
    _, t_coarse, coarse_z = arrays["w160_dz050_dt050"]
    _, t_fine, fine_t = arrays["w160_dz025_dt025"]
    x_wide, t_wide, wide = arrays["w320_dz025_dt050"]
    if (not np.allclose(t, t_coarse, rtol=1e-12, atol=1e-27)
            or not np.allclose(t, t_fine[::2], rtol=1e-12, atol=1e-27)
            or not np.allclose(t, t_wide, rtol=1e-12, atol=1e-27)
            or not np.allclose(x, x_wide[len(x_wide)//2-len(x)//2:
                                         len(x_wide)//2+len(x)//2],
                               rtol=1e-12, atol=1e-24)):
        raise ValueError("Long-range runs do not share the intended grids")
    centered_wide = wide[len(x_wide)//2-len(x)//2:
                         len(x_wide)//2+len(x)//2]
    central = np.abs(x) <= 5e-6
    outer = np.abs(x_wide) >= 0.4 * (x_wide[-1] - x_wide[0])
    return {
        "scope": "100 µm numerical sensitivity of the calibrated local-SBE model",
        "central_region_um": [-5, 5],
        "relative_rms": {
            "dz_0p5_vs_0p25_um": relative_rms(coarse_z[central], baseline[central]),
            "dt_0p05_vs_0p025_fs": relative_rms(baseline[central], fine_t[central, ::2]),
            "box_160_vs_320_um": relative_rms(baseline[central], centered_wide[central]),
        },
        "outer_fifth_energy_fraction_w320": float(
            np.sum(wide[outer] ** 2) / np.sum(wide ** 2)
        ),
        "interpretation": "Numerical sensitivity only; no first-principles or experimental transmission validation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--case-timeout-hours", type=float, default=8.0)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="validate existing output and continue a failed sweep")
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    inputs = root / "archive/2026-10-19-local-coupling/inputs"
    code = root / "analysis/gaas_local_sbe_uppe.py"
    dependencies = (root / "analysis/gaas_linear_material_window.py",
                    root / "sbe/local_response.py",
                    root / "uppe/dispersive_residual.py")
    for path in (args.python, code, *dependencies, *(inputs / name for name in INPUTS)):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (0 < args.case_timeout_hours <= 12):
        parser.error("Case timeout must be positive and at most 12 hours")
    input_hashes = {key: digest(inputs / key) for key in INPUTS}
    aliases = {"time": "time_s.npy", "field": "E_t.npy",
               "omega": "transition_omega_rad_s_used.npy",
               "dipole": "transition_d_eff_Cm_used.npy",
               "weights": "transition_weight_used.npy",
               "model": "transition_model.json"}
    expected_input_hashes = {key: input_hashes[value] for key, value in aliases.items()}
    records = []
    for label, width, dz, stride in CASES:
        prefix = out_dir / label
        command = [str(args.python), "-m", "analysis.gaas_local_sbe_uppe",
                   "--sbe-dir", str(inputs), "--out-prefix", str(prefix),
                   "--width-um", str(width), "--dx-nm", "156.25",
                   "--waist-um", "2", "--z-um", "100", "--dz-um", str(dz),
                   "--stride", str(stride), "--tail-fs", "150",
                   "--max-field-vpm", "1e9"]
        records.append({"case": label, "state": "pending", "command": command,
                        "log": str(out_dir / f"{label}.log")})
    state = {"purpose": "Bounded 100 µm reduced-model numerical sensitivity, not physical validation",
             "state": "planned", "started_utc": now(), "pid": os.getpid(),
             "source_sha256": digest(code), "runner_sha256": digest(Path(__file__)),
             "dependency_sha256": {str(path.relative_to(root)): digest(path)
                                   for path in dependencies},
             "input_sha256": expected_input_hashes,
             "case_timeout_hours": args.case_timeout_hours, "cases": records}
    if args.plan:
        print(json.dumps(state, indent=2))
        return 0
    status = out_dir / "longrange_status.json"
    if args.resume:
        if not status.is_file():
            raise FileNotFoundError(f"No sweep status to resume: {status}")
        previous = json.loads(status.read_text())
        if previous.get("state") != "failed":
            raise ValueError("Resume requires a previously failed sweep")
        for key in ("source_sha256", "dependency_sha256", "input_sha256"):
            if previous.get(key) != state[key]:
                raise ValueError(f"Refusing resume: {key} changed")
        if [entry["case"] for entry in previous["cases"]] != [entry["case"] for entry in records]:
            raise ValueError("Refusing resume: case plan changed")
        state = previous
        records = state["cases"]
        state.setdefault("recovery_history", []).append({
            "started_utc": now(), "previous_state": "failed",
            "previous_error": records[0].get("error"),
            "resume_runner_sha256": digest(Path(__file__))})
        state["state"] = "running"
        save(status, state)
    else:
        if out_dir.exists() or status.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {out_dir}")
        out_dir.mkdir(parents=True)
        save(status, state)
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    for record, (label, width, dz, stride) in zip(records, CASES):
        prefix = out_dir / label
        if args.resume and prefix.with_suffix(".json").is_file() and prefix.with_suffix(".npz").is_file():
            record["output"] = validate(prefix, width, dz, stride,
                                        expected_input_hashes)
            record["state"] = "completed"
            record["reused_existing_output"] = True
            record.pop("error", None)
            record["finished_utc"] = now()
            save(status, state)
            continue
        if prefix.with_suffix(".json").exists() or prefix.with_suffix(".npz").exists():
            raise FileExistsError(f"Refusing to overwrite partial output: {prefix}")
        record["state"] = "running"
        record["started_utc"] = now()
        state["state"] = "running"
        save(status, state)
        try:
            with (out_dir / f"{label}.log").open("w") as stream:
                completed = subprocess.run(record["command"], cwd=root,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           env=env, timeout=args.case_timeout_hours * 3600,
                                           check=False)
            record["exit_code"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(f"Case exited {completed.returncode}")
            record["output"] = validate(out_dir / label, width, dz, stride,
                                        expected_input_hashes)
        except Exception as exc:
            record["state"] = "failed"
            record["error"] = str(exc)
            record["finished_utc"] = now()
            state["state"] = "failed"
            save(status, state)
            return 1
        record["state"] = "completed"
        record["finished_utc"] = now()
        save(status, state)
    try:
        result = report_sensitivity(out_dir)
        result_path = out_dir / "longrange_sensitivity.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        state["sensitivity_report"] = {"path": str(result_path),
                                       "sha256": digest(result_path)}
    except Exception as exc:
        state["state"] = "failed"
        state["error"] = f"Comparison failed: {exc}"
        save(status, state)
        return 2
    state["state"] = "completed"
    state["finished_utc"] = now()
    save(status, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
