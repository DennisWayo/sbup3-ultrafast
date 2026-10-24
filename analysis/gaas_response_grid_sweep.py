#!/usr/bin/env python3
"""Run one bounded, serial GaAs response-grid sweep on a prepared GPAW host.

This runner deliberately keeps the ground state, bands, response cutoff,
broadening, and frequency grid fixed. It records provenance and file hashes;
it does not declare numerical convergence from a single grid difference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_status(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


def validate_output(csv_path: Path, meta_path: Path, grid: int) -> dict:
    meta = json.loads(meta_path.read_text())
    expected = {
        "gs_kpts": [6, 6, 6],
        "kpts_response": [grid] * 3,
        "nbands": 48,
        "eta_eV": 0.05,
        "ecut_eV": 150.0,
        "integrationmode": "point integration",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{meta_path}: {key}={meta.get(key)!r}, expected {value!r}")
    if meta.get("frequencies") != {
        "type": "nonlinear", "domega0_eV": 0.02,
        "omega2_eV": 5.0, "omegamax_eV": 8.0,
    }:
        raise ValueError(f"{meta_path}: frequency grid differs from baseline")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if fields != ["energy_eV", "eps1_nlf", "eps2_nlf", "eps1_lf",
                      "eps2_lf", "n", "k", "alpha_m_inv"]:
            raise ValueError(f"Unexpected CSV columns: {fields}")
        rows = list(reader)
    if len(rows) != 242 or meta.get("n_rows") != 242:
        raise ValueError(f"Expected 242 spectral rows, got {len(rows)}")
    energies = [float(row["energy_eV"]) for row in rows]
    if energies[0] != 0.0 or energies[-1] < 8.0:
        raise ValueError("Spectral range differs from the baseline")
    return {
        "csv": str(csv_path), "csv_sha256": digest(csv_path),
        "metadata": str(meta_path), "metadata_sha256": digest(meta_path),
        "rows": len(rows), "energy_max_eV": energies[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Prepared validation directory with dft/ and logs/")
    parser.add_argument("--python", type=Path, required=True,
                        help="Python interpreter in the GPAW environment")
    parser.add_argument("--grids", default="32,36",
                        help="Exactly two serial response grids (default: 32,36)")
    parser.add_argument("--plan", action="store_true", help="Validate and print plan only")
    args = parser.parse_args()
    root = args.root.resolve()
    grids = [int(value) for value in args.grids.split(",")]
    if len(grids) != 2 or grids != sorted(set(grids)) or grids[0] <= 28:
        parser.error("This bounded sweep requires two increasing grids above 28")
    script = root / "dft/gaas_dielectric_response.py"
    gs = root / "dft/gaas_pw.gpw"
    gs_meta = root / "dft/gaas_ground_state_meta.json"
    baseline_csv = root / "dft/gaas_dielectric_gs6_k28_b48_e150.csv"
    baseline_meta = root / "dft/gaas_dielectric_gs6_k28_b48_e150.meta.json"
    for path in (args.python, script, gs, gs_meta, baseline_csv, baseline_meta):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline = validate_output(baseline_csv, baseline_meta, 28)
    outputs = []
    for grid in grids:
        stem = f"gaas_dielectric_gs6_k{grid}_b48_e150"
        response = root / f"dft/gaas_response_gs6_k{grid}_b48.gpw"
        csv_path = root / f"dft/{stem}.csv"
        meta_path = root / f"dft/{stem}.meta.json"
        log = root / f"logs/k{grid}_eta005.log"
        if any(path.exists() for path in (response, csv_path, meta_path, log)):
            raise FileExistsError(f"Refusing to overwrite an existing k={grid} output")
        command = [str(args.python), str(script), "--gs-gpw", str(gs),
                   "--gs-meta", str(gs_meta), "--response-gpw", str(response),
                   "--out-csv", str(csv_path), "--meta-json", str(meta_path),
                   "--kpts", f"{grid},{grid},{grid}", "--nbands", "48",
                   "--eta", "0.05", "--ecut", "150", "--domega0", "0.02",
                   "--omega2", "5", "--omegamax", "8",
                   "--integrationmode", "point integration"]
        outputs.append((grid, csv_path, meta_path, log, command))
    status_path = root / "response_grid_sweep_32_36.json"
    if status_path.exists():
        raise FileExistsError(f"Refusing to overwrite sweep record: {status_path}")
    state = {
        "purpose": "One bounded response-grid sensitivity sweep; not a convergence certification",
        "started_utc": utc_now(), "state": "planned", "pid": os.getpid(),
        "ground_state_sha256": digest(gs),
        "ground_state_metadata_sha256": digest(gs_meta),
        "response_script_sha256": digest(script),
        "baseline_k28": baseline,
        "points": [{"grid": grid, "state": "pending", "command": command,
                    "log": str(log)} for grid, _, _, log, command in outputs],
    }
    if args.plan:
        print(json.dumps(state, indent=2))
        return 0
    (root / "logs").mkdir(exist_ok=True)
    save_status(status_path, state)
    for point, (grid, csv_path, meta_path, log, command) in zip(state["points"], outputs):
        point["state"] = "running"
        point["started_utc"] = utc_now()
        state["state"] = "running"
        save_status(status_path, state)
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=root, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
        point["exit_code"] = completed.returncode
        point["finished_utc"] = utc_now()
        if completed.returncode != 0:
            point["state"] = "failed"
            state["state"] = "failed"
            save_status(status_path, state)
            return completed.returncode
        try:
            point["output"] = validate_output(csv_path, meta_path, grid)
        except Exception as exc:
            point["state"] = "invalid"
            point["error"] = str(exc)
            state["state"] = "failed"
            save_status(status_path, state)
            return 2
        point["state"] = "completed"
        save_status(status_path, state)
    state["state"] = "completed"
    state["finished_utc"] = utc_now()
    save_status(status_path, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
