"""Compare the archived GaAs dielectric curve with measured optical data.

This is a direct comparison: no channel rescaling, fitting, or extrapolation.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "data"
LOW_EV = 1.5715537238681454
HIGH_EV = 2.977680739960696


def load_csv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty reference: {path}")
    return rows


def evaluate(reference_path):
    dft_path = DATA / "gaas_dielectric_reference.csv"
    dft = load_csv(dft_path)
    reference = load_csv(reference_path)
    energies_dft = np.array([float(row["energy_eV"]) for row in dft])
    energies_ref = np.array([float(row["energy_eV"]) for row in reference])
    mask = ((energies_ref >= LOW_EV) & (energies_ref <= HIGH_EV)
            & (energies_ref >= energies_dft.min()) & (energies_ref <= energies_dft.max()))
    energies = energies_ref[mask]
    if len(energies) < 2 or not np.all(np.diff(energies_dft) > 0):
        raise ValueError("Insufficient ordered overlap")

    metrics = {}
    for label, dft_field, measured_field in (
        ("n", "n", "n"),
        ("alpha", "alpha_m_inv", "alpha"),
    ):
        measured = np.array([float(row[measured_field]) for row in reference])[mask]
        predicted = np.interp(energies, energies_dft,
                              [float(row[dft_field]) for row in dft])
        residual = predicted - measured
        rmse = float(np.sqrt(np.mean(residual**2)))
        metrics[label] = {
            "rmse": rmse,
            "nrmse_max_reference": rmse / float(np.max(np.abs(measured))),
            "mean_bias": float(np.mean(residual)),
            "mae": float(np.mean(np.abs(residual))),
            "n_points": len(energies),
        }
    return {
        "dft_file": str(dft_path.relative_to(ROOT)),
        "dft_sha256": hashlib.sha256(dft_path.read_bytes()).hexdigest(),
        "measurement_file": str(reference_path.relative_to(ROOT)),
        "measurement_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "window_eV": [float(energies[0]), float(energies[-1])],
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = {
        "protocol": "direct DFT vs measurement on archived probe-support window; no fitting",
        "requested_window_eV": [LOW_EV, HIGH_EV],
        "comparisons": [
            evaluate(DATA / "gaas_exp_jellison_1992.csv"),
            evaluate(DATA / "gaas_exp_aspnes_1986.csv"),
        ],
    }
    print(json.dumps(result, indent=2))
