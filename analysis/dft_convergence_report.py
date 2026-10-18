#!/usr/bin/env python3
"""Compare archived GaAs dielectric sweeps on one measured energy grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOW_EV = 1.5715537
HIGH_EV = 2.9776807
CASES = {
    "legacy_gs4_resp6_b48_e150": ROOT / "analysis/data/gaas_dielectric_reference.csv",
    "gs6_resp6_b48_e150": "gaas_dielectric_gs6_k6_b48_e150.csv",
    "gs6_resp6_b48_e200": "gaas_dielectric_gs6_k6_b48_e200.csv",
    "gs6_resp6_b48_eta003": "gaas_dielectric_gs6_k6_b48_eta003.csv",
    "gs6_resp6_b48_freq001": "gaas_dielectric_gs6_k6_b48_freq001.csv",
    "gs6_resp6_b48_freq0005": "gaas_dielectric_gs6_k6_b48_freq0005.csv",
    "gs6_resp8_b48_e150": "gaas_dielectric_gs6_k8_b48_e150.csv",
    "gs6_resp10_b48_e150": "gaas_dielectric_gs6_k10_b48_e150.csv",
    "gs6_resp6_b64_e150": "gaas_dielectric_gs6_k6_b64_e150.csv",
    "gs6_pw250_resp6_b48_e150": "gaas_dielectric_gs6_pw250_k6_b48_e150.csv",
    "gs8_resp8_b48_e150": "gaas_dielectric_gs8_k8_b48_e150.csv",
}
PAIRS = [
    ("legacy_gs4_resp6_b48_e150", "gs6_resp6_b48_e150", "combined GS k-grid and band change"),
    ("gs6_resp6_b48_e150", "gs6_resp6_b48_e200", "response cutoff"),
    ("gs6_resp6_b48_e150", "gs6_resp6_b48_eta003", "broadening"),
    ("gs6_resp6_b48_e150", "gs6_resp6_b48_freq001", "frequency grid seed"),
    ("gs6_resp6_b48_freq001", "gs6_resp6_b48_freq0005", "frequency grid seed"),
    ("gs6_resp6_b48_e150", "gs6_resp8_b48_e150", "response k-grid"),
    ("gs6_resp8_b48_e150", "gs6_resp10_b48_e150", "response k-grid"),
    ("gs6_resp6_b48_e150", "gs6_resp6_b64_e150", "response bands"),
    ("gs6_resp6_b48_e150", "gs6_pw250_resp6_b48_e150", "ground-state PW cutoff"),
    ("gs6_resp8_b48_e150", "gs8_resp8_b48_e150", "ground-state k-grid"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_rmse(delta: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean(delta**2)) / np.max(np.abs(reference)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True,
                        help="Directory containing the sweep CSV files")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    study_root = args.study_root.resolve()
    out_json = args.out_json.resolve()
    reference_path = ROOT / "analysis/data/gaas_exp_jellison_1992.csv"
    reference = np.genfromtxt(reference_path, delimiter=",", names=True)
    mask = (reference["energy_eV"] >= LOW_EV) & (reference["energy_eV"] <= HIGH_EV)
    energy = reference["energy_eV"][mask]
    observed = {"n": reference["n"][mask], "alpha_m_inv": reference["alpha"][mask]}
    if len(energy) != 84:
        raise RuntimeError(f"Expected 84 Jellison energies, found {len(energy)}")

    cases = {}
    curves = {}
    missing = []
    for name, configured in CASES.items():
        path = configured if isinstance(configured, Path) else study_root / configured
        if not path.exists():
            missing.append(name)
            continue
        table = np.genfromtxt(path, delimiter=",", names=True)
        model_energy = table["energy_eV"]
        if energy[0] < model_energy[0] or energy[-1] > model_energy[-1]:
            raise ValueError(f"Model does not span measured comparison window: {path}")
        curves[name] = {key: np.interp(energy, model_energy, table[key]) for key in observed}
        cases[name] = {
            "csv_path": str(path), "csv_sha256": sha256(path),
            "rows": len(table),
            "nrmse_against_jellison": {
                key: norm_rmse(curves[name][key] - observed[key], observed[key])
                for key in observed
            },
        }
        meta = path.with_suffix(".meta.json")
        if meta.exists():
            cases[name]["meta_path"] = str(meta)
            cases[name]["meta_sha256"] = sha256(meta)

    comparisons = []
    for first, second, axis in PAIRS:
        if first not in curves or second not in curves:
            continue
        comparisons.append({
            "first": first, "second": second, "parameter_changed": axis,
            "nrmse_between_curves_normalized_to_measured_max": {
                key: norm_rmse(curves[first][key] - curves[second][key], observed[key])
                for key in observed
            },
        })
    report = {
        "window_eV": [LOW_EV, HIGH_EV], "measured_points": len(energy),
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
        "normalization": "RMSE divided by the maximum absolute Jellison value in the same channel",
        "interpolation": "linear, model to measured energy grid; no extrapolation or fitting",
        "interpretation": "Finite parameter sensitivities, not a declaration of convergence or experimental agreement",
        "cases": cases, "comparisons": comparisons, "missing_cases": missing,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"comparisons": comparisons, "missing_cases": missing}, indent=2))


if __name__ == "__main__":
    main()
