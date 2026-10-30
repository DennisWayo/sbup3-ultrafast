#!/usr/bin/env python3
"""Compare GaAs optical references and one PBE curve on identical energies.

The cross-study difference is a descriptive spread, not an experimental
uncertainty estimate: the cited samples and protocols are not identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.dft_convergence_report import HIGH_EV, LOW_EV, ROOT


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(predicted: np.ndarray, reference: np.ndarray) -> dict:
    residual = predicted - reference
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "rmse": rmse,
        "nrmse_max_reference": rmse / float(np.max(np.abs(reference))),
        "mean_bias": float(np.mean(residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dft-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    dft_path = args.dft_csv.resolve()
    meta_path = dft_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    expected = {"gs_kpts": [6, 6, 6], "kpts_response": [36, 36, 36],
                "nbands": 48, "eta_eV": 0.05, "ecut_eV": 150.0,
                "integrationmode": "point integration", "intraband": False}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("DFT metadata do not describe the 36-cubed comparison")

    jellison_path = ROOT / "analysis/data/gaas_exp_jellison_1992.csv"
    aspnes_path = ROOT / "analysis/data/gaas_exp_aspnes_1986.csv"
    jellison = np.genfromtxt(jellison_path, delimiter=",", names=True)
    aspnes = np.genfromtxt(aspnes_path, delimiter=",", names=True)
    dft = np.genfromtxt(dft_path, delimiter=",", names=True)
    if len(dft) != meta["n_rows"]:
        raise ValueError("DFT CSV row count does not match metadata")
    mask = (aspnes["energy_eV"] >= LOW_EV) & (aspnes["energy_eV"] <= HIGH_EV)
    energy = aspnes["energy_eV"][mask]
    if (len(energy) != 14 or energy[0] < jellison["energy_eV"][0]
            or energy[-1] > jellison["energy_eV"][-1]
            or energy[0] < dft["energy_eV"][0]
            or energy[-1] > dft["energy_eV"][-1]):
        raise ValueError("Unexpected common-energy coverage")
    for table in (jellison, aspnes, dft):
        if not np.all(np.diff(table["energy_eV"]) > 0):
            raise ValueError("Input energies must be strictly increasing")

    channels = {}
    for label, measured_field, dft_field in (
        ("n", "n", "n"), ("alpha_m_inv", "alpha", "alpha_m_inv")
    ):
        a = aspnes[measured_field][mask]
        j = np.interp(energy, jellison["energy_eV"], jellison[measured_field])
        d = np.interp(energy, dft["energy_eV"], dft[dft_field])
        channels[label] = {
            "Jellison_minus_Aspnes": score(j, a),
            "PBE36_minus_Aspnes": score(d, a),
            "PBE36_minus_Jellison": score(d, j),
        }
    result = {
        "purpose": "Common-energy optical comparison; no fitting or extrapolation",
        "interpretation": "Cross-study spread is not a measurement uncertainty bound or an additive error-budget term",
        "window_eV": [LOW_EV, HIGH_EV],
        "common_points": len(energy),
        "common_energy_minmax_eV": [float(energy[0]), float(energy[-1])],
        "normalization": "RMSE divided by maximum absolute reference value in each pair and channel",
        "sources": {label: {"path": str(path), "sha256": digest(path)}
                    for label, path in (("Jellison", jellison_path),
                                        ("Aspnes", aspnes_path),
                                        ("PBE36", dft_path),
                                        ("PBE36_metadata", meta_path))},
        "channels": channels,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(channels, indent=2))


if __name__ == "__main__":
    main()
