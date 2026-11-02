#!/usr/bin/env python3
"""Compare matched GaAs response grids on the Jellison optical window."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.dft_convergence_report import HIGH_EV, LOW_EV, ROOT, norm_rmse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="K=CSV",
                        help="Point-integration response with an isotropic K-cubed grid")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    reference_path = ROOT / "analysis/data/gaas_exp_jellison_1992.csv"
    reference = np.genfromtxt(reference_path, delimiter=",", names=True)
    mask = (reference["energy_eV"] >= LOW_EV) & (reference["energy_eV"] <= HIGH_EV)
    energy = reference["energy_eV"][mask]
    if len(energy) != 84:
        raise RuntimeError("Expected 84 measured comparison energies")
    measured = {"n": reference["n"][mask],
                "alpha_m_inv": reference["alpha"][mask]}
    cases = {}
    settings = None
    for configured in args.case:
        label, sep, raw_path = configured.partition("=")
        if not sep or not label.isdigit() or not raw_path:
            raise ValueError(f"Invalid case {configured!r}; expected K=CSV")
        k = int(label)
        if k in cases:
            raise ValueError(f"Repeated response grid: {k}")
        path = Path(raw_path).resolve()
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        expected = {"gs_kpts": [6, 6, 6], "kpts_response": [k, k, k],
                    "nbands": 48, "eta_eV": 0.05, "ecut_eV": 150.0,
                    "integrationmode": "point integration", "intraband": False}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Response settings do not match the stated sweep: {path}")
        if settings is None:
            settings = meta["frequencies"]
        elif meta["frequencies"] != settings:
            raise ValueError("Frequency grids differ across response cases")
        table = np.genfromtxt(path, delimiter=",", names=True)
        if len(table) != meta["n_rows"] or energy[0] < table["energy_eV"][0] or energy[-1] > table["energy_eV"][-1]:
            raise ValueError(f"CSV coverage or row count is invalid: {path}")
        curve = {channel: np.interp(energy, table["energy_eV"], table[channel])
                 for channel in measured}
        cases[k] = {"csv_path": str(path), "csv_sha256": sha256(path),
                    "meta_path": str(meta_path), "meta_sha256": sha256(meta_path),
                    "direct_nrmse": {channel: norm_rmse(curve[channel] - observed, observed)
                                     for channel, observed in measured.items()},
                    "curve": curve}
    ordered = sorted(cases)
    if len(ordered) < 2:
        raise ValueError("At least two response grids are needed")
    comparisons = []
    for first, second in zip(ordered, ordered[1:]):
        curves_a = cases[first]["curve"]
        curves_b = cases[second]["curve"]
        comparisons.append({"from_k": first, "to_k": second,
                            "nrmse_between_curves_normalized_to_measured_max": {
                                channel: norm_rmse(curves_a[channel] - curves_b[channel], observed)
                                for channel, observed in measured.items()}})
    for case in cases.values():
        case.pop("curve")
    report = {
        "window_eV": [LOW_EV, HIGH_EV], "measured_points": len(energy),
        "reference_path": str(reference_path), "reference_sha256": sha256(reference_path),
        "common_settings": {"gs_kpts": [6, 6, 6], "nbands": 48,
                            "eta_eV": 0.05, "ecut_eV": 150.0,
                            "integrationmode": "point integration", "frequencies": settings},
        "normalization": "RMSE divided by maximum absolute Jellison value in each channel",
        "cases": {str(k): cases[k] for k in ordered},
        "comparisons": comparisons,
        "interpretation": "Finite response-grid changes, not a certification of convergence or experimental agreement",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
