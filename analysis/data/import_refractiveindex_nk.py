#!/usr/bin/env python3
"""
Convert refractiveindex.info tabulated nk YAML files to SBUP3 reference CSV.

Output CSV columns are compatible with analysis/calibrate_polarization_scale.py
and analysis/validate_sbup3.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HC_EV_UM = 1.2398419843320026  # eV*um
TWO_PI = 2.0 * np.pi


def _extract_reference_block(text: str, key: str) -> str | None:
    lines = text.splitlines()
    out: list[str] = []
    capture = False
    prefix = f"{key}: |"
    for line in lines:
        if capture:
            if line and not line.startswith(" "):
                break
            out.append(line.strip())
        elif line.startswith(prefix):
            capture = True
    joined = " ".join(tok for tok in out if tok)
    return joined if joined else None


def _parse_tabulated_nk(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = text.splitlines()
    in_block = False
    found_tabulated = False
    values: list[float] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- type:") and "tabulated nk" in stripped:
            found_tabulated = True
            continue
        if not found_tabulated:
            continue
        if stripped.startswith("data: |"):
            in_block = True
            continue
        if in_block:
            if not line.startswith("        "):
                break
            row = stripped
            if not row:
                continue
            for tok in row.split():
                values.append(float(tok))

    if not values:
        raise RuntimeError("No tabulated nk data found in YAML input.")
    if len(values) % 3 != 0:
        raise RuntimeError(f"Parsed nk data length is not divisible by 3 (got {len(values)}).")

    arr = np.asarray(values, dtype=float).reshape(-1, 3)
    wl_um = arr[:, 0]
    n = arr[:, 1]
    k = arr[:, 2]
    return wl_um, n, k


def _to_rows(wl_um: np.ndarray, n: np.ndarray, k: np.ndarray) -> np.ndarray:
    energy_eV = HC_EV_UM / wl_um
    alpha_m_inv = TWO_PI * 2.0 * k / (wl_um * 1.0e-6)
    eps1 = n * n - k * k
    eps2 = 2.0 * n * k
    out = np.column_stack((energy_eV, wl_um, n, k, alpha_m_inv, eps1, eps2))
    order = np.argsort(out[:, 0])
    return out[order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert refractiveindex.info nk YAML to SBUP3 CSV")
    parser.add_argument("--in-yaml", required=True, help="Input YAML file with tabulated nk data")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument("--source-label", required=True, help="Short source label")
    parser.add_argument("--source-url", default=None, help="Source URL for metadata")
    parser.add_argument("--out-meta", default=None, help="Optional output metadata JSON path")
    args = parser.parse_args()

    in_yaml = Path(args.in_yaml).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_meta = Path(args.out_meta).resolve() if args.out_meta else out_csv.with_suffix(".meta.json")

    text = in_yaml.read_text()
    wl_um, n, k = _parse_tabulated_nk(text)
    rows = _to_rows(wl_um, n, k)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "energy_eV",
                "wavelength_um",
                "n",
                "k",
                "alpha",
                "eps1",
                "eps2",
                "source_label",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row[0]:.9f}",
                    f"{row[1]:.6f}",
                    f"{row[2]:.9f}",
                    f"{row[3]:.9f}",
                    f"{row[4]:.9f}",
                    f"{row[5]:.9f}",
                    f"{row[6]:.9f}",
                    args.source_label,
                ]
            )

    meta = {
        "input_yaml": str(in_yaml),
        "output_csv": str(out_csv),
        "source_label": args.source_label,
        "source_url": args.source_url,
        "references": _extract_reference_block(text, "REFERENCES"),
        "comments": _extract_reference_block(text, "COMMENTS"),
        "n_rows": int(rows.shape[0]),
        "energy_min_eV": float(rows[:, 0].min()),
        "energy_max_eV": float(rows[:, 0].max()),
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"[OK] Wrote CSV: {out_csv}")
    print(f"[OK] Wrote metadata: {out_meta}")
    print(f"rows={rows.shape[0]}, energy=[{rows[:, 0].min():.6f}, {rows[:, 0].max():.6f}] eV")


if __name__ == "__main__":
    main()
