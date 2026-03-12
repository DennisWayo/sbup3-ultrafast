#!/usr/bin/env python3
"""
Export TDDFT excitations from a GPAW log to CSV.

Parses either:
  - LrTDDFT "om=... [eV] |me|=..." lines, or
  - KSS/RPA "eji=... [eV] (...)" transition vectors.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def parse_tddft_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pattern_om = re.compile(r"om=([0-9.+-Ee]+)\[eV\]\s+\|me\|=([0-9.+-Ee]+)")
    pattern_kss = re.compile(r"eji=([0-9.+-Ee]+)\[eV\]\s+\(([^)]+)\)")

    energies: list[float] = []
    osc: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern_om.search(line)
        if match:
            energies.append(float(match.group(1)))
            osc.append(float(match.group(2)))
            continue
        match_kss = pattern_kss.search(line)
        if match_kss:
            try:
                energy = float(match_kss.group(1))
                vec = [float(v.strip()) for v in match_kss.group(2).split(",")]
                if len(vec) == 3:
                    energies.append(energy)
                    osc.append(float(np.linalg.norm(vec)))
            except Exception:
                continue

    if not energies:
        raise ValueError(f"No TDDFT transitions found in {path}")

    e = np.asarray(energies, dtype=float)
    f = np.asarray(osc, dtype=float)
    idx = np.argsort(e)
    return e[idx], f[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TDDFT excitations to CSV")
    parser.add_argument(
        "--log",
        default=None,
        help="Path to TDDFT log (defaults to dft/gaas_lrtddft.log)",
    )
    parser.add_argument(
        "--out",
        default="analysis/data/gaas_tddft_reference.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--osc-min",
        type=float,
        default=0.0,
        help="Drop entries with oscillator strength below this threshold",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    if args.log:
        log_path = Path(args.log)
    else:
        log_path = base / "dft" / "gaas_lrtddft.log"

    if not log_path.exists():
        raise FileNotFoundError(f"Missing log: {log_path}")

    e, f = parse_tddft_log(log_path)
    if args.osc_min > 0:
        mask = f >= args.osc_min
        e = e[mask]
        f = f[mask]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["energy_eV", "osc"])
        writer.writerows(zip(e.tolist(), f.tolist()))

    print(f"Wrote {len(e)} excitations to {out_path}")


if __name__ == "__main__":
    main()
