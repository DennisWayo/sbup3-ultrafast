#!/usr/bin/env python3
"""
Build reduced SBE transition arrays from periodic DFT dielectric-reference data.

This script converts peaks of epsilon2/absorption into a compact transition list:
  - transitions_E_eV.npy
  - transitions_f.npy
  - transitions_meta.json

The outputs are consumed by `sbe/sbe_rk4.py` in strict ground-truth mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent

DEFAULT_IN_CSV = BASE / "analysis" / "data" / "gaas_dielectric_reference.csv"
DEFAULT_OUT_E = HERE / "transitions_E_eV.npy"
DEFAULT_OUT_F = HERE / "transitions_f.npy"
DEFAULT_OUT_META = HERE / "transitions_meta.json"


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV missing headers: {path}")
        keys = {_normalize_key(k): k for k in reader.fieldnames}

        def pick(*names: str) -> str | None:
            for n in names:
                if n in keys:
                    return keys[n]
            return None

        col_energy = pick("energyev", "energy", "hvev", "e")
        if col_energy is None:
            raise ValueError(f"CSV missing energy column: {path}")

        # Prefer imaginary dielectric response, then absorption.
        col_strength = (
            pick("eps2lf", "eps2nlf", "eps2")
            or pick("alphaminv", "alpha", "absorption", "absorptioncoeff")
        )
        if col_strength is None:
            raise ValueError(
                "CSV must include eps2_lf/eps2_nlf (preferred) or alpha/alpha_m_inv."
            )

        energy: list[float] = []
        strength: list[float] = []
        for row in reader:
            try:
                e = float(row[col_energy])
                s = float(row[col_strength])
            except Exception:
                continue
            if np.isfinite(e) and np.isfinite(s):
                energy.append(e)
                strength.append(s)

    if not energy:
        raise ValueError(f"No numeric rows in {path}")

    e_arr = np.asarray(energy, dtype=float)
    s_arr = np.asarray(strength, dtype=float)
    order = np.argsort(e_arr)
    e_arr = e_arr[order]
    s_arr = s_arr[order]
    return e_arr, s_arr, col_strength


def _pick_peaks(
    energy_eV: np.ndarray,
    strength: np.ndarray,
    *,
    e_min: float,
    e_max: float,
    min_rel_height: float,
    min_sep_eV: float,
    min_count: int,
    max_count: int,
) -> np.ndarray:
    mask = np.isfinite(energy_eV) & np.isfinite(strength)
    mask &= energy_eV >= e_min
    mask &= energy_eV <= e_max
    e = energy_eV[mask]
    y = strength[mask]
    if len(e) < 3:
        raise RuntimeError("Not enough samples for peak extraction.")

    # Physical response strength should be non-negative for our use here.
    y = np.maximum(y, 0.0)
    ymax = float(np.max(y))
    if ymax <= 0.0:
        raise RuntimeError("Reference response is non-positive in selected window.")
    ythr = max(0.0, float(min_rel_height)) * ymax

    local = []
    for i in range(1, len(y) - 1):
        if y[i] < ythr:
            continue
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            local.append(i)

    if not local:
        local = [int(np.argmax(y))]

    local_arr = np.asarray(local, dtype=int)
    # Greedy separation by descending peak height.
    order = np.argsort(y[local_arr])[::-1]
    chosen: list[int] = []
    for idx in local_arr[order]:
        ei = e[idx]
        if all(abs(ei - e[j]) >= min_sep_eV for j in chosen):
            chosen.append(int(idx))
        if max_count > 0 and len(chosen) >= max_count:
            break

    target_count = max(int(min_count), 1)
    if max_count > 0:
        target_count = min(target_count, int(max_count))
    if len(chosen) < target_count:
        # Backfill from strongest spectral points to avoid under-resolved
        # reduced models when local maxima are too sparse.
        all_order = np.argsort(y)[::-1]
        for idx in all_order:
            i = int(idx)
            ei = e[i]
            if any(abs(ei - e[j]) < min_sep_eV for j in chosen):
                continue
            chosen.append(i)
            if len(chosen) >= target_count:
                break

    if not chosen:
        raise RuntimeError("Peak extraction produced no transitions.")

    chosen_arr = np.asarray(sorted(chosen, key=lambda j: e[j]), dtype=int)
    # Map back to original array indices
    original_idx = np.nonzero(mask)[0][chosen_arr]
    return original_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Build transition arrays from dielectric reference CSV")
    parser.add_argument("--in-csv", default=str(DEFAULT_IN_CSV), help="Input dielectric reference CSV")
    parser.add_argument("--out-e", default=str(DEFAULT_OUT_E), help="Output transitions_E_eV.npy")
    parser.add_argument("--out-f", default=str(DEFAULT_OUT_F), help="Output transitions_f.npy")
    parser.add_argument("--out-meta", default=str(DEFAULT_OUT_META), help="Output transitions metadata JSON")
    parser.add_argument("--e-min", type=float, default=0.0, help="Minimum transition energy [eV]")
    parser.add_argument("--e-max", type=float, default=3.0, help="Maximum transition energy [eV]")
    parser.add_argument(
        "--min-rel-height",
        type=float,
        default=0.01,
        help="Minimum relative peak height vs max in selected window",
    )
    parser.add_argument(
        "--min-sep-ev",
        type=float,
        default=0.02,
        help="Minimum peak-to-peak separation [eV]",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=16,
        help="Minimum number of transitions to emit (backfilled by strongest points).",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=24,
        help="Maximum number of transitions to emit (0 means unlimited)",
    )
    parser.add_argument(
        "--sum-f",
        type=float,
        default=1.0,
        help="Normalize output oscillator strengths to this total.",
    )
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_e = Path(args.out_e)
    out_f = Path(args.out_f)
    out_meta = Path(args.out_meta)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    energy_eV, strength_raw, strength_col = _load_reference(in_csv)
    peaks_idx = _pick_peaks(
        energy_eV,
        strength_raw,
        e_min=float(args.e_min),
        e_max=float(args.e_max),
        min_rel_height=float(args.min_rel_height),
        min_sep_eV=max(float(args.min_sep_ev), 0.0),
        min_count=max(int(args.min_count), 1),
        max_count=max(int(args.max_count), 0),
    )

    trans_e = np.asarray(energy_eV[peaks_idx], dtype=float)
    peak_strength = np.asarray(np.maximum(strength_raw[peaks_idx], 0.0), dtype=float)
    if float(np.sum(peak_strength)) <= 0.0:
        peak_strength = np.ones_like(peak_strength, dtype=float)
    trans_f = peak_strength / float(np.sum(peak_strength))
    trans_f *= max(float(args.sum_f), 1e-30)

    out_e.parent.mkdir(parents=True, exist_ok=True)
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_e, trans_e)
    np.save(out_f, trans_f)

    meta: dict[str, Any] = {
        "source_csv": str(in_csv),
        "strength_column": strength_col,
        "energy_window_eV": [float(args.e_min), float(args.e_max)],
        "min_rel_height": float(args.min_rel_height),
        "min_sep_eV": float(args.min_sep_ev),
        "min_count": int(args.min_count),
        "max_count": int(args.max_count),
        "sum_f": float(args.sum_f),
        "n_transitions": int(len(trans_e)),
        "energy_min_eV": float(np.min(trans_e)) if len(trans_e) else None,
        "energy_max_eV": float(np.max(trans_e)) if len(trans_e) else None,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    print("[OK] Built transition arrays from dielectric reference")
    print(f"  source: {in_csv}")
    print(f"  column: {strength_col}")
    print(f"  n_transitions: {len(trans_e)}")
    print(f"  transitions_E_eV: {out_e}")
    print(f"  transitions_f: {out_f}")
    print(f"  metadata: {out_meta}")


if __name__ == "__main__":
    main()
