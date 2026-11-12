#!/usr/bin/env python3
"""Compare saved local material-field solutions without rerunning propagation.

This is a numerical diagnostic of the calibrated reduced model. It does not
establish dielectric convergence or experimental GaAs agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, expected_sha256: str) -> tuple[np.ndarray, ...]:
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    with np.load(path) as saved:
        x, t, field = (saved[key] for key in ("x_m", "time_s", "field_xt"))
    if (field.shape != (len(x), len(t)) or not np.isfinite(field).all()
            or not np.isfinite(x).all() or not np.isfinite(t).all()):
        raise ValueError(f"Invalid coordinates or field: {path}")
    return x, t, field


def compare(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...],
            center_um: float, max_lag: int) -> dict:
    xa, ta, a = left
    xb, tb, b = right
    if (a.shape != b.shape
            or not np.allclose(xa, xb, atol=1e-24, rtol=1e-12)
            or not np.allclose(ta, tb, atol=1e-27, rtol=1e-12)):
        raise ValueError("Saved comparison grids differ")
    center = np.abs(xa) <= center_um / 1e6
    if not center.any():
        raise ValueError("No samples in central spatial region")
    a, b = a[center], b[center]
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0:
        raise ValueError("Reference field has zero norm")
    raw = float(np.linalg.norm(a - b) / norm_b)
    best_lag, best_error = 0, raw
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[:, :lag], b[:, -lag:]
        elif lag > 0:
            aa, bb = a[:, lag:], b[:, :-lag]
        else:
            aa, bb = a, b
        error = float(np.linalg.norm(aa - bb) / np.linalg.norm(bb))
        if error < best_error:
            best_lag, best_error = lag, error
    spectral_a = np.fft.rfft(a, axis=1)
    spectral_b = np.fft.rfft(b, axis=1)
    dt_fs = float(np.median(np.diff(ta))) * 1e15
    return {
        "raw_relative_rms": raw,
        "spectral_magnitude_relative_rms": float(
            np.linalg.norm(np.abs(spectral_a) - np.abs(spectral_b))
            / np.linalg.norm(np.abs(spectral_b))),
        "best_integer_time_lag_samples": best_lag,
        "best_integer_time_lag_fs": best_lag * dt_fs,
        "relative_rms_after_lag": best_error,
        "field_energy_ratio": float(np.sum(a * a) / np.sum(b * b)),
        "peak_magnitude_ratio": float(np.max(np.abs(a)) / np.max(np.abs(b))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("coarse", "medium", "fine"):
        parser.add_argument(f"--{label}", type=Path, required=True)
        parser.add_argument(f"--{label}-sha256", required=True)
    parser.add_argument("--center-um", type=float, default=5.0)
    parser.add_argument("--max-lag-samples", type=int, default=40)
    args = parser.parse_args()
    if args.center_um <= 0 or args.max_lag_samples < 0:
        parser.error("Require a positive center width and nonnegative lag")
    arrays = {label: read(getattr(args, label), getattr(args, f"{label}_sha256"))
              for label in ("coarse", "medium", "fine")}
    report = {
        "scope": "Saved-field numerical comparison of the calibrated local material-field model",
        "central_region_um": [-args.center_um, args.center_um],
        "source_files": {
            label: {"path": str(getattr(args, label)),
                    "sha256": getattr(args, f"{label}_sha256")}
            for label in arrays},
        "comparisons": {
            f"{a}_vs_{b}": compare(arrays[a], arrays[b], args.center_um,
                                     args.max_lag_samples)
            for a, b in (("coarse", "medium"), ("medium", "fine"))},
        "interpretation_limit": "Numerical diagnosis only; no physical GaAs validation",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
