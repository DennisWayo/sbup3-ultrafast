#!/usr/bin/env python3
"""Converge the frozen-polarization Gaussian UPPE result with FFT box width.

The archived SBE traces have a separable transverse Gaussian profile.  Process
temporal frequencies in chunks so a wide spatial box need not allocate an
Nx-by-Nt complex array.  The Fourier propagator and source integral match one
100-um step of ``uppe.uppe_2d.propagate_uppe_2d`` exactly for a fixed source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


C0 = 2.99792458e8
EPS0 = 8.8541878128e-12
MU0 = 1.0 / (EPS0 * C0**2)
HERE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def propagate_central(
    time_s: np.ndarray,
    electric_t: np.ndarray,
    polarization_t: np.ndarray,
    *,
    width_m: float,
    dx_m: float,
    center_nx: int,
    z_m: float,
    waist_m: float,
    chunk: int,
) -> tuple[np.ndarray, dict]:
    nx = round(width_m / dx_m)
    if nx < center_nx or nx % 2 != 0 or center_nx % 2 != 0:
        raise ValueError("Width and central crop must yield even grid sizes")
    nt = len(time_s)
    dt = float(time_s[1] - time_s[0])
    if not np.allclose(np.diff(time_s), dt, rtol=1e-10, atol=0):
        raise ValueError("Time samples must be uniform")
    x_m = (np.arange(nx) - nx // 2) * dx_m
    spatial = np.fft.fft(np.exp(-(x_m / waist_m) ** 2))
    electric_w = np.fft.fft(electric_t)
    polarization_w = np.fft.fft(polarization_t)
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx_m)[:, None]
    omega = 2 * np.pi * np.fft.fftfreq(nt, dt)
    lo = nx // 2 - center_nx // 2
    hi = lo + center_nx
    center_w = np.empty((center_nx, nt), dtype=complex)
    energy_x = np.zeros(nx, dtype=float)

    for first in range(0, nt, chunk):
        last = min(first + chunk, nt)
        om = omega[None, first:last]
        sign = np.sign(om)
        q2 = (om / C0) ** 2 - kx**2
        q = np.sqrt(np.maximum(q2, 0.0))
        decay = np.sqrt(np.maximum(-q2, 0.0))
        phase = np.exp((-1j * sign * q - decay) * z_m)
        source = np.zeros_like(phase)
        propagating = (q > 0) & (om != 0)
        omega_b = np.broadcast_to(om, q.shape)
        sign_b = np.broadcast_to(sign, q.shape)
        source[propagating] = (
            -1j * sign_b[propagating] * MU0 * omega_b[propagating] ** 2
            / (2 * q[propagating]) * z_m
            * np.exp(-0.5j * sign_b[propagating] * q[propagating] * z_m)
            * np.sinc(q[propagating] * z_m / (2 * np.pi))
        )
        if nt % 2 == 0 and first <= nt // 2 < last:
            local = nt // 2 - first
            phase[:, local] = np.exp(-decay[:, local] * z_m) * np.cos(q[:, local] * z_m)
            source[:, local] = 0
        spectral_x = np.fft.ifft(
            spatial[:, None]
            * (phase * electric_w[None, first:last]
               + source * polarization_w[None, first:last]),
            axis=0,
        )
        center_w[:, first:last] = spectral_x[lo:hi]
        energy_x += np.sum(np.abs(spectral_x) ** 2, axis=1) / nt

    central_complex = np.fft.ifft(center_w, axis=1)
    imaginary_ratio = float(
        np.max(np.abs(central_complex.imag))
        / max(np.max(np.abs(central_complex.real)), 1e-300)
    )
    if imaginary_ratio > 1e-10:
        raise RuntimeError(f"Real-field conjugate symmetry failed: {imaginary_ratio}")
    edge_mask = np.abs(x_m) >= 0.4 * width_m
    metrics = {
        "width_um": width_m * 1e6,
        "nx": nx,
        "central_rms_v_per_m": float(np.sqrt(np.mean(central_complex.real**2))),
        "outer_20_percent_energy_fraction": float(energy_x[edge_mask].sum() / energy_x.sum()),
        "imaginary_to_real_max_ratio": imaginary_ratio,
    }
    return central_complex.real, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbe-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--widths-um", nargs="+", type=float, default=[10, 20, 40, 80, 160])
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--verify-direct", action="store_true",
                        help="Compare 10-um chunked result with the original full-grid solver")
    args = parser.parse_args()
    sbe = args.sbe_dir.resolve()
    paths = {key: sbe / filename for key, filename in {
        "time_s": "time_s.npy", "E_t": "E_t.npy", "polarization": "polarization.npy"
    }.items()}
    t = np.asarray(np.load(paths["time_s"]), dtype=float)
    e = np.asarray(np.load(paths["E_t"]), dtype=float)
    p = np.asarray(np.load(paths["polarization"]), dtype=complex).real
    if len(t) != 10000 or e.shape != t.shape or p.shape != t.shape:
        raise ValueError("Expected the archived 10,000-sample GaAs SBE traces")
    previous = None
    cases = []
    direct_relative_rms = None
    for width_um in args.widths_um:
        center, metrics = propagate_central(
            t, e, p, width_m=width_um * 1e-6,
            dx_m=10e-6 / 256, center_nx=256,
            z_m=100e-6, waist_m=2e-6, chunk=args.chunk,
        )
        if previous is not None:
            metrics["central_relative_rms_vs_previous"] = float(
                np.linalg.norm(center - previous) / np.linalg.norm(center)
            )
        if args.verify_direct and width_um == 10.0:
            sys.path.insert(0, str(HERE))
            from uppe.uppe_2d import propagate_uppe_2d

            x = (np.arange(256) - 128) * (10e-6 / 256)
            g = np.exp(-(x / 2e-6) ** 2)
            direct = propagate_uppe_2d(
                x, t, g[:, None] * e[None, :], g[:, None] * p[None, :],
                100e-6, 1, background_index=1.0, polarization_kind="full",
            )
            direct_relative_rms = float(np.linalg.norm(center - direct) /
                                        np.linalg.norm(direct))
            print(json.dumps({"direct_solver_relative_rms": direct_relative_rms}), flush=True)
        cases.append(metrics)
        print(json.dumps(metrics), flush=True)
        previous = center
    report = {
        "model": "Sequential 100-um vacuum-background UPPE with frozen SBE full polarization and 2-um Gaussian transverse profile",
        "meaning": "Periodic-box sensitivity of this restricted model; not a self-consistent optical prediction",
        "central_crop_um": 10.0,
        "dx_nm": 1e9 * 10e-6 / 256,
        "input_sha256": {key: sha256(path) for key, path in paths.items()},
        "solver_sha256": sha256(HERE / "uppe/uppe_2d.py"),
        "direct_solver_relative_rms": direct_relative_rms,
        "cases": cases,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
