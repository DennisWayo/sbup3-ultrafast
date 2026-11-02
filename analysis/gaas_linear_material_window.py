#!/usr/bin/env python3
"""Converge a Gaussian pulse in the local linear limit of the reduced SBE.

The polarization follows the propagated field through its analytic causal
susceptibility.  This removes the frozen-polarization assumption but retains
the effective transition and density parameters of the archived SBE model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HBAR = 1.054571817e-34
EPS0 = 8.8541878128e-12
C0 = 299792458.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_sbe_chi(
    omega: np.ndarray,
    transition_omega: np.ndarray,
    dipole_cm: np.ndarray,
    weights: np.ndarray,
    *,
    density_m3: float,
    polarization_scale: float,
    gamma_s: float,
) -> np.ndarray:
    """Causal susceptibility for the implemented real-polarization SBE ODE.

    The FFT convention is e^(+i omega t), so a passive response has Im chi < 0.
    Population is linearized at its ground-state value.
    """
    omega = np.asarray(omega, dtype=float)
    w0 = np.asarray(transition_omega, dtype=float).reshape(-1)
    d = np.asarray(dipole_cm, dtype=float).reshape(-1)
    weight = np.asarray(weights, dtype=float).reshape(-1)
    if w0.shape != d.shape or w0.shape != weight.shape or len(w0) < 1:
        raise ValueError("Transition arrays must have equal nonzero length")
    if np.any(omega < 0) or np.any(w0 <= 0) or np.any(d < 0) or np.any(weight < 0):
        raise ValueError("Frequencies and transition parameters are invalid")
    if not np.isclose(weight.sum(), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("Transition weights must sum to one")
    if gamma_s <= 0 or density_m3 <= 0 or polarization_scale <= 0:
        raise ValueError("Dephasing, density, and scale must be positive")
    pos = 1j / (gamma_s + 1j * (w0[None, :] + omega[:, None]))
    neg = 1j / (gamma_s + 1j * (omega[:, None] - w0[None, :]))
    chi = (polarization_scale * density_m3 / (2 * EPS0 * HBAR)
           * np.sum(weight[None, :] * d[None, :] ** 2 * (pos - neg), axis=1))
    if np.any(chi.imag > 1e-10 * np.maximum(1.0, np.abs(chi))):
        raise RuntimeError("Nonpassive susceptibility from local SBE parameters")
    return chi


def propagate_central(
    time_s: np.ndarray,
    electric_t: np.ndarray,
    chi: np.ndarray,
    *,
    width_m: float,
    dx_m: float,
    center_nx: int,
    z_m: float,
    waist_m: float,
    chunk: int,
) -> tuple[np.ndarray, dict]:
    nt = len(time_s)
    nx = round(width_m / dx_m)
    if (nx < center_nx or nx % 2 or center_nx % 2 or chunk < 1
            or len(electric_t) != nt or len(chi) != nt // 2 + 1):
        raise ValueError("Incompatible spatial or temporal grid")
    dt = float(time_s[1] - time_s[0])
    if dt <= 0 or not np.allclose(np.diff(time_s), dt, rtol=1e-10, atol=0):
        raise ValueError("Time grid must be uniform")
    x = (np.arange(nx) - nx // 2) * dx_m
    spatial = np.fft.fft(np.exp(-(x / waist_m) ** 2))
    electric_w = np.fft.rfft(electric_t)
    omega = 2 * np.pi * np.fft.rfftfreq(nt, dt)
    index = np.sqrt(1.0 + chi)
    if np.any(index.imag > 1e-10) or np.any(index.real <= 0):
        raise RuntimeError("Optical index is not passive with a positive real branch")
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx_m)[:, None]
    lo = nx // 2 - center_nx // 2
    hi = lo + center_nx
    center_w = np.empty((center_nx, len(omega)), dtype=complex)
    energy_x = np.zeros(nx, dtype=float)

    for first in range(0, len(omega), chunk):
        last = min(first + chunk, len(omega))
        om = omega[None, first:last]
        n = index[None, first:last]
        q2 = (n * om / C0) ** 2 - kx**2
        q = np.sqrt(q2)
        q = np.where(q.imag > 0, -q, q)
        transfer = np.exp(-1j * q * z_m)
        if first == 0:
            transfer[:, 0] = np.exp(-np.abs(kx[:, 0]) * z_m)
        if nt % 2 == 0 and first <= nt // 2 < last:
            local = nt // 2 - first
            transfer[:, local] = (np.exp(q[:, local].imag * z_m)
                                  * np.cos(q[:, local].real * z_m))
        output_wx = np.fft.ifft(
            spatial[:, None] * electric_w[None, first:last] * transfer,
            axis=0,
        )
        center_w[:, first:last] = output_wx[lo:hi]
        factor = 1.0 if first == 0 else 2.0
        weights = np.full(last - first, factor)
        if first == 0:
            weights[0] = 1.0
        if nt % 2 == 0 and first <= nt // 2 < last:
            weights[nt // 2 - first] = 1.0
        energy_x += np.sum(weights[None, :] * np.abs(output_wx) ** 2, axis=1) / nt

    central = np.fft.irfft(center_w, n=nt, axis=1)
    edge = np.abs(x) >= 0.4 * width_m
    metrics = {
        "width_um": width_m * 1e6,
        "nx": nx,
        "central_rms_vpm": float(np.sqrt(np.mean(central**2))),
        "outer_20_percent_energy_fraction": float(energy_x[edge].sum() / energy_x.sum()),
    }
    return central, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbe-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--widths-um", type=float, nargs="+", default=[10, 20, 40, 80, 160, 320, 640])
    parser.add_argument("--dx-nm", type=float, default=39.0625)
    parser.add_argument("--waist-um", type=float, default=2.0)
    parser.add_argument("--z-um", type=float, default=100.0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--tail-fs", type=float, default=150.0)
    parser.add_argument("--gamma-s", type=float, default=5.0e13)
    parser.add_argument("--chunk", type=int, default=32)
    args = parser.parse_args()
    sbe = args.sbe_dir.resolve()
    paths = {key: sbe / name for key, name in {
        "time": "time_s.npy", "field": "E_t.npy",
        "omega": "transition_omega_rad_s_used.npy",
        "dipole": "transition_d_eff_Cm_used.npy",
        "weights": "transition_weight_used.npy",
        "model": "transition_model.json",
    }.items()}
    time = np.load(paths["time"])[::args.stride]
    electric = np.load(paths["field"])[::args.stride]
    dt = float(time[1] - time[0])
    tail_n = round(args.tail_fs * 1e-15 / dt)
    if tail_n:
        time = np.concatenate((time, time[-1] + dt * np.arange(1, tail_n + 1)))
        electric = np.pad(electric, (0, tail_n))
    meta = json.loads(paths["model"].read_text())
    freq = 2 * np.pi * np.fft.rfftfreq(len(time), dt)
    chi = linear_sbe_chi(
        freq, np.load(paths["omega"]), np.load(paths["dipole"]),
        np.load(paths["weights"]), density_m3=float(meta["density_m3"]),
        polarization_scale=float(meta["polarization_scale"]), gamma_s=args.gamma_s,
    )
    previous = None
    cases = []
    for width in args.widths_um:
        central, metrics = propagate_central(
            time, electric, chi, width_m=width * 1e-6,
            dx_m=args.dx_nm * 1e-9, center_nx=round(10e3 / args.dx_nm),
            z_m=args.z_um * 1e-6, waist_m=args.waist_um * 1e-6,
            chunk=args.chunk,
        )
        if previous is not None:
            metrics["central_relative_rms_vs_previous"] = float(
                np.linalg.norm(central - previous) / np.linalg.norm(central)
            )
        cases.append(metrics)
        print(json.dumps(metrics), flush=True)
        previous = central
    report = {
        "model": "Exact forward Fourier propagation with analytic local linear SBE susceptibility; effective oscillator parameters",
        "meaning": "Self-consistent linear constitutive response, not nonlinear field-SBE feedback or first-principles prediction",
        "z_um": args.z_um,
        "dx_nm": args.dx_nm,
        "waist_um": args.waist_um,
        "time_stride": args.stride,
        "time_samples": len(time),
        "zero_field_tail_fs": tail_n * dt * 1e15,
        "gamma_s": args.gamma_s,
        "chi_imag_max": float(np.max(chi.imag)),
        "input_sha256": {key: sha256(path) for key, path in paths.items()},
        "cases": cases,
    }
    out = args.out_json.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
