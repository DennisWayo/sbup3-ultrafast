#!/usr/bin/env python3
"""Propagate a Gaussian GaAs pulse with a local SBE update at every z step.

This is a reduced-model calculation: the dielectric-derived transition weights
and calibrated density are effective parameters. It is not a converged or
ab-initio prediction of GaAs pulse transmission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.gaas_linear_material_window import linear_sbe_chi
from sbe.local_response import local_polarization
from uppe.dispersive_residual import DispersiveResidualStep


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_sbe_uppe(
    x_m: np.ndarray,
    time_s: np.ndarray,
    initial_xt: np.ndarray,
    omega_rad_s: np.ndarray,
    dipole_cm: np.ndarray,
    weights: np.ndarray,
    *,
    density_m3: float,
    polarization_scale: float,
    gamma_s: float,
    chi_positive: np.ndarray,
    dz_m: float,
    z_steps: int,
    max_field_vpm: float,
) -> tuple[np.ndarray, dict]:
    """Advance a dispersive UPPE with a local temporal Bloch response.

    The analytic linear SBE response defines the background index. A predictor
    supplies the nonlinear residual polarization at the end of the step; its
    average with the beginning residual enters the source integral. The
    corrected field becomes the next step's local SBE drive.
    Population is reset only at the start of each local time trace, so this
    model has no carrier transport or memory between distinct z locations.
    """
    field = np.asarray(initial_xt, dtype=float).copy()
    if not np.isfinite(field).all() or max_field_vpm <= 0:
        raise ValueError("Initial field and maximum field must be finite")
    if z_steps < 0 or dz_m <= 0:
        raise ValueError("Require a positive step and nonnegative step count")

    def response(e: np.ndarray) -> tuple[np.ndarray, float]:
        return local_polarization(
            e, time_s, omega_rad_s, dipole_cm, weights,
            density_m3=density_m3, polarization_scale=polarization_scale,
            gamma_s=gamma_s,
        )

    propagator = DispersiveResidualStep(x_m, time_s, chi_positive, dz_m)

    def residual(e: np.ndarray) -> tuple[np.ndarray, float]:
        nonlinear_p, peak = response(e)
        return nonlinear_p - propagator.linear_polarization(e), peak

    polarization, peak_population = residual(field)
    initial_polarization = polarization.copy()
    largest_field = float(np.max(np.abs(field)))
    step_diagnostics: list[dict] = []
    for step in range(z_steps):
        predicted = propagator.advance(field, polarization)
        predicted_p, pop_pred = residual(predicted)
        corrected = propagator.advance(field, 0.5 * (polarization + predicted_p))
        next_p, pop_next = residual(corrected)
        field_peak = float(np.max(np.abs(corrected)))
        if not np.isfinite(corrected).all() or field_peak > max_field_vpm:
            raise FloatingPointError(
                f"Field diverged or exceeded the declared bound at step {step + 1}"
            )
        largest_field = max(largest_field, field_peak)
        peak_population = max(peak_population, pop_pred, pop_next)
        step_diagnostics.append({
            "step": step + 1,
            "z_um": (step + 1) * dz_m * 1e6,
            "peak_field_vpm": field_peak,
            "relative_polarization_change": float(
                np.linalg.norm(next_p - polarization)
                / max(np.linalg.norm(next_p), 1e-300)
            ),
        })
        field = corrected
        polarization = next_p
    frozen = np.asarray(initial_xt, dtype=float).copy()
    for _ in range(z_steps):
        frozen = propagator.advance(frozen, initial_polarization)
    report = {
        "model": "Dispersive linear reduced-SBE background plus local nonlinear SBE residual recomputed at each z step",
        "source_rule": "Predictor-corrector average of beginning and predicted end-step residual polarization",
        "z_um": dz_m * z_steps * 1e6,
        "dz_um": dz_m * 1e6,
        "z_steps": z_steps,
        "peak_field_vpm": largest_field,
        "peak_population": peak_population,
        "relative_rms_vs_frozen_residual": float(
            np.linalg.norm(field - frozen) / max(np.linalg.norm(field), 1e-300)
        ),
        "step_diagnostics": step_diagnostics,
    }
    return field, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbe-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--width-um", type=float, default=10.0)
    parser.add_argument("--dx-nm", type=float, default=156.25)
    parser.add_argument("--waist-um", type=float, default=2.0)
    parser.add_argument("--z-um", type=float, default=2.0)
    parser.add_argument("--dz-um", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--tail-fs", type=float, default=150.0,
                        help="Zero-field tail for the SBE polarization to decay before the temporal FFT wraps")
    parser.add_argument("--gamma-s", type=float, default=5.0e13)
    parser.add_argument("--max-field-vpm", type=float, default=1.0e9)
    args = parser.parse_args()
    if (args.stride < 1 or args.width_um <= 0 or args.dx_nm <= 0
            or args.waist_um <= 0 or args.tail_fs < 0):
        raise ValueError("Invalid grid or pulse arguments")
    steps = round(args.z_um / args.dz_um)
    if steps < 1 or not np.isclose(steps * args.dz_um, args.z_um):
        raise ValueError("z-um must be a positive multiple of dz-um")
    nx = round(args.width_um * 1e3 / args.dx_nm)
    if nx < 4 or nx % 2 or not np.isclose(nx * args.dx_nm, args.width_um * 1e3):
        raise ValueError("Width/dx must give an even transverse grid")

    sbe = args.sbe_dir.resolve()
    paths = {key: sbe / name for key, name in {
        "time": "time_s.npy", "field": "E_t.npy",
        "omega": "transition_omega_rad_s_used.npy",
        "dipole": "transition_d_eff_Cm_used.npy",
        "weights": "transition_weight_used.npy",
        "model": "transition_model.json",
    }.items()}
    time = np.load(paths["time"])[::args.stride]
    drive = np.load(paths["field"])[::args.stride]
    dt = float(time[1] - time[0])
    tail_n = round(args.tail_fs * 1e-15 / dt)
    if tail_n:
        time = np.concatenate((time, time[-1] + dt * np.arange(1, tail_n + 1)))
        drive = np.pad(drive, (0, tail_n))
    omega = np.load(paths["omega"])
    dipole = np.load(paths["dipole"])
    weight = np.load(paths["weights"])
    meta = json.loads(paths["model"].read_text())
    chi = linear_sbe_chi(
        2 * np.pi * np.fft.rfftfreq(len(time), dt), omega, dipole, weight,
        density_m3=float(meta["density_m3"]),
        polarization_scale=float(meta["polarization_scale"]),
        gamma_s=args.gamma_s,
    )
    x = (np.arange(nx) - nx // 2) * args.dx_nm * 1e-9
    gaussian = np.exp(-(x / (args.waist_um * 1e-6)) ** 2)
    initial = gaussian[:, None] * drive[None, :]
    field, report = local_sbe_uppe(
        x, time, initial, omega, dipole, weight,
        density_m3=float(meta["density_m3"]),
        polarization_scale=float(meta["polarization_scale"]),
        gamma_s=args.gamma_s,
        chi_positive=chi,
        dz_m=args.dz_um * 1e-6,
        z_steps=steps,
        max_field_vpm=args.max_field_vpm,
    )
    report.update({
        "width_um": args.width_um,
        "dx_nm": args.dx_nm,
        "waist_um": args.waist_um,
        "time_stride": args.stride,
        "zero_field_tail_fs": tail_n * dt * 1e15,
        "time_samples": len(time),
        "density_m3": float(meta["density_m3"]),
        "polarization_scale": float(meta["polarization_scale"]),
        "gamma_s": args.gamma_s,
        "input_sha256": {key: sha256(value) for key, value in paths.items()},
    })
    output = args.out_prefix.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), x_m=x, time_s=time, field_xt=field)
    report["output_npz_sha256"] = sha256(output.with_suffix(".npz"))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "step_diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
