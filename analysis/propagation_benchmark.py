"""Reproducible manufactured-field benchmark for the corrected UPPE operator."""

import hashlib
import json
from pathlib import Path

import numpy as np

from uppe.uppe_2d import c0, eps0, propagate_uppe_2d


def main():
    wavelength = 1.55e-6
    period = wavelength / c0
    time = np.arange(256) * period / 64
    x = np.arange(8) * 0.5e-6
    input_field = np.broadcast_to(np.cos(2 * np.pi * time / period),
                                  (len(x), len(time))).copy()
    zero = np.zeros_like(input_field)
    lossless = propagate_uppe_2d(x, time, input_field, zero,
                                 wavelength / (4 * 3.6), 1,
                                 background_index=3.6)
    expected_lossless = np.broadcast_to(np.cos(2 * np.pi * time / period - np.pi / 2),
                                        input_field.shape)

    susceptibility = 0.02
    source = eps0 * susceptibility * input_field
    distance = wavelength / 8
    sourced_one = propagate_uppe_2d(x, time, input_field, source, distance, 1,
                                    background_index=1.0, polarization_kind="full")
    sourced_16 = propagate_uppe_2d(x, time, input_field, source, distance / 16, 16,
                                   background_index=1.0, polarization_kind="full")
    phase = 2 * np.pi * distance / wavelength
    expected_source = ((1 + susceptibility / 2)
                       * np.cos(2 * np.pi * time / period - phase)
                       - susceptibility / 2 * np.cos(2 * np.pi * time / period))

    x2 = np.arange(64) * 0.1e-6
    kx = 2 * np.pi / (len(x2) * (x2[1] - x2[0]))
    transverse = np.cos(kx * x2)[:, None]
    field_2d = transverse * input_field[0][None, :]
    q = np.sqrt((3.6 * 2 * np.pi / wavelength)**2 - kx**2)
    distance_2d = 1.2e-6
    out_2d = propagate_uppe_2d(x2, time, field_2d, np.zeros_like(field_2d),
                               distance_2d, 1, background_index=3.6)
    expected_2d = transverse * np.cos(2 * np.pi * time[None, :] / period
                                      - q * distance_2d)

    solver_file = Path(__file__).resolve().parents[1] / "uppe" / "uppe_2d.py"
    report = {
        "solver_sha256": hashlib.sha256(solver_file.read_bytes()).hexdigest(),
        "input": {"wavelength_m": wavelength, "samples_per_period": 64,
                  "time_samples": 256, "x_samples": 8},
        "lossless": {
            "index": 3.6,
            "distance_m": wavelength / (4 * 3.6),
            "rms_ratio": float(np.sqrt(np.mean(lossless**2)) /
                               np.sqrt(np.mean(input_field**2))),
            "max_abs_error_v_per_m_for_unit_input": float(
                np.max(np.abs(lossless - expected_lossless))),
        },
        "prescribed_source": {
            "background_index": 1.0,
            "source_polarization": "eps0 * 0.02 * cos(omega*t) C/m^2",
            "distance_m": distance,
            "one_step_max_abs_error": float(
                np.max(np.abs(sourced_one - expected_source[None, :]))),
            "one_vs_16_max_abs_difference": float(
                np.max(np.abs(sourced_one - sourced_16))),
        },
        "transverse_mode": {
            "mode_kx_rad_per_m": float(kx),
            "distance_m": distance_2d,
            "max_abs_error_v_per_m_for_unit_input": float(
                np.max(np.abs(out_2d - expected_2d))),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
