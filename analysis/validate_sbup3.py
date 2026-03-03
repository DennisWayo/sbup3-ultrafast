#!/usr/bin/env python3
"""
SBUP3 validation script.

Computes stability and consistency metrics from SBE/UPPE outputs.
Optionally runs parameter sweeps for convergence studies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


HBAR = 1.054571817e-34  # J*s
E0 = 1.602176634e-19    # C


def _git_info(base: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(base),
            check=True,
            capture_output=True,
            text=True,
        )
        info["commit"] = out.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(base),
            check=True,
            capture_output=True,
            text=True,
        )
        info["dirty"] = bool(status.stdout.strip())
    except Exception:
        return info
    return info


def _parse_list(value: str, cast):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("\n[CMD]", " ".join(cmd))
    print("[CWD]", str(cwd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _load_outputs(base: Path) -> dict[str, Any]:
    sbe = base / "sbe"
    uppe = base / "uppe" / "outputs"
    data = {
        "time_s": np.load(sbe / "time_s.npy"),
        "E_t": np.load(sbe / "E_t.npy"),
        "P_t": np.load(sbe / "polarization.npy"),
        "n_t": np.load(sbe / "population.npy"),
        "omega": np.load(sbe / "omega_rad_s.npy"),
        "chi_r": np.load(sbe / "chi_eff_real.npy"),
        "chi_i": np.load(sbe / "chi_eff_imag.npy"),
        "alpha": np.load(sbe / "alpha_rel.npy"),
        "n_rel": np.load(sbe / "n_rel.npy"),
        "E_xt": np.load(uppe / "E_xt_out.npy"),
        "x_m": np.load(uppe / "x_m.npy"),
        "time_s_uppe": np.load(uppe / "time_s.npy"),
        "E_t_feedback": (sbe / "E_t_feedback.npy"),
    }
    return data


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def _spectral_moments(freq_eV: np.ndarray, spectrum: np.ndarray) -> tuple[float, float]:
    s = spectrum / np.sum(spectrum)
    mu = float(np.sum(freq_eV * s))
    sigma = float(np.sqrt(np.sum((freq_eV - mu) ** 2 * s)))
    return mu, sigma


def compute_metrics(base: Path) -> dict[str, Any]:
    data = _load_outputs(base)

    time_s = data["time_s"]
    E_t = data["E_t"]
    omega = data["omega"]
    E_xt = data["E_xt"]
    x_m = data["x_m"]
    time_s_uppe = data["time_s_uppe"]

    dt_in = float(time_s[1] - time_s[0])
    dt_out = float(time_s_uppe[1] - time_s_uppe[0])
    dx = float(x_m[1] - x_m[0])

    # Spectral metrics
    E_in = E_t - E_t.mean()
    win_in = _hann(len(E_in))
    Ew_in = np.abs(np.fft.rfft(E_in * win_in)) ** 2
    freq_in = np.fft.rfftfreq(len(E_in), dt_in)
    freq_in_eV = HBAR * (2 * np.pi * freq_in) / E0

    ix0 = int(np.argmin(np.abs(x_m)))
    E_out_t = E_xt[ix0, :]
    E_out = E_out_t - E_out_t.mean()
    win_out = _hann(len(E_out))
    Ew_out = np.abs(np.fft.rfft(E_out * win_out)) ** 2
    freq_out = np.fft.rfftfreq(len(E_out), dt_out)
    freq_out_eV = HBAR * (2 * np.pi * freq_out) / E0

    mu_in, sig_in = _spectral_moments(freq_in_eV, Ew_in)
    mu_out, sig_out = _spectral_moments(freq_out_eV, Ew_out)

    # Energy proxy
    energy_xt = np.trapezoid(np.abs(E_xt) ** 2, x_m, axis=0)
    energy_mean = float(np.mean(energy_xt))
    energy_min = float(np.min(energy_xt))
    energy_max = float(np.max(energy_xt))
    energy_std = float(np.std(energy_xt))

    # Feedback vs drive
    fb_rel_l2 = None
    fb_path = data["E_t_feedback"]
    if fb_path.exists():
        E_fb = np.load(fb_path)
        n = min(len(E_fb), len(E_t))
        denom = max(float(np.linalg.norm(E_t[:n])), 1e-30)
        fb_rel_l2 = float(np.linalg.norm(E_fb[:n] - E_t[:n]) / denom)

    metrics = {
        "dt_in_s": dt_in,
        "dt_out_s": dt_out,
        "dx_m": dx,
        "spectral": {
            "input_centroid_eV": mu_in,
            "input_rms_width_eV": sig_in,
            "output_centroid_eV": mu_out,
            "output_rms_width_eV": sig_out,
        },
        "energy_proxy": {
            "mean": energy_mean,
            "min": energy_min,
            "max": energy_max,
            "std": energy_std,
            "drift_rel": (energy_max - energy_min) / energy_mean if energy_mean else None,
            "std_rel": energy_std / energy_mean if energy_mean else None,
        },
        "feedback_rel_l2": fb_rel_l2,
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SBUP3 outputs")
    parser.add_argument(
        "--report-dir",
        default="analysis/reports",
        help="Directory for validation reports",
    )
    parser.add_argument("--sweep-dt-as", default=None, help="Comma-separated dt [as]")
    parser.add_argument("--sweep-dz", default=None, help="Comma-separated dz [m]")
    parser.add_argument("--sweep-nx", default=None, help="Comma-separated NX values")
    parser.add_argument("--sweep-z-steps", default=None, help="Comma-separated z-steps")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    sbe_dir = base / "sbe"

    dt_list = _parse_list(args.sweep_dt_as, float) if args.sweep_dt_as else []
    dz_list = _parse_list(args.sweep_dz, float) if args.sweep_dz else []
    nx_list = _parse_list(args.sweep_nx, int) if args.sweep_nx else []
    z_steps_list = _parse_list(args.sweep_z_steps, int) if args.sweep_z_steps else []

    do_sweep = any([dt_list, dz_list, nx_list, z_steps_list])

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "git": _git_info(base),
        "sweep": [],
        "metrics": None,
    }

    if not do_sweep:
        report["metrics"] = compute_metrics(base)
    else:
        # Ensure internal drive during sweeps
        env_base = os.environ.copy()
        env_base["SBUP3_IGNORE_FEEDBACK"] = "1"

        if not dt_list:
            dt_list = [None]
        if not dz_list:
            dz_list = [None]
        if not nx_list:
            nx_list = [None]
        if not z_steps_list:
            z_steps_list = [None]

        for dt_as in dt_list:
            for dz in dz_list:
                for nx in nx_list:
                    for z_steps in z_steps_list:
                        env = env_base.copy()
                        params = {}
                        if dt_as is not None:
                            env["SBUP3_DT_AS"] = str(dt_as)
                            params["dt_as"] = dt_as
                        if dz is not None:
                            env["SBUP3_UPPE_DZ"] = str(dz)
                            params["dz"] = dz
                        if nx is not None:
                            env["SBUP3_UPPE_NX"] = str(nx)
                            params["nx"] = nx
                        if z_steps is not None:
                            env["SBUP3_UPPE_Z_STEPS"] = str(z_steps)
                            params["z_steps"] = z_steps

                        _run([sys.executable, str(sbe_dir / "sbe_rk4.py")], cwd=sbe_dir, env=env)
                        _run([sys.executable, "-m", "uppe.uppe_2d"], cwd=base, env=env)

                        metrics = compute_metrics(base)
                        report["sweep"].append({"params": params, "metrics": metrics})

    report_dir = base / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"validation_{report['timestamp_utc']}.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\n[OK] Wrote validation report: {report_path}")


if __name__ == "__main__":
    main()
