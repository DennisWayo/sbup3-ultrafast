#!/usr/bin/env python3
"""
SBUP3 validation script.

Computes stability and consistency metrics from SBE/UPPE outputs.
Optionally runs parameter sweeps for convergence studies.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _load_reference_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Reference CSV missing headers")

        keys = {_normalize_key(k): k for k in reader.fieldnames}

        def pick(*names: str) -> str | None:
            for n in names:
                if n in keys:
                    return keys[n]
            return None

        col_energy = pick("energyev", "energy", "photonenergy", "hv", "hvev", "e")
        if col_energy is None:
            raise ValueError("Reference CSV missing energy column (energy_eV)")

        cols = {
            "energy_eV": col_energy,
            "chi_real": pick("chireal", "rechi", "chire"),
            "chi_imag": pick("chiimag", "imchi", "chiim"),
            "alpha": pick("alpha", "absorption", "absorptioncoeff"),
            "n": pick("n", "refractiveindex"),
            "osc": pick("osc", "f", "oscillatorstrength", "oscstrength"),
        }

        data = {k: [] for k, v in cols.items() if v is not None}

        for row in reader:
            try:
                data["energy_eV"].append(float(row[cols["energy_eV"]]))
                for k, v in cols.items():
                    if k == "energy_eV" or v is None:
                        continue
                    val = row.get(v, "")
                    if val == "":
                        data[k].append(float("nan"))
                    else:
                        data[k].append(float(val))
            except Exception:
                # Skip malformed row
                continue

    out = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    # Sort by energy
    idx = np.argsort(out["energy_eV"])
    for k in out:
        out[k] = out[k][idx]

    return out


def _load_reference_log(base: Path) -> tuple[dict[str, np.ndarray], str] | None:
    log_paths = [
        base / "dft" / "lrtddft.log",
        base / "dft" / "gaas_lrtddft.log",
    ]
    pattern = re.compile(r"om=([0-9.+-Ee]+)\[eV\]\s+\|me\|=([0-9.+-Ee]+)")

    for path in log_paths:
        if not path.exists():
            continue
        energies: list[float] = []
        osc: list[float] = []
        for line in path.read_text(errors="ignore").splitlines():
            match = pattern.search(line)
            if match:
                energies.append(float(match.group(1)))
                osc.append(float(match.group(2)))
        if energies:
            ref = {
                "energy_eV": np.asarray(energies, dtype=float),
                "osc": np.asarray(osc, dtype=float),
            }
            idx = np.argsort(ref["energy_eV"])
            for k in ref:
                ref[k] = ref[k][idx]
            return ref, str(path)
    return None


def _spectral_moments(freq_eV: np.ndarray, spectrum: np.ndarray) -> tuple[float, float]:
    s = spectrum / np.sum(spectrum)
    mu = float(np.sum(freq_eV * s))
    sigma = float(np.sqrt(np.sum((freq_eV - mu) ** 2 * s)))
    return mu, sigma


def _compare_series(
    sim_energy: np.ndarray,
    sim_values: np.ndarray,
    ref_energy: np.ndarray,
    ref_values: np.ndarray,
) -> dict[str, float] | None:
    if len(sim_energy) < 2 or len(ref_energy) < 2:
        return None

    e_min = max(sim_energy.min(), ref_energy.min())
    e_max = min(sim_energy.max(), ref_energy.max())
    mask = (ref_energy >= e_min) & (ref_energy <= e_max)
    if mask.sum() < 5:
        return None

    ref_e = ref_energy[mask]
    ref_v = ref_values[mask]
    sim_v = np.interp(ref_e, sim_energy, sim_values)

    # Fit scale factor (least squares)
    denom = float(np.dot(sim_v, sim_v))
    scale = float(np.dot(ref_v, sim_v) / denom) if denom > 0 else 1.0

    resid = ref_v - scale * sim_v
    rmse = float(np.sqrt(np.mean(resid**2)))
    ref_norm = float(np.max(np.abs(ref_v))) or 1.0
    rmse_norm = rmse / ref_norm

    # Shape-only comparison
    sim_norm = sim_v / (float(np.max(np.abs(sim_v))) or 1.0)
    ref_normed = ref_v / ref_norm
    rmse_shape = float(np.sqrt(np.mean((ref_normed - sim_norm) ** 2)))

    return {
        "energy_min_eV": float(e_min),
        "energy_max_eV": float(e_max),
        "scale": scale,
        "rmse": rmse,
        "rmse_norm": rmse_norm,
        "rmse_shape": rmse_shape,
        "n_points": int(mask.sum()),
    }


def compute_metrics(base: Path) -> dict[str, Any]:
    data = _load_outputs(base)

    time_s = data["time_s"]
    E_t = data["E_t"]
    P_t = data["P_t"]
    n_t = data["n_t"]
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

    # Dephasing check (tail vs peak)
    absP = np.abs(P_t)
    peak = float(absP.max()) if len(absP) else 0.0
    tail_start = int(0.7 * len(absP)) if len(absP) else 0
    tail = absP[tail_start:] if len(absP) else absP
    tail_max = float(tail.max()) if len(tail) else 0.0
    tail_mean = float(tail.mean()) if len(tail) else 0.0
    tail_ratio = (tail_max / peak) if peak else None

    # Tail decay slope (log envelope)
    decay_slope = None
    if len(tail) > 3 and peak > 0:
        t_tail = time_s[tail_start:]
        eps = 1e-30
        decay_slope = float(np.polyfit(t_tail, np.log(tail + eps), 1)[0])

    # NaN/Inf checks
    def _finite_stats(arr: np.ndarray) -> dict[str, int]:
        return {
            "nan": int(np.isnan(arr).sum()),
            "inf": int(np.isinf(arr).sum()),
        }

    metrics = {
        "dt_in_s": dt_in,
        "dt_out_s": dt_out,
        "dx_m": dx,
        "finite": {
            "E_t": _finite_stats(E_t),
            "P_t": _finite_stats(P_t),
            "n_t": _finite_stats(n_t),
            "E_xt": _finite_stats(E_xt),
        },
        "dephasing": {
            "tail_max_over_peak": tail_ratio,
            "tail_mean": tail_mean,
            "decay_slope": decay_slope,
        },
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


def _oscillator_spectrum(energy_grid: np.ndarray, e0: np.ndarray, osc: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        sigma = 0.05
    spec = np.zeros_like(energy_grid, dtype=float)
    for e, f in zip(e0, osc):
        spec += f * np.exp(-0.5 * ((energy_grid - e) / sigma) ** 2)
    return spec


def _baseline_from_ref(
    data: dict[str, Any],
    ref: dict[str, np.ndarray],
    label: str,
    osc_sigma: float,
) -> dict[str, Any]:
    omega = data["omega"]
    omega_eV = HBAR * omega / E0

    baseline: dict[str, Any] = {"path": label, "metrics": {}}
    if label.endswith(".log"):
        baseline["notes"] = "Reference built from LR-TDDFT log (|me| used as oscillator proxy)."

    if "chi_real" in ref:
        baseline["metrics"]["chi_real"] = _compare_series(
            omega_eV, data["chi_r"], ref["energy_eV"], ref["chi_real"]
        )
    if "chi_imag" in ref:
        baseline["metrics"]["chi_imag"] = _compare_series(
            omega_eV, data["chi_i"], ref["energy_eV"], ref["chi_imag"]
        )
    if "alpha" in ref:
        baseline["metrics"]["alpha"] = _compare_series(
            omega_eV, data["alpha"], ref["energy_eV"], ref["alpha"]
        )
    elif "osc" in ref:
        ref_alpha = _oscillator_spectrum(omega_eV, ref["energy_eV"], ref["osc"], osc_sigma)
        baseline["metrics"]["alpha_from_osc"] = _compare_series(
            omega_eV, data["alpha"], omega_eV, ref_alpha
        )
    if "n" in ref:
        baseline["metrics"]["n"] = _compare_series(
            omega_eV, data["n_rel"], ref["energy_eV"], ref["n"]
        )

    # Recommend polarization scaling from alpha (preferred) or chi_imag
    if baseline["metrics"].get("alpha") and baseline["metrics"]["alpha"] is not None:
        scale = baseline["metrics"]["alpha"]["scale"]
        baseline["recommended_polarization_scale"] = scale
        baseline["recommended_source"] = "alpha"
    elif baseline["metrics"].get("alpha_from_osc") and baseline["metrics"]["alpha_from_osc"] is not None:
        scale = baseline["metrics"]["alpha_from_osc"]["scale"]
        baseline["recommended_polarization_scale"] = scale
        baseline["recommended_source"] = "alpha_from_osc"
    elif baseline["metrics"].get("chi_imag") and baseline["metrics"]["chi_imag"] is not None:
        scale = baseline["metrics"]["chi_imag"]["scale"]
        baseline["recommended_polarization_scale"] = scale
        baseline["recommended_source"] = "chi_imag"

    return baseline


def compute_baseline(base: Path, ref_csv: Path, osc_sigma: float) -> dict[str, Any]:
    data = _load_outputs(base)
    ref = _load_reference_csv(ref_csv)
    return _baseline_from_ref(data, ref, str(ref_csv), osc_sigma)


def _run_sbe_only(sbe_dir: Path, env: dict[str, str]) -> float:
    _run([sys.executable, str(sbe_dir / "sbe_rk4.py")], cwd=sbe_dir, env=env)
    pol = np.load(sbe_dir / "polarization.npy")
    return float(np.max(np.abs(pol)))


def compute_linear_check(base: Path, amplitude: float) -> dict[str, Any]:
    sbe_dir = base / "sbe"
    env = os.environ.copy()
    env["SBUP3_IGNORE_FEEDBACK"] = "1"

    env["SBUP3_E_FIELD_AMPL"] = str(amplitude)
    p1 = _run_sbe_only(sbe_dir, env)

    env["SBUP3_E_FIELD_AMPL"] = str(0.5 * amplitude)
    p2 = _run_sbe_only(sbe_dir, env)

    ratio = p1 / p2 if p2 != 0 else None
    return {"amplitude": amplitude, "ratio_Pmax": ratio}


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
    parser.add_argument("--sweep-x-window", default=None, help="Comma-separated x-window [m]")
    parser.add_argument("--sweep-mixing", default=None, help="Comma-separated MIXING values")
    parser.add_argument("--sweep-feedback-beta", default=None, help="Comma-separated FEEDBACK_BETA values")
    parser.add_argument(
        "--sweep-feedback-norm",
        default=None,
        help="Comma-separated FEEDBACK_NORM values (rms,peak,none)",
    )
    parser.add_argument("--sweep-coupling-iters", default=None, help="Coupling iters for sweeps")
    parser.add_argument("--ref-csv", default=None, help="Reference CSV for baseline comparison")
    parser.add_argument("--osc-broadening", type=float, default=0.1, help="Sigma [eV] for oscillator spectrum")
    parser.add_argument("--linear-check", action="store_true", help="Run linear-regime check")
    parser.add_argument("--linear-amp", type=float, default=None, help="Base amplitude for linear check")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    sbe_dir = base / "sbe"

    dt_list = _parse_list(args.sweep_dt_as, float) if args.sweep_dt_as else []
    dz_list = _parse_list(args.sweep_dz, float) if args.sweep_dz else []
    nx_list = _parse_list(args.sweep_nx, int) if args.sweep_nx else []
    z_steps_list = _parse_list(args.sweep_z_steps, int) if args.sweep_z_steps else []
    x_window_list = _parse_list(args.sweep_x_window, float) if args.sweep_x_window else []
    mixing_list = _parse_list(args.sweep_mixing, float) if args.sweep_mixing else []
    beta_list = _parse_list(args.sweep_feedback_beta, float) if args.sweep_feedback_beta else []
    norm_list = _parse_list(args.sweep_feedback_norm, str) if args.sweep_feedback_norm else []
    coupling_iters = int(args.sweep_coupling_iters) if args.sweep_coupling_iters else None

    do_sweep = any([dt_list, dz_list, nx_list, z_steps_list, x_window_list, mixing_list, beta_list, norm_list])

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "git": _git_info(base),
        "scaling": {
            "density_m3": os.getenv("SBUP3_DENSITY_M3"),
            "polarization_scale": os.getenv("SBUP3_POLARIZATION_SCALE"),
        },
        "sweep": [],
        "metrics": None,
        "baseline": None,
        "linear_check": None,
    }

    if not do_sweep:
        report["metrics"] = compute_metrics(base)
        ref_path = Path(args.ref_csv) if args.ref_csv else (base / "analysis" / "data" / "gaas_tddft_reference.csv")
        if ref_path.exists():
            report["baseline"] = compute_baseline(base, ref_path, args.osc_broadening)
        else:
            fallback = _load_reference_log(base)
            if fallback:
                ref, label = fallback
                report["baseline"] = _baseline_from_ref(_load_outputs(base), ref, label, args.osc_broadening)
        if args.linear_check:
            base_amp = args.linear_amp
            if base_amp is None:
                base_amp = float(os.getenv("SBUP3_E_FIELD_AMPL", "1.0e7"))
            report["linear_check"] = compute_linear_check(base, base_amp)
    else:
        # Ensure internal drive during sweeps
        env_base = os.environ.copy()
        use_coupling = any([mixing_list, beta_list, norm_list])
        if not use_coupling:
            env_base["SBUP3_IGNORE_FEEDBACK"] = "1"

        if not dt_list:
            dt_list = [None]
        if not dz_list:
            dz_list = [None]
        if not nx_list:
            nx_list = [None]
        if not z_steps_list:
            z_steps_list = [None]
        if not x_window_list:
            x_window_list = [None]
        if not mixing_list:
            mixing_list = [None]
        if not beta_list:
            beta_list = [None]
        if not norm_list:
            norm_list = [None]

        for dt_as in dt_list:
            for dz in dz_list:
                for nx in nx_list:
                    for z_steps in z_steps_list:
                        for x_window in x_window_list:
                            for mixing in mixing_list:
                                for beta in beta_list:
                                    for norm in norm_list:
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
                                        if x_window is not None:
                                            env["SBUP3_UPPE_X_WINDOW"] = str(x_window)
                                            params["x_window"] = x_window
                                        if mixing is not None:
                                            env["SBUP3_MIXING"] = str(mixing)
                                            params["mixing"] = mixing
                                        if beta is not None:
                                            env["SBUP3_FEEDBACK_BETA"] = str(beta)
                                            params["feedback_beta"] = beta
                                        if norm is not None:
                                            env["SBUP3_FEEDBACK_NORM"] = str(norm)
                                            params["feedback_norm"] = norm
                                        if coupling_iters is not None:
                                            env["SBUP3_COUPLING_ITERS"] = str(coupling_iters)
                                            params["coupling_iters"] = coupling_iters

                                        if use_coupling:
                                            _run([sys.executable, "-m", "coupling.sbup3_loop"], cwd=base, env=env)
                                        else:
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
