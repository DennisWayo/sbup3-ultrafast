#!/usr/bin/env python3
"""
SBUP3 pipeline runner.

Runs optional TDDFT -> SBE -> UPPE -> coupling and writes a run manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import importlib.metadata

def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("\n[CMD]", " ".join(cmd))
    print("[CWD]", str(cwd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _sha256(path: Path, max_mb: int = 50) -> str | None:
    if not path.exists():
        return None
    size = path.stat().st_size
    if size > max_mb * 1024 * 1024:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        info["size_bytes"] = path.stat().st_size
        info["sha256"] = _sha256(path)
    return info


def _load_scalar(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(np.load(path).item())
    except Exception:
        return None


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


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _find_conda_exe() -> str:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        for candidate in [prefix / "bin" / "conda", prefix.parent / "bin" / "conda"]:
            if candidate.exists():
                return str(candidate)

    return "conda"


def _conda_for_env(env_name: str) -> str | None:
    home = Path.home()
    bases = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        bases.append(Path(conda_prefix).parent)

    # Common install locations
    bases.extend(
        [
            home / "miniforge3",
            home / "miniforge-x86_64",
            home / "miniforge",
            home / "miniconda3",
            home / "mambaforge",
            home / "anaconda3",
        ]
    )

    for base in bases:
        env_path = base / "envs" / env_name
        conda_exe = base / "bin" / "conda"
        if env_path.exists() and conda_exe.exists():
            return str(conda_exe)

    return None


def _parse_py_constants(path: Path, keys: set[str]) -> dict[str, str]:
    constants: dict[str, str] = {}
    if not path.exists():
        return constants
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name in keys:
            constants[name] = value.strip()
    return constants


def _set_env(env: dict[str, str], key: str, value: Any) -> None:
    env[key] = str(value)


def _env_float(env: dict[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(env: dict[str, str], key: str, default: str) -> str:
    raw = env.get(key)
    if raw is None:
        return default
    return str(raw)


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _parse_calibration_channels(raw: str) -> list[str]:
    supported = {"alpha", "n", "chi_real", "chi_imag"}
    channels = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if not channels:
        return ["alpha", "n"]
    invalid = [c for c in channels if c not in supported]
    if invalid:
        raise ValueError(
            "Unsupported calibration channel(s): "
            + ", ".join(sorted(invalid))
            + f". Supported: {sorted(supported)}"
        )
    # Keep order while deduplicating.
    seen: set[str] = set()
    unique = []
    for ch in channels:
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)
    return unique


def _reference_channels_from_csv(path: Path) -> set[str]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Reference CSV missing headers: {path}")
        keys = {_normalize_key(k) for k in reader.fieldnames}

    channels: set[str] = set()
    if {"alpha", "alphaminv", "absorption", "absorptioncoeff"} & keys:
        channels.add("alpha")
    if {"n", "refractiveindex"} & keys:
        channels.add("n")
    if {"chireal", "rechi", "chire"} & keys:
        channels.add("chi_real")
    if {"chiimag", "imchi", "chiim"} & keys:
        channels.add("chi_imag")
    return channels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBUP3 pipeline")
    parser.add_argument(
        "--mode",
        choices=["sequential", "coupling", "full"],
        default="coupling",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--allow-feedback-in-sequential",
        action="store_true",
        help="Allow sequential mode to consume sbe/E_t_feedback.npy if present",
    )
    parser.add_argument("--run-dft", action="store_true", help="Run DFT optical-response stage")
    parser.add_argument("--dft-env", default=None, help="Conda env for DFT stage")
    parser.add_argument(
        "--dft-kpts",
        default="8,8,8",
        help="Ground-state Monkhorst-Pack k-grid for DFT stage as kx,ky,kz.",
    )
    parser.add_argument(
        "--dft-gs-mode",
        choices=["pw", "fd"],
        default="pw",
        help="Ground-state mode for DFT stage. 'pw' is required for dielectric response in this GPAW build.",
    )
    parser.add_argument(
        "--dft-gs-nbands",
        type=int,
        default=32,
        help="Number of bands for the ground-state stage.",
    )
    parser.add_argument(
        "--dft-pw-ecut-ev",
        type=float,
        default=200.0,
        help="PW cutoff [eV] for DFT ground-state mode when --dft-gs-mode pw.",
    )
    parser.add_argument(
        "--dft-response-kpts",
        default="10,10,10",
        help="Dense response Monkhorst-Pack k-grid as kx,ky,kz.",
    )
    parser.add_argument(
        "--dft-response-nbands",
        type=int,
        default=120,
        help="Number of bands for dielectric-response full diagonalization.",
    )
    parser.add_argument(
        "--dft-response-out-csv",
        default="analysis/data/gaas_dielectric_reference.csv",
        help="Output CSV for dielectric optical constants.",
    )
    parser.add_argument(
        "--dft-transition-e-min",
        type=float,
        default=0.0,
        help="Minimum energy [eV] for transition extraction from dielectric reference.",
    )
    parser.add_argument(
        "--dft-transition-e-max",
        type=float,
        default=3.0,
        help="Maximum energy [eV] for transition extraction from dielectric reference.",
    )
    parser.add_argument(
        "--dft-transition-min-rel-height",
        type=float,
        default=0.01,
        help="Minimum relative peak height for dielectric transition extraction.",
    )
    parser.add_argument(
        "--dft-transition-min-sep-ev",
        type=float,
        default=0.02,
        help="Minimum peak separation [eV] for dielectric transition extraction.",
    )
    parser.add_argument(
        "--dft-transition-min-count",
        type=int,
        default=16,
        help="Minimum number of transitions to extract (backfilled by strongest points).",
    )
    parser.add_argument(
        "--dft-transition-max-count",
        type=int,
        default=24,
        help="Maximum number of extracted transitions from dielectric data.",
    )
    parser.add_argument(
        "--dft-transition-sum-f",
        type=float,
        default=1.0,
        help="Normalize extracted transition strengths to this oscillator sum.",
    )
    parser.add_argument(
        "--dft-force-ground-state",
        action="store_true",
        help="Recompute ground-state DFT artifacts instead of reusing existing files.",
    )
    parser.add_argument(
        "--dft-allow-missing-gs-meta",
        action="store_true",
        help="Allow dielectric stage without gaas_ground_state_meta.json (debug only).",
    )
    parser.add_argument(
        "--run-lrtddft-stage",
        action="store_true",
        help="Run legacy LR-TDDFT transition extraction stage (gamma-only workflow).",
    )
    parser.add_argument(
        "--dft-scissor-ev",
        type=float,
        default=None,
        help="Rigid energy shift [eV] for legacy LR-TDDFT excitations.",
    )
    parser.add_argument("--plots", action="store_true", help="Generate figures")
    parser.add_argument("--validate", action="store_true", help="Run validation report")
    parser.add_argument(
        "--report-dir",
        default="analysis/reports",
        help="Directory for run manifests",
    )
    parser.add_argument(
        "--calibrate-scale",
        action="store_true",
        help="Fit polarization scale against reference optical data.",
    )
    parser.add_argument(
        "--calibration-ref-csv",
        default=None,
        help="Reference CSV for scale calibration (defaults to --ref-csv then analysis/data/gaas_dielectric_reference.csv).",
    )
    parser.add_argument(
        "--calibration-fit-channels",
        default="alpha,n",
        help="Comma-separated channels for calibration (alpha,n,chi_real,chi_imag). "
        "Reference CSV alpha can be named alpha or alpha_m_inv.",
    )
    parser.add_argument(
        "--calibration-log10-ratio-min",
        type=float,
        default=-8.0,
        help="Calibration search lower bound in log10 ratio.",
    )
    parser.add_argument(
        "--calibration-log10-ratio-max",
        type=float,
        default=8.0,
        help="Calibration search upper bound in log10 ratio.",
    )
    parser.add_argument(
        "--calibration-grid",
        type=int,
        default=801,
        help="Calibration search grid points.",
    )
    parser.add_argument(
        "--calibration-target-density-m3",
        type=float,
        default=None,
        help="Optional target density for reporting scale adjusted to physical density.",
    )
    parser.add_argument(
        "--calibration-out-json",
        default=None,
        help="Optional explicit path for calibration JSON report.",
    )
    parser.add_argument(
        "--calibration-write-env",
        default=None,
        help="Optional file path to write export commands for calibrated parameters.",
    )

    # SBE overrides
    parser.add_argument("--t-fs", type=float)
    parser.add_argument("--dt-as", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--e-field-ampl", type=float)
    parser.add_argument("--omega-l", type=float)
    parser.add_argument("--mixing", type=float)
    parser.add_argument("--ignore-feedback", action="store_true")
    parser.add_argument("--e-max", type=float)
    parser.add_argument("--density-m3", type=float)
    parser.add_argument("--polarization-scale", type=float)
    parser.add_argument("--osc-norm-factor", type=float)
    parser.add_argument("--chi-eps", type=float)
    parser.add_argument("--response-power-cutoff", type=float)
    parser.add_argument("--chi-max-ev", type=float)

    # UPPE overrides
    parser.add_argument("--x-window", type=float)
    parser.add_argument("--w0", type=float)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--dz", type=float)
    parser.add_argument("--z-steps", type=int)

    # Coupling overrides
    parser.add_argument("--iters", type=int)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--feedback-beta", type=float)
    parser.add_argument("--feedback-norm", choices=["rms", "peak", "none"])
    parser.add_argument("--feedback-norm-eps", type=float)

    # Validation + plotting extras
    parser.add_argument("--ref-csv", default=None, help="Reference CSV for baseline comparison")
    parser.add_argument("--linear-check", action="store_true", help="Run linear-regime check in validation")
    parser.add_argument("--osc-broadening", type=float, default=None, help="Sigma [eV] for oscillator spectrum")
    parser.add_argument("--overlay-e-min", type=float, default=None, help="Overlay min energy [eV] for plots")
    parser.add_argument("--overlay-e-max", type=float, default=None, help="Overlay max energy [eV] for plots")
    parser.add_argument(
        "--overlay-window",
        default=None,
        help="Overlay window as min,max eV (passed to visualize script)",
    )
    parser.add_argument("--overlay-scale", choices=["fit", "peak", "none"], default=None, help="Overlay scaling mode for plots")
    parser.add_argument("--overlay-abs", action="store_true", help="Use absolute alpha in overlay")

    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    sbe_dir = base / "sbe"
    uppe_dir = base / "uppe"
    dft_dir = base / "dft"

    env = os.environ.copy()

    if args.t_fs is not None:
        _set_env(env, "SBUP3_T_FS", args.t_fs)
    if args.dt_as is not None:
        _set_env(env, "SBUP3_DT_AS", args.dt_as)
    if args.gamma is not None:
        _set_env(env, "SBUP3_GAMMA", args.gamma)
    if args.e_field_ampl is not None:
        _set_env(env, "SBUP3_E_FIELD_AMPL", args.e_field_ampl)
    if args.omega_l is not None:
        _set_env(env, "SBUP3_OMEGA_L", args.omega_l)
    if args.mixing is not None:
        _set_env(env, "SBUP3_MIXING", args.mixing)
    if args.ignore_feedback:
        _set_env(env, "SBUP3_IGNORE_FEEDBACK", 1)
    if args.e_max is not None:
        _set_env(env, "SBUP3_E_MAX", args.e_max)
    if args.density_m3 is not None:
        _set_env(env, "SBUP3_DENSITY_M3", args.density_m3)
    if args.polarization_scale is not None:
        _set_env(env, "SBUP3_POLARIZATION_SCALE", args.polarization_scale)
    if args.osc_norm_factor is not None:
        _set_env(env, "SBUP3_OSC_NORM_FACTOR", args.osc_norm_factor)
    if args.chi_eps is not None:
        _set_env(env, "SBUP3_CHI_EPS", args.chi_eps)
    if args.response_power_cutoff is not None:
        _set_env(env, "SBUP3_RESPONSE_POWER_CUTOFF", args.response_power_cutoff)
    if args.chi_max_ev is not None:
        _set_env(env, "SBUP3_CHI_MAX_EV", args.chi_max_ev)

    if args.x_window is not None:
        _set_env(env, "SBUP3_UPPE_X_WINDOW", args.x_window)
    if args.w0 is not None:
        _set_env(env, "SBUP3_UPPE_W0", args.w0)
    if args.nx is not None:
        _set_env(env, "SBUP3_UPPE_NX", args.nx)
    if args.dz is not None:
        _set_env(env, "SBUP3_UPPE_DZ", args.dz)
    if args.z_steps is not None:
        _set_env(env, "SBUP3_UPPE_Z_STEPS", args.z_steps)

    if args.iters is not None:
        _set_env(env, "SBUP3_COUPLING_ITERS", args.iters)
    if args.rtol is not None:
        _set_env(env, "SBUP3_COUPLING_RTOL", args.rtol)
    if args.feedback_beta is not None:
        _set_env(env, "SBUP3_FEEDBACK_BETA", args.feedback_beta)
    if args.feedback_norm is not None:
        _set_env(env, "SBUP3_FEEDBACK_NORM", args.feedback_norm)
    if args.feedback_norm_eps is not None:
        _set_env(env, "SBUP3_FEEDBACK_NORM_EPS", args.feedback_norm_eps)

    # Reproducibility guardrail: sequential runs are baseline calibrations unless
    # explicitly configured to consume stored feedback.
    if args.mode == "sequential" and not args.allow_feedback_in_sequential:
        env.setdefault("SBUP3_IGNORE_FEEDBACK", "1")

    calibration_ref_path: Path | None = None
    if args.calibrate_scale:
        default_ref = base / "analysis" / "data" / "gaas_dielectric_reference.csv"
        calibration_ref = args.calibration_ref_csv or args.ref_csv or str(default_ref)
        calibration_ref_path = Path(calibration_ref)
        if not calibration_ref_path.is_absolute():
            calibration_ref_path = base / calibration_ref_path
        calibration_ref_path = calibration_ref_path.resolve()
        if not calibration_ref_path.exists():
            raise FileNotFoundError(f"Calibration reference CSV not found: {calibration_ref_path}")

        requested_channels = _parse_calibration_channels(str(args.calibration_fit_channels))
        available_channels = _reference_channels_from_csv(calibration_ref_path)
        missing = [ch for ch in requested_channels if ch not in available_channels]
        if missing:
            raise RuntimeError(
                "Calibration reference CSV is missing required channels.\n"
                f"  file: {calibration_ref_path}\n"
                f"  requested: {', '.join(requested_channels)}\n"
                f"  missing: {', '.join(missing)}\n"
                f"  available: {', '.join(sorted(available_channels)) or '(none)'}\n"
                "  alpha aliases accepted: alpha, alpha_m_inv, absorption, absorptioncoeff"
            )

    # Optional DFT stage
    if args.run_dft:
        dft_python = [sys.executable]
        if args.dft_env:
            conda_exe = _conda_for_env(args.dft_env) or _find_conda_exe()
            dft_python = [conda_exe, "run", "-n", args.dft_env, "python"]

        _run(
            dft_python
            + [
                str(dft_dir / "gaas_bulk_gpaw.py"),
                "--gs-mode",
                str(args.dft_gs_mode),
                "--kpts",
                str(args.dft_kpts),
                "--nbands",
                str(args.dft_gs_nbands),
                "--pw-ecut-ev",
                str(args.dft_pw_ecut_ev),
                *(["--force"] if args.dft_force_ground_state else []),
            ],
            cwd=dft_dir,
            env=env,
        )

        gs_gpw = dft_dir / ("gaas_pw.gpw" if args.dft_gs_mode == "pw" else "gaas_fd.gpw")
        if not gs_gpw.exists():
            raise FileNotFoundError(f"Expected ground-state file not found after DFT stage: {gs_gpw}")

        export_csv = Path(args.dft_response_out_csv)
        if not export_csv.is_absolute():
            export_csv = base / export_csv
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        _run(
            dft_python
            + [
                str(dft_dir / "gaas_dielectric_response.py"),
                "--gs-gpw",
                str(gs_gpw),
                "--gs-meta",
                str(dft_dir / "gaas_ground_state_meta.json"),
                "--kpts",
                str(args.dft_response_kpts),
                "--nbands",
                str(args.dft_response_nbands),
                "--out-csv",
                str(export_csv),
                *(["--allow-missing-gs-meta"] if args.dft_allow_missing_gs_meta else []),
            ],
            cwd=dft_dir,
            env=env,
        )
        _run(
            dft_python
            + [
                str(dft_dir / "build_transitions_from_dielectric.py"),
                "--in-csv",
                str(export_csv),
                "--out-e",
                str(dft_dir / "transitions_E_eV.npy"),
                "--out-f",
                str(dft_dir / "transitions_f.npy"),
                "--out-meta",
                str(dft_dir / "transitions_meta.json"),
                "--e-min",
                str(args.dft_transition_e_min),
                "--e-max",
                str(args.dft_transition_e_max),
                "--min-rel-height",
                str(args.dft_transition_min_rel_height),
                "--min-sep-ev",
                str(args.dft_transition_min_sep_ev),
                "--min-count",
                str(args.dft_transition_min_count),
                "--max-count",
                str(args.dft_transition_max_count),
                "--sum-f",
                str(args.dft_transition_sum_f),
            ],
            cwd=dft_dir,
            env=env,
        )

        if args.run_lrtddft_stage:
            if args.dft_gs_mode != "fd":
                raise RuntimeError(
                    "--run-lrtddft-stage requires --dft-gs-mode fd (legacy gamma-only path)."
                )
            legacy_csv = base / "analysis" / "data" / "gaas_tddft_reference.csv"
            legacy_csv.parent.mkdir(parents=True, exist_ok=True)
            _run(
                dft_python
                + [
                    str(dft_dir / "gaas_lrtddft.py"),
                    "--gpw",
                    str(dft_dir / "gaas_fd.gpw"),
                    *(
                        ["--scissor-ev", str(args.dft_scissor_ev)]
                        if args.dft_scissor_ev is not None
                        else []
                    ),
                    "--export-csv",
                    str(legacy_csv),
                ],
                cwd=dft_dir,
                env=env,
            )

    # Main pipeline
    if args.mode in {"sequential", "full"}:
        _run([sys.executable, str(sbe_dir / "sbe_rk4.py")], cwd=sbe_dir, env=env)
        _run([sys.executable, "-m", "uppe.uppe_2d"], cwd=base, env=env)

    if args.mode in {"coupling", "full"}:
        _run([sys.executable, "-m", "coupling.sbup3_loop"], cwd=base, env=env)

    # Optional plots
    if args.plots:
        plot_cmd = [sys.executable, str(base / "analysis" / "visualize_sbup3.py")]
        if args.overlay_e_min is not None:
            plot_cmd += ["--overlay-e-min", str(args.overlay_e_min)]
        if args.overlay_e_max is not None:
            plot_cmd += ["--overlay-e-max", str(args.overlay_e_max)]
        if args.overlay_window is not None:
            plot_cmd += ["--overlay-window", str(args.overlay_window)]
        if args.overlay_scale is not None:
            plot_cmd += ["--overlay-scale", str(args.overlay_scale)]
        if args.overlay_abs:
            plot_cmd += ["--overlay-abs"]
        if args.osc_broadening is not None:
            plot_cmd += ["--osc-broadening", str(args.osc_broadening)]
        _run(plot_cmd, cwd=base, env=env)

    # Optional absolute-scale calibration
    calibration_report = None
    calibration_ref_effective = None
    if args.calibrate_scale:
        if calibration_ref_path is None:
            raise RuntimeError("Internal error: calibration reference path was not initialized.")
        calibration_ref_effective = str(calibration_ref_path)
        ccmd = [
            sys.executable,
            str(base / "analysis" / "calibrate_polarization_scale.py"),
            "--ref-csv",
            str(calibration_ref_path),
            "--fit-channels",
            str(args.calibration_fit_channels),
            "--log10-ratio-min",
            str(args.calibration_log10_ratio_min),
            "--log10-ratio-max",
            str(args.calibration_log10_ratio_max),
            "--grid",
            str(args.calibration_grid),
        ]
        if args.calibration_target_density_m3 is not None:
            ccmd += ["--target-density-m3", str(args.calibration_target_density_m3)]
        if args.calibration_out_json:
            ccmd += ["--out-json", str(args.calibration_out_json)]
        if args.calibration_write_env:
            ccmd += ["--write-env", str(args.calibration_write_env)]
        _run(ccmd, cwd=base, env=env)

        if args.calibration_out_json:
            cpath = Path(args.calibration_out_json)
            calibration_report = str(cpath if cpath.is_absolute() else base / cpath)
        else:
            latest = sorted((base / "analysis" / "reports").glob("calibration_*.json"))
            if latest:
                calibration_report = str(latest[-1])

    # Optional validation
    validation_report = None
    if args.validate:
        vcmd = [sys.executable, str(base / "analysis" / "validate_sbup3.py")]
        if args.ref_csv:
            vcmd += ["--ref-csv", args.ref_csv]
        if args.linear_check:
            vcmd += ["--linear-check"]
        _run(vcmd, cwd=base, env=env)
        latest = sorted((base / "analysis" / "reports").glob("validation_*.json"))
        if latest:
            validation_report = str(latest[-1])

    # Manifest
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = base / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / f"run_{now}.json"

    dft_constants = {
        "gaas_bulk_gpaw.py": _parse_py_constants(
            dft_dir / "gaas_bulk_gpaw.py",
            {"DEFAULT_XC", "DEFAULT_KPTS", "DEFAULT_NBANDS", "DEFAULT_SMEARING", "DEFAULT_H"},
        ),
        "gaas_lrtddft.py": _parse_py_constants(
            dft_dir / "gaas_lrtddft.py", {"OSC_MIN"}
        ),
    }

    omega0_default = _load_scalar(dft_dir / "omega0_rad_s.npy")
    if omega0_default is None:
        omega0_default = 0.0

    effective_params = {
        "sbe": {
            "t_fs": _env_float(env, "SBUP3_T_FS", 50.0),
            "dt_as": _env_float(env, "SBUP3_DT_AS", 5.0),
            "gamma": _env_float(env, "SBUP3_GAMMA", 5.0e13),
            "e_field_ampl": _env_float(env, "SBUP3_E_FIELD_AMPL", 1.0e7),
            "omega_l": _env_float(env, "SBUP3_OMEGA_L", omega0_default),
            "mixing": _env_float(env, "SBUP3_MIXING", 0.05),
            "ignore_feedback": _env_bool(env, "SBUP3_IGNORE_FEEDBACK", False),
            "e_max": _env_float(env, "SBUP3_E_MAX", 5.0e7),
            "density_m3": _env_float(env, "SBUP3_DENSITY_M3", 1.0),
            "polarization_scale": _env_float(env, "SBUP3_POLARIZATION_SCALE", 1.0),
            "osc_norm_factor": _env_float(env, "SBUP3_OSC_NORM_FACTOR", 1.0),
            "chi_eps": _env_float(env, "SBUP3_CHI_EPS", 1e-3),
            "response_power_cutoff": _env_float(env, "SBUP3_RESPONSE_POWER_CUTOFF", 0.999),
            "chi_max_ev": _env_float(env, "SBUP3_CHI_MAX_EV", 3.0),
        },
        "uppe": {
            "x_window": _env_float(env, "SBUP3_UPPE_X_WINDOW", 10e-6),
            "w0": _env_float(env, "SBUP3_UPPE_W0", 2e-6),
            "nx": _env_int(env, "SBUP3_UPPE_NX", 256),
            "dz": _env_float(env, "SBUP3_UPPE_DZ", 1e-6),
            "z_steps": _env_int(env, "SBUP3_UPPE_Z_STEPS", 100),
        },
        "coupling": {
            "iters": _env_int(env, "SBUP3_COUPLING_ITERS", 10),
            "rtol": _env_float(env, "SBUP3_COUPLING_RTOL", 1e-3),
            "feedback_beta": _env_float(env, "SBUP3_FEEDBACK_BETA", 0.2),
            "feedback_norm": _env_str(env, "SBUP3_FEEDBACK_NORM", "rms"),
            "feedback_norm_eps": _env_float(env, "SBUP3_FEEDBACK_NORM_EPS", 1e-30),
        },
    }

    manifest: dict[str, Any] = {
        "timestamp_utc": now,
        "mode": args.mode,
        "git": _git_info(base),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "packages": {
            "numpy": _pkg_version("numpy"),
            "matplotlib": _pkg_version("matplotlib"),
            "gpaw": _pkg_version("gpaw"),
        },
        "params": {
            "calibration": {
                "enabled": args.calibrate_scale,
                "ref_csv": args.calibration_ref_csv,
                "ref_csv_effective": calibration_ref_effective,
                "fit_channels": args.calibration_fit_channels,
                "log10_ratio_min": args.calibration_log10_ratio_min,
                "log10_ratio_max": args.calibration_log10_ratio_max,
                "grid": args.calibration_grid,
                "target_density_m3": args.calibration_target_density_m3,
                "out_json": args.calibration_out_json,
                "write_env": args.calibration_write_env,
            },
            "dft": {
                "kpts": args.dft_kpts,
                "gs_mode": args.dft_gs_mode,
                "gs_nbands": args.dft_gs_nbands,
                "pw_ecut_ev": args.dft_pw_ecut_ev,
                "response_kpts": args.dft_response_kpts,
                "response_nbands": args.dft_response_nbands,
                "response_out_csv": args.dft_response_out_csv,
                "transition_e_min": args.dft_transition_e_min,
                "transition_e_max": args.dft_transition_e_max,
                "transition_min_rel_height": args.dft_transition_min_rel_height,
                "transition_min_sep_ev": args.dft_transition_min_sep_ev,
                "transition_min_count": args.dft_transition_min_count,
                "transition_max_count": args.dft_transition_max_count,
                "transition_sum_f": args.dft_transition_sum_f,
                "force_ground_state": args.dft_force_ground_state,
                "allow_missing_gs_meta": args.dft_allow_missing_gs_meta,
                "run_lrtddft_stage": args.run_lrtddft_stage,
                "scissor_ev": args.dft_scissor_ev,
            },
            "sbe": {
                "t_fs": args.t_fs,
                "dt_as": args.dt_as,
                "gamma": args.gamma,
                "e_field_ampl": args.e_field_ampl,
                "omega_l": args.omega_l,
                "mixing": args.mixing,
                "ignore_feedback": args.ignore_feedback,
                "e_max": args.e_max,
                "density_m3": args.density_m3,
                "polarization_scale": args.polarization_scale,
                "osc_norm_factor": args.osc_norm_factor,
                "chi_eps": args.chi_eps,
                "response_power_cutoff": args.response_power_cutoff,
                "chi_max_ev": args.chi_max_ev,
                "allow_feedback_in_sequential": args.allow_feedback_in_sequential,
            },
            "uppe": {
                "x_window": args.x_window,
                "w0": args.w0,
                "nx": args.nx,
                "dz": args.dz,
                "z_steps": args.z_steps,
            },
            "coupling": {
                "iters": args.iters,
                "rtol": args.rtol,
                "feedback_beta": args.feedback_beta,
                "feedback_norm": args.feedback_norm,
                "feedback_norm_eps": args.feedback_norm_eps,
            },
        },
        "effective_params": effective_params,
        "dft": {
            "ran": args.run_dft,
            "env": args.dft_env,
            "kpts": args.dft_kpts,
            "gs_mode": args.dft_gs_mode,
            "gs_nbands": args.dft_gs_nbands,
            "pw_ecut_ev": args.dft_pw_ecut_ev,
            "response_kpts": args.dft_response_kpts,
            "response_nbands": args.dft_response_nbands,
            "response_out_csv": args.dft_response_out_csv,
            "transition_e_min": args.dft_transition_e_min,
            "transition_e_max": args.dft_transition_e_max,
            "transition_min_rel_height": args.dft_transition_min_rel_height,
            "transition_min_sep_ev": args.dft_transition_min_sep_ev,
            "transition_min_count": args.dft_transition_min_count,
            "transition_max_count": args.dft_transition_max_count,
            "transition_sum_f": args.dft_transition_sum_f,
            "force_ground_state": args.dft_force_ground_state,
            "allow_missing_gs_meta": args.dft_allow_missing_gs_meta,
            "run_lrtddft_stage": args.run_lrtddft_stage,
            "scissor_ev": args.dft_scissor_ev,
            "constants": dft_constants,
            "scripts": {
                "gaas_bulk_gpaw.py": _file_info(dft_dir / "gaas_bulk_gpaw.py"),
                "gaas_dielectric_response.py": _file_info(dft_dir / "gaas_dielectric_response.py"),
                "build_transitions_from_dielectric.py": _file_info(
                    dft_dir / "build_transitions_from_dielectric.py"
                ),
                "gaas_lrtddft.py": _file_info(dft_dir / "gaas_lrtddft.py"),
            },
            "values": {
                "omega0_rad_s": _load_scalar(dft_dir / "omega0_rad_s.npy"),
                "d_eff_Cm": _load_scalar(dft_dir / "d_eff_Cm.npy"),
                "E0_eV": _load_scalar(dft_dir / "E0_eV.npy"),
                "f0": _load_scalar(dft_dir / "f0.npy"),
            },
            "outputs": {
                "omega0_rad_s": _file_info(dft_dir / "omega0_rad_s.npy"),
                "d_eff_Cm": _file_info(dft_dir / "d_eff_Cm.npy"),
                "E0_eV": _file_info(dft_dir / "E0_eV.npy"),
                "f0": _file_info(dft_dir / "f0.npy"),
                "transitions_E_eV": _file_info(dft_dir / "transitions_E_eV.npy"),
                "transitions_f": _file_info(dft_dir / "transitions_f.npy"),
                "transitions_meta": _file_info(dft_dir / "transitions_meta.json"),
                "dielectric_meta": _file_info(dft_dir / "gaas_dielectric_meta.json"),
                "dielectric_csv": _file_info(base / args.dft_response_out_csv),
            },
        },
        "outputs": {
            "sbe": {
                "time_s": _file_info(sbe_dir / "time_s.npy"),
                "E_t": _file_info(sbe_dir / "E_t.npy"),
                "polarization": _file_info(sbe_dir / "polarization.npy"),
                "population": _file_info(sbe_dir / "population.npy"),
                "omega_rad_s": _file_info(sbe_dir / "omega_rad_s.npy"),
                "chi_eff_complex": _file_info(sbe_dir / "chi_eff_complex.npy"),
                "chi_eff_real": _file_info(sbe_dir / "chi_eff_real.npy"),
                "chi_eff_imag": _file_info(sbe_dir / "chi_eff_imag.npy"),
                "probe_chi_eff_complex": _file_info(sbe_dir / "probe_chi_eff_complex.npy"),
                "probe_chi_eff_real": _file_info(sbe_dir / "probe_chi_eff_real.npy"),
                "probe_chi_eff_imag": _file_info(sbe_dir / "probe_chi_eff_imag.npy"),
                "probe_chi_support_mask": _file_info(sbe_dir / "probe_chi_support_mask.npy"),
                "alpha_rel": _file_info(sbe_dir / "alpha_rel.npy"),
                "n_rel": _file_info(sbe_dir / "n_rel.npy"),
                "probe_alpha_rel": _file_info(sbe_dir / "probe_alpha_rel.npy"),
                "probe_n_rel": _file_info(sbe_dir / "probe_n_rel.npy"),
                "probe_alpha_m_inv": _file_info(sbe_dir / "probe_alpha_m_inv.npy"),
                "probe_n_real": _file_info(sbe_dir / "probe_n_real.npy"),
                "E_t_feedback": _file_info(sbe_dir / "E_t_feedback.npy"),
                "transition_model": _file_info(sbe_dir / "transition_model.json"),
                "transition_omega_rad_s_used": _file_info(sbe_dir / "transition_omega_rad_s_used.npy"),
                "transition_d_eff_Cm_used": _file_info(sbe_dir / "transition_d_eff_Cm_used.npy"),
                "transition_osc_used": _file_info(sbe_dir / "transition_osc_used.npy"),
                "transition_weight_used": _file_info(sbe_dir / "transition_weight_used.npy"),
                "transition_energy_eV_used": _file_info(sbe_dir / "transition_energy_eV_used.npy"),
            },
            "uppe": {
                "E_xt_out": _file_info(uppe_dir / "outputs" / "E_xt_out.npy"),
                "x_m": _file_info(uppe_dir / "outputs" / "x_m.npy"),
                "time_s": _file_info(uppe_dir / "outputs" / "time_s.npy"),
            },
        },
        "validation_report": validation_report,
        "calibration_report": calibration_report,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[OK] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
