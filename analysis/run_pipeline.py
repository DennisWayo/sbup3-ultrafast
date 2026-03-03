#!/usr/bin/env python3
"""
SBUP3 pipeline runner.

Runs optional TDDFT -> SBE -> UPPE -> coupling and writes a run manifest.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBUP3 pipeline")
    parser.add_argument(
        "--mode",
        choices=["sequential", "coupling", "full"],
        default="coupling",
        help="Pipeline mode",
    )
    parser.add_argument("--run-dft", action="store_true", help="Run DFT/TDDFT stage")
    parser.add_argument("--dft-env", default=None, help="Conda env for DFT stage")
    parser.add_argument("--plots", action="store_true", help="Generate figures")
    parser.add_argument("--validate", action="store_true", help="Run validation report")
    parser.add_argument(
        "--report-dir",
        default="analysis/reports",
        help="Directory for run manifests",
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

    # Optional DFT stage
    if args.run_dft:
        dft_python = [sys.executable]
        if args.dft_env:
            conda_exe = _conda_for_env(args.dft_env) or _find_conda_exe()
            dft_python = [conda_exe, "run", "-n", args.dft_env, "python"]

        _run(dft_python + [str(dft_dir / "gaas_bulk_gpaw.py")], cwd=dft_dir, env=env)
        _run(dft_python + [str(dft_dir / "gaas_lrtddft.py")], cwd=dft_dir, env=env)

    # Main pipeline
    if args.mode in {"sequential", "full"}:
        _run([sys.executable, str(sbe_dir / "sbe_rk4.py")], cwd=sbe_dir, env=env)
        _run([sys.executable, "-m", "uppe.uppe_2d"], cwd=base, env=env)

    if args.mode in {"coupling", "full"}:
        _run([sys.executable, "-m", "coupling.sbup3_loop"], cwd=base, env=env)

    # Optional plots
    if args.plots:
        _run([sys.executable, str(base / "analysis" / "visualize_sbup3.py")], cwd=base, env=env)

    # Optional validation
    validation_report = None
    if args.validate:
        _run([sys.executable, str(base / "analysis" / "validate_sbup3.py")], cwd=base, env=env)
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
            dft_dir / "gaas_bulk_gpaw.py", {"XC", "KPTS", "NBANDS", "SMEARING"}
        ),
        "gaas_lrtddft.py": _parse_py_constants(
            dft_dir / "gaas_lrtddft.py", {"OSC_MIN"}
        ),
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
            "sbe": {
                "t_fs": args.t_fs,
                "dt_as": args.dt_as,
                "gamma": args.gamma,
                "e_field_ampl": args.e_field_ampl,
                "omega_l": args.omega_l,
                "mixing": args.mixing,
                "ignore_feedback": args.ignore_feedback,
                "e_max": args.e_max,
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
        "dft": {
            "ran": args.run_dft,
            "env": args.dft_env,
            "constants": dft_constants,
            "scripts": {
                "gaas_bulk_gpaw.py": _file_info(dft_dir / "gaas_bulk_gpaw.py"),
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
                "alpha_rel": _file_info(sbe_dir / "alpha_rel.npy"),
                "n_rel": _file_info(sbe_dir / "n_rel.npy"),
                "E_t_feedback": _file_info(sbe_dir / "E_t_feedback.npy"),
            },
            "uppe": {
                "E_xt_out": _file_info(uppe_dir / "outputs" / "E_xt_out.npy"),
                "x_m": _file_info(uppe_dir / "outputs" / "x_m.npy"),
                "time_s": _file_info(uppe_dir / "outputs" / "time_s.npy"),
            },
        },
        "validation_report": validation_report,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[OK] Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
