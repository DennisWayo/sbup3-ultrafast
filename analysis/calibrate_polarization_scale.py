#!/usr/bin/env python3
"""
Calibrate SBUP3 optical normalization against reference data.

Modes:
1) physical (default): fit density + oscillator normalization under bounds/prior.
2) free: legacy single scalar ratio fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


HBAR = 1.054571817e-34  # J*s
E0 = 1.602176634e-19    # C
C0 = 299792458.0        # m/s
DEFAULT_LATTICE_A_ANG = 5.6533


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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
            "alpha": pick("alpha", "alphaminv", "absorption", "absorptioncoeff"),
            "n": pick("n", "refractiveindex"),
            "chi_real": pick("chireal", "rechi", "chire"),
            "chi_imag": pick("chiimag", "imchi", "chiim"),
        }

        data = {k: [] for k, v in cols.items() if v is not None}
        for row in reader:
            try:
                data["energy_eV"].append(float(row[cols["energy_eV"]]))
                for k, v in cols.items():
                    if k == "energy_eV" or v is None:
                        continue
                    raw = row.get(v, "")
                    data[k].append(float(raw) if raw != "" else float("nan"))
            except Exception:
                continue

    out = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    if len(out["energy_eV"]) == 0:
        raise ValueError("Reference CSV has no valid numeric rows")

    supported = [k for k in ("alpha", "n", "chi_real", "chi_imag") if k in out]
    if not supported:
        raise ValueError(
            "Reference CSV must include at least one channel: "
            "alpha (or alpha_m_inv), n, chi_real, chi_imag"
        )

    order = np.argsort(out["energy_eV"])
    for k in out:
        out[k] = out[k][order]
    return out


def _load_response_set(sbe: Path, prefix: str) -> dict[str, np.ndarray] | None:
    omega_path = sbe / f"{prefix}omega_rad_s.npy"
    if not omega_path.exists():
        return None
    omega = np.load(omega_path)

    chi_path = sbe / f"{prefix}chi_eff_complex.npy"
    if chi_path.exists():
        chi = np.load(chi_path)
    else:
        chi_r_path = sbe / f"{prefix}chi_eff_real.npy"
        chi_i_path = sbe / f"{prefix}chi_eff_imag.npy"
        if not (chi_r_path.exists() and chi_i_path.exists()):
            return None
        chi = np.load(chi_r_path) + 1j * np.load(chi_i_path)

    return {"omega_rad_s": omega, "chi": chi}


def _load_simulation(base: Path, lattice_a_ang: float) -> dict[str, Any]:
    sbe = base / "sbe"
    probe = _load_response_set(sbe, "probe_")
    drive = _load_response_set(sbe, "")
    if probe is not None:
        response = probe
        response_source = "probe"
    elif drive is not None:
        response = drive
        response_source = "drive"
    else:
        raise FileNotFoundError("Missing SBE response outputs (drive/probe).")

    transition_model = {}
    tmodel_path = sbe / "transition_model.json"
    if tmodel_path.exists():
        try:
            transition_model = json.loads(tmodel_path.read_text())
        except Exception:
            transition_model = {}

    base_scale = _safe_float(
        transition_model.get("polarization_scale", os.getenv("SBUP3_POLARIZATION_SCALE", "1.0")),
        1.0,
    )
    base_density = _safe_float(
        transition_model.get("density_m3", os.getenv("SBUP3_DENSITY_M3", "1.0")),
        1.0,
    )
    base_osc_norm = _safe_float(
        transition_model.get("osc_norm_factor", os.getenv("SBUP3_OSC_NORM_FACTOR", "1.0")),
        1.0,
    )
    if base_density <= 0.0:
        base_density = 1.0
    if base_osc_norm <= 0.0:
        base_osc_norm = 1.0

    lattice_m = max(float(lattice_a_ang), 1e-9) * 1e-10
    pair_density_m3 = 4.0 / max(lattice_m**3, 1e-60)

    omega = response["omega_rad_s"]
    chi = response["chi"]
    return {
        "omega_rad_s": omega,
        "energy_eV": HBAR * omega / E0,
        "chi": chi,
        "response_source": response_source,
        "base_scale": base_scale,
        "base_density_m3": base_density,
        "base_osc_norm_factor": base_osc_norm,
        "pair_density_m3": pair_density_m3,
        "transition_model": transition_model,
    }


def _compare_series(
    sim_energy: np.ndarray,
    sim_values: np.ndarray,
    ref_energy: np.ndarray,
    ref_values: np.ndarray,
) -> dict[str, float] | None:
    finite_sim = np.isfinite(sim_energy) & np.isfinite(sim_values)
    finite_ref = np.isfinite(ref_energy) & np.isfinite(ref_values)
    sim_energy = sim_energy[finite_sim]
    sim_values = sim_values[finite_sim]
    ref_energy = ref_energy[finite_ref]
    ref_values = ref_values[finite_ref]

    if len(sim_energy) < 2 or len(ref_energy) < 2:
        return None

    e_min = max(float(np.min(sim_energy)), float(np.min(ref_energy)))
    e_max = min(float(np.max(sim_energy)), float(np.max(ref_energy)))
    mask = (ref_energy >= e_min) & (ref_energy <= e_max)
    if int(mask.sum()) < 5:
        return None

    ref_e = ref_energy[mask]
    ref_v = ref_values[mask]
    sim_v = np.interp(ref_e, sim_energy, sim_values)

    resid = sim_v - ref_v
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    ref_norm = float(np.max(np.abs(ref_v))) or 1.0
    rmse_norm = rmse / ref_norm
    mae_norm = mae / ref_norm

    return {
        "energy_min_eV": e_min,
        "energy_max_eV": e_max,
        "rmse": rmse,
        "rmse_norm": rmse_norm,
        "mae": mae,
        "mae_norm": mae_norm,
        "n_points": int(mask.sum()),
    }


def _predicted_channels(omega: np.ndarray, chi_base: np.ndarray, ratio: float) -> dict[str, np.ndarray]:
    chi = ratio * chi_base
    n_tilde = np.sqrt(1.0 + chi)
    alpha = 2.0 * omega * np.imag(n_tilde) / C0
    return {
        "alpha": alpha,
        "n": np.real(n_tilde),
        "chi_real": np.real(chi),
        "chi_imag": np.imag(chi),
    }


def _score_ratio(
    ratio: float,
    sim_energy: np.ndarray,
    omega: np.ndarray,
    chi_base: np.ndarray,
    ref: dict[str, np.ndarray],
    channels: list[str],
    weights: dict[str, float],
) -> tuple[float, dict[str, dict[str, float]]]:
    pred = _predicted_channels(omega, chi_base, ratio)
    per_channel: dict[str, dict[str, float]] = {}
    total = 0.0
    wsum = 0.0

    for ch in channels:
        if ch not in ref:
            continue
        comp = _compare_series(sim_energy, pred[ch], ref["energy_eV"], ref[ch])
        if comp is None:
            continue
        per_channel[ch] = comp
        w = max(float(weights.get(ch, 1.0)), 0.0)
        total += w * comp["rmse_norm"]
        wsum += w

    if wsum <= 0.0:
        return float("inf"), per_channel
    return total / wsum, per_channel


def _fit_ratio_free(
    sim_energy: np.ndarray,
    omega: np.ndarray,
    chi_base: np.ndarray,
    ref: dict[str, np.ndarray],
    channels: list[str],
    weights: dict[str, float],
    log10_min: float,
    log10_max: float,
    grid: int,
) -> dict[str, Any]:
    ratios = np.logspace(log10_min, log10_max, grid)
    scores = np.full_like(ratios, np.nan, dtype=float)
    details: list[dict[str, dict[str, float]]] = []
    for i, ratio in enumerate(ratios):
        score, per_channel = _score_ratio(
            float(ratio),
            sim_energy,
            omega,
            chi_base,
            ref,
            channels,
            weights,
        )
        scores[i] = score
        details.append(per_channel)

    finite = np.isfinite(scores)
    if not np.any(finite):
        raise RuntimeError("Calibration failed: no finite score in scalar-ratio search grid.")

    idx = int(np.nanargmin(scores))
    best_ratio = float(ratios[idx])
    best_score = float(scores[idx])
    best_details = details[idx]
    best_on_boundary = bool(idx == 0 or idx == len(ratios) - 1)

    if 0 < idx < len(ratios) - 1:
        lo = float(ratios[idx - 1])
        hi = float(ratios[idx + 1])
        local_ratios = np.geomspace(lo, hi, max(201, grid // 4))
        local_scores = np.full_like(local_ratios, np.nan, dtype=float)
        local_details: list[dict[str, dict[str, float]]] = []
        for i_local, ratio in enumerate(local_ratios):
            score, per_channel = _score_ratio(
                float(ratio),
                sim_energy,
                omega,
                chi_base,
                ref,
                channels,
                weights,
            )
            local_scores[i_local] = score
            local_details.append(per_channel)
        local_idx = int(np.nanargmin(local_scores))
        if np.isfinite(local_scores[local_idx]) and float(local_scores[local_idx]) < best_score:
            best_ratio = float(local_ratios[local_idx])
            best_score = float(local_scores[local_idx])
            best_details = local_details[local_idx]
            best_on_boundary = bool(
                np.isclose(best_ratio, ratios[0]) or np.isclose(best_ratio, ratios[-1])
            )

    return {
        "best_ratio": best_ratio,
        "objective": best_score,
        "metrics": best_details,
        "best_on_boundary": best_on_boundary,
        "search": {
            "log10_min": float(log10_min),
            "log10_max": float(log10_max),
            "n_points": int(grid),
        },
    }


def _prior_term(value: float, prior: float, rel_sigma: float) -> float:
    if value <= 0.0 or prior <= 0.0 or rel_sigma <= 0.0:
        return 0.0
    sigma_ln = np.log1p(rel_sigma)
    if not np.isfinite(sigma_ln) or sigma_ln <= 0.0:
        return 0.0
    z = np.log(value / prior) / sigma_ln
    return float(z * z)


def _bounds_from_mask(
    density_grid: np.ndarray,
    osc_grid: np.ndarray,
    ratio_grid: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    if not np.any(mask):
        return {
            "count": 0,
            "density_m3_min": None,
            "density_m3_max": None,
            "osc_norm_min": None,
            "osc_norm_max": None,
            "ratio_min": None,
            "ratio_max": None,
        }
    ii, jj = np.where(mask)
    dens = density_grid[ii]
    osc = osc_grid[jj]
    ratio = ratio_grid[ii, jj]
    return {
        "count": int(mask.sum()),
        "density_m3_min": float(np.min(dens)),
        "density_m3_max": float(np.max(dens)),
        "osc_norm_min": float(np.min(osc)),
        "osc_norm_max": float(np.max(osc)),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
    }


def _fit_physical_grid(
    sim_energy: np.ndarray,
    omega: np.ndarray,
    chi_base: np.ndarray,
    ref: dict[str, np.ndarray],
    channels: list[str],
    weights: dict[str, float],
    base_density_m3: float,
    base_osc_norm: float,
    density_grid: np.ndarray,
    osc_grid: np.ndarray,
    density_prior_m3: float,
    density_rel_sigma: float,
    osc_prior: float,
    osc_rel_sigma: float,
    prior_weight: float,
    delta68: float,
    delta95: float,
) -> dict[str, Any]:
    total_obj = np.full((len(density_grid), len(osc_grid)), np.inf, dtype=float)
    data_obj = np.full_like(total_obj, np.inf)
    prior_obj = np.full_like(total_obj, np.inf)
    ratio_grid = np.full_like(total_obj, np.nan)
    metrics_grid: list[list[dict[str, dict[str, float]] | None]] = [
        [None for _ in range(len(osc_grid))] for _ in range(len(density_grid))
    ]

    best_total = float("inf")
    best_data = float("inf")
    best_prior = float("inf")
    best_i = -1
    best_j = -1
    best_ratio = float("nan")
    best_metrics: dict[str, dict[str, float]] = {}

    ratio_cache: dict[float, tuple[float, dict[str, dict[str, float]]]] = {}

    base_density_safe = max(base_density_m3, 1e-300)
    base_osc_safe = max(base_osc_norm, 1e-300)
    for i, density_m3 in enumerate(density_grid):
        for j, osc_norm in enumerate(osc_grid):
            ratio = (density_m3 / base_density_safe) * (osc_norm / base_osc_safe)
            ratio_grid[i, j] = ratio
            if not np.isfinite(ratio) or ratio <= 0.0:
                continue

            key = round(np.log10(ratio), 12)
            cached = ratio_cache.get(key)
            if cached is None:
                score_data, per_channel = _score_ratio(
                    ratio,
                    sim_energy,
                    omega,
                    chi_base,
                    ref,
                    channels,
                    weights,
                )
                ratio_cache[key] = (score_data, per_channel)
            else:
                score_data, per_channel = cached

            if not np.isfinite(score_data):
                continue

            score_prior = (
                _prior_term(float(density_m3), density_prior_m3, density_rel_sigma)
                + _prior_term(float(osc_norm), osc_prior, osc_rel_sigma)
            )
            score_total = float(score_data + max(prior_weight, 0.0) * score_prior)

            data_obj[i, j] = score_data
            prior_obj[i, j] = score_prior
            total_obj[i, j] = score_total
            metrics_grid[i][j] = per_channel

            if score_total < best_total:
                best_total = score_total
                best_data = score_data
                best_prior = score_prior
                best_i = i
                best_j = j
                best_ratio = float(ratio)
                best_metrics = per_channel

    if best_i < 0:
        raise RuntimeError("Calibration failed: no finite point in physical parameter grid.")

    finite = np.isfinite(total_obj)
    delta68 = max(float(delta68), 0.0)
    delta95 = max(float(delta95), delta68)
    mask68 = finite & (total_obj <= (best_total + delta68))
    mask95 = finite & (total_obj <= (best_total + delta95))
    if not np.any(mask68):
        mask68[best_i, best_j] = True
    if not np.any(mask95):
        mask95[best_i, best_j] = True

    best_on_density_boundary = bool(best_i == 0 or best_i == len(density_grid) - 1)
    best_on_osc_boundary = bool(best_j == 0 or best_j == len(osc_grid) - 1)

    return {
        "best_density_m3": float(density_grid[best_i]),
        "best_osc_norm_factor": float(osc_grid[best_j]),
        "best_ratio": best_ratio,
        "objective_total": best_total,
        "objective_data": best_data,
        "objective_prior": best_prior,
        "metrics": best_metrics,
        "best_on_boundary": bool(best_on_density_boundary or best_on_osc_boundary),
        "best_on_density_boundary": best_on_density_boundary,
        "best_on_osc_boundary": best_on_osc_boundary,
        "uncertainty": {
            "delta68": delta68,
            "delta95": delta95,
            "bounds68": _bounds_from_mask(density_grid, osc_grid, ratio_grid, mask68),
            "bounds95": _bounds_from_mask(density_grid, osc_grid, ratio_grid, mask95),
        },
        "grid": {
            "density_m3_min": float(np.min(density_grid)),
            "density_m3_max": float(np.max(density_grid)),
            "density_points": int(len(density_grid)),
            "osc_norm_min": float(np.min(osc_grid)),
            "osc_norm_max": float(np.max(osc_grid)),
            "osc_points": int(len(osc_grid)),
        },
    }


def _write_env_file(
    path: Path,
    density_m3: float,
    osc_norm_factor: float,
    polarization_scale: float,
) -> None:
    lines = [
        "# SBUP3 calibration recommendation",
        f"export SBUP3_DENSITY_M3={density_m3:.16g}",
        f"export SBUP3_OSC_NORM_FACTOR={osc_norm_factor:.16g}",
        f"export SBUP3_POLARIZATION_SCALE={polarization_scale:.16g}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_channels(raw: str) -> list[str]:
    supported = {"alpha", "n", "chi_real", "chi_imag"}
    channels = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if not channels:
        return ["alpha", "n"]
    for ch in channels:
        if ch not in supported:
            raise ValueError(f"Unsupported channel '{ch}'. Supported: {sorted(supported)}")
    return channels


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SBUP3 normalization")
    parser.add_argument("--ref-csv", required=True, help="Reference CSV (energy_eV + optical columns)")
    parser.add_argument(
        "--fit-mode",
        choices=["physical", "free"],
        default="physical",
        help="physical: constrained density+osc fit; free: legacy scalar ratio fit.",
    )
    parser.add_argument(
        "--fit-channels",
        default="alpha,n",
        help="Comma-separated channels: alpha,n,chi_real,chi_imag",
    )
    parser.add_argument("--alpha-weight", type=float, default=1.0, help="Weight for alpha channel")
    parser.add_argument("--n-weight", type=float, default=1.0, help="Weight for n channel")
    parser.add_argument("--chi-real-weight", type=float, default=1.0, help="Weight for chi_real channel")
    parser.add_argument("--chi-imag-weight", type=float, default=1.0, help="Weight for chi_imag channel")

    parser.add_argument("--log10-ratio-min", type=float, default=-16.0, help="Legacy/free ratio lower log10 bound")
    parser.add_argument("--log10-ratio-max", type=float, default=16.0, help="Legacy/free ratio upper log10 bound")
    parser.add_argument("--grid", type=int, default=801, help="Legacy/free ratio search grid size")

    parser.add_argument("--lattice-a-ang", type=float, default=DEFAULT_LATTICE_A_ANG, help="Lattice constant for default pair-density prior [A]")
    parser.add_argument("--density-prior-m3", type=float, default=None, help="Density prior [1/m^3] for physical mode")
    parser.add_argument("--density-min-m3", type=float, default=None, help="Density lower bound [1/m^3]")
    parser.add_argument("--density-max-m3", type=float, default=None, help="Density upper bound [1/m^3]")
    parser.add_argument("--density-grid", type=int, default=101, help="Density grid points")
    parser.add_argument("--density-rel-sigma", type=float, default=0.25, help="Relative 1-sigma prior width for density (log-space)")

    parser.add_argument("--osc-prior", type=float, default=1.0, help="Oscillator normalization prior")
    parser.add_argument("--osc-min", type=float, default=0.5, help="Oscillator normalization lower bound")
    parser.add_argument("--osc-max", type=float, default=2.0, help="Oscillator normalization upper bound")
    parser.add_argument("--osc-grid", type=int, default=101, help="Oscillator normalization grid points")
    parser.add_argument("--osc-rel-sigma", type=float, default=0.25, help="Relative 1-sigma prior width for oscillator normalization (log-space)")

    parser.add_argument("--prior-weight", type=float, default=0.05, help="Weight for prior penalty in physical mode objective")
    parser.add_argument("--objective-delta-68", type=float, default=0.02, help="Objective delta for 68%% interval surrogate")
    parser.add_argument("--objective-delta-95", type=float, default=0.05, help="Objective delta for 95%% interval surrogate")

    parser.add_argument(
        "--target-density-m3",
        type=float,
        default=None,
        help="If set, also report scale adjusted to this target density",
    )
    parser.add_argument(
        "--report-dir",
        default="analysis/reports",
        help="Output directory for calibration report",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional explicit output JSON path",
    )
    parser.add_argument(
        "--write-env",
        default=None,
        help="Optional path to write export commands",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    ref_path = Path(args.ref_csv)
    if not ref_path.is_absolute():
        ref_path = base / ref_path
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {ref_path}")

    channels = _parse_channels(args.fit_channels)
    weights = {
        "alpha": float(args.alpha_weight),
        "n": float(args.n_weight),
        "chi_real": float(args.chi_real_weight),
        "chi_imag": float(args.chi_imag_weight),
    }

    ref = _load_reference_csv(ref_path)
    sim = _load_simulation(base, lattice_a_ang=float(args.lattice_a_ang))

    missing = [ch for ch in channels if ch not in ref]
    if missing:
        raise RuntimeError(
            "Reference CSV is missing required channels for calibration: " + ", ".join(missing)
        )

    base_scale = float(sim["base_scale"])
    base_density = float(sim["base_density_m3"])
    base_osc_norm = float(sim["base_osc_norm_factor"])

    fit_result: dict[str, Any]
    fit_summary: dict[str, Any]
    if args.fit_mode == "free":
        fit_result = _fit_ratio_free(
            sim_energy=sim["energy_eV"],
            omega=sim["omega_rad_s"],
            chi_base=sim["chi"],
            ref=ref,
            channels=channels,
            weights=weights,
            log10_min=float(args.log10_ratio_min),
            log10_max=float(args.log10_ratio_max),
            grid=max(int(args.grid), 21),
        )
        best_ratio = float(fit_result["best_ratio"])
        recommended_density_m3 = base_density
        recommended_osc_norm = base_osc_norm
        recommended_scale = base_scale * best_ratio
        fit_summary = {
            "mode": "free",
            "best_on_boundary": bool(fit_result["best_on_boundary"]),
            "objective": float(fit_result["objective"]),
            "channel_metrics": fit_result["metrics"],
            "search": fit_result["search"],
        }
    else:
        density_prior = (
            float(args.density_prior_m3)
            if args.density_prior_m3 is not None and args.density_prior_m3 > 0.0
            else (base_density if base_density >= 1e20 else float(sim["pair_density_m3"]))
        )
        density_min = (
            float(args.density_min_m3)
            if args.density_min_m3 is not None and args.density_min_m3 > 0.0
            else density_prior / 10.0
        )
        density_max = (
            float(args.density_max_m3)
            if args.density_max_m3 is not None and args.density_max_m3 > density_min
            else density_prior * 10.0
        )
        if not (density_min > 0.0 and density_max > density_min):
            raise ValueError("Invalid density bounds.")
        if not (args.osc_min > 0.0 and args.osc_max > args.osc_min):
            raise ValueError("Invalid oscillator normalization bounds.")

        density_grid = np.logspace(
            np.log10(density_min),
            np.log10(density_max),
            max(int(args.density_grid), 11),
        )
        osc_grid = np.logspace(
            np.log10(float(args.osc_min)),
            np.log10(float(args.osc_max)),
            max(int(args.osc_grid), 11),
        )

        fit_result = _fit_physical_grid(
            sim_energy=sim["energy_eV"],
            omega=sim["omega_rad_s"],
            chi_base=sim["chi"],
            ref=ref,
            channels=channels,
            weights=weights,
            base_density_m3=base_density,
            base_osc_norm=base_osc_norm,
            density_grid=density_grid,
            osc_grid=osc_grid,
            density_prior_m3=density_prior,
            density_rel_sigma=max(float(args.density_rel_sigma), 0.0),
            osc_prior=max(float(args.osc_prior), 1e-30),
            osc_rel_sigma=max(float(args.osc_rel_sigma), 0.0),
            prior_weight=max(float(args.prior_weight), 0.0),
            delta68=float(args.objective_delta_68),
            delta95=float(args.objective_delta_95),
        )

        best_ratio = float(fit_result["best_ratio"])
        recommended_density_m3 = float(fit_result["best_density_m3"])
        recommended_osc_norm = float(fit_result["best_osc_norm_factor"])
        recommended_scale = base_scale
        fit_summary = {
            "mode": "physical",
            "best_on_boundary": bool(fit_result["best_on_boundary"]),
            "best_on_density_boundary": bool(fit_result["best_on_density_boundary"]),
            "best_on_osc_boundary": bool(fit_result["best_on_osc_boundary"]),
            "objective_total": float(fit_result["objective_total"]),
            "objective_data": float(fit_result["objective_data"]),
            "objective_prior": float(fit_result["objective_prior"]),
            "channel_metrics": fit_result["metrics"],
            "search": fit_result["grid"],
            "priors": {
                "density_prior_m3": density_prior,
                "density_rel_sigma": float(args.density_rel_sigma),
                "osc_prior": float(args.osc_prior),
                "osc_rel_sigma": float(args.osc_rel_sigma),
                "prior_weight": float(args.prior_weight),
            },
            "uncertainty": fit_result["uncertainty"],
        }

    effective_factor = recommended_scale * recommended_density_m3 * recommended_osc_norm
    target_density = args.target_density_m3
    scale_for_target_density = None
    if target_density is not None and target_density > 0.0:
        target_ratio_term = (target_density / max(base_density, 1e-300)) * (
            recommended_osc_norm / max(base_osc_norm, 1e-300)
        )
        scale_for_target_density = (
            base_scale * best_ratio / max(target_ratio_term, 1e-300)
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = base / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else report_dir / f"calibration_{timestamp}.json"
    if not out_json.is_absolute():
        out_json = base / out_json

    report = {
        "timestamp_utc": timestamp,
        "reference_csv": str(ref_path),
        "response_source": sim["response_source"],
        "fit_mode": args.fit_mode,
        "fit_channels": channels,
        "weights": weights,
        "fit": fit_summary,
        "base_state": {
            "base_scale": base_scale,
            "base_density_m3": base_density,
            "base_osc_norm_factor": base_osc_norm,
            "pair_density_m3": float(sim["pair_density_m3"]),
            "transition_model": sim.get("transition_model"),
        },
        "best_ratio": best_ratio,
        "recommended_density_m3": recommended_density_m3,
        "recommended_osc_norm_factor": recommended_osc_norm,
        "recommended_scale": recommended_scale,
        "recommended_effective_factor_scale_density_osc": effective_factor,
        "target_density_m3": target_density,
        "recommended_scale_for_target_density": scale_for_target_density,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    if args.write_env:
        env_path = Path(args.write_env)
        if not env_path.is_absolute():
            env_path = base / env_path
        density_for_env = recommended_density_m3
        scale_for_env = recommended_scale
        if target_density is not None and target_density > 0.0 and scale_for_target_density is not None:
            density_for_env = target_density
            scale_for_env = scale_for_target_density
        _write_env_file(
            env_path,
            density_m3=density_for_env,
            osc_norm_factor=recommended_osc_norm,
            polarization_scale=scale_for_env,
        )
        print(f"[OK] Wrote env recommendations: {env_path}")

    print(f"[OK] Wrote calibration report: {out_json}")
    print(f"fit_mode = {args.fit_mode}")
    print(f"response_source = {sim['response_source']}")
    print(f"best_ratio = {best_ratio:.12e}")
    print(f"recommended_density_m3 = {recommended_density_m3:.12e}")
    print(f"recommended_osc_norm_factor = {recommended_osc_norm:.12e}")
    print(f"recommended_scale = {recommended_scale:.12e}")
    if fit_summary.get("best_on_boundary"):
        print("[WARN] Best fit is on at least one search boundary.")
    if target_density is not None and scale_for_target_density is not None:
        print(f"target_density_m3 = {target_density:.12e}")
        print(f"recommended_scale_for_target_density = {scale_for_target_density:.12e}")


if __name__ == "__main__":
    main()
