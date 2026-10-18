"""
SBUP3 - Semiconductor Bloch Equations (RK4, TDDFT-driven)
==========================================================

This solver supports both:
1) a legacy single-transition two-level model, and
2) a multi-transition reduced model built from TDDFT transition lists.

Outputs are kept backward-compatible with existing SBUP3 analysis scripts.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


# ============================================================
# Environment overrides (optional)
# ============================================================
def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    return float(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return int(default)
    return int(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# ============================================================
# Constants
# ============================================================
HBAR = 1.054571817e-34     # J*s
E0 = 1.602176634e-19       # C
EPS0 = 8.8541878128e-12    # F/m
C0 = 299792458.0           # m/s
ME = 9.1093837015e-31      # kg


# ============================================================
# Paths
# ============================================================
BASE = Path(__file__).resolve().parents[1]
HERE = BASE / "sbe"
DFT = BASE / "dft"
ANALYSIS_DATA = BASE / "analysis" / "data"


def _save_output(name: str, array: np.ndarray) -> None:
    np.save(HERE / name, array)


def _load_scalar(path: Path, default: float | None = None) -> float | None:
    if not path.exists():
        return default
    try:
        return float(np.load(path).item())
    except Exception:
        return default


omega0_scalar = _load_scalar(DFT / "omega0_rad_s.npy", 0.0) or 0.0
d_eff_scalar = _load_scalar(DFT / "d_eff_Cm.npy", 0.0) or 0.0
E0_scalar_eV = _load_scalar(DFT / "E0_eV.npy", 0.0) or 0.0
f0_scalar = _load_scalar(DFT / "f0.npy", 0.0) or 0.0


def dipole_from_f(omega_rad_s: np.ndarray, osc_strength: np.ndarray) -> np.ndarray:
    """SI-consistent oscillator-strength inversion."""
    omega_safe = np.maximum(omega_rad_s, 1e-30)
    f_safe = np.maximum(osc_strength, 0.0)
    return np.sqrt((3.0 * HBAR * E0 * E0 / (2.0 * ME * omega_safe)) * f_safe)


def _load_transition_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    energies: list[float] = []
    osc: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        if "energy_eV" not in reader.fieldnames or "osc" not in reader.fieldnames:
            return None
        for row in reader:
            try:
                energies.append(float(row["energy_eV"]))
                osc.append(float(row["osc"]))
            except Exception:
                continue
    if not energies:
        return None
    return np.asarray(energies, dtype=float), np.asarray(osc, dtype=float)


def _aggregate_transitions(
    energies_eV: np.ndarray,
    osc: np.ndarray,
    tol_eV: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge near-degenerate transitions by summing oscillator strengths.
    """
    if len(energies_eV) <= 1:
        return energies_eV, osc
    order = np.argsort(energies_eV)
    e_sorted = energies_eV[order]
    f_sorted = osc[order]

    out_e: list[float] = []
    out_f: list[float] = []

    i = 0
    while i < len(e_sorted):
        j = i + 1
        members_e = [e_sorted[i]]
        members_f = [f_sorted[i]]
        while j < len(e_sorted) and abs(e_sorted[j] - e_sorted[i]) <= tol_eV:
            members_e.append(e_sorted[j])
            members_f.append(f_sorted[j])
            j += 1
        f_sum = float(np.sum(members_f))
        if f_sum > 0.0:
            e_eff = float(np.dot(members_e, members_f) / f_sum)
        else:
            e_eff = float(np.mean(members_e))
        out_e.append(e_eff)
        out_f.append(f_sum)
        i = j

    return np.asarray(out_e, dtype=float), np.asarray(out_f, dtype=float)


def load_transition_model() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Returns (omega_vec, d_eff_vec, osc_vec, metadata).
    """
    multi_enable = _env_bool("SBUP3_MULTI_ENABLE", True)
    osc_min = _env_float("SBUP3_MULTI_OSC_MIN", 1e-6)
    max_ev = _env_float("SBUP3_MULTI_MAX_EV", 3.0)
    merge_tol = _env_float("SBUP3_MULTI_MERGE_TOL_EV", 1e-3)
    max_transitions = _env_int("SBUP3_MULTI_MAX_TRANSITIONS", 0)
    auto_gap_target = _env_float("SBUP3_MULTI_TARGET_GAP_EV", 0.0)
    allow_csv_fallback = _env_bool("SBUP3_ALLOW_TRANSITION_CSV_FALLBACK", False)
    require_ground_truth = _env_bool("SBUP3_REQUIRE_GROUND_TRUTH_TRANSITIONS", True)

    meta: dict[str, object] = {
        "multi_enable": bool(multi_enable),
        "source": "single_transition",
    }

    if not multi_enable:
        if require_ground_truth:
            raise RuntimeError(
                "SBUP3_MULTI_ENABLE=0 is incompatible with strict ground-truth mode. "
                "Set SBUP3_MULTI_ENABLE=1 and provide transition arrays."
            )
        omega_vec = np.asarray([omega0_scalar], dtype=float)
        d_vec = np.asarray([abs(d_eff_scalar)], dtype=float)
        f_vec = np.asarray([max(f0_scalar, 0.0)], dtype=float)
        return omega_vec, d_vec, f_vec, meta

    energies_eV = None
    osc = None
    source = None

    npy_e = DFT / "transitions_E_eV.npy"
    npy_f = DFT / "transitions_f.npy"
    if npy_e.exists() and npy_f.exists():
        try:
            energies_eV = np.asarray(np.load(npy_e), dtype=float).reshape(-1)
            osc = np.asarray(np.load(npy_f), dtype=float).reshape(-1)
            source = str(npy_e)
        except Exception:
            energies_eV = None
            osc = None

    if (energies_eV is None or osc is None) and allow_csv_fallback:
        csv_paths = [
            ANALYSIS_DATA / "gaas_tddft_reference.csv",
            DFT / "gaas_tddft_reference.csv",
        ]
        for csv_path in csv_paths:
            parsed = _load_transition_csv(csv_path)
            if parsed is not None:
                energies_eV, osc = parsed
                source = str(csv_path)
                break

    if energies_eV is None or osc is None:
        if require_ground_truth:
            raise RuntimeError(
                "Missing transition ground truth. Expected "
                "dft/transitions_E_eV.npy and dft/transitions_f.npy "
                "(or disable strict mode with SBUP3_REQUIRE_GROUND_TRUTH_TRANSITIONS=0)."
            )
        omega_vec = np.asarray([omega0_scalar], dtype=float)
        d_vec = np.asarray([abs(d_eff_scalar)], dtype=float)
        f_vec = np.asarray([max(f0_scalar, 0.0)], dtype=float)
        meta["source"] = "single_transition_fallback"
        meta["csv_fallback_enabled"] = bool(allow_csv_fallback)
        return omega_vec, d_vec, f_vec, meta

    finite = (
        np.isfinite(energies_eV)
        & np.isfinite(osc)
        & (energies_eV > 0.0)
        & (osc >= 0.0)
    )
    energies_eV = energies_eV[finite]
    osc = osc[finite]

    if max_ev > 0.0:
        sel = energies_eV <= max_ev
        energies_eV = energies_eV[sel]
        osc = osc[sel]

    if osc_min > 0.0:
        sel = osc >= osc_min
        energies_eV = energies_eV[sel]
        osc = osc[sel]

    if len(energies_eV) == 0:
        if require_ground_truth:
            raise RuntimeError(
                "Transition arrays were found but no entries survived filtering. "
                "Relax filters or regenerate transitions."
            )
        omega_vec = np.asarray([omega0_scalar], dtype=float)
        d_vec = np.asarray([abs(d_eff_scalar)], dtype=float)
        f_vec = np.asarray([max(f0_scalar, 0.0)], dtype=float)
        meta["source"] = "single_transition_empty_filter_fallback"
        return omega_vec, d_vec, f_vec, meta

    energies_eV, osc = _aggregate_transitions(energies_eV, osc, merge_tol)

    # Optional rigid correction to align the first bright transition to a target gap.
    if auto_gap_target > 0.0:
        shift = auto_gap_target - float(np.min(energies_eV))
        energies_eV = energies_eV + shift
        meta["auto_gap_target_eV"] = float(auto_gap_target)
        meta["auto_gap_shift_eV"] = float(shift)

    if max_transitions > 0 and len(energies_eV) > max_transitions:
        keep = np.argsort(osc)[-max_transitions:]
        energies_eV = energies_eV[keep]
        osc = osc[keep]

    order = np.argsort(energies_eV)
    energies_eV = energies_eV[order]
    osc = osc[order]

    omega_vec = energies_eV * E0 / HBAR
    d_vec = dipole_from_f(omega_vec, osc)

    meta["source"] = source or "unknown"
    meta["n_transitions"] = int(len(energies_eV))
    meta["energy_min_eV"] = float(np.min(energies_eV))
    meta["energy_max_eV"] = float(np.max(energies_eV))
    return omega_vec, d_vec, osc, meta


# ============================================================
# Simulation parameters
# ============================================================
T_FS = _env_float("SBUP3_T_FS", 50.0)         # total time [fs]
DT_AS = _env_float("SBUP3_DT_AS", 5.0)        # timestep [as]
GAMMA = _env_float("SBUP3_GAMMA", 5.0e13)     # dephasing rate [1/s]

E_FIELD_AMPL = _env_float("SBUP3_E_FIELD_AMPL", 1.0e7)  # V/m
OMEGA_L = _env_float("SBUP3_OMEGA_L", omega0_scalar)

use_physical_density = _env_bool("SBUP3_USE_PHYSICAL_DENSITY", False)
lattice_a_ang = _env_float("SBUP3_LATTICE_A_ANG", 5.6533)
lattice_a_m = lattice_a_ang * 1e-10
pair_density_m3 = 4.0 / max(lattice_a_m**3, 1e-60)

if os.getenv("SBUP3_DENSITY_M3") is None and use_physical_density:
    N_DENSITY = pair_density_m3
else:
    N_DENSITY = _env_float("SBUP3_DENSITY_M3", 1.0)

P_SCALE = _env_float("SBUP3_POLARIZATION_SCALE", 1.0)
N_CLIP = _env_float("SBUP3_N_CLIP", 1.0)
OSC_NORM_FACTOR = _env_float("SBUP3_OSC_NORM_FACTOR", 1.0)
if OSC_NORM_FACTOR <= 0.0:
    raise ValueError("SBUP3_OSC_NORM_FACTOR must be > 0.")


# ============================================================
# Transition model
# ============================================================
omega_vec, d_vec, f_vec, transition_meta = load_transition_model()
if OSC_NORM_FACTOR != 1.0:
    d_vec = d_vec * np.sqrt(OSC_NORM_FACTOR)
    f_vec = f_vec * OSC_NORM_FACTOR

n_trans = len(omega_vec)
f_sum = float(np.sum(f_vec))
if f_sum > 0.0:
    weights = f_vec / f_sum
else:
    weights = np.full(n_trans, 1.0 / max(n_trans, 1), dtype=float)

_save_output("transition_omega_rad_s_used.npy", omega_vec)
_save_output("transition_d_eff_Cm_used.npy", d_vec)
_save_output("transition_osc_used.npy", f_vec)
_save_output("transition_weight_used.npy", weights)
if n_trans > 0:
    _save_output("transition_energy_eV_used.npy", HBAR * omega_vec / E0)

transition_meta.update(
    {
        "n_transitions_used": int(n_trans),
        "oscillator_sum_used": float(np.sum(f_vec)),
        "density_m3": float(N_DENSITY),
        "polarization_scale": float(P_SCALE),
        "osc_norm_factor": float(OSC_NORM_FACTOR),
    }
)


# ============================================================
# Time grid
# ============================================================
dt = DT_AS * 1e-18
tmax = T_FS * 1e-15
time_s = np.arange(0.0, tmax, dt)
nt = len(time_s)


# ============================================================
# Driving field (Gaussian pulse definition)
# ============================================================
def E_of_t(t: float) -> float:
    t0 = 0.5 * tmax
    tau = 0.15 * tmax
    envelope = np.exp(-((t - t0) / tau) ** 2)
    return E_FIELD_AMPL * envelope * np.cos(OMEGA_L * (t - t0))


E_t_internal = np.asarray([E_of_t(t) for t in time_s], dtype=float)
E_FEEDBACK = BASE / "sbe" / "E_t_feedback.npy"
MIXING = _env_float("SBUP3_MIXING", 0.05)  # 0 < MIXING <= 1
IGNORE_FEEDBACK = _env_bool("SBUP3_IGNORE_FEEDBACK", False)

if E_FEEDBACK.exists() and not IGNORE_FEEDBACK:
    E_fb = np.load(E_FEEDBACK)
    if len(E_fb) != len(time_s):
        E_t = E_t_internal
        print(
            "[WARN] Feedback field length mismatch: "
            f"{len(E_fb)} vs time grid {len(time_s)}. "
            "Ignoring stale feedback and using internal Gaussian pulse."
        )
    else:
        E_t = (1.0 - MIXING) * E_t_internal + MIXING * E_fb
        print(f"[LOAD] Using mixed UPPE feedback (alpha = {MIXING}) -> {E_FEEDBACK}")
elif E_FEEDBACK.exists() and IGNORE_FEEDBACK:
    E_t = E_t_internal
    print("[INFO] Feedback field found but ignored by SBUP3_IGNORE_FEEDBACK.")
else:
    E_t = E_t_internal
    print("[INFO] No feedback field found; using internal Gaussian pulse.")


# ------------------------------------------------------------
# Field stabilization (important for SBE <-> UPPE coupling)
# ------------------------------------------------------------
E_MAX = _env_float("SBUP3_E_MAX", 5.0e7)  # V/m
E_t = np.clip(E_t, -E_MAX, E_MAX)


# ============================================================
# Semiconductor Bloch Equations (multi-transition reduced model)
# ============================================================
gamma_slope_per_eV = _env_float("SBUP3_GAMMA_SLOPE_PER_EV", 0.0)
transition_eV = HBAR * omega_vec / E0
if n_trans > 0:
    e_ref = float(np.median(transition_eV))
else:
    e_ref = 0.0
gamma_vec = GAMMA * (1.0 + gamma_slope_per_eV * (transition_eV - e_ref))
gamma_vec = np.maximum(gamma_vec, 0.0)


def rhs(p: np.ndarray, n: np.ndarray, e_field: float) -> tuple[np.ndarray, np.ndarray]:
    coupling = (d_vec / HBAR) * e_field
    dp = -(1j * omega_vec + gamma_vec) * p + 1j * coupling * (1.0 - 2.0 * n)
    dn = -2.0 * np.imag(coupling * np.conj(p))
    return dp, dn


# ============================================================
# Spectral extraction controls
# ============================================================
CHI_EPS = _env_float("SBUP3_CHI_EPS", 1e-3)
RESPONSE_POWER_CUTOFF = _env_float("SBUP3_RESPONSE_POWER_CUTOFF", 0.999)
CHI_MAX_EV = _env_float("SBUP3_CHI_MAX_EV", 3.0)
DEBUG_SUPPORT = _env_bool("SBUP3_DEBUG_SUPPORT", False)

PROBE_ENABLE = _env_bool("SBUP3_PROBE_ENABLE", True)
PROBE_E_FIELD_AMPL = _env_float("SBUP3_PROBE_E_FIELD_AMPL", 5.0e4)
PROBE_T0_FS = _env_float("SBUP3_PROBE_T0_FS", 0.5 * T_FS)
PROBE_TAU_FS = max(_env_float("SBUP3_PROBE_TAU_FS", 0.35), 1e-6)
PROBE_OMEGA_L = _env_float("SBUP3_PROBE_OMEGA_L", 0.0)
PROBE_CHI_EPS = _env_float("SBUP3_PROBE_CHI_EPS", min(CHI_EPS, 1e-6))
PROBE_RESPONSE_POWER_CUTOFF = _env_float("SBUP3_PROBE_RESPONSE_POWER_CUTOFF", 1.0)
PROBE_CHI_MIN_EV = _env_float("SBUP3_PROBE_CHI_MIN_EV", 1.5)
PROBE_CHI_MAX_EV = _env_float("SBUP3_PROBE_CHI_MAX_EV", 3.0)

transition_meta.update(
    {
        "chi_eps": float(CHI_EPS),
        "response_power_cutoff": float(RESPONSE_POWER_CUTOFF),
        "chi_max_ev": float(CHI_MAX_EV),
        "probe_enable": bool(PROBE_ENABLE),
        "probe_e_field_ampl": float(PROBE_E_FIELD_AMPL),
        "probe_t0_fs": float(PROBE_T0_FS),
        "probe_tau_fs": float(PROBE_TAU_FS),
        "probe_omega_l": float(PROBE_OMEGA_L),
        "probe_chi_eps": float(PROBE_CHI_EPS),
        "probe_response_power_cutoff": float(PROBE_RESPONSE_POWER_CUTOFF),
        "probe_chi_min_ev": float(PROBE_CHI_MIN_EV),
        "probe_chi_max_ev": float(PROBE_CHI_MAX_EV),
    }
)
(HERE / "transition_model.json").write_text(json.dumps(transition_meta, indent=2))


def run_rk4(e_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros(n_trans, dtype=np.complex128)
    n = np.zeros(n_trans, dtype=float)

    polarization = np.zeros(nt, dtype=np.complex128)
    population = np.zeros(nt, dtype=float)

    for i in range(nt):
        e_field = e_t[i]

        k1p, k1n = rhs(p, n, e_field)
        k2p, k2n = rhs(p + 0.5 * dt * k1p, n + 0.5 * dt * k1n, e_field)
        k3p, k3n = rhs(p + 0.5 * dt * k2p, n + 0.5 * dt * k2n, e_field)
        k4p, k4n = rhs(p + dt * k3p, n + dt * k3n, e_field)

        p = p + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        n = n + (dt / 6.0) * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
        n = np.clip(n, 0.0, N_CLIP)

        p_eff = np.sum(weights * d_vec * p)
        n_eff = float(np.sum(weights * n))

        polarization[i] = P_SCALE * N_DENSITY * p_eff
        population[i] = n_eff

    return polarization, population


def hann(npts: int) -> np.ndarray:
    if npts <= 1:
        return np.ones(npts)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(npts) / (npts - 1))


def extract_chi(
    polarization_t: np.ndarray,
    e_t: np.ndarray,
    chi_eps: float,
    response_power_cutoff: float,
    chi_min_ev: float,
    chi_max_ev: float,
    debug_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = hann(nt)
    p_win = np.real(polarization_t - polarization_t.mean()) * w
    e_win = (e_t - e_t.mean()) * w

    p_w = np.fft.rfft(p_win)
    e_w = np.fft.rfft(e_win)
    freq = np.fft.rfftfreq(nt, dt)
    omega = 2.0 * np.pi * freq

    chi = np.full_like(p_w, np.nan + 1j * np.nan, dtype=complex)
    e_w_abs = np.abs(e_w)
    e_w_max = float(e_w_abs.max()) if e_w_abs.size else 0.0
    support_mask = e_w_abs > max(1e-30, chi_eps * e_w_max)

    # Restrict inversion to field spectral support.
    drive_power = e_w_abs**2
    power_sum = float(np.sum(drive_power))
    if power_sum > 0.0:
        cumulative = np.cumsum(drive_power) / power_sum
        cutoff = min(max(response_power_cutoff, 0.0), 1.0)
        if cutoff >= 1.0 - 1e-12:
            idx_cut = len(omega) - 1
        else:
            idx_cut = int(np.searchsorted(cumulative, cutoff, side="left"))
        support_mask &= np.arange(len(omega)) <= idx_cut
    else:
        idx_cut = len(omega) - 1

    if chi_min_ev > 0.0:
        omega_min = chi_min_ev * E0 / HBAR
        support_mask &= omega >= omega_min
    if chi_max_ev > 0.0:
        omega_max = chi_max_ev * E0 / HBAR
        support_mask &= omega <= omega_max

    if DEBUG_SUPPORT:
        print(
            f"[DEBUG] {debug_label} support params: "
            f"CHI_EPS={chi_eps:.3e}, RESPONSE_POWER_CUTOFF={response_power_cutoff:.6g}, "
            f"CHI_MIN_EV={chi_min_ev:.6g}, CHI_MAX_EV={chi_max_ev:.6g}, "
            f"idx_cut={idx_cut}, support_count={int(np.sum(support_mask))}"
        )

    chi[support_mask] = p_w[support_mask] / (EPS0 * e_w[support_mask])
    return omega, chi, support_mask


def optics_from_chi(omega: np.ndarray, chi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_tilde = np.sqrt(1.0 + chi)
    n_real = np.real(n_tilde)
    kappa = np.imag(n_tilde)
    alpha_m_inv = 2.0 * omega * kappa / C0
    alpha_rel = alpha_m_inv
    n_rel = n_real - 1.0
    return alpha_rel, n_rel, alpha_m_inv, kappa, n_real


# ============================================================
# RK4 propagation (drive)
# ============================================================
polarization, population = run_rk4(E_t)

# ============================================================
# Save time-domain data
# ============================================================
_save_output("time_s.npy", time_s)
_save_output("E_t.npy", E_t)
_save_output("polarization.npy", polarization)
_save_output("population.npy", population)


# ============================================================
# chi(omega) extraction from propagation drive
# ============================================================
omega, chi, support_mask = extract_chi(
    polarization_t=polarization,
    e_t=E_t,
    chi_eps=CHI_EPS,
    response_power_cutoff=RESPONSE_POWER_CUTOFF,
    chi_min_ev=0.0,
    chi_max_ev=CHI_MAX_EV,
    debug_label="drive",
)

_save_output("omega_rad_s.npy", omega)
_save_output("chi_eff_complex.npy", chi)
_save_output("chi_eff_real.npy", np.real(chi))
_save_output("chi_eff_imag.npy", np.imag(chi))
_save_output("chi_support_mask.npy", support_mask)


# ------------------------------------------------------------
# Optics-ready derived quantities
# ------------------------------------------------------------
alpha_rel, n_rel, alpha_m_inv, kappa, n_real = optics_from_chi(omega, chi)

_save_output("alpha_rel.npy", alpha_rel)
_save_output("n_rel.npy", n_rel)
_save_output("alpha_m_inv.npy", alpha_m_inv)
_save_output("kappa.npy", kappa)
_save_output("n_real.npy", n_real)


# ============================================================
# Dedicated broadband linear-response probe
# ============================================================
if PROBE_ENABLE:
    probe_t0 = PROBE_T0_FS * 1e-15
    probe_tau = PROBE_TAU_FS * 1e-15
    probe_env = np.exp(-((time_s - probe_t0) / max(probe_tau, 1e-30)) ** 2)
    E_probe = PROBE_E_FIELD_AMPL * probe_env * np.cos(PROBE_OMEGA_L * (time_s - probe_t0))
    E_probe = np.clip(E_probe, -E_MAX, E_MAX)

    probe_polarization, probe_population = run_rk4(E_probe)
    probe_omega, probe_chi, probe_support = extract_chi(
        polarization_t=probe_polarization,
        e_t=E_probe,
        chi_eps=PROBE_CHI_EPS,
        response_power_cutoff=PROBE_RESPONSE_POWER_CUTOFF,
        chi_min_ev=PROBE_CHI_MIN_EV,
        chi_max_ev=PROBE_CHI_MAX_EV,
        debug_label="probe",
    )
    (
        probe_alpha_rel,
        probe_n_rel,
        probe_alpha_m_inv,
        probe_kappa,
        probe_n_real,
    ) = optics_from_chi(probe_omega, probe_chi)

    _save_output("probe_time_s.npy", time_s)
    _save_output("probe_E_t.npy", E_probe)
    _save_output("probe_polarization.npy", probe_polarization)
    _save_output("probe_population.npy", probe_population)
    _save_output("probe_omega_rad_s.npy", probe_omega)
    _save_output("probe_chi_eff_complex.npy", probe_chi)
    _save_output("probe_chi_eff_real.npy", np.real(probe_chi))
    _save_output("probe_chi_eff_imag.npy", np.imag(probe_chi))
    _save_output("probe_chi_support_mask.npy", probe_support)
    _save_output("probe_alpha_rel.npy", probe_alpha_rel)
    _save_output("probe_n_rel.npy", probe_n_rel)
    _save_output("probe_alpha_m_inv.npy", probe_alpha_m_inv)
    _save_output("probe_kappa.npy", probe_kappa)
    _save_output("probe_n_real.npy", probe_n_real)


# ============================================================
# Summary
# ============================================================
print("\nSBUP3 SBE initialized with TDDFT parameters")
print("-------------------------------------------")
print(f"omega0_scalar = {omega0_scalar:.3e} rad/s")
print(f"E0_scalar     = {E0_scalar_eV:.3f} eV")
print(f"|d|_scalar    = {abs(d_eff_scalar):.3e} C*m ({abs(d_eff_scalar) / 3.33564e-30:.2f} D)")
print(f"N_density     = {N_DENSITY:.3e} 1/m^3 (P scale = {P_SCALE:.3e})")
print(f"osc_norm      = {OSC_NORM_FACTOR:.3e}")
print(f"n_transitions = {n_trans}")
if n_trans > 0:
    print(
        "transition span = "
        f"{transition_eV.min():.3f} -> {transition_eV.max():.3f} eV, "
        f"sum f = {np.sum(f_vec):.3e}"
    )
print(f"transition source = {transition_meta.get('source')}")

print("\nSBE propagation complete.")
print("Saved:")
print("  time_s.npy")
print("  E_t.npy")
print("  polarization.npy")
print("  population.npy")
print("  omega_rad_s.npy")
print("  chi_eff_complex.npy")
print("  chi_eff_real.npy")
print("  chi_eff_imag.npy")
print("  chi_support_mask.npy")
print("  alpha_rel.npy")
print("  n_rel.npy")
print("  alpha_m_inv.npy")
print("  kappa.npy")
print("  n_real.npy")
if PROBE_ENABLE:
    print("\nProbe-mode outputs:")
    print("  probe_time_s.npy")
    print("  probe_E_t.npy")
    print("  probe_polarization.npy")
    print("  probe_population.npy")
    print("  probe_omega_rad_s.npy")
    print("  probe_chi_eff_complex.npy")
    print("  probe_chi_eff_real.npy")
    print("  probe_chi_eff_imag.npy")
    print("  probe_chi_support_mask.npy")
    print("  probe_alpha_rel.npy")
    print("  probe_n_rel.npy")
    print("  probe_alpha_m_inv.npy")
    print("  probe_kappa.npy")
    print("  probe_n_real.npy")
print("  transition_model.json")
