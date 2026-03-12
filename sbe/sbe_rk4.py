"""
SBUP³ — Semiconductor Bloch Equations (RK4, TDDFT-driven)
=========================================================

Inputs (from sbup3/dft/):
  - omega0_rad_s.npy   : transition frequency ω0 [rad/s]
  - d_eff_Cm.npy       : effective interband dipole [C·m]
  - E0_eV.npy          : excitation energy [eV]
  - f0.npy             : oscillator strength

Outputs:
  - time_s.npy
  - E_t.npy
  - polarization.npy (macroscopic; scaled by SBUP3_DENSITY_M3 and SBUP3_POLARIZATION_SCALE)
  - population.npy
  - omega_rad_s.npy
  - chi_eff_complex.npy
  - chi_eff_real.npy
  - chi_eff_imag.npy

Author: Dennis Wayo
"""

from __future__ import annotations
import numpy as np
import os

# ============================================================
# Environment overrides (optional)
# ============================================================
def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}

# ============================================================
# Constants
# ============================================================
HBAR = 1.054571817e-34     # J·s
E0   = 1.602176634e-19    # C
EPS0 = 8.8541878128e-12   # F/m
C0   = 299792458.0        # m/s

# ============================================================
# Paths
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DFT  = os.path.join(HERE, "..", "dft")

omega0 = np.load(os.path.join(DFT, "omega0_rad_s.npy")).item()
d_eff  = np.load(os.path.join(DFT, "d_eff_Cm.npy")).item()
E0_eV  = np.load(os.path.join(DFT, "E0_eV.npy")).item()
f0     = np.load(os.path.join(DFT, "f0.npy")).item()

# ============================================================
# Simulation parameters
# ============================================================
T_FS   = _env_float("SBUP3_T_FS", 50.0)      # total time [fs]
DT_AS  = _env_float("SBUP3_DT_AS", 5.0)       # timestep [as]
GAMMA  = _env_float("SBUP3_GAMMA", 5.0e13)    # dephasing rate [1/s]

E_FIELD_AMPL = _env_float("SBUP3_E_FIELD_AMPL", 1.0e7)  # V/m (linear regime)
OMEGA_L      = _env_float("SBUP3_OMEGA_L", omega0)      # resonant drive
N_DENSITY    = _env_float("SBUP3_DENSITY_M3", 1.0)      # m^-3 (macroscopic scaling)
P_SCALE      = _env_float("SBUP3_POLARIZATION_SCALE", 1.0)


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
def E_of_t(t):
    t0 = 0.5 * tmax
    tau = 0.15 * tmax
    envelope = np.exp(-((t - t0) / tau)**2)
    return E_FIELD_AMPL * envelope * np.cos(OMEGA_L * (t - t0))


# ============================================================
# Driving field (internal pulse OR UPPE feedback)
# ============================================================
E_t_internal = np.array([E_of_t(t) for t in time_s])

from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
E_FEEDBACK = BASE / "sbe" / "E_t_feedback.npy"

MIXING = _env_float("SBUP3_MIXING", 0.05)  # 0 < MIXING ≤ 1
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
        E_t = (1 - MIXING) * E_t_internal + MIXING * E_fb
        print(f"[LOAD] Using mixed UPPE feedback (α = {MIXING}) → {E_FEEDBACK}")

elif E_FEEDBACK.exists() and IGNORE_FEEDBACK:
    E_t = E_t_internal
    print("[INFO] Feedback field found but ignored by SBUP3_IGNORE_FEEDBACK.")

else:
    E_t = E_t_internal
    print("[INFO] No feedback field found; using internal Gaussian pulse.")

# ------------------------------------------------------------
# Field stabilization (important for SBE ↔ UPPE coupling)
# ------------------------------------------------------------
E_MAX = _env_float("SBUP3_E_MAX", 5.0e7)  # V/m, safe nonlinear GaAs regime
E_t = np.clip(E_t, -E_MAX, E_MAX)

# ============================================================
# Semiconductor Bloch Equations (two-level)
# ============================================================
def rhs(p, n, t, E):
    dp = -(1j * omega0 + GAMMA) * p + 1j * (d_eff / HBAR) * E * (1.0 - 2.0 * n)
    dn = -2.0 * np.imag((d_eff / HBAR) * E * np.conj(p))
    return dp, dn

# ============================================================
# RK4 propagation
# ============================================================
p = 0.0 + 0.0j
n = 0.0

polarization = np.zeros(nt, dtype=np.complex128)
population   = np.zeros(nt)

for i, t in enumerate(time_s):
    E = E_t[i]

    k1p, k1n = rhs(p, n, t, E)
    k2p, k2n = rhs(p + 0.5*dt*k1p, n + 0.5*dt*k1n, t + 0.5*dt, E)
    k3p, k3n = rhs(p + 0.5*dt*k2p, n + 0.5*dt*k2n, t + 0.5*dt, E)
    k4p, k4n = rhs(p + dt*k3p,     n + dt*k3n,     t + dt,     E)

    p += (dt / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    n += (dt / 6.0) * (k1n + 2*k2n + 2*k3n + k4n)

    polarization[i] = P_SCALE * N_DENSITY * d_eff * p
    population[i]   = n

# ============================================================
# Save time-domain data
# ============================================================
np.save("time_s.npy", time_s)
np.save("E_t.npy", E_t)
np.save("polarization.npy", polarization)
np.save("population.npy", population)

# ============================================================
# χ(ω) extraction
# ============================================================
def hann(n):
    return 0.5 - 0.5 * np.cos(2*np.pi*np.arange(n)/(n-1))

w = hann(nt)
P = np.real(polarization - polarization.mean()) * w
E = (E_t - E_t.mean()) * w

P_w = np.fft.rfft(P)
E_w = np.fft.rfft(E)

freq = np.fft.rfftfreq(nt, dt)
omega = 2*np.pi*freq

chi = np.full_like(P_w, np.nan + 1j * np.nan, dtype=complex)
E_w_abs = np.abs(E_w)
E_w_max = float(E_w_abs.max()) if E_w_abs.size else 0.0
CHI_EPS = _env_float("SBUP3_CHI_EPS", 1e-3)
RESPONSE_POWER_CUTOFF = _env_float("SBUP3_RESPONSE_POWER_CUTOFF", 0.999)
CHI_MAX_EV = _env_float("SBUP3_CHI_MAX_EV", 3.0)

support_mask = E_w_abs > max(1e-30, CHI_EPS * E_w_max)

# Restrict inversion to the spectral support of the driving field.
drive_power = E_w_abs**2
power_sum = float(np.sum(drive_power))
if power_sum > 0:
    cum = np.cumsum(drive_power) / power_sum
    cutoff = min(max(RESPONSE_POWER_CUTOFF, 0.0), 1.0)
    idx_cut = int(np.searchsorted(cum, cutoff))
    support_mask &= (np.arange(len(omega)) <= idx_cut)

if CHI_MAX_EV > 0:
    omega_max = CHI_MAX_EV * E0 / HBAR
    support_mask &= (omega <= omega_max)

chi[support_mask] = P_w[support_mask] / (EPS0 * E_w[support_mask])

np.save("omega_rad_s.npy", omega)
np.save("chi_eff_complex.npy", chi)
np.save("chi_eff_real.npy", np.real(chi))
np.save("chi_eff_imag.npy", np.imag(chi))
np.save("chi_support_mask.npy", support_mask)


# ------------------------------------------------------------
# Optics-ready derived quantities (physical + legacy aliases)
# ------------------------------------------------------------
chi_real = np.real(chi)
chi_imag = np.imag(chi)

# Complex refractive index n~ = n + i*kappa from epsilon_r = 1 + chi
n_tilde = np.sqrt(1.0 + chi)
n_real = np.real(n_tilde)
kappa = np.imag(n_tilde)

# Physical absorption coefficient alpha [1/m]
alpha_m_inv = 2.0 * omega * kappa / C0

# Legacy aliases to keep downstream scripts compatible
alpha_rel = alpha_m_inv
n_rel = n_real - 1.0

np.save("alpha_rel.npy", alpha_rel)
np.save("n_rel.npy", n_rel)
np.save("alpha_m_inv.npy", alpha_m_inv)
np.save("kappa.npy", kappa)
np.save("n_real.npy", n_real)

# ============================================================
# Summary
# ============================================================
print("\nSBUP³ SBE initialized with TDDFT parameters")
print("-------------------------------------------")
print(f"ω0  = {omega0:.3e} rad/s")
print(f"E0  = {E0_eV:.3f} eV")
print(f"|d| = {abs(d_eff):.3e} C·m ({abs(d_eff)/3.33564e-30:.2f} D)")
print(f"N   = {N_DENSITY:.3e} 1/m^3 (P scale = {P_SCALE:.3e})")
print(f"f0  = {f0:.3e}")

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
