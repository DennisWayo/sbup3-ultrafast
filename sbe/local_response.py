"""Side-effect-free local response of the reduced multi-transition SBE model.

Each transverse location has its own temporal Bloch state.  This module uses
the same RK4 equations and effective polarization convention as sbe_rk4.py,
without importing that script (which writes the historical output files).
"""

from __future__ import annotations

import numpy as np


HBAR = 1.054571817e-34


def local_polarization(
    field_xt: np.ndarray,
    time_s: np.ndarray,
    omega_rad_s: np.ndarray,
    dipole_cm: np.ndarray,
    weights: np.ndarray,
    *,
    density_m3: float,
    polarization_scale: float,
    gamma_s: float,
    population_clip: float = 1.0,
    field_sampling: str = "linear",
) -> tuple[np.ndarray, float]:
    """Return real P(x,t) and the largest local excited population.

    The sampled dielectric weights and inferred dipoles are effective model
    parameters; this function does not turn them into ab-initio matrix elements.
    ``field_sampling='left'`` reproduces the historical constant-within-step
    drive; the default linearly samples the drive at RK4 substeps.
    """
    field = np.asarray(field_xt, dtype=float)
    time = np.asarray(time_s, dtype=float)
    omega = np.asarray(omega_rad_s, dtype=float).reshape(-1)
    dipole = np.asarray(dipole_cm, dtype=float).reshape(-1)
    weight = np.asarray(weights, dtype=float).reshape(-1)
    if field.ndim != 2 or time.ndim != 1 or field.shape[1] != len(time):
        raise ValueError("field_xt must have shape (Nx, len(time_s))")
    if len(time) < 2 or len(omega) < 1 or dipole.shape != omega.shape or weight.shape != omega.shape:
        raise ValueError("Invalid temporal or transition dimensions")
    dt = float(time[1] - time[0])
    if dt <= 0 or not np.allclose(np.diff(time), dt, rtol=1e-10, atol=0):
        raise ValueError("time_s must be uniform and increasing")
    if not all(np.isfinite(a).all() for a in (field, time, omega, dipole, weight)):
        raise ValueError("SBE inputs must be finite")
    if np.any(omega <= 0) or np.any(dipole < 0) or np.any(weight < 0):
        raise ValueError("Transition energies, dipoles, and weights must be nonnegative")
    if not np.isclose(weight.sum(), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("Transition weights must sum to one")
    if not (np.isfinite(density_m3) and density_m3 > 0 and
            np.isfinite(polarization_scale) and polarization_scale > 0 and
            np.isfinite(gamma_s) and gamma_s >= 0 and
            np.isfinite(population_clip) and population_clip > 0):
        raise ValueError("Invalid density, polarization scale, dephasing, or population clip")
    if field_sampling not in {"linear", "left"}:
        raise ValueError("field_sampling must be 'linear' or 'left'")

    nx, nt = field.shape
    p = np.zeros((nx, len(omega)), dtype=complex)
    population = np.zeros((nx, len(omega)), dtype=float)
    output = np.empty((nx, nt), dtype=float)
    coupling_factor = dipole / HBAR
    weighted_dipole = weight * dipole
    decay = 1j * omega + gamma_s
    peak_population = 0.0

    def rhs(p_state: np.ndarray, n_state: np.ndarray,
            coupling: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dp = -decay * p_state + 1j * coupling * (1.0 - 2.0 * n_state)
        dn = -2.0 * np.imag(coupling * np.conj(p_state))
        return dp, dn

    for i in range(nt):
        if field_sampling == "linear":
            output[:, i] = (polarization_scale * density_m3
                            * np.real(np.sum(weighted_dipole[None, :] * p, axis=1)))
            peak_population = max(peak_population, float(population.max()))
            if i == nt - 1:
                break
        coupling0 = field[:, i, None] * coupling_factor[None, :]
        coupling1 = (field[:, min(i + 1, nt - 1), None] * coupling_factor[None, :]
                     if field_sampling == "linear" else coupling0)
        coupling_half = 0.5 * (coupling0 + coupling1)
        k1p, k1n = rhs(p, population, coupling0)
        k2p, k2n = rhs(p + 0.5 * dt * k1p, population + 0.5 * dt * k1n,
                       coupling_half)
        k3p, k3n = rhs(p + 0.5 * dt * k2p, population + 0.5 * dt * k2n,
                       coupling_half)
        k4p, k4n = rhs(p + dt * k3p, population + dt * k3n, coupling1)
        p += (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        population += (dt / 6.0) * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
        np.clip(population, 0.0, population_clip, out=population)
        if field_sampling == "left":
            output[:, i] = (polarization_scale * density_m3
                            * np.real(np.sum(weighted_dipole[None, :] * p, axis=1)))
            peak_population = max(peak_population, float(population.max()))

    if not np.isfinite(output).all():
        raise FloatingPointError("Local SBE polarization became non-finite")
    return output, peak_population
