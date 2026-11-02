import os
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from pathlib import Path

from uppe.dispersion import refractive_index_gaas

# =================================================
# Environment overrides (optional)
# =================================================
def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)

# =================================================
# Constants
# =================================================
c0 = 2.99792458e8  # m/s
eps0 = 8.8541878128e-12  # F/m
mu0 = 1.0 / (eps0 * c0**2)

# =================================================
# UPPE 2D solver: x–t–z
# =================================================
def propagate_uppe_2d(
    x_m,
    time_s,
    E_xt0,
    P_xt,
    dz,
    z_steps,
    *,
    background_index=None,
    polarization_kind=None,
):
    """
    2D UPPE propagation (x, t, z)

    Parameters
    ----------
    x_m : (Nx,) array
        Transverse coordinate (m)
    time_s : (Nt,) array
        Time grid (s)
    E_xt0 : (Nx, Nt) array
        Input electric field at z=0
    P_xt : (Nx, Nt) array
        Material polarization
    dz : float
        Propagation step (m)
    z_steps : int
        Number of propagation steps
    background_index : float or None
        Constant background index. None uses the tabulated GaAs index.
    polarization_kind : {"residual", "full"} or None
        A full material polarization is compatible only with a vacuum
        background. A non-vacuum index requires polarization with its
        overlapping linear response removed. State the convention explicitly
        when a nonzero polarization is supplied.

    Returns
    -------
    E_xt : (Nx, Nt) array
        Electric field at final z
    """

    x_m = np.asarray(x_m, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    E_xt0 = np.asarray(E_xt0)
    P_xt = np.asarray(P_xt)
    if not np.isrealobj(E_xt0) or not np.isrealobj(P_xt):
        raise ValueError("Use real physical E and P traces in the time domain")
    E_xt0 = E_xt0.astype(float, copy=False)
    P_xt = P_xt.astype(float, copy=False)
    nx, nt = E_xt0.shape
    if nx < 2 or nt < 2 or P_xt.shape != (nx, nt):
        raise ValueError("E and P must share an (Nx, Nt) grid with at least two points per axis")
    dx = x_m[1] - x_m[0]
    dt = time_s[1] - time_s[0]
    if len(x_m) != nx or len(time_s) != nt or dx <= 0 or dt <= 0:
        raise ValueError("Coordinate grids must match the field and increase")
    if not np.allclose(np.diff(x_m), dx, rtol=1e-10, atol=0) or not np.allclose(np.diff(time_s), dt, rtol=1e-10, atol=0):
        raise ValueError("FFT coordinate grids must be uniform")
    if not np.isfinite(E_xt0).all() or not np.isfinite(P_xt).all():
        raise ValueError("E and P must be finite real arrays")
    if not np.isfinite(dz) or dz <= 0 or not isinstance(z_steps, (int, np.integer)) or z_steps < 0:
        raise ValueError("dz must be positive and z_steps a nonnegative integer")
    if polarization_kind is None and np.any(P_xt != 0):
        raise ValueError("Specify full or residual polarization for a nonzero source")
    if polarization_kind not in {None, "residual", "full"}:
        raise ValueError("polarization_kind must be 'residual' or 'full'")

    kx = 2 * np.pi * fftfreq(nx, dx)
    omega = 2 * np.pi * fftfreq(nt, dt)

    KX, OMEGA = np.meshgrid(kx, omega, indexing="ij")

    n = (refractive_index_gaas(np.abs(OMEGA)) if background_index is None
         else np.full_like(OMEGA, float(background_index)))
    if not np.isfinite(n).all() or np.any(n <= 0):
        raise ValueError("Background index must be finite and positive")
    if polarization_kind == "full" and not np.allclose(n, 1.0, rtol=0, atol=1e-12):
        raise ValueError("Full polarization requires a vacuum background to avoid double counting")
    q2 = (n * OMEGA / c0)**2 - KX**2
    q = np.sqrt(np.maximum(q2, 0.0))
    decay = np.sqrt(np.maximum(-q2, 0.0))
    sign = np.sign(OMEGA)
    phase = np.exp((-1j * sign * q - decay) * dz)
    # The temporal Nyquist bin of an even grid has no distinct conjugate bin.
    # Its real projection is used; a physical carrier should be well below it.
    if nt % 2 == 0:
        phase[:, nt // 2] = np.exp(-decay[:, nt // 2] * dz) * np.cos(q[:, nt // 2] * dz)

    source_step = np.zeros_like(phase)
    propagating = (q > 0) & (OMEGA != 0)
    source_step[propagating] = (
        -1j * sign[propagating] * mu0 * OMEGA[propagating]**2
        / (2 * q[propagating]) * dz
        * np.exp(-0.5j * sign[propagating] * q[propagating] * dz)
        * np.sinc(q[propagating] * dz / (2 * np.pi))
    )
    if nt % 2 == 0:
        source_step[:, nt // 2] = 0.0

    # Fourier transform to (kx, ω)
    E_kw = fft2(E_xt0)
    P_kw = fft2(P_xt)

    for _ in range(z_steps):
        # Linear propagation
        # Exact step for a polarization held fixed over this interval.
        E_kw = phase * E_kw + source_step * P_kw

    E_xt = ifft2(E_kw)
    if np.max(np.abs(E_xt.imag)) > 1e-10 * max(1.0, np.max(np.abs(E_xt.real))):
        raise RuntimeError("Propagation broke real-field conjugate symmetry")
    return E_xt.real


# =================================================
# Standalone execution (SBE → UPPE)
# =================================================
if __name__ == "__main__":

    print("SBUP³ UPPE 2D propagation (SBE-coupled)")
    print("--------------------------------------")

    BASE = Path(__file__).resolve().parents[1]

    # --- Load SBE outputs ---
    time_s = np.load(BASE / "sbe" / "time_s.npy")
    E_t    = np.load(BASE / "sbe" / "E_t.npy")
    P_t    = np.load(BASE / "sbe" / "polarization.npy")

    # --- Transverse grid ---
    nx = _env_int("SBUP3_UPPE_NX", 256)
    x_window = _env_float("SBUP3_UPPE_X_WINDOW", 10e-6)  # full window (m)
    w0 = _env_float("SBUP3_UPPE_W0", 2e-6)               # beam waist (m)
    x_m = np.linspace(-0.5 * x_window, 0.5 * x_window, nx, endpoint=False)

    # --- Lift to 2D ---
    envelope = np.exp(-(x_m[:, None]**2) / w0**2)
    E_xt0 = envelope * E_t[None, :]
    P_xt  = envelope * np.real(P_t)[None, :]

    # --- Propagation ---
    dz = _env_float("SBUP3_UPPE_DZ", 1e-6)
    z_steps = _env_int("SBUP3_UPPE_Z_STEPS", 100)

    E_xt_out = propagate_uppe_2d(
        x_m=x_m,
        time_s=time_s,
        E_xt0=E_xt0,
        P_xt=P_xt,
        dz=dz,
        z_steps=z_steps,
        background_index=1.0,
        polarization_kind="full",
    )

    # --- Save ---
    OUTDIR = BASE / "uppe" / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTDIR / "E_xt_out.npy", E_xt_out)
    np.save(OUTDIR / "x_m.npy", x_m)
    np.save(OUTDIR / "time_s.npy", time_s)

    print("UPPE 2D propagation complete.")
    print(f"Saved to {OUTDIR}")
