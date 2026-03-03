import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from pathlib import Path

from uppe.dispersion import refractive_index_gaas

# =================================================
# Constants
# =================================================
c0 = 2.99792458e8  # m/s

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

    Returns
    -------
    E_xt : (Nx, Nt) array
        Electric field at final z
    """

    dx = x_m[1] - x_m[0]
    dt = time_s[1] - time_s[0]

    nx = len(x_m)
    nt = len(time_s)

    kx = 2 * np.pi * fftfreq(nx, dx)
    omega = 2 * np.pi * fftfreq(nt, dt)

    KX, OMEGA = np.meshgrid(kx, omega, indexing="ij")

    n = refractive_index_gaas(np.abs(OMEGA))
    k0 = n * OMEGA / c0
    kz = np.sqrt(np.maximum(k0**2 - KX**2, 0.0))

    # Fourier transform to (kx, ω)
    E_kw = fft2(E_xt0)
    P_kw = fft2(P_xt)

    for _ in range(z_steps):
        # Linear propagation
        E_kw *= np.exp(-1j * kz * dz)

        # Source term (linear UPPE)
        E_kw += 1j * dz * (OMEGA**2 / (2 * kz + 1e-30)) * P_kw

    return np.real(ifft2(E_kw))


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
    x_m = np.linspace(-5e-6, 5e-6, 256)  # 10 µm window
    w0 = 2e-6                            # beam waist (m)

    # --- Lift to 2D ---
    envelope = np.exp(-(x_m[:, None]**2) / w0**2)
    E_xt0 = envelope * E_t[None, :]
    P_xt  = envelope * P_t[None, :]

    # --- Propagation ---
    E_xt_out = propagate_uppe_2d(
        x_m=x_m,
        time_s=time_s,
        E_xt0=E_xt0,
        P_xt=P_xt,
        dz=1e-6,
        z_steps=100,
    )

    # --- Save ---
    OUTDIR = BASE / "uppe" / "outputs"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTDIR / "E_xt_out.npy", E_xt_out)
    np.save(OUTDIR / "x_m.npy", x_m)
    np.save(OUTDIR / "time_s.npy", time_s)

    print("UPPE 2D propagation complete.")
    print(f"Saved to {OUTDIR}")