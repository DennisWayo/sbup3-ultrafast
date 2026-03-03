#!/usr/bin/env python3
"""
SBUP3 visualization script.

Generates publication-ready figures from SBE and UPPE outputs.
Run from the repo root:
  python analysis/visualize_sbup3.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Ensure matplotlib can write its cache in constrained environments
_BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_BASE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_BASE / ".cache"))

import matplotlib.pyplot as plt


HBAR = 1.054571817e-34  # J*s
E0 = 1.602176634e-19    # C


def hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def load_outputs(base: Path) -> dict:
    sbe = base / "sbe"
    uppe = base / "uppe" / "outputs"

    required = [
        sbe / "time_s.npy",
        sbe / "E_t.npy",
        sbe / "polarization.npy",
        sbe / "population.npy",
        sbe / "omega_rad_s.npy",
        sbe / "chi_eff_complex.npy",
        sbe / "chi_eff_real.npy",
        sbe / "chi_eff_imag.npy",
        sbe / "alpha_rel.npy",
        sbe / "n_rel.npy",
        uppe / "E_xt_out.npy",
        uppe / "x_m.npy",
        uppe / "time_s.npy",
    ]

    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing outputs:")
        for p in missing:
            print(" -", p)
        print("\nRun from repo root:")
        print(f"  cd {sbe} && python sbe_rk4.py")
        print(f"  cd {base} && python -m uppe.uppe_2d")
        raise FileNotFoundError("Required outputs not found.")

    data = {
        "sbe": sbe,
        "uppe": uppe,
        "time_s": np.load(sbe / "time_s.npy"),
        "E_t": np.load(sbe / "E_t.npy"),
        "P_t": np.load(sbe / "polarization.npy"),
        "n_t": np.load(sbe / "population.npy"),
        "omega": np.load(sbe / "omega_rad_s.npy"),
        "chi": np.load(sbe / "chi_eff_complex.npy"),
        "chi_r": np.load(sbe / "chi_eff_real.npy"),
        "chi_i": np.load(sbe / "chi_eff_imag.npy"),
        "alpha": np.load(sbe / "alpha_rel.npy"),
        "n_rel": np.load(sbe / "n_rel.npy"),
        "E_xt": np.load(uppe / "E_xt_out.npy"),
        "x_m": np.load(uppe / "x_m.npy"),
        "time_s_uppe": np.load(uppe / "time_s.npy"),
    }

    return data


def savefig(fig: plt.Figure, outdir: Path, name: str, show: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / name, bbox_inches="tight")
    if not show:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="SBUP3 visualization")
    parser.add_argument(
        "--outdir",
        default="analysis/figures",
        help="Output directory for figures (relative to repo root)",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    outdir = base / args.outdir

    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 4),
            "font.size": 11,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.linewidth": 1.0,
            "savefig.dpi": 300,
        }
    )

    data = load_outputs(base)

    time_s = data["time_s"]
    E_t = data["E_t"]
    P_t = data["P_t"]
    n_t = data["n_t"]
    omega = data["omega"]
    chi_r = data["chi_r"]
    chi_i = data["chi_i"]
    alpha = data["alpha"]
    n_rel = data["n_rel"]
    E_xt = data["E_xt"]
    x_m = data["x_m"]
    time_s_uppe = data["time_s_uppe"]

    time_fs = time_s * 1e15
    omega_eV = HBAR * omega / E0

    # Time-domain SBE dynamics
    fig = plt.figure()
    E_norm = np.max(np.abs(E_t)) or 1.0
    P_norm = np.max(np.abs(P_t)) or 1.0
    plt.plot(time_fs, E_t / E_norm, label="E(t) [norm]")
    plt.plot(time_fs, np.real(P_t) / P_norm, label="Re P(t)")
    plt.plot(time_fs, np.imag(P_t) / P_norm, label="Im P(t)")
    plt.plot(time_fs, n_t, label="Population n(t)")
    plt.xlabel("Time (fs)")
    plt.ylabel("Normalized units")
    plt.legend()
    plt.title("SBE Time-Domain Dynamics")
    savefig(fig, outdir, "sbe_time_domain.png", args.show)

    # Susceptibility
    fig = plt.figure()
    plt.plot(omega_eV, chi_r, label="Re chi(omega)")
    plt.plot(omega_eV, chi_i, label="Im chi(omega)")
    plt.xlim(0, 3)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Susceptibility")
    plt.legend()
    plt.title("Effective Susceptibility")
    savefig(fig, outdir, "sbe_chi.png", args.show)

    # Refractive index modification
    fig = plt.figure()
    plt.plot(omega_eV, n_rel)
    plt.xlim(0, 3)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Delta n (relative)")
    plt.title("Refractive Index Modification")
    savefig(fig, outdir, "sbe_n_rel.png", args.show)

    # Absorption coefficient proxy
    fig = plt.figure()
    plt.plot(omega_eV, alpha)
    plt.xlim(0, 3)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Alpha (arb. units)")
    plt.title("Absorption Coefficient")
    savefig(fig, outdir, "sbe_alpha.png", args.show)

    # UPPE spatiotemporal field
    x_um = x_m * 1e6
    time_fs_uppe = time_s_uppe * 1e15
    fig = plt.figure(figsize=(7.5, 4))
    plt.imshow(
        np.real(E_xt),
        extent=[time_fs_uppe.min(), time_fs_uppe.max(), x_um.min(), x_um.max()],
        aspect="auto",
        cmap="RdBu",
        origin="lower",
    )
    plt.colorbar(label="E(x,t)")
    plt.xlabel("Time (fs)")
    plt.ylabel("x (um)")
    plt.title("UPPE Spatiotemporal Field")
    savefig(fig, outdir, "uppe_field.png", args.show)

    # On-axis output time trace
    ix0 = int(np.argmin(np.abs(x_m)))
    E_out_t = E_xt[ix0, :]
    fig = plt.figure()
    plt.plot(time_fs, E_t / E_norm, label="Input E(t) [norm]")
    out_norm = np.max(np.abs(E_out_t)) or 1.0
    plt.plot(time_fs_uppe, E_out_t / out_norm, label="Output E(t) on-axis [norm]")
    plt.xlabel("Time (fs)")
    plt.ylabel("Normalized field")
    plt.legend()
    plt.title("Input vs Output Temporal Field")
    savefig(fig, outdir, "time_compare.png", args.show)

    # Feedback vs drive
    E_fb_path = data["sbe"] / "E_t_feedback.npy"
    if E_fb_path.exists():
        E_fb = np.load(E_fb_path)
        n = min(len(E_fb), len(E_t))
        rel = np.linalg.norm(E_fb[:n] - E_t[:n]) / max(np.linalg.norm(E_t[:n]), 1e-30)

        fig = plt.figure()
        plt.plot(time_fs[:n], E_t[:n] / (np.max(np.abs(E_t[:n])) or 1.0), label="Drive E(t)")
        plt.plot(time_fs[:n], E_fb[:n] / (np.max(np.abs(E_fb[:n])) or 1.0), label="Feedback E(t)")
        plt.xlabel("Time (fs)")
        plt.ylabel("Normalized field")
        plt.legend()
        plt.title(f"Feedback vs Drive (rel L2 = {rel:.2e})")
        savefig(fig, outdir, "feedback_compare.png", args.show)

    # Spectral evolution
    dt_in = time_s[1] - time_s[0]
    dt_out = time_s_uppe[1] - time_s_uppe[0]
    if not np.isclose(dt_in, dt_out, rtol=1e-6, atol=0):
        print(f"Warning: dt mismatch (SBE {dt_in:.3e} s vs UPPE {dt_out:.3e} s)")

    E_in = E_t - E_t.mean()
    win_in = hann(len(E_in))
    Ew_in = np.abs(np.fft.rfft(E_in * win_in)) ** 2

    E_out = E_out_t - E_out_t.mean()
    win_out = hann(len(E_out))
    Ew_out = np.abs(np.fft.rfft(E_out * win_out)) ** 2

    freq_in = np.fft.rfftfreq(len(E_in), dt_in)
    freq_out = np.fft.rfftfreq(len(E_out), dt_out)
    freq_in_eV = HBAR * (2 * np.pi * freq_in) / E0
    freq_out_eV = HBAR * (2 * np.pi * freq_out) / E0

    fig = plt.figure()
    plt.plot(freq_in_eV, Ew_in / (Ew_in.max() or 1.0), label="Input")
    plt.plot(freq_out_eV, Ew_out / (Ew_out.max() or 1.0), label="After UPPE")
    plt.xlim(0, 3)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Normalized spectral power")
    plt.legend()
    plt.title("Spectral Evolution")
    savefig(fig, outdir, "spectral_evolution.png", args.show)

    def spectral_moments(freq_eV: np.ndarray, spectrum: np.ndarray) -> tuple[float, float]:
        s = spectrum / np.sum(spectrum)
        mu = float(np.sum(freq_eV * s))
        sigma = float(np.sqrt(np.sum((freq_eV - mu) ** 2 * s)))
        return mu, sigma

    mu_in, sig_in = spectral_moments(freq_in_eV, Ew_in)
    mu_out, sig_out = spectral_moments(freq_out_eV, Ew_out)
    metrics = [
        f"Input centroid: {mu_in:.3f} eV, RMS width: {sig_in:.3f} eV",
        f"Output centroid: {mu_out:.3f} eV, RMS width: {sig_out:.3f} eV",
    ]
    print("\n".join(metrics))
    (outdir / "spectral_metrics.txt").write_text("\n".join(metrics) + "\n")

    # Transverse energy proxy
    energy_xt = np.trapezoid(np.abs(E_xt) ** 2, x_m, axis=0)
    fig = plt.figure()
    plt.plot(time_fs_uppe, energy_xt / (energy_xt.max() or 1.0))
    plt.xlabel("Time (fs)")
    plt.ylabel("Normalized integral |E|^2 dx")
    plt.title("Transverse Energy Proxy")
    savefig(fig, outdir, "energy_proxy.png", args.show)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
