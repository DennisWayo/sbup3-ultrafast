#!/usr/bin/env python3
"""
SBUP3 visualization script.

Run from the repo root:
  python analysis/visualize_sbup3.py
"""

from __future__ import annotations

import argparse
import os
import re
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


def _load_tddft_log(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    pattern_om = re.compile(r"om=([0-9.+-Ee]+)\[eV\]\s+\|me\|=([0-9.+-Ee]+)")
    pattern_kss = re.compile(r"eji=([0-9.+-Ee]+)\[eV\]\s+\(([^)]+)\)")

    energies: list[float] = []
    osc: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern_om.search(line)
        if match:
            energies.append(float(match.group(1)))
            osc.append(float(match.group(2)))
            continue
        match_kss = pattern_kss.search(line)
        if match_kss:
            try:
                energy = float(match_kss.group(1))
                vec = [float(v.strip()) for v in match_kss.group(2).split(",")]
                if len(vec) == 3:
                    energies.append(energy)
                    osc.append(float(np.linalg.norm(vec)))
            except Exception:
                continue

    if not energies:
        return None
    e = np.asarray(energies, dtype=float)
    f = np.asarray(osc, dtype=float)
    idx = np.argsort(e)
    return e[idx], f[idx]


def _oscillator_spectrum(energy_grid: np.ndarray, e0: np.ndarray, osc: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        sigma = 0.1
    spec = np.zeros_like(energy_grid, dtype=float)
    for e, f in zip(e0, osc):
        spec += f * np.exp(-0.5 * ((energy_grid - e) / sigma) ** 2)
    return spec


def load_outputs(base: Path) -> dict:
    sbe = base / "sbe"
    uppe = base / "uppe" / "outputs"

    alpha_path = sbe / "alpha_m_inv.npy"
    if not alpha_path.exists():
        alpha_path = sbe / "alpha_rel.npy"

    n_path = sbe / "n_real.npy"
    n_is_absolute = True
    if not n_path.exists():
        n_path = sbe / "n_rel.npy"
        n_is_absolute = False

    support_path = sbe / "chi_support_mask.npy"

    required = [
        sbe / "time_s.npy",
        sbe / "E_t.npy",
        sbe / "polarization.npy",
        sbe / "population.npy",
        sbe / "omega_rad_s.npy",
        sbe / "chi_eff_complex.npy",
        sbe / "chi_eff_real.npy",
        sbe / "chi_eff_imag.npy",
        alpha_path,
        n_path,
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
        "alpha": np.load(alpha_path),
        "n_rel": np.load(n_path),
        "n_is_absolute": n_is_absolute,
        "chi_support_mask": np.load(support_path).astype(bool) if support_path.exists() else None,
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
    parser.add_argument(
        "--baseline-log",
        default=None,
        help="Path to TDDFT log for overlay (defaults to dft/gaas_lrtddft.log)",
    )
    parser.add_argument(
        "--osc-broadening",
        type=float,
        default=0.1,
        help="Gaussian sigma [eV] for TDDFT oscillator spectrum",
    )
    parser.add_argument(
        "--overlay-scale",
        choices=["fit", "peak", "none"],
        default="fit",
        help="Scale SBE alpha to TDDFT for overlay (fit, peak) or use raw amplitude (none)",
    )
    parser.add_argument(
        "--overlay-abs",
        action="store_true",
        help="Use |alpha| for the SBE overlay (useful when sign conventions differ)",
    )
    parser.add_argument(
        "--overlay-e-min",
        type=float,
        default=0.0,
        help="Energy window min [eV] for overlay scaling/plot",
    )
    parser.add_argument(
        "--overlay-e-max",
        type=float,
        default=3.0,
        help="Energy window max [eV] for overlay scaling/plot",
    )
    parser.add_argument(
        "--overlay-window",
        default=None,
        help="Convenience window in eV as min,max (overrides overlay-e-min/max)",
    )
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
    n_is_absolute = data["n_is_absolute"]
    chi_support_mask = data["chi_support_mask"]
    E_xt = data["E_xt"]
    x_m = data["x_m"]
    time_s_uppe = data["time_s_uppe"]

    time_fs = time_s * 1e15
    omega_eV = HBAR * omega / E0
    valid = np.isfinite(chi_r) & np.isfinite(chi_i) & np.isfinite(alpha) & np.isfinite(n_rel)
    if chi_support_mask is not None and len(chi_support_mask) == len(valid):
        valid &= chi_support_mask

    # TDDFT reference (optional overlay)
    ref_energy = None
    ref_osc = None
    ref_label = None
    if args.baseline_log:
        ref_path = Path(args.baseline_log)
        ref = _load_tddft_log(ref_path)
        if ref:
            ref_energy, ref_osc = ref
            ref_label = str(ref_path)
    else:
        default_paths = [
            base / "dft" / "gaas_lrtddft.log",
            base / "dft" / "lrtddft.log",
        ]
        for path in default_paths:
            ref = _load_tddft_log(path)
            if ref:
                ref_energy, ref_osc = ref
                ref_label = str(path)
                break

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
    plt.plot(omega_eV[valid], chi_r[valid], label="Re chi(omega)")
    plt.plot(omega_eV[valid], chi_i[valid], label="Im chi(omega)")
    plt.xlim(0, 3)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Susceptibility")
    plt.legend()
    plt.title("Effective Susceptibility")
    savefig(fig, outdir, "sbe_chi.png", args.show)

    # Refractive index modification
    fig = plt.figure()
    plt.plot(omega_eV[valid], n_rel[valid])
    plt.xlim(0, 3)
    plt.xlabel("Photon energy (eV)")
    if n_is_absolute:
        plt.ylabel("n(omega)")
        plt.title("Refractive Index")
    else:
        plt.ylabel("Delta n (relative)")
        plt.title("Refractive Index Modification")
    savefig(fig, outdir, "sbe_n_rel.png", args.show)

    # Absorption coefficient (SBE only)
    fig = plt.figure()
    alpha_mask = (omega_eV >= 0) & (omega_eV <= 3) & valid
    plt.plot(omega_eV[alpha_mask], alpha[alpha_mask])
    plt.xlim(0, 3)
    plt.xlabel("Photon energy (eV)")
    plt.ylabel("Alpha (1/m)")
    savefig(fig, outdir, "sbe_alpha.png", args.show)

    # TDDFT overlay (separate file)
    if args.overlay_window:
        try:
            wmin, wmax = [float(v.strip()) for v in args.overlay_window.split(",")]
            args.overlay_e_min = wmin
            args.overlay_e_max = wmax
        except Exception:
            pass

    if ref_energy is not None and ref_osc is not None:
        ref_alpha = _oscillator_spectrum(omega_eV, ref_energy, ref_osc, args.osc_broadening)
        sim_alpha = np.abs(alpha) if args.overlay_abs else alpha
        finite_mask = np.isfinite(sim_alpha) & np.isfinite(ref_alpha)
        mask = (omega_eV >= args.overlay_e_min) & (omega_eV <= args.overlay_e_max) & valid & finite_mask
        if not np.any(mask):
            mask = finite_mask
        scale = 1.0
        if args.overlay_scale == "fit":
            denom = float(np.dot(sim_alpha[mask], sim_alpha[mask])) or 1.0
            scale = float(np.dot(ref_alpha[mask], sim_alpha[mask]) / denom)
        elif args.overlay_scale == "peak":
            sim_peak = float(np.max(np.abs(sim_alpha[mask]))) if hasattr(mask, "__len__") else float(np.max(np.abs(sim_alpha)))
            ref_peak = float(np.max(np.abs(ref_alpha[mask]))) if hasattr(mask, "__len__") else float(np.max(np.abs(ref_alpha)))
            if sim_peak > 0:
                scale = ref_peak / sim_peak
        fig, ax = plt.subplots()
        apply_scale = args.overlay_scale in {"fit", "peak"}
        if args.overlay_abs:
            label = "SBE |alpha| (scaled)" if apply_scale else "SBE |alpha|"
        else:
            label = "SBE alpha (scaled)" if apply_scale else "SBE alpha"
        sim_plot = sim_alpha * scale if apply_scale else sim_alpha
        ref_plot = ref_alpha
        ax.plot(omega_eV, sim_plot, label=label)
        ax.plot(omega_eV, ref_plot, label="TDDFT osc", linestyle="--")
        ax.set_xlim(args.overlay_e_min, args.overlay_e_max)
        # Auto-scale y limits to the visible window for clarity
        if hasattr(mask, "__len__"):
            visible = np.concatenate([sim_plot[mask], ref_plot[mask]])
        else:
            visible = np.concatenate([sim_plot, ref_plot])
        vmin = float(np.min(visible))
        vmax = float(np.max(visible))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            pad = 0.05 * (vmax - vmin)
            ax.set_ylim(vmin - pad, vmax + pad)

        # Region bands (IR / Visible / UV)
        regions = [
            ("IR", 0.0, 1.65, "#e8f2ff"),
            ("Visible", 1.65, 3.1, "#fff3d9"),
            ("UV", 3.1, 100.0, "#f7e9ff"),
        ]
        xmin, xmax = args.overlay_e_min, args.overlay_e_max
        for name, a, b, color in regions:
            left = max(xmin, a)
            right = min(xmax, b)
            if right > left:
                ax.axvspan(left, right, color=color, alpha=0.18, zorder=0)
                mid = 0.5 * (left + right)
                ax.text(
                    mid,
                    0.95,
                    name,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="0.35",
                )

        # Top axis with wavelength (um) using explicit tick mapping
        ev_min = float(args.overlay_e_min)
        ev_max = float(args.overlay_e_max)
        tick_start = max(ev_min, 0.2)
        tick_end = ev_max
        if tick_end - tick_start >= 0.1:
            step = 0.2 if (tick_end - tick_start) >= 0.6 else 0.1
            ticks = np.arange(tick_start, tick_end + 1e-9, step)
            labels = [f"{(1.23984193 / t):.2f}" for t in ticks]
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            ax_top.set_xticks(ticks)
            ax_top.set_xticklabels(labels)
            ax_top.set_xlabel("Wavelength (um)")

        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("Alpha (arb. units)")
        ax.set_title("Absorption Overlay (TDDFT)")
        ax.legend()
        savefig(fig, outdir, "sbe_alpha_tddft_overlay.png", args.show)

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
