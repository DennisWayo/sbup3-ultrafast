#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle


HBAR = 1.054571817e-34
E0 = 1.602176634e-19
C0 = 299792458.0

PALETTE = {
    "sim": "#1f77b4",
    "ref_primary": "#d62728",
    "ref_secondary": "#2ca02c",
    "accent": "#9467bd",
    "muted": "#7f7f7f",
}

EXPORT_DPI = 360


def _set_pub_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 120,
            "savefig.dpi": EXPORT_DPI,
            "savefig.bbox": "tight",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.9,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "grid.color": "#c8c8c8",
            "axes.grid": True,
            "axes.axisbelow": True,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        out: dict[str, list[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in reader.fieldnames:
                raw = row.get(k, "")
                try:
                    out[k].append(float(raw))
                except Exception:
                    out[k].append(float("nan"))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _norm_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _pick_col(data: dict[str, np.ndarray], *aliases: str) -> np.ndarray:
    by_norm = {_norm_key(k): v for k, v in data.items()}
    for name in aliases:
        key = _norm_key(name)
        if key in by_norm:
            return by_norm[key]
    raise KeyError(f"Missing required column aliases: {aliases}")


def _interp_overlap(
    sim_e: np.ndarray,
    sim_y: np.ndarray,
    ref_e: np.ndarray,
    ref_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim_mask = np.isfinite(sim_e) & np.isfinite(sim_y)
    ref_mask = np.isfinite(ref_e) & np.isfinite(ref_y)
    se = sim_e[sim_mask]
    sy = sim_y[sim_mask]
    re = ref_e[ref_mask]
    ry = ref_y[ref_mask]
    if len(se) < 2 or len(re) < 2:
        return np.asarray([]), np.asarray([]), np.asarray([])
    e_min = max(float(np.min(se)), float(np.min(re)))
    e_max = min(float(np.max(se)), float(np.max(re)))
    m = (re >= e_min) & (re <= e_max)
    if int(np.sum(m)) < 2:
        return np.asarray([]), np.asarray([]), np.asarray([])
    re2 = re[m]
    ry2 = ry[m]
    sy2 = np.interp(re2, se, sy)
    return re2, sy2, ry2


def _load_study_paths(base: Path) -> dict[str, Path]:
    return {
        "run": base / "analysis/reports/study/run_release_primary_jellison_20260520_081916.json",
        "cal": base / "analysis/reports/study/calibration_release_primary_jellison.json",
        "val_jel": base / "analysis/reports/study/validation_release_primary_jellison_20260520_081916_104478.json",
        "val_asp": base / "analysis/reports/study/validation_secondary_aspnes_release_20260520_082112_511494.json",
        "val_linear": base / "analysis/reports/study/validation_linearity_primary_jellison_20260520_081928_899730.json",
        "val_conv": base / "analysis/reports/study/validation_convergence_primary_jellison_20260520_081928_899769.json",
        "csv_dielectric": base / "analysis/data/gaas_dielectric_reference.csv",
        "csv_jel": base / "analysis/data/gaas_exp_jellison_1992.csv",
        "csv_asp": base / "analysis/data/gaas_exp_aspnes_1986.csv",
        "trans_e": base / "dft/transitions_E_eV.npy",
        "trans_f": base / "dft/transitions_f.npy",
        "trans_meta": base / "dft/transitions_meta.json",
        "sbe_probe_omega": base / "sbe/probe_omega_rad_s.npy",
        "sbe_probe_alpha": base / "sbe/probe_alpha_m_inv.npy",
        "sbe_probe_n": base / "sbe/probe_n_real.npy",
        "sbe_probe_chi_r": base / "sbe/probe_chi_eff_real.npy",
        "sbe_probe_chi_i": base / "sbe/probe_chi_eff_imag.npy",
        "sbe_probe_support": base / "sbe/probe_chi_support_mask.npy",
        "sbe_drive_chi_r": base / "sbe/chi_eff_real.npy",
        "sbe_drive_chi_i": base / "sbe/chi_eff_imag.npy",
        "sbe_drive_support": base / "sbe/chi_support_mask.npy",
        "sbe_drive_omega": base / "sbe/omega_rad_s.npy",
        "uppe_Ext": base / "uppe/outputs/E_xt_out.npy",
        "uppe_x": base / "uppe/outputs/x_m.npy",
        "uppe_t": base / "uppe/outputs/time_s.npy",
    }


def _cal_objective_grid(
    cal: dict[str, Any],
    probe_omega: np.ndarray,
    probe_chi: np.ndarray,
    ref_e: np.ndarray,
    ref_alpha: np.ndarray,
    ref_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    search = cal["fit"]["search"]
    priors = cal["fit"]["priors"]
    d_grid = np.logspace(
        np.log10(float(search["density_m3_min"])),
        np.log10(float(search["density_m3_max"])),
        int(search["density_points"]),
    )
    o_grid = np.logspace(
        np.log10(float(search["osc_norm_min"])),
        np.log10(float(search["osc_norm_max"])),
        int(search["osc_points"]),
    )

    base_d = float(cal["base_state"]["base_density_m3"])
    base_o = float(cal["base_state"]["base_osc_norm_factor"])
    prior_w = float(priors["prior_weight"])
    prior_d = float(priors["density_prior_m3"])
    prior_d_sig = float(priors["density_rel_sigma"])
    prior_o = float(priors["osc_prior"])
    prior_o_sig = float(priors["osc_rel_sigma"])

    sim_e = HBAR * probe_omega / E0
    z = np.full((len(d_grid), len(o_grid)), np.nan, dtype=float)

    def prior_term(v: float, p: float, rel_sigma: float) -> float:
        if v <= 0.0 or p <= 0.0 or rel_sigma <= 0.0:
            return 0.0
        sigma_ln = np.log1p(rel_sigma)
        if sigma_ln <= 0.0:
            return 0.0
        u = np.log(v / p) / sigma_ln
        return float(u * u)

    for i, d in enumerate(d_grid):
        for j, o in enumerate(o_grid):
            ratio = (d / base_d) * (o / base_o)
            chi = ratio * probe_chi
            n_tilde = np.sqrt(1.0 + chi)
            pred_alpha = 2.0 * probe_omega * np.imag(n_tilde) / C0
            pred_n = np.real(n_tilde)

            e_a, sa, ra = _interp_overlap(sim_e, np.abs(pred_alpha), ref_e, ref_alpha)
            e_n, sn, rn = _interp_overlap(sim_e, pred_n, ref_e, ref_n)
            if len(e_a) < 5 or len(e_n) < 5:
                continue

            rmse_a = float(np.sqrt(np.mean((sa - ra) ** 2)))
            rmse_n = float(np.sqrt(np.mean((sn - rn) ** 2)))
            norm_a = float(np.max(np.abs(ra))) or 1.0
            norm_n = float(np.max(np.abs(rn))) or 1.0
            data_obj = 0.5 * (rmse_a / norm_a + rmse_n / norm_n)
            prior_obj = prior_term(float(d), prior_d, prior_d_sig) + prior_term(float(o), prior_o, prior_o_sig)
            z[i, j] = data_obj + prior_w * prior_obj

    return d_grid, o_grid, z


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    out_dir = base / "analysis/figures/study_20260520"
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_pub_style()

    p = _load_study_paths(base)
    run = _load_json(p["run"])
    cal = _load_json(p["cal"])
    val_jel = _load_json(p["val_jel"])
    val_asp = _load_json(p["val_asp"])
    val_linear = _load_json(p["val_linear"])
    val_conv = _load_json(p["val_conv"])
    trans_meta = _load_json(p["trans_meta"])

    dielectric = _load_csv(p["csv_dielectric"])
    jel = _load_csv(p["csv_jel"])
    asp = _load_csv(p["csv_asp"])

    die_e = _pick_col(dielectric, "energy_eV")
    die_alpha = _pick_col(dielectric, "alpha_m_inv", "alpha")
    die_n = _pick_col(dielectric, "n")
    die_eps2 = _pick_col(dielectric, "eps2_lf", "eps2")

    jel_e = _pick_col(jel, "energy_eV")
    jel_alpha = _pick_col(jel, "alpha", "alpha_m_inv")
    jel_n = _pick_col(jel, "n")

    asp_e = _pick_col(asp, "energy_eV")
    asp_alpha = _pick_col(asp, "alpha", "alpha_m_inv")
    asp_n = _pick_col(asp, "n")

    trans_e = np.asarray(np.load(p["trans_e"]), dtype=float)
    trans_f = np.asarray(np.load(p["trans_f"]), dtype=float)

    probe_omega = np.asarray(np.load(p["sbe_probe_omega"]), dtype=float)
    probe_e = HBAR * probe_omega / E0
    probe_alpha = np.asarray(np.load(p["sbe_probe_alpha"]), dtype=float)
    probe_n = np.asarray(np.load(p["sbe_probe_n"]), dtype=float)
    probe_chi_r = np.asarray(np.load(p["sbe_probe_chi_r"]), dtype=float)
    probe_chi_i = np.asarray(np.load(p["sbe_probe_chi_i"]), dtype=float)
    probe_support = np.asarray(np.load(p["sbe_probe_support"]), dtype=bool)
    probe_chi = probe_chi_r + 1j * probe_chi_i

    drive_omega = np.asarray(np.load(p["sbe_drive_omega"]), dtype=float)
    drive_e = HBAR * drive_omega / E0
    drive_chi_r = np.asarray(np.load(p["sbe_drive_chi_r"]), dtype=float)
    drive_chi_i = np.asarray(np.load(p["sbe_drive_chi_i"]), dtype=float)
    drive_support = np.asarray(np.load(p["sbe_drive_support"]), dtype=bool)

    E_xt = np.asarray(np.load(p["uppe_Ext"]))
    x_m = np.asarray(np.load(p["uppe_x"]), dtype=float)
    t_s = np.asarray(np.load(p["uppe_t"]), dtype=float)
    energy_proxy = np.trapezoid(np.abs(E_xt) ** 2, x_m, axis=0)

    # F0
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 4.8), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    source_box = FancyBboxPatch(
        (0.03, 0.28),
        0.19,
        0.44,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.1,
        edgecolor="#3b3b3b",
        facecolor="#f9f5f2",
    )
    ax.add_patch(source_box)
    ax.text(0.125, 0.67, "Ultrafast laser", ha="center", va="center", fontsize=10.0, weight="bold")
    ax.text(0.125, 0.61, r"$E_{\mathrm{in}}(t)$", ha="center", va="center", fontsize=9.8, color=PALETTE["ref_primary"])
    u = np.linspace(-2.8, 2.8, 500)
    pulse = np.exp(-(u / 1.15) ** 2) * np.cos(10.0 * u)
    xw = 0.05 + (u - np.min(u)) / (np.max(u) - np.min(u)) * 0.15
    yw = 0.46 + 0.11 * pulse
    ax.plot(xw, yw, color=PALETTE["ref_primary"], lw=2.0)
    ax.text(0.125, 0.33, "fs pulse driver", ha="center", va="center", fontsize=8.7, color="#555555")

    ax.annotate(
        "",
        xy=(0.36, 0.50),
        xytext=(0.23, 0.50),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#4a4a4a"),
    )
    ax.text(0.265, 0.55, "incident field", fontsize=8.8, color="#4a4a4a")

    block_front = Rectangle((0.36, 0.26), 0.24, 0.46, linewidth=1.2, edgecolor="#2f2f2f", facecolor="#eef2f8")
    block_top = Polygon(
        [(0.36, 0.72), (0.43, 0.79), (0.67, 0.79), (0.60, 0.72)],
        closed=True,
        linewidth=1.0,
        edgecolor="#2f2f2f",
        facecolor="#e2e8f3",
    )
    block_side = Polygon(
        [(0.60, 0.26), (0.67, 0.33), (0.67, 0.79), (0.60, 0.72)],
        closed=True,
        linewidth=1.0,
        edgecolor="#2f2f2f",
        facecolor="#d8dfec",
    )
    ax.add_patch(block_front)
    ax.add_patch(block_top)
    ax.add_patch(block_side)

    gx = np.linspace(0.40, 0.56, 5)
    gy = np.linspace(0.34, 0.64, 4)
    for xv in gx:
        for yv in gy:
            ax.add_patch(Circle((xv, yv), 0.007, edgecolor="none", facecolor="#7d8fb3", alpha=0.92))
    ax.text(0.48, 0.74, "Bulk GaAs crystal", ha="center", va="center", fontsize=9.8, weight="bold")
    ax.annotate(
        "",
        xy=(0.50, 0.64),
        xytext=(0.50, 0.36),
        arrowprops=dict(arrowstyle="<->", lw=1.3, color=PALETTE["accent"]),
    )
    ax.text(0.515, 0.65, r"$P(t),\, n(t)$", fontsize=9.2, color=PALETTE["accent"], va="bottom")
    ax.annotate(
        "",
        xy=(0.39, 0.30),
        xytext=(0.34, 0.30),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#595959"),
    )
    ax.text(0.338, 0.285, r"$x$", fontsize=8.5, color="#595959")
    ax.annotate(
        "",
        xy=(0.39, 0.35),
        xytext=(0.39, 0.30),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#595959"),
    )
    ax.text(0.394, 0.352, r"$z$", fontsize=8.5, color="#595959")

    ax.annotate(
        "",
        xy=(0.72, 0.50),
        xytext=(0.60, 0.50),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#4a4a4a"),
    )
    ax.text(0.63, 0.55, "material response", fontsize=8.8, color="#4a4a4a")

    out_box = FancyBboxPatch(
        (0.73, 0.22),
        0.24,
        0.56,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.1,
        edgecolor="#3b3b3b",
        facecolor="#f6faf5",
    )
    ax.add_patch(out_box)
    ax.text(0.85, 0.73, "Measured outputs", ha="center", va="center", fontsize=10.0, weight="bold")
    ax.plot([0.75, 0.82], [0.62, 0.62], color=PALETTE["sim"], lw=2.4)
    ax.text(0.835, 0.615, r"$n(\omega)$", fontsize=9.3, va="center")
    ax.plot([0.75, 0.82], [0.53, 0.53], color=PALETTE["ref_primary"], lw=2.4)
    ax.text(0.835, 0.525, r"$\alpha(\omega)$", fontsize=9.3, va="center")
    ax.plot([0.75, 0.82], [0.44, 0.44], color=PALETTE["accent"], lw=2.4)
    ax.text(0.835, 0.435, r"$E(x,t,z)$", fontsize=9.3, va="center")
    ax.text(0.85, 0.33, "Calibration and validation", ha="center", va="center", fontsize=8.8, color="#555555")

    _save(fig, out_dir, "F0_intro_gaas_interaction")
    plt.close(fig)
    user_flow = out_dir / "flow.png"
    if user_flow.exists():
        flow_img = plt.imread(user_flow)
        flow_h, flow_w = flow_img.shape[:2]
        fig_w = 12.8
        fig_h = max(3.8, fig_w * (float(flow_h) / float(max(flow_w, 1))))
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True)
        ax.imshow(flow_img)
        ax.set_axis_off()
        _save(fig, out_dir, "F0_intro_gaas_interaction")
        plt.close(fig)

    # F1
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.set_axis_off()
    stages = [
        (0.03, 0.62, "Periodic DFT\nReference"),
        (0.27, 0.62, "Transition\nReduction"),
        (0.51, 0.62, "SBE\nDrive/Probe"),
        (0.75, 0.62, "UPPE\nPropagation"),
        (0.27, 0.24, "Physical\nCalibration"),
        (0.51, 0.24, "Validation\n(Primary/Secondary)"),
        (0.75, 0.24, "Stability +\nLinearity Checks"),
    ]
    for x, y, text in stages:
        patch = FancyBboxPatch(
            (x, y),
            0.2,
            0.2,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#2f2f2f",
            facecolor="#f7f7f7",
        )
        ax.add_patch(patch)
        ax.text(x + 0.1, y + 0.1, text, ha="center", va="center", fontsize=9.2)

    arrows = [
        ((0.23, 0.72), (0.27, 0.72)),
        ((0.47, 0.72), (0.51, 0.72)),
        ((0.71, 0.72), (0.75, 0.72)),
        ((0.37, 0.62), (0.37, 0.44)),
        ((0.61, 0.62), (0.61, 0.44)),
        ((0.85, 0.62), (0.85, 0.44)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=1.15, color="#444444"),
        )
    ax.text(0.0, 1.03, "A) Workflow Structure", transform=ax.transAxes, fontsize=11, weight="bold")

    ax = axes[1]
    ax.set_axis_off()
    transition_summary = val_jel["metrics"]["transition_summary"]
    digest = run["dft"]["outputs"]["dielectric_csv"]["sha256"][:12]
    rows = [
        ("Run identifier", Path(p["run"]).stem.replace("run_", "")),
        ("Calibration set", "Jellison (primary)"),
        ("Secondary check", "Aspnes"),
        ("Transition count", f"{transition_summary['n_transitions']}"),
        (
            "Transition range [eV]",
            f"{transition_summary['energy_min_eV']:.3f} - {transition_summary['energy_max_eV']:.3f}",
        ),
        ("Density [m^-3]", f"{run['effective_params']['sbe']['density_m3']:.3e}"),
        ("Oscillator norm", f"{run['effective_params']['sbe']['osc_norm_factor']:.6f}"),
        ("Dielectric digest", digest),
    ]
    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Pinned provenance field", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.28)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#ececec")
            cell.set_text_props(weight="bold")
    ax.text(0.0, 1.03, "B) Provenance Snapshot", transform=ax.transAxes, fontsize=11, weight="bold")
    _save(fig, out_dir, "F1_workflow_provenance")
    plt.close(fig)

    # F2
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(die_e, np.abs(die_alpha), label="DFT dielectric reference", color=PALETTE["sim"])
    axes[0].plot(jel_e, jel_alpha, label="Jellison 1992", color=PALETTE["ref_primary"])
    axes[0].plot(asp_e, asp_alpha, label="Aspnes 1986", color=PALETTE["ref_secondary"])
    axes[0].axvspan(1.5, 3.0, color=PALETTE["muted"], alpha=0.10, zorder=0)
    axes[0].set_ylabel(r"$\alpha$ [m$^{-1}$]")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].set_title("A) Absorption Comparison")

    axes[1].plot(die_e, die_n, label="DFT dielectric reference", color=PALETTE["sim"])
    axes[1].plot(jel_e, jel_n, label="Jellison 1992", color=PALETTE["ref_primary"])
    axes[1].plot(asp_e, asp_n, label="Aspnes 1986", color=PALETTE["ref_secondary"])
    axes[1].axvspan(1.5, 3.0, color=PALETTE["muted"], alpha=0.10, zorder=0)
    axes[1].set_ylabel("n")
    axes[1].set_xlabel("Photon energy [eV]")
    axes[1].set_title("B) Refractive Index Comparison")
    _save(fig, out_dir, "F2_reference_optics_overlay")
    plt.close(fig)

    # F3
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), constrained_layout=True)
    axes[0].plot(die_e, die_eps2, color=PALETTE["sim"])
    stem_heights = np.interp(trans_e, die_e, die_eps2)
    axes[0].vlines(trans_e, 0.0, stem_heights, color=PALETTE["ref_primary"], alpha=0.72, lw=1.2)
    axes[0].set_xlim(0.0, 3.1)
    axes[0].set_xlabel("Energy [eV]")
    axes[0].set_ylabel(r"$\epsilon_2^{LF}$")
    axes[0].set_title("A) Transition-Energy Selection")

    markerline, stemlines, _ = axes[1].stem(trans_e, trans_f, basefmt=" ")
    plt.setp(markerline, markersize=4.2, markerfacecolor=PALETTE["accent"], markeredgecolor=PALETTE["accent"])
    plt.setp(stemlines, color=PALETTE["accent"], linewidth=1.2)
    axes[1].set_xlim(0.0, 3.1)
    axes[1].set_xlabel("Energy [eV]")
    axes[1].set_ylabel("Oscillator weight")
    axes[1].set_title("B) Reduced Weights")

    csum = np.cumsum(trans_f)
    axes[2].plot(np.arange(1, len(trans_f) + 1), csum, "o-", color=PALETTE["ref_secondary"])
    axes[2].set_xlabel("Transition index")
    axes[2].set_ylabel("Cumulative weight")
    axes[2].set_ylim(0.0, max(1.02, float(np.max(csum) * 1.05)))
    axes[2].set_title("C) Cumulative Oscillator Sum")
    _save(fig, out_dir, "F3_transition_model")
    plt.close(fig)

    # F4
    d_grid, o_grid, z = _cal_objective_grid(cal, probe_omega, probe_chi, jel_e, jel_alpha, jel_n)
    D, O = np.meshgrid(o_grid, d_grid)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    pcm = axes[0].pcolormesh(D, O, z, shading="auto", cmap="viridis")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Oscillator normalization factor")
    axes[0].set_ylabel(r"Density [m$^{-3}$]")
    axes[0].set_title("A) Physical-Fit Objective")
    axes[0].plot(
        float(cal["recommended_osc_norm_factor"]),
        float(cal["recommended_density_m3"]),
        "r*",
        ms=12,
        label="Best fit",
    )
    axes[0].legend(fontsize=8)
    fig.colorbar(pcm, ax=axes[0], label="Objective")

    b68 = cal["fit"]["uncertainty"]["bounds68"]
    b95 = cal["fit"]["uncertainty"]["bounds95"]
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Oscillator normalization factor")
    axes[1].set_ylabel(r"Density [m$^{-3}$]")
    axes[1].set_title("B) Surrogate Uncertainty Bounds")
    axes[1].add_patch(
        plt.Rectangle(
            (b95["osc_norm_min"], b95["density_m3_min"]),
            b95["osc_norm_max"] - b95["osc_norm_min"],
            b95["density_m3_max"] - b95["density_m3_min"],
            fill=False,
            edgecolor=PALETTE["ref_primary"],
            lw=1.4,
            label="95% surrogate",
        )
    )
    axes[1].add_patch(
        plt.Rectangle(
            (b68["osc_norm_min"], b68["density_m3_min"]),
            b68["osc_norm_max"] - b68["osc_norm_min"],
            b68["density_m3_max"] - b68["density_m3_min"],
            fill=False,
            edgecolor=PALETTE["ref_secondary"],
            lw=1.8,
            label="68% surrogate",
        )
    )
    axes[1].plot(
        float(cal["recommended_osc_norm_factor"]),
        float(cal["recommended_density_m3"]),
        "r*",
        ms=12,
        label="Best fit",
    )
    axes[1].legend(fontsize=8)
    _save(fig, out_dir, "F4_calibration_landscape")
    plt.close(fig)

    # F5
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True, constrained_layout=True)
    e, sim, ref = _interp_overlap(probe_e, np.abs(probe_alpha), jel_e, jel_alpha)
    axes[0].plot(e, ref, label="Jellison reference", color=PALETTE["ref_primary"])
    axes[0].plot(e, sim, label="SBUP$^3$ probe |alpha|", color=PALETTE["sim"])
    axes[0].set_ylabel(r"$\alpha$ [1/m]")
    m = val_jel["baseline"]["metrics"]["alpha_abs"]
    axes[0].set_title(
        f"A) |alpha| fit: RMSE_norm={m['rmse_norm']:.3f}, RMSE_shape={m['rmse_shape']:.3f}"
    )
    axes[0].legend(loc="upper left", frameon=True)

    e, sim, ref = _interp_overlap(probe_e, probe_n, jel_e, jel_n)
    axes[1].plot(e, ref, label="Jellison reference", color=PALETTE["ref_primary"])
    axes[1].plot(e, sim, label="SBUP$^3$ probe n", color=PALETTE["sim"])
    axes[1].set_ylabel("n")
    axes[1].set_xlabel("Photon energy [eV]")
    m = val_jel["baseline"]["metrics"]["n"]
    axes[1].set_title(f"B) n fit: RMSE_norm={m['rmse_norm']:.3f}, RMSE_shape={m['rmse_shape']:.3f}")
    axes[1].legend(loc="upper left", frameon=True)
    _save(fig, out_dir, "F5_postcal_fit_jellison")
    plt.close(fig)

    # F6
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    field_mag = np.abs(E_xt)
    vmax = float(np.nanpercentile(field_mag, 99.5))
    im = axes[0].imshow(
        field_mag,
        aspect="auto",
        extent=[float(t_s[0] * 1e15), float(t_s[-1] * 1e15), float(x_m[0] * 1e6), float(x_m[-1] * 1e6)],
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    axes[0].set_xlabel("Time [fs]")
    axes[0].set_ylabel(r"$x$ [$\mu$m]")
    axes[0].set_title("A) |E(x,t)| at UPPE Output")
    fig.colorbar(im, ax=axes[0], label="|E|")
    axes[1].plot(t_s * 1e15, energy_proxy, color=PALETTE["sim"])
    axes[1].set_xlabel("Time [fs]")
    axes[1].set_ylabel("Transverse energy proxy")
    axes[1].set_title("B) Transverse Energy Proxy")
    _save(fig, out_dir, "F6_uppe_field_evolution")
    plt.close(fig)

    # S1
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True, constrained_layout=True)
    e, sim, ref = _interp_overlap(probe_e, np.abs(probe_alpha), asp_e, asp_alpha)
    axes[0].plot(e, ref, label="Aspnes reference", color=PALETTE["ref_secondary"])
    axes[0].plot(e, sim, label="SBUP$^3$ probe |alpha|", color=PALETTE["sim"])
    axes[0].set_ylabel(r"$\alpha$ [1/m]")
    m = val_asp["baseline"]["metrics"]["alpha_abs"]
    axes[0].set_title(
        f"A) |alpha| fit: RMSE_norm={m['rmse_norm']:.3f}, RMSE_shape={m['rmse_shape']:.3f}"
    )
    axes[0].legend(loc="upper left", frameon=True)

    e, sim, ref = _interp_overlap(probe_e, probe_n, asp_e, asp_n)
    axes[1].plot(e, ref, label="Aspnes reference", color=PALETTE["ref_secondary"])
    axes[1].plot(e, sim, label="SBUP$^3$ probe n", color=PALETTE["sim"])
    axes[1].set_ylabel("n")
    axes[1].set_xlabel("Photon energy [eV]")
    m = val_asp["baseline"]["metrics"]["n"]
    axes[1].set_title(f"B) n fit: RMSE_norm={m['rmse_norm']:.3f}, RMSE_shape={m['rmse_shape']:.3f}")
    axes[1].legend(loc="upper left", frameon=True)
    _save(fig, out_dir, "S1_postcal_fit_aspnes")
    plt.close(fig)

    # S2
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), constrained_layout=True)
    axes[0].plot(drive_e, drive_chi_r, label="drive", color=PALETTE["muted"])
    axes[0].plot(probe_e, probe_chi_r, label="probe", color=PALETTE["sim"])
    axes[0].set_xlim(0.0, 3.2)
    axes[0].set_xlabel("Energy [eV]")
    axes[0].set_ylabel(r"$\chi'$")
    axes[0].set_title(r"A) $\chi'$: Drive vs Probe")
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].plot(drive_e, drive_chi_i, label="drive", color=PALETTE["muted"])
    axes[1].plot(probe_e, probe_chi_i, label="probe", color=PALETTE["sim"])
    axes[1].set_xlim(0.0, 3.2)
    axes[1].set_xlabel("Energy [eV]")
    axes[1].set_ylabel(r"$\chi''$")
    axes[1].set_title(r"B) $\chi''$: Drive vs Probe")
    axes[1].legend(loc="upper right", frameon=True)

    axes[2].step(probe_e, probe_support.astype(float), where="mid", lw=1.5, label="probe support", color=PALETTE["sim"])
    axes[2].step(drive_e, drive_support.astype(float), where="mid", lw=1.5, label="drive support", color=PALETTE["accent"])
    axes[2].set_xlim(0.0, 3.2)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Energy [eV]")
    axes[2].set_ylabel("Support mask")
    axes[2].set_title("C) Trusted Support Masks")
    axes[2].legend(loc="upper right", frameon=True)
    _save(fig, out_dir, "S2_probe_vs_drive_response")
    plt.close(fig)

    # S3
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.6), constrained_layout=True)
    metrics = val_jel["metrics"]
    probe_window = metrics.get("probe_response", {})
    finite_probe = np.isfinite(probe_e) & np.isfinite(probe_chi_i)
    probe_e_f = probe_e[finite_probe]
    probe_chi_i_f = probe_chi_i[finite_probe]
    probe_support_f = probe_support[finite_probe]
    axes[0].plot(probe_e_f, np.abs(probe_chi_i_f), label=r"$|\chi''_{\mathrm{probe}}|$", color=PALETTE["sim"])
    axes[0].fill_between(
        probe_e_f,
        0.0,
        np.maximum(np.nanmax(np.abs(probe_chi_i_f)), 1e-20),
        where=probe_support_f,
        alpha=0.20,
        label="trusted support",
        color=PALETTE["accent"],
    )
    axes[0].set_ylabel(r"$|\chi''|$")
    axes[0].set_xlabel("Photon energy [eV]")
    if "min_energy_eV" in probe_window and "max_energy_eV" in probe_window:
        x_lo = float(probe_window["min_energy_eV"])
        x_hi = float(probe_window["max_energy_eV"])
    else:
        x_lo = float(np.nanmin(probe_e_f))
        x_hi = float(np.nanmax(probe_e_f))
    axes[0].set_xlim(x_lo, x_hi)
    axes[0].set_title("A) Probe Trusted-Support Region")
    axes[0].legend(loc="upper right", frameon=True)

    x = ["drive coverage", "probe coverage"]
    y = [
        float(metrics.get("response_support", {}).get("coverage_fraction", np.nan)),
        float(metrics.get("probe_response", {}).get("coverage_fraction", np.nan)),
    ]
    axes[1].bar(x, y, color=[PALETTE["muted"], PALETTE["sim"]], width=0.55)
    axes[1].set_ylim(0.0, max(0.01, float(max(y) * 1.25)))
    axes[1].set_ylabel("Coverage fraction")
    axes[1].set_xlabel("Coverage metric")
    axes[1].set_title(
        "B) Coverage Summary "
        f"(probe window {probe_window.get('min_energy_eV', np.nan):.2f}-{probe_window.get('max_energy_eV', np.nan):.2f} eV)"
    )
    _save(fig, out_dir, "S3_support_window_diagnostics")
    plt.close(fig)

    # S4
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    sweep = val_conv.get("sweep", [])
    labels: list[str] = []
    alpha_shape: list[float] = []
    n_shape: list[float] = []
    support_cov: list[float] = []
    for item in sweep:
        params = item.get("params", {})
        dt_as = params.get("dt_as")
        dz = params.get("dz")
        dz_um = float(dz) * 1e6 if dz is not None else np.nan
        labels.append(f"dt={dt_as:g} as\ndz={dz_um:.1f} μm")
        bmet = item.get("baseline", {}).get("metrics", {})
        alpha_shape.append(float(bmet.get("alpha_abs", {}).get("rmse_shape", np.nan)))
        n_shape.append(float(bmet.get("n", {}).get("rmse_shape", np.nan)))
        support_cov.append(float(item.get("metrics", {}).get("probe_response", {}).get("coverage_fraction", np.nan)))

    x = np.arange(len(labels))
    axes[0].plot(x, alpha_shape, "o-", label=r"$|\alpha|$ shape RMSE", color=PALETTE["sim"])
    axes[0].plot(x, n_shape, "s-", label=r"$n$ shape RMSE", color=PALETTE["ref_primary"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Shape RMSE")
    axes[0].set_title("A) Convergence Sweep")
    ax2 = axes[0].twinx()
    ax2.plot(x, support_cov, "^-", color=PALETTE["ref_secondary"], lw=1.2, label="probe support coverage")
    ax2.set_ylabel("Support coverage")
    ax2.tick_params(axis="y", labelsize=8)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[0].legend(
        h1 + h2,
        l1 + l2,
        loc="center right",
        bbox_to_anchor=(0.99, 0.58),
        ncol=1,
        frameon=True,
        framealpha=0.92,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=1.8,
    )

    lc = val_linear.get("linear_check")
    ratio = float(lc["ratio_Pmax"]) if isinstance(lc, dict) and lc.get("ratio_Pmax") is not None else np.nan
    amp = float(lc["amplitude"]) if isinstance(lc, dict) and lc.get("amplitude") is not None else np.nan
    axes[1].bar(["Measured", "Ideal"], [ratio, 2.0], color=[PALETTE["sim"], PALETTE["muted"]], width=0.58)
    axes[1].set_ylabel("Pmax ratio")
    axes[1].set_title("B) Linearity Check")
    rel_err = abs(ratio - 2.0) / 2.0 if np.isfinite(ratio) else np.nan
    axes[1].set_title(
        "B) Linearity Check\n"
        f"Amplitude={amp:.3e} V/m, ratio={ratio:.9f}, |ratio-2|/2={rel_err:.3e}"
    )
    _save(fig, out_dir, "S4_stability_linearity")
    plt.close(fig)

    print(f"[OK] Wrote figures to: {out_dir}")
    for fn in sorted(out_dir.glob("*.png")):
        print(f" - {fn.name}")
    print("[OK] All figures use pinned study artifacts only (no synthetic proxy data).")


if __name__ == "__main__":
    main()
