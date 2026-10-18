#!/usr/bin/env python3
"""
Compute bulk GaAs dielectric/optical response from converged periodic DFT.

Outputs
-------
- CSV with energy-resolved optical constants:
  energy_eV, eps1_nlf, eps2_nlf, eps1_lf, eps2_lf, n, k, alpha_m_inv
- JSON metadata with computational settings and provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from gpaw import GPAW
from gpaw.response.df import DielectricFunction


HBAR = 1.054571817e-34  # J*s
E0 = 1.602176634e-19    # C
C0 = 299792458.0        # m/s

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

DEFAULT_GS_GPW = HERE / "gaas_pw.gpw"
DEFAULT_GS_META = HERE / "gaas_ground_state_meta.json"
DEFAULT_RESPONSE_GPW = HERE / "gaas_response.gpw"
DEFAULT_META = HERE / "gaas_dielectric_meta.json"
DEFAULT_OUT_CSV = BASE / "analysis" / "data" / "gaas_dielectric_reference.csv"


def _parse_kpts(raw: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected k-points as 'kx,ky,kz', got: {raw}")
    kpts = (int(parts[0]), int(parts[1]), int(parts[2]))
    if min(kpts) < 1:
        raise ValueError(f"k-points must be >= 1, got: {kpts}")
    return kpts


def _is_gamma_only(kpts: tuple[int, int, int]) -> bool:
    return kpts == (1, 1, 1)


def _optical_constants_from_eps(
    energy_eV: np.ndarray,
    eps_lf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_complex = np.sqrt(eps_lf.astype(complex))
    n = np.real(n_complex)
    k = np.imag(n_complex)
    omega = np.maximum(energy_eV, 0.0) * E0 / HBAR
    alpha_m_inv = 2.0 * omega * k / C0
    return n, k, alpha_m_inv


def _write_csv(
    path: Path,
    energy_eV: np.ndarray,
    eps_nlf: np.ndarray,
    eps_lf: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    alpha_m_inv: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "energy_eV",
                "eps1_nlf",
                "eps2_nlf",
                "eps1_lf",
                "eps2_lf",
                "n",
                "k",
                "alpha_m_inv",
            ]
        )
        for row in zip(
            energy_eV.tolist(),
            np.real(eps_nlf).tolist(),
            np.imag(eps_nlf).tolist(),
            np.real(eps_lf).tolist(),
            np.imag(eps_lf).tolist(),
            n.tolist(),
            k.tolist(),
            alpha_m_inv.tolist(),
        ):
            writer.writerow(row)


def _validate_ground_state_meta(
    path: Path,
    *,
    allow_missing: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(
            f"Ground-state provenance metadata missing: {path}. "
            "Run dft/gaas_bulk_gpaw.py first."
        )
    try:
        meta = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Could not parse ground-state metadata: {path}") from exc

    raw_kpts = meta.get("kpts")
    if not isinstance(raw_kpts, list) or len(raw_kpts) != 3:
        raise RuntimeError(
            f"Ground-state metadata is missing valid kpts triplet: {path}"
        )
    gs_kpts = (int(raw_kpts[0]), int(raw_kpts[1]), int(raw_kpts[2]))
    if _is_gamma_only(gs_kpts):
        raise RuntimeError(
            "Ground-state metadata reports gamma-only kpts=(1,1,1). "
            "Refusing publication workflow."
        )
    if not bool(meta.get("publication_grade_kmesh", True)):
        raise RuntimeError(
            "Ground-state metadata does not certify publication-grade k-point sampling."
        )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="GaAs dielectric response from periodic DFT")
    parser.add_argument("--gs-gpw", default=str(DEFAULT_GS_GPW), help="Ground-state .gpw file")
    parser.add_argument(
        "--gs-meta",
        default=str(DEFAULT_GS_META),
        help="Ground-state provenance metadata JSON from gaas_bulk_gpaw.py",
    )
    parser.add_argument(
        "--response-gpw",
        default=str(DEFAULT_RESPONSE_GPW),
        help="Output .gpw for fixed-density dense-k response stage",
    )
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV), help="Output optical constants CSV")
    parser.add_argument("--meta-json", default=str(DEFAULT_META), help="Output metadata JSON")
    parser.add_argument("--kpts", default="10,10,10", help="Dense response k-grid kx,ky,kz")
    parser.add_argument("--nbands", type=int, default=120, help="Bands for full diagonalization")
    parser.add_argument("--eta", type=float, default=0.05, help="Broadening [eV]")
    parser.add_argument("--ecut", type=float, default=150.0, help="Response cutoff [eV]")
    parser.add_argument("--domega0", type=float, default=0.02, help="Nonlinear frequency grid seed [eV]")
    parser.add_argument("--omega2", type=float, default=5.0, help="Nonlinear frequency scale [eV]")
    parser.add_argument("--omegamax", type=float, default=8.0, help="Max frequency [eV]")
    parser.add_argument(
        "--integrationmode",
        choices=["point integration", "tetrahedron integration"],
        default="point integration",
        help="k-space integration mode for dielectric response",
    )
    parser.add_argument(
        "--allow-gamma-only",
        action="store_true",
        help="Allow 1,1,1 response k-grid (debug only; not publication-grade).",
    )
    parser.add_argument(
        "--allow-missing-gs-meta",
        action="store_true",
        help="Allow missing ground-state metadata (debug only).",
    )
    args = parser.parse_args()

    gs_gpw = Path(args.gs_gpw)
    gs_meta = Path(args.gs_meta)
    response_gpw = Path(args.response_gpw)
    out_csv = Path(args.out_csv)
    meta_json = Path(args.meta_json)
    kpts = _parse_kpts(args.kpts)

    if _is_gamma_only(kpts) and not args.allow_gamma_only:
        raise RuntimeError(
            "Refusing dielectric-response stage with kpts=(1,1,1). "
            "Use a converged periodic grid."
        )
    if not gs_gpw.exists():
        raise FileNotFoundError(f"Ground-state .gpw not found: {gs_gpw}")
    gs_meta_payload = _validate_ground_state_meta(
        gs_meta,
        allow_missing=bool(args.allow_missing_gs_meta),
    )

    print("SBUP3 dielectric response: GaAs")
    print("-----------------------------")
    print(f"[LOAD] {gs_gpw}")

    calc = GPAW(str(gs_gpw))
    calc_resp = calc.fixed_density(kpts=kpts, nbands=int(args.nbands))
    # Ensure requested bands are initialized for response calculations.
    calc_resp.get_potential_energy()
    try:
        calc_resp.diagonalize_full_hamiltonian(nbands=int(args.nbands))
    except Exception as exc:
        print(
            "[WARN] diagonalize_full_hamiltonian unavailable for this GPAW mode; "
            f"continuing with fixed-density eigensolver state ({exc})."
        )
    if not hasattr(calc_resp.wfs, "pd"):
        raise RuntimeError(
            "DielectricFunction response requires plane-wave wavefunctions in this GPAW build. "
            "Current fixed-density result is not PW-compatible (missing wfs.pd). "
            "Run the ground-state in PW mode before dielectric extraction."
        )
    response_gpw.parent.mkdir(parents=True, exist_ok=True)
    calc_resp.write(str(response_gpw), mode="all")
    print(f"[SAVE] response wavefunctions -> {response_gpw}")

    df = DielectricFunction(
        calc=str(response_gpw),
        eta=float(args.eta),
        ecut=float(args.ecut),
        intraband=False,
        frequencies={
            "type": "nonlinear",
            "domega0": float(args.domega0),
            "omega2": float(args.omega2),
            "omegamax": float(args.omegamax),
        },
        integrationmode=args.integrationmode,
    )

    tmp_csv = response_gpw.with_suffix(".dielectric_raw.csv")
    df.get_dielectric_function(filename=str(tmp_csv))
    raw = np.loadtxt(tmp_csv, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] < 5:
        raise RuntimeError(f"Unexpected dielectric CSV format in {tmp_csv}")
    energy_eV = np.asarray(raw[:, 0], dtype=float)
    eps_nlf = np.asarray(raw[:, 1] + 1j * raw[:, 2], dtype=complex)
    eps_lf = np.asarray(raw[:, 3] + 1j * raw[:, 4], dtype=complex)

    n, k, alpha_m_inv = _optical_constants_from_eps(energy_eV, eps_lf)
    _write_csv(
        out_csv,
        energy_eV=energy_eV,
        eps_nlf=eps_nlf,
        eps_lf=eps_lf,
        n=n,
        k=k,
        alpha_m_inv=alpha_m_inv,
    )

    meta: dict[str, Any] = {
        "gs_gpw": str(gs_gpw),
        "gs_meta_json": str(gs_meta),
        "gs_meta_loaded": gs_meta_payload is not None,
        "gs_kpts": (gs_meta_payload or {}).get("kpts"),
        "response_gpw": str(response_gpw),
        "out_csv": str(out_csv),
        "kpts_response": list(kpts),
        "nbands": int(args.nbands),
        "eta_eV": float(args.eta),
        "ecut_eV": float(args.ecut),
        "frequencies": {
            "type": "nonlinear",
            "domega0_eV": float(args.domega0),
            "omega2_eV": float(args.omega2),
            "omegamax_eV": float(args.omegamax),
        },
        "intraband": False,
        "integrationmode": args.integrationmode,
        "publication_grade_kmesh": not _is_gamma_only(kpts),
        "n_rows": int(len(energy_eV)),
    }
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta_json.write_text(json.dumps(meta, indent=2))
    try:
        tmp_csv.unlink()
    except Exception:
        pass

    print(f"[OK] Wrote optical constants: {out_csv}")
    print(f"[OK] Wrote metadata: {meta_json}")


if __name__ == "__main__":
    main()
