"""
SBUP3 ground-state DFT preparation for bulk GaAs.

This stage prepares periodic-wavefunction files for optical-response
calculations with converged k-point sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.build import bulk
from gpaw import GPAW, FermiDirac, PW

# -------------------------------------------------
# Paths
# -------------------------------------------------
HERE = Path(__file__).resolve().parent

LCAO_GPW = HERE / "gaas_lcao.gpw"
FD_GPW = HERE / "gaas_fd.gpw"
PW_GPW = HERE / "gaas_pw.gpw"
META_JSON = HERE / "gaas_ground_state_meta.json"

# Defaults for publication-grade periodic calculations.
DEFAULT_XC = "PBE"
DEFAULT_KPTS = (8, 8, 8)
DEFAULT_NBANDS = 32
DEFAULT_SMEARING = 0.01
DEFAULT_H = 0.25
DEFAULT_GS_MODE = "pw"
DEFAULT_PW_ECUT_EV = 200.0


def _validate_existing_artifacts(
    *,
    gs_mode: str,
    xc: str,
    kpts: tuple[int, int, int],
    nbands: int,
    smearing_ev: float,
    h: float,
    pw_ecut_ev: float,
    a_ang: float,
) -> None:
    artifacts = [PW_GPW] if gs_mode == "pw" else [LCAO_GPW, FD_GPW]
    existing = [p for p in artifacts if p.exists()]
    if not existing:
        return
    if not META_JSON.exists():
        raise RuntimeError(
            "Found existing DFT artifacts without provenance metadata "
            f"({META_JSON}). Refusing reuse; rerun with --force."
        )

    try:
        meta = json.loads(META_JSON.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse provenance metadata {META_JSON}; rerun with --force."
        ) from exc

    expected = {
        "gs_mode": str(gs_mode),
        "xc": str(xc),
        "kpts": [int(kpts[0]), int(kpts[1]), int(kpts[2])],
        "nbands": int(nbands),
        "smearing_ev": float(smearing_ev),
        "h_ang": float(h) if gs_mode == "fd" else None,
        "pw_ecut_ev": float(pw_ecut_ev) if gs_mode == "pw" else None,
        "lattice_a_ang": float(a_ang),
    }

    mismatches: list[str] = []
    for key, exp in expected.items():
        got = meta.get(key)
        if isinstance(exp, float):
            try:
                if abs(float(got) - exp) > 1e-12:
                    mismatches.append(f"{key}: existing={got} requested={exp}")
            except Exception:
                mismatches.append(f"{key}: existing={got} requested={exp}")
        else:
            if got != exp:
                mismatches.append(f"{key}: existing={got} requested={exp}")

    if mismatches:
        raise RuntimeError(
            "Existing DFT artifacts do not match requested settings. "
            "Refusing mixed provenance; rerun with --force.\n"
            + "\n".join(mismatches)
        )

# -------------------------------------------------
# Build bulk GaAs
# -------------------------------------------------
def build_gaas(a: float = 5.653):
    atoms = bulk("GaAs", "zincblende", a=a)
    atoms.pbc = True
    return atoms

def _parse_kpts(raw: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected k-points as 'kx,ky,kz', got: {raw}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _is_gamma_only(kpts: tuple[int, int, int]) -> bool:
    return kpts == (1, 1, 1)


def run_lcao(
    atoms,
    *,
    xc: str,
    kpts: tuple[int, int, int],
    nbands: int,
    smearing_ev: float,
    force: bool,
):
    if LCAO_GPW.exists() and not force:
        print(f"[SKIP] LCAO -> {LCAO_GPW}")
        return GPAW(str(LCAO_GPW))

    print("[RUN ] LCAO ground state")
    calc = GPAW(
        mode="lcao",
        basis="dzp",
        xc=xc,
        kpts=kpts,
        nbands=nbands,
        occupations=FermiDirac(smearing_ev),
        symmetry="off",
        txt=str(HERE / "gaas_lcao.txt"),
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(str(LCAO_GPW), mode="all")
    return calc


def run_fd(
    *,
    h: float,
    smearing_ev: float,
    force: bool,
):
    if FD_GPW.exists() and not force:
        print(f"[SKIP] FD -> {FD_GPW}")
        return GPAW(str(FD_GPW))

    print("[RUN ] FD restart")
    calc = GPAW(
        str(LCAO_GPW),
        mode="fd",
        h=h,
        occupations=FermiDirac(smearing_ev),
        txt=str(HERE / "gaas_fd.txt"),
    )
    calc.get_potential_energy()
    calc.write(str(FD_GPW), mode="all")
    return calc


def run_pw(
    atoms,
    *,
    xc: str,
    kpts: tuple[int, int, int],
    nbands: int,
    smearing_ev: float,
    pw_ecut_ev: float,
    force: bool,
):
    if PW_GPW.exists() and not force:
        print(f"[SKIP] PW -> {PW_GPW}")
        return GPAW(str(PW_GPW))

    print("[RUN ] PW ground state")
    calc = GPAW(
        mode=PW(float(pw_ecut_ev)),
        xc=xc,
        kpts={"size": kpts, "gamma": True},
        nbands=nbands,
        occupations=FermiDirac(smearing_ev),
        symmetry="off",
        txt=str(HERE / "gaas_pw.txt"),
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(str(PW_GPW), mode="all")
    return calc


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GaAs ground-state wavefunctions")
    parser.add_argument("--a-ang", type=float, default=5.6533, help="Lattice constant [Angstrom]")
    parser.add_argument("--xc", default=DEFAULT_XC, help="XC functional")
    parser.add_argument("--kpts", default="8,8,8", help="Monkhorst-Pack k-points as kx,ky,kz")
    parser.add_argument(
        "--gs-mode",
        choices=["pw", "fd"],
        default=DEFAULT_GS_MODE,
        help="Ground-state mode. Use 'pw' for dielectric response workflows.",
    )
    parser.add_argument("--nbands", type=int, default=DEFAULT_NBANDS, help="Number of bands for GS step")
    parser.add_argument("--smearing-ev", type=float, default=DEFAULT_SMEARING, help="Fermi smearing [eV]")
    parser.add_argument("--h", type=float, default=DEFAULT_H, help="FD grid spacing [Angstrom]")
    parser.add_argument(
        "--pw-ecut-ev",
        type=float,
        default=DEFAULT_PW_ECUT_EV,
        help="PW cutoff [eV] for PW ground-state mode.",
    )
    parser.add_argument(
        "--allow-gamma-only",
        action="store_true",
        help="Allow kpts=1,1,1 (debug only; not publication-grade).",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if .gpw files exist")
    args = parser.parse_args()

    kpts = _parse_kpts(args.kpts)
    if _is_gamma_only(kpts) and not args.allow_gamma_only:
        raise RuntimeError(
            "Refusing to run with kpts=(1,1,1). "
            "Use a converged periodic grid (for example 6,6,6 or 8,8,8)."
        )
    if not args.force:
        _validate_existing_artifacts(
            gs_mode=args.gs_mode,
            xc=args.xc,
            kpts=kpts,
            nbands=args.nbands,
            smearing_ev=args.smearing_ev,
            h=args.h,
            pw_ecut_ev=args.pw_ecut_ev,
            a_ang=args.a_ang,
        )

    print("SBUP3 DFT ground-state setup: GaAs")
    print("----------------------------------")
    print(
        f"mode={args.gs_mode}, XC={args.xc}, kpts={kpts}, nbands={args.nbands}, "
        f"h={args.h}, ecut={args.pw_ecut_ev}, smearing={args.smearing_ev} eV"
    )

    atoms = build_gaas(a=args.a_ang)
    if args.gs_mode == "pw":
        run_pw(
            atoms,
            xc=args.xc,
            kpts=kpts,
            nbands=args.nbands,
            smearing_ev=args.smearing_ev,
            pw_ecut_ev=args.pw_ecut_ev,
            force=args.force,
        )
    else:
        run_lcao(
            atoms,
            xc=args.xc,
            kpts=kpts,
            nbands=args.nbands,
            smearing_ev=args.smearing_ev,
            force=args.force,
        )
        run_fd(
            h=args.h,
            smearing_ev=args.smearing_ev,
            force=args.force,
        )

    meta = {
        "gs_mode": args.gs_mode,
        "xc": args.xc,
        "kpts": list(kpts),
        "nbands": int(args.nbands),
        "smearing_ev": float(args.smearing_ev),
        "h_ang": float(args.h) if args.gs_mode == "fd" else None,
        "pw_ecut_ev": float(args.pw_ecut_ev) if args.gs_mode == "pw" else None,
        "lattice_a_ang": float(args.a_ang),
        "lcao_gpw": str(LCAO_GPW) if args.gs_mode == "fd" else None,
        "fd_gpw": str(FD_GPW) if args.gs_mode == "fd" else None,
        "pw_gpw": str(PW_GPW) if args.gs_mode == "pw" else None,
        "publication_grade_kmesh": not _is_gamma_only(kpts),
    }
    META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"[OK] Wrote metadata: {META_JSON}")
    print("[OK] Ground-state preparation complete.")


if __name__ == "__main__":
    main()
