import csv
import os
from pathlib import Path

import numpy as np

c0 = 2.99792458e8  # m/s
EPS0 = 8.8541878128e-12  # F/m
QE = 1.602176634e-19  # C
ME = 9.1093837015e-31  # kg


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    return float(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _omega_to_lambda_um(omega: np.ndarray) -> np.ndarray:
    abs_omega = np.maximum(np.abs(omega), 1e-30)
    return (2.0 * np.pi * c0 / abs_omega) * 1e6


def _internal_gaas_table() -> tuple[np.ndarray, np.ndarray]:
    """
    Coarse room-temperature GaAs index trend in the NIR/MIR.
    Values are intentionally smooth and conservative for propagation stability.
    """
    wavelength_um = np.asarray(
        [0.90, 1.00, 1.30, 1.55, 2.00, 3.00, 4.00, 5.00, 8.00, 10.6, 14.0],
        dtype=float,
    )
    n_values = np.asarray(
        [3.63, 3.57, 3.49, 3.43, 3.39, 3.34, 3.32, 3.31, 3.29, 3.285, 3.27],
        dtype=float,
    )
    return wavelength_um, n_values


def _load_custom_table_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    wavelength: list[float] = []
    nvals: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        fields = {k.lower().strip(): k for k in reader.fieldnames}

        key_l = fields.get("wavelength_um") or fields.get("lambda_um")
        key_e = fields.get("energy_ev")
        key_n = fields.get("n")
        if key_n is None:
            return None

        for row in reader:
            try:
                n = float(row[key_n])
                if key_l is not None:
                    lam = float(row[key_l])
                elif key_e is not None:
                    e_ev = float(row[key_e])
                    lam = 1.239841984 / max(e_ev, 1e-30)
                else:
                    continue
                if np.isfinite(lam) and np.isfinite(n) and lam > 0.0 and n > 0.0:
                    wavelength.append(lam)
                    nvals.append(n)
            except Exception:
                continue

    if not wavelength:
        return None
    wl = np.asarray(wavelength, dtype=float)
    nn = np.asarray(nvals, dtype=float)
    order = np.argsort(wl)
    return wl[order], nn[order]


def _interp_n_from_table(omega: np.ndarray, wl_um: np.ndarray, nn: np.ndarray) -> np.ndarray:
    lam_um = _omega_to_lambda_um(omega)
    lam_safe = np.clip(lam_um, wl_um[0], wl_um[-1])
    # Interpolate in log-lambda space for smoother broadband behavior.
    return np.interp(np.log(lam_safe), np.log(wl_um), nn)


def _apply_drude_real_correction(omega: np.ndarray, n_host: np.ndarray) -> np.ndarray:
    if not _env_bool("SBUP3_DRUDE_ENABLE", False):
        return n_host

    carrier_n = _env_float("SBUP3_DRUDE_N_M3", 0.0)
    tau_s = _env_float("SBUP3_DRUDE_TAU_S", 100e-15)
    mstar = _env_float("SBUP3_DRUDE_MSTAR", 0.067)

    if carrier_n <= 0.0:
        return n_host

    omega_abs = np.maximum(np.abs(omega), 1e9)  # avoid singularity at DC
    eps_host = np.maximum(n_host * n_host, 1e-12)
    wp2 = carrier_n * QE * QE / (EPS0 * max(mstar * ME, 1e-30))
    # Real part of epsilon for Drude model.
    eps_real = eps_host - wp2 / (omega_abs * omega_abs + (1.0 / max(tau_s, 1e-30)) ** 2)
    return np.sqrt(np.maximum(eps_real, 1e-6))


def refractive_index_gaas(omega: np.ndarray) -> np.ndarray:
    """
    Configurable GaAs refractive index model.

    Modes:
      - constant: n = SBUP3_GAAS_N0 (default 3.6)
      - table:    interpolated wavelength-dependent n(lambda) (default)
    """
    model = _env_str("SBUP3_DISPERSION_MODEL", "table").strip().lower()

    if model == "constant":
        n0 = _env_float("SBUP3_GAAS_N0", 3.6)
        n = n0 * np.ones_like(omega, dtype=float)
        return _apply_drude_real_correction(omega, n)

    wl_um, nn = _internal_gaas_table()
    csv_path = _env_str("SBUP3_DISPERSION_TABLE_CSV", "").strip()
    if csv_path:
        custom = _load_custom_table_from_csv(Path(csv_path))
        if custom is not None:
            wl_um, nn = custom

    n = _interp_n_from_table(omega, wl_um, nn)
    return _apply_drude_real_correction(omega, n)


def propagation_constant(omega: np.ndarray) -> np.ndarray:
    n = refractive_index_gaas(omega)
    return n * omega / c0
