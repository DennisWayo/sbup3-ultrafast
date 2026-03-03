#!/usr/bin/env python3
"""
SBUP³ — Self-consistent SBE ↔ UPPE coupling loop
================================================

Loop:
  1) Run SBE (RK4) -> produces E_t.npy, polarization.npy in sbup3/sbe/
  2) Run UPPE 2D   -> produces E_xt_out.npy in sbup3/uppe/outputs/
  3) Extract on-axis field E_out(t) at x≈0
  4) Feed back as next iteration drive via sbup3/sbe/E_t_feedback.npy

Notes:
  - This script runs SBE from the sbe/ directory (so outputs land in sbe/)
    and runs UPPE as a module from the repo root.

  - Minimal hook needed in sbe_rk4.py:
      if sbe/E_t_feedback.npy exists, load it instead of generating E_t internally.

Author: Dennis Wayo
"""

from __future__ import annotations

import os
import sys
import subprocess
import numpy as np
from pathlib import Path


# -----------------------------
# Config
# -----------------------------
N_ITERS = 5
RTOL = 1e-3               # convergence tolerance (relative L2 change)
USE_ABS_TOL = 1e-12       # prevents division by tiny norms
VERBOSE = True


# -----------------------------
# Paths (assumes this file is sbup3/coupling/sbup3_loop.py)
# -----------------------------
BASE = Path(__file__).resolve().parents[1]          # .../sbup3
SBE_DIR = BASE / "sbe"
UPPE_DIR = BASE / "uppe"
UPPE_OUT = UPPE_DIR / "outputs"

SBE_SCRIPT = SBE_DIR / "sbe_rk4.py"
UPPE_SCRIPT = UPPE_DIR / "uppe_2d.py"

# SBE outputs
SBE_TIME = SBE_DIR / "time_s.npy"
SBE_E_T = SBE_DIR / "E_t.npy"
SBE_P_T = SBE_DIR / "polarization.npy"

# Feedback file (read by sbe_rk4.py if present)
SBE_E_FEEDBACK = SBE_DIR / "E_t_feedback.npy"

# UPPE outputs
UPPE_E_XT = UPPE_OUT / "E_xt_out.npy"
UPPE_X = UPPE_OUT / "x_m.npy"
UPPE_TIME = UPPE_OUT / "time_s.npy"


def _run(cmd: list[str], cwd: Path) -> None:
    """Run command, fail loudly."""
    if VERBOSE:
        print("\n[CMD]", " ".join(cmd))
        print("[CWD]", str(cwd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 change ||a-b||/||a|| with safe floor."""
    num = np.linalg.norm(a - b)
    den = max(np.linalg.norm(a), USE_ABS_TOL)
    return float(num / den)


def _extract_on_axis_field() -> tuple[np.ndarray, np.ndarray]:
    """
    Load UPPE outputs and return (time_s, E_onaxis_t) where on-axis means x≈0.
    """
    if not UPPE_E_XT.exists():
        raise FileNotFoundError(f"Missing UPPE output: {UPPE_E_XT}")
    if not UPPE_X.exists():
        raise FileNotFoundError(f"Missing UPPE grid: {UPPE_X}")
    if not UPPE_TIME.exists():
        raise FileNotFoundError(f"Missing UPPE time: {UPPE_TIME}")

    E_xt = np.load(UPPE_E_XT)      # (Nx, Nt)
    x_m = np.load(UPPE_X)          # (Nx,)
    time_s = np.load(UPPE_TIME)    # (Nt,)

    ix0 = int(np.argmin(np.abs(x_m - 0.0)))
    E_onaxis = E_xt[ix0, :].astype(np.float64)

    return time_s, E_onaxis


def main() -> None:
    print("SBUP³ coupling loop: SBE ↔ UPPE")
    print("--------------------------------")

    # Ensure expected scripts exist
    for p in [SBE_SCRIPT, UPPE_SCRIPT]:
        if not p.exists():
            raise FileNotFoundError(f"Missing script: {p}")

    prev_E_drive = None

    for it in range(1, N_ITERS + 1):
        print(f"\n=== Iteration {it}/{N_ITERS} ===")

        # 1) Run SBE (will generate E_t.npy and polarization.npy)
        _run([sys.executable, str(SBE_SCRIPT)], cwd=SBE_DIR)

        if not SBE_E_T.exists() or not SBE_P_T.exists() or not SBE_TIME.exists():
            raise RuntimeError("SBE did not produce required outputs: time_s.npy, E_t.npy, polarization.npy")

        # 2) Run UPPE (consumes SBE outputs internally as you coded it)
        _run([sys.executable, "-m", "uppe.uppe_2d"], cwd=BASE)

        # 3) Extract feedback field from UPPE output
        time_uppe, E_onaxis = _extract_on_axis_field()

        # 4) Save feedback for next SBE pass
        np.save(SBE_E_FEEDBACK, E_onaxis)
        if VERBOSE:
            print(f"[SAVE] Feedback field -> {SBE_E_FEEDBACK}")

        # Convergence check: compare current drive to previous drive
        # We compare the *feedback* drive each iteration (on-axis propagated field).
        if prev_E_drive is not None:
            # Make sure shapes match; if not, truncate to min length
            n = min(len(prev_E_drive), len(E_onaxis))
            rel = _relative_l2(prev_E_drive[:n], E_onaxis[:n])
            print(f"[CHK ] Relative L2 change = {rel:.3e}")

            if rel < RTOL:
                print(f"[DONE] Converged (rel < {RTOL}). Stopping.")
                break

        prev_E_drive = E_onaxis.copy()

    print("\nSBUP³ coupling loop complete.")
    print("Key files:")
    print(f"  Feedback drive: {SBE_E_FEEDBACK}")
    print(f"  UPPE output:    {UPPE_E_XT}")


if __name__ == "__main__":
    main()
