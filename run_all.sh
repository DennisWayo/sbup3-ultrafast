#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${SBUP3_RUN_MODE:-sequential}"
REF_CSV="${SBUP3_REF_CSV:-analysis/data/gaas_dielectric_reference.csv}"
PYTHON_BIN="${PYTHON:-}"
SETUP=1
RUN_DFT=0
DFT_ENV="${SBUP3_DFT_ENV:-}"
EXTRA_ARGS=()

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

usage() {
  cat <<'EOF'
Usage: ./run_all.sh [options] [-- extra analysis/run_pipeline.py args]

Runs the SBUP3 workflow from a fresh clone:
  setup/check Python deps -> SBE -> UPPE -> plots -> validation report

Default run is portable and does not require GPAW/DFT:
  ./run_all.sh

Options:
  --mode sequential|coupling|full
      Pipeline mode passed to analysis/run_pipeline.py. Default: sequential.

  --run-dft
      Also regenerate DFT-derived transition inputs. Requires GPAW and usually a
      separate conda environment. Use with --dft-env when needed.

  --dft-env NAME
      Conda environment name used for the DFT stage.

  --ref-csv PATH
      Reference CSV for validation/plots. Default:
      analysis/data/gaas_dielectric_reference.csv if present.

  --python PATH
      Python executable to use. Default: .venv/bin/python if available,
      otherwise python3/python.

  --no-setup
      Do not create .venv or install minimal Python dependencies.

  -h, --help
      Show this help.

Examples:
  ./run_all.sh
  ./run_all.sh --mode coupling -- --iters 3
  ./run_all.sh --mode sequential --ref-csv analysis/data/gaas_exp_aspnes_1986.csv
  ./run_all.sh --run-dft --dft-env gpaw --mode sequential

Outputs:
  sbe/*.npy
  uppe/outputs/*
  analysis/figures/*
  analysis/reports/*
EOF
}

log() {
  printf '\n[run_all] %s\n' "$*"
}

die() {
  printf '\n[run_all] ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --run-dft)
      RUN_DFT=1
      shift
      ;;
    --dft-env)
      [[ $# -ge 2 ]] || die "--dft-env requires a value"
      DFT_ENV="$2"
      shift 2
      ;;
    --ref-csv)
      [[ $# -ge 2 ]] || die "--ref-csv requires a value"
      REF_CSV="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-setup)
      SETUP=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$MODE" in
  sequential|coupling|full) ;;
  *) die "Unknown mode: $MODE" ;;
esac

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    die "No Python executable found. Install Python 3.10+ or pass --python PATH."
  fi
fi

if [[ "$SETUP" -eq 1 && ! -x ".venv/bin/python" ]]; then
  log "Creating local virtual environment in .venv"
  "$PYTHON_BIN" -m venv .venv
  PYTHON_BIN=".venv/bin/python"
fi

if [[ "$SETUP" -eq 1 ]]; then
  log "Installing minimal Python dependencies"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install "numpy>=1.22" "matplotlib>=3.7"
fi

log "Checking Python dependencies"
"$PYTHON_BIN" - <<'PY'
import sys
import numpy
import matplotlib

print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("matplotlib:", matplotlib.__version__)
PY

PIPELINE_ARGS=(
  --mode "$MODE"
  --plots
  --validate
)

if [[ -f "$REF_CSV" ]]; then
  PIPELINE_ARGS+=(--ref-csv "$REF_CSV")
else
  log "Reference CSV not found at $REF_CSV; running without --ref-csv"
fi

if [[ "$RUN_DFT" -eq 1 ]]; then
  PIPELINE_ARGS+=(--run-dft)
  if [[ -n "$DFT_ENV" ]]; then
    PIPELINE_ARGS+=(--dft-env "$DFT_ENV")
  fi
else
  # Keep fresh clones runnable without GPAW-generated dft/*.npy artifacts.
  export SBUP3_ALLOW_TRANSITION_CSV_FALLBACK="${SBUP3_ALLOW_TRANSITION_CSV_FALLBACK:-1}"
  export SBUP3_REQUIRE_GROUND_TRUTH_TRANSITIONS="${SBUP3_REQUIRE_GROUND_TRUTH_TRANSITIONS:-0}"
fi

log "Running pipeline"
printf '[run_all] Command: %q' "$PYTHON_BIN"
printf ' %q' analysis/run_pipeline.py "${PIPELINE_ARGS[@]}" "${EXTRA_ARGS[@]}"
printf '\n'

"$PYTHON_BIN" analysis/run_pipeline.py "${PIPELINE_ARGS[@]}" "${EXTRA_ARGS[@]}"

log "Done"
log "Figures: analysis/figures/"
log "Reports: analysis/reports/"
