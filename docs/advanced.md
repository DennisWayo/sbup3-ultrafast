## Advanced SBUP3 Usage

This document keeps advanced options out of the top-level README.

## Pipeline Flags

Main entry point:

```bash
python analysis/run_pipeline.py [flags]
```

Important flags:
- `--mode sequential|coupling|full`
- `--plots`
- `--validate`
- `--ref-csv <path>`
- `--allow-feedback-in-sequential`

SBE override flags:
- `--t-fs --dt-as --gamma --e-field-ampl --omega-l --mixing`
- `--ignore-feedback --e-max --density-m3 --polarization-scale`
- `--osc-norm-factor --chi-eps --response-power-cutoff --chi-max-ev`

UPPE override flags:
- `--x-window --w0 --nx --dz --z-steps`

Coupling override flags:
- `--iters --rtol --feedback-beta --feedback-norm --feedback-norm-eps`

Overlay/plot flags:
- `--overlay-e-min --overlay-e-max --overlay-window`
- `--overlay-scale fit|peak|none`
- `--overlay-abs --osc-broadening`

## Calibration

Standalone calibration:

```bash
python analysis/calibrate_polarization_scale.py \
  --ref-csv analysis/data/gaas_dielectric_reference.csv \
  --fit-channels alpha,n \
  --write-env analysis/reports/polarization_recommendation.env
```

Pipeline-triggered calibration:

```bash
python analysis/run_pipeline.py --mode sequential --validate \
  --ref-csv analysis/data/gaas_dielectric_reference.csv \
  --calibrate-scale --calibration-fit-channels alpha,n \
  --calibration-write-env analysis/reports/polarization_recommendation.env
```

## Optional DFT Regeneration

Through pipeline (recommended for publication-grade dielectric path):

```bash
python analysis/run_pipeline.py --run-dft --dft-env <gpaw-env> --mode sequential \
  --dft-gs-mode pw --dft-pw-ecut-ev 200 \
  --dft-kpts 8,8,8 --dft-response-kpts 10,10,10 \
  --dft-force-ground-state
```

This writes:
- `analysis/data/gaas_dielectric_reference.csv`
- `dft/transitions_E_eV.npy`
- `dft/transitions_f.npy`
- `dft/transitions_meta.json`

## Legacy LR-TDDFT Compatibility Path

```bash
# 1) Build gamma-only FD ground state
python dft/gaas_bulk_gpaw.py --gs-mode fd --kpts 1,1,1 --force

# 2) Extract transitions with legacy LR-TDDFT stage
python dft/gaas_lrtddft.py \
  --gpw dft/gaas_fd.gpw \
  --export-csv analysis/data/gaas_tddft_reference.csv
```

## Useful Environment Variables

Multi-transition controls:
- `SBUP3_MULTI_ENABLE`
- `SBUP3_MULTI_OSC_MIN`
- `SBUP3_MULTI_MAX_EV`
- `SBUP3_MULTI_MAX_TRANSITIONS`
- `SBUP3_REQUIRE_GROUND_TRUTH_TRANSITIONS`
- `SBUP3_MULTI_TARGET_GAP_EV`

Physical scaling controls:
- `SBUP3_DENSITY_M3`
- `SBUP3_OSC_NORM_FACTOR`
- `SBUP3_POLARIZATION_SCALE`
- `SBUP3_USE_PHYSICAL_DENSITY`

Probe-response controls:
- `SBUP3_PROBE_ENABLE`
- `SBUP3_PROBE_E_FIELD_AMPL`
- `SBUP3_PROBE_TAU_FS`
- `SBUP3_PROBE_CHI_MIN_EV`
- `SBUP3_PROBE_CHI_MAX_EV`

UPPE dispersion controls:
- `SBUP3_DISPERSION_MODEL=table|constant`
- `SBUP3_GAAS_N0` (constant model)
- `SBUP3_DISPERSION_TABLE_CSV` (custom table)

## Visualization

```bash
python analysis/visualize_sbup3.py
python analysis/visualize_sbup3.py --show
```

Study figure pack:

```bash
python analysis/plot_study_figures.py
```
