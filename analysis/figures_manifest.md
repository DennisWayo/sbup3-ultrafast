# SBUP3 Study Figure Manifest (2026-05-20 Snapshot)

This manifest defines the publication figure set, exact output filenames, panel layouts, and metric/report sources.

## 1) Canonical Study Artifacts

Use these pinned artifacts for all figure labels and quantitative annotations:

- Run manifest: `analysis/reports/study/run_release_primary_jellison_20260520_081916.json`
- Primary calibration (Jellison): `analysis/reports/study/calibration_release_primary_jellison.json`
- Primary validation (Jellison): `analysis/reports/study/validation_release_primary_jellison_20260520_081916_104478.json`
- Secondary validation (Aspnes): `analysis/reports/study/validation_secondary_aspnes_release_20260520_082112_511494.json`
- Linearity validation: `analysis/reports/study/validation_linearity_primary_jellison_20260520_081928_899730.json`
- Convergence validation: `analysis/reports/study/validation_convergence_primary_jellison_20260520_081928_899769.json`
- Primary env recommendation: `analysis/reports/study/polarization_release_primary_jellison.env`
- Release freeze bundle: `analysis/reports/study/release_freeze_primary_jellison_20260520_081916.json`

Core data files:

- `analysis/data/gaas_dielectric_reference.csv`
- `analysis/data/gaas_exp_jellison_1992.csv`
- `analysis/data/gaas_exp_aspnes_1986.csv`
- `dft/transitions_E_eV.npy`
- `dft/transitions_f.npy`
- `dft/transitions_meta.json`
- `sbe/probe_alpha_m_inv.npy`, `sbe/probe_n_real.npy`, `sbe/probe_omega_rad_s.npy`
- `uppe/outputs/E_xt_out.npy`, `uppe/outputs/x_m.npy`, `uppe/outputs/time_s.npy`

All figure outputs go to:

- `analysis/figures/study_20260520/`

## 2) Main Figures

### Figure 1 — Workflow + Provenance
- Output: `analysis/figures/study_20260520/F1_workflow_provenance.png`
- Layout: 2 panels (A-B), horizontal.
- Panel A: DFT -> dielectric CSV -> transition extraction -> SBE (drive/probe) -> UPPE -> validation/calibration flowchart.
- Panel B: provenance table (code/data hashes, transition count, k-grid, selected env values).
- Metrics/labels sourced from:
  - `run_primary_jellison...json`:
    - `dft.outputs.dielectric_csv.sha256`
    - `dft.outputs.transitions_meta.sha256`
    - `dft.transition_min_count`, `dft.transition_max_count`
    - `effective_params.sbe.density_m3`, `effective_params.sbe.osc_norm_factor`
  - `validation_primary_jellison...json`:
    - `metrics.transition_summary.n_transitions`

### Figure 2 — Optical Reference Context
- Output: `analysis/figures/study_20260520/F2_reference_optics_overlay.png`
- Layout: 2 panels (A-B), vertical.
- Panel A: `alpha` vs energy: dielectric reference vs Jellison vs Aspnes.
- Panel B: `n` vs energy: dielectric reference vs Jellison vs Aspnes.
- Mark fit window: 1.5–3.0 eV.
- Data: the three CSV files above.

### Figure 3 — Reduced Transition Model
- Output: `analysis/figures/study_20260520/F3_transition_model.png`
- Layout: 3 panels (A-C), horizontal.
- Panel A: dielectric strength (`eps2_lf`) with selected transition energies (stems).
- Panel B: transition oscillator weights (`f_i`) vs transition energy.
- Panel C: cumulative oscillator weight vs transition index.
- Data:
  - `analysis/data/gaas_dielectric_reference.csv` (`energy_eV`, `eps2_lf`)
  - `dft/transitions_E_eV.npy`, `dft/transitions_f.npy`
  - `dft/transitions_meta.json` (`n_transitions`, extraction params)

### Figure 4 — Calibration Landscape (Physical Fit)
- Output: `analysis/figures/study_20260520/F4_calibration_landscape.png`
- Layout: 2 panels (A-B), horizontal.
- Panel A: objective heatmap over (`density_m3`, `osc_norm_factor`) with best point.
- Panel B: uncertainty boxes for 68% and 95% surrogate bounds.
- Source JSON:
  - `calibration_primary_jellison...json`:
    - `fit.search.*`
    - `fit.uncertainty.bounds68.*`
    - `fit.uncertainty.bounds95.*`
    - `recommended_density_m3`, `recommended_osc_norm_factor`, `best_ratio`

### Figure 5 — Post-Calibration Primary Fit (Jellison)
- Output: `analysis/figures/study_20260520/F5_postcal_fit_jellison.png`
- Layout: 2 panels (A-B), vertical.
- Panel A: `|alpha_sim|` vs `alpha_ref` (Jellison), 1.5–3.0 eV.
- Panel B: `n_sim` vs `n_ref` (Jellison), same window.
- Annotate:
  - `alpha_abs.rmse_norm`, `alpha_abs.rmse_shape`
  - `n.rmse_norm`, `n.rmse_shape`
- Source JSON:
  - `validation_primary_jellison...json`:
    - `baseline.metrics.alpha_abs.*`
    - `baseline.metrics.n.*`
    - `baseline.recommended_polarization_scale`

### Figure 6 — Propagation Result (UPPE)
- Output: `analysis/figures/study_20260520/F6_uppe_field_evolution.png`
- Layout: 2 panels (A-B), horizontal.
- Panel A: `|E(x,t)|` map at UPPE output.
- Panel B: transverse energy proxy vs time.
- Data:
  - `uppe/outputs/E_xt_out.npy`
  - `uppe/outputs/x_m.npy`
  - `uppe/outputs/time_s.npy`
- Optional annotation from primary validation:
  - `metrics.energy_proxy.drift_rel`
  - `metrics.energy_proxy.std_rel`

## 3) Supplemental Figures

### Figure S1 — Secondary Experimental Overlay (Aspnes)
- Output: `analysis/figures/study_20260520/S1_postcal_fit_aspnes.png`
- Layout: same as Figure 5.
- Source JSON:
  - `validation_secondary_aspnes...json`:
    - `baseline.metrics.alpha_abs.*`
    - `baseline.metrics.n.*`

### Figure S2 — Probe vs Drive Extraction Consistency
- Output: `analysis/figures/study_20260520/S2_probe_vs_drive_response.png`
- Layout: 3 panels (A-C), horizontal.
- Panel A: `chi_real` probe vs drive.
- Panel B: `chi_imag` probe vs drive.
- Panel C: support masks and valid energy range.
- Data:
  - `sbe/chi_eff_real.npy`, `sbe/chi_eff_imag.npy`, `sbe/chi_support_mask.npy`
  - `sbe/probe_chi_eff_real.npy`, `sbe/probe_chi_eff_imag.npy`, `sbe/probe_chi_support_mask.npy`

### Figure S3 — Support-Mask / Window Diagnostics
- Output: `analysis/figures/study_20260520/S3_support_window_diagnostics.png`
- Layout: 2 panels (A-B), vertical.
- Panel A: drive/probe spectral support coverage.
- Panel B: trusted energy interval boundaries.
- Source JSON:
  - `validation_primary_jellison...json`:
    - `metrics.response_support.*`
    - `metrics.probe_response.*`

### Figure S4 — Numerical Stability + Linearity
- Output: `analysis/figures/study_20260520/S4_stability_linearity.png`
- Layout: 2 panels (A-B), horizontal.
- Panel A: convergence sweep summary with explicit per-point values (`alpha_abs rmse_shape`, `n rmse_shape`, support coverage) across `dt_as,dz` combinations.
- Panel B: linearity bar (`ratio_Pmax` vs ideal 2.0) with explicit amplitude and relative error text.
- Source JSON:
  - `validation_convergence_primary_jellison...json`:
    - `sweep[].params.dt_as`
    - `sweep[].params.dz`
    - `sweep[].baseline.metrics.alpha_abs.rmse_shape`
    - `sweep[].baseline.metrics.n.rmse_shape`
    - `sweep[].metrics.probe_response.coverage_fraction`
  - `validation_linearity_primary_jellison...json`:
    - `linear_check.amplitude`
    - `linear_check.ratio_Pmax`

## 4) Figure Annotation Checklist

For each figure, include:

- Run ID (`run_release_primary_jellison_20260520_081916`)
- Calibration ID (`calibration_release_primary_jellison`)
- Validation ID(s) used for annotations
- Fit window (1.5–3.0 eV)
- Transition count (`n_transitions = 16`)

## 5) Minimal Repro Command Block

Use this before generating figures to ensure state matches the manifest:

```bash
export SBUP3_DENSITY_M3=1.7585466510231837e+28
export SBUP3_OSC_NORM_FACTOR=0.9592641193252643
export SBUP3_POLARIZATION_SCALE=1
export SBUP3_IGNORE_FEEDBACK=1
export SBUP3_CHI_EPS=1e-12
export SBUP3_RESPONSE_POWER_CUTOFF=1.0
export SBUP3_CHI_MAX_EV=3.0
```
