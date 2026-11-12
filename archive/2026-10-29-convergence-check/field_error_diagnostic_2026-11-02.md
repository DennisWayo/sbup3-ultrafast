# Saved-field error diagnostic (2 November 2026)

The already completed 100 µm local material-field calculations were compared
without another propagation run. `analysis/gaas_field_refinement_diagnostic.py`
checks the three NPZ SHA-256 digests recorded in
`coupled_100um_wide_status.json` and
`coupled_100um_quarter_step_2026-11-01.md`, and verifies finite arrays and
matching spatial and temporal coordinates. The comparison uses the central
±5 µm field on the 160 µm transverse domain at temporal stride 10 (0.05 fs).

| Propagation step pair (µm) | Field relative RMS | Spectral-magnitude relative RMS | Best integer time lag within ±2 fs | Central field-energy ratio |
|---|---:|---:|---:|---:|
| 0.25 → 0.125 | 0.105289 | 0.105131 | 0 samples | 1.02853 |
| 0.125 → 0.0625 | 0.094055 | 0.094036 | 0 samples | 1.01355 |

The spectral-magnitude differences are almost as large as the waveform
differences, and an integer time shift does not reduce either field error.
The discrepancy therefore cannot be explained as a simple sampled pulse
delay. This does not isolate the solver defect: accumulated amplitude or
spectral-shape error, including its interaction with the local polarization
update, still needs diagnosis. The 100 µm result remains unconverged. No
claim of quantitative GaAs transmission or dielectric convergence follows
from this check.

The full field arrays remain on the Mac Studio at
`/Users/dela/sbup3-coupled-100um-wide-20261030/` and
`/Users/dela/sbup3-coupled-100um-quarter-20261101/`.
