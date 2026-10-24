# GaAs local-response propagation checks (19 October 2026)

This record tests two related, reduced-model calculations using the archived
16-transition GaAs SBE calibration. The inputs and JSON reports in this folder
are content-hashed in `manifest.json`. The field arrays from the four short
nonlinear runs are not committed; their SHA-256 digests are in the reports,
and the commands below regenerate them. The code is in
`sbe/local_response.py`, `uppe/dispersive_residual.py`, and `analysis/`.

## What was checked

1. `analysis.gaas_linear_material_window` propagates the field with the
   analytic linear susceptibility of the implemented SBE equations. The
   material polarization changes with the propagated field; it is not the
   historical frozen polarization. At 100 micrometres of propagation and
   fixed 39.0625 nm transverse spacing, the central 10 micrometre field
   changes by relative RMS 0.000125 between 320 and 640 micrometre boxes.
   The outer 20% of the 640 micrometre box contains 4.98e-8 of the field
   energy. This is a **linear constitutive** calculation with effective
   transition parameters, not nonlinear SBE feedback.
2. `analysis.gaas_local_sbe_uppe` recomputes a local Bloch polarization from
   the field at each propagation step. The exact linear response is included
   in the dispersive background; only the nonlinear residual is an explicit
   source. Predictor-corrector stepping updates that residual. For a
   2 micrometre propagation, 2 micrometre beam waist, and 10 micrometre box,
   the comparison changes are: 6.29e-5 under 0.25 to 0.125 micrometre
   propagation-step refinement; 2.28e-4 under 0.05 to 0.025 fs time-step
   refinement; and 0.00141 in the central region when the box doubles from
   10 to 20 micrometres. Peak excited population is about 4.4e-7. The
   weak-drive field agrees with the exact linear-background result at
   relative RMS 3.04e-4 in the baseline case.

The finite-beam nonlinear run covers only 2 micrometres. It does **not**
validate nonlinear 100 micrometre transmission. The upstream 0.05 eV
dielectric response remains under k-grid convergence study, and the SBE
transition weights and dipoles remain effective, calibrated quantities. No
first-principles or experimental accuracy claim follows from these checks.

## Reproduce

Run from the repository root with NumPy installed. The local solver needs
enough memory for the chosen transverse and temporal grids.

```sh
python -m analysis.gaas_linear_material_window \
  --sbe-dir archive/2026-10-19-local-coupling/inputs \
  --out-json OUTPUT_LINEAR.json --widths-um 10 20 40 80 160 320 640 \
  --dx-nm 39.0625 --z-um 100 --stride 5 --tail-fs 150

python -m analysis.gaas_local_sbe_uppe \
  --sbe-dir archive/2026-10-19-local-coupling/inputs \
  --out-prefix OUTPUT_BASE --width-um 10 --dx-nm 156.25 \
  --z-um 2 --dz-um 0.25 --stride 10

python -m analysis.gaas_local_sbe_uppe \
  --sbe-dir archive/2026-10-19-local-coupling/inputs \
  --out-prefix OUTPUT_HALF_DZ --width-um 10 --dx-nm 156.25 \
  --z-um 2 --dz-um 0.125 --stride 10

python -m analysis.gaas_local_sbe_uppe \
  --sbe-dir archive/2026-10-19-local-coupling/inputs \
  --out-prefix OUTPUT_HALF_DT --width-um 10 --dx-nm 156.25 \
  --z-um 2 --dz-um 0.25 --stride 5

python -m analysis.gaas_local_sbe_uppe \
  --sbe-dir archive/2026-10-19-local-coupling/inputs \
  --out-prefix OUTPUT_DOUBLE_WIDTH --width-um 20 --dx-nm 156.25 \
  --z-um 2 --dz-um 0.25 --stride 10

python -m analysis.gaas_local_convergence_report \
  --base OUTPUT_BASE --half-dz OUTPUT_HALF_DZ \
  --half-dt OUTPUT_HALF_DT --double-width OUTPUT_DOUBLE_WIDTH \
  --out-json OUTPUT_CONVERGENCE.json
```

`reports/` contains the original run metadata, convergence comparison, and
linear-window sweep. The recorded absolute paths point to the machines and
temporary run files used at the time; the `inputs/` directory here contains
matching SBE arrays and model metadata.
