# SBUP3 solver and optical-reference audit (17 October 2026)

This archive adds two checks to the historical GaAs optical release. It does
not replace or reinterpret the May 2026 run. The `legacy/` files are byte-for-byte
copies of the earlier release freeze, run, calibration, validation, and DFT
metadata. The older run used a field operator whose signed-frequency phase and
polarization source were incorrect. Its propagated field is not a physical
result of the corrected solver.

## New evidence

- `propagation_benchmark.json` records a lossless 1.55 μm real-wave test and a
  prescribed SI-polarization test. For a unit-amplitude input, the quarter-wave
  RMS ratio is 1.0 and the maximum phase-field error is about `1.04e-14`.
  The known-source solution has maximum error about `6.99e-15`; one and 16
  exact frozen-source steps differ by about `5.55e-16`. A nonzero transverse
  Fourier mode has maximum error about `4.80e-15`. These are operator checks,
  not a GaAs pulse simulation.
- `optical_error_budget.json` compares the archived PBE dielectric table with
  Jellison and Aspnes measurements on the archived probe-support window
  (1.5715537–2.9776807 eV). It interpolates PBE onto measured energies,
  without extrapolation or fitted scaling. RMSE is divided by the maximum
  absolute measured value in each channel. The Jellison errors are 0.360
  (`alpha`) and 0.179 (`n`) at 84 energies; the Aspnes errors are 0.452 and
  0.173 at 14 energies.
- `legacy/calibration_release_primary_jellison.json` reports direct joint-fit
  errors of 0.48173 (`alpha`) and 0.62021 (`n`) on the 84-point Jellison
  comparison. These are stage-specific discrepancies, not terms in an
  additive error decomposition.
- `provenance_audit.json` compares current local output arrays with the
  historical run report: 4 of 28 hashes match and 24 differ. All three
  checked UPPE output arrays differ. No array was overwritten to obtain this
  result.

The DFT metadata record a 4×4×4 ground-state grid, 6×6×6 response grid,
200 eV ground-state cutoff, and 150 eV response cutoff. The metadata's
`publication_grade_kmesh` flag is not evidence of convergence; no mesh or
basis sweep accompanies this archive. The current field arrays also do not
match the historical run-report hashes, so no corrected GaAs propagation
curve is archived here.

## Reproduce

From the repository root, using an environment with NumPy:

```sh
python -m unittest tests.test_uppe_real_field -v
python -m analysis.propagation_benchmark
python analysis/optical_error_budget.py
python analysis/audit_release_outputs.py
```

The first two commands use manufactured fields and do not overwrite the SBE
or UPPE output arrays. The optical comparison reads the three tracked CSVs in
`analysis/data/`. The JSON outputs give their SHA-256 checksums and the solver
checksum. The provenance audit reads the local ignored output arrays without
changing them; its match counts are specific to the 17 October checkout.
`manifest.json` lists checksums for all files in this archive, the scripts,
and the input CSVs.
