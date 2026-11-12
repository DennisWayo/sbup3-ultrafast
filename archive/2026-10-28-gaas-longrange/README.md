# GaAs local Bloch response over 100 µm (28 October 2026)

This folder records a bounded numerical-sensitivity sweep of the calibrated,
effective GaAs Bloch--field model. It does not validate measured transmission
or establish convergence of the upstream PBE dielectric spectrum.

The four JSON case reports, `longrange_status.json`, and
`longrange_sensitivity.json` preserve the parameters, input/source hashes,
field-array SHA-256 digests, and the measured differences. The full `.npz`
field arrays (about 180 MB in total) remain on the Mac Studio calculation
host and are **not** included in this repository. Accordingly, the committed
summary can be checked against its hashes, while recomputing the field
comparison requires rerunning the four cases or obtaining those arrays.

The local report for the first case was initially marked failed because its
reported propagation length was `99.99999999999999` µm rather than exactly
`100.0` µm. `analysis/gaas_longrange_miner_resume.py` checks physical scalar
metadata with a tight floating-point tolerance, while retaining strict
source, input, field-array checksum, shape, and finite-value checks. Its
SHA-256 digest is recorded in `longrange_status.json` under
`recovery_history`. The original runner remains in
`analysis/gaas_longrange_miner.py` for provenance.

For the central `-5` to `5` µm region after 100 µm propagation, the relative
RMS changes are 0.1095 when the propagation step falls from 0.50 to 0.25 µm,
0.0720 when the sampled temporal step falls from 0.050 to 0.025 fs, and
0.000916 when the transverse box grows from 160 to 320 µm. The outer fifth
of the 320 µm box contains 2.35e-6 of the output field energy. The box test
supports spatial containment at these settings; the step and time tests do
not support long-distance nonlinear convergence.

To inspect the exact four-case plan without executing the sweep, from the
repository root run:

```sh
python analysis/gaas_longrange_miner_resume.py \
  --root . --out-dir /path/to/new-output \
  --python /path/to/python-with-numpy --plan
```

Remove `--plan` to run the four cases in a **new** output directory. The
script refuses to overwrite an existing directory. `--resume` is reserved
for a previously failed sweep with unchanged source, dependencies, inputs,
and case plan. The repository's
`archive/2026-10-19-local-coupling/inputs/` contains the required small
input arrays and model metadata.
