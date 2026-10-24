# GaAs propagation and dielectric sensitivity record (17–18 October 2026)

This archive supports the revised SBUP3 manuscript. `manifest.json` gives a
relative path, size, and SHA-256 digest for every other file in this directory.
The archived inputs, output, and source snapshots are small enough to keep in
Git. GPAW `.gpw` restart files are excluded; the CSVs can be regenerated with
the recorded settings and GPAW 25.7.0, but their exact numerical reproduction
may depend on GPAW, BLAS, and platform versions.

## Propagation

`inputs/sequential_run_manifest.json` records a clean run at commit `2395fd0`
using the previously archived 16 GaAs transition energies and oscillator
weights. `inputs/time_s.npy`, `E_t.npy`, and `polarization.npy` are the
10,000-sample SBE arrays used for the field calculation. The two transition
arrays match the earlier frozen transition hashes. `code_used/` contains the
exact SBE and corrected UPPE scripts from that run.

Run `python -m analysis.gaas_uniform_propagation --sbe-dir
archive/2026-10-17-gaas-validation/inputs --out-dir OUTPUT_DIR` from the repo
root. This calculation applies a frozen, full SBE polarization to a uniform
transverse field over 100 micrometres in a vacuum-background real-field UPPE.
The output is `gaas_uniform_100um.npz`; its archived SHA-256 is
`a9a0df39b3bf1ca607f187d628af2c47ef02eadabb6cfe1bd47152ca2b61da6a`.
An independent replay from the archived inputs reproduced that digest.
`gaas_uniform_propagation.json` records the run metrics and original absolute
paths; the same files are available here under relative paths.

The uniform field agrees under 100 versus 200 propagation steps to relative
RMS `7.53e-15` and under 10 versus 80 micrometre transverse windows exactly
at stored precision. It is a restricted numerical calculation, not a
self-consistent pulse prediction. The original 2 micrometre Gaussian result
changed by relative RMS `0.920` when its periodic FFT box grew from 10 to 20
micrometres at fixed spacing. `gaas_gaussian_box_diagnostic.json` records this
failure. The wider sweep in `gaas_gaussian_window_sweep.json` holds that
spacing and the central 10 micrometre comparison region fixed while widening
the box to 640 micrometres. The central field changes by relative RMS `0.00589`
from 320 to 640 micrometres; the outer 20 percent contains `0.000133` of the
output field energy at 640 micrometres. Its chunked Fourier calculation agrees
with the original full-grid solver at 10 micrometres to relative RMS
`1.54e-12`. This supports spatial stability of the **frozen-source
diagnostic** to within 1 percent between the two widest tested boxes. It does not supply a
self-consistent finite-beam GaAs result or experimental field validation.
Recompute the sweep with:

```sh
python -m analysis.gaas_gaussian_window_sweep \
  --sbe-dir archive/2026-10-17-gaas-validation/inputs \
  --out-json OUTPUT.json --widths-um 10 20 40 80 160 320 640 \
  --verify-direct
```

## Dielectric response

`dft/` holds the exported optical curves and metadata for controlled changes
of ground-state grid/cutoff, response grid/bands/cutoff, broadening, and
frequency sampling. `dft_convergence_report.json` compares them on the same
84 measured Jellison energies, from 1.5715537 to 2.9776807 eV. Its channel
differences are RMS curve differences divided by that measured channel's
maximum absolute value; there is no fitting or extrapolation. Recompute the
report with:

```sh
python -m analysis.dft_convergence_report \
  --study-root archive/2026-10-17-gaas-validation/dft \
  --out-json archive/2026-10-17-gaas-validation/dft_convergence_report.json
```

At fixed 6-cubed ground state and 0.05 eV broadening, response-grid changes
6 to 8, 8 to 10, 10 to 12, and 12 to 14 give absorption differences of
0.176, 0.196, 0.185, and 0.155 in that normalization. With an 8-cubed
response grid, the ground-state grid change 6 to 8 gives 0.0057. Thus the
response k-grid is not converged at the manuscript's linewidth. Reprocessing
the 8, 10, 12, and 14-cubed response restarts at 0.20 eV broadening gives
successive absorption differences 0.053, 0.041, and 0.028, but changing
broadening from 0.05 to 0.20 eV at fixed 10-cubed grid itself changes
absorption by 0.108. The broader curves are a **sensitivity study**, not a
replacement for the original calibrated SBE input. Direct 14-cubed PBE
comparisons against Jellison still give absorption/index errors 0.318/0.159
at 0.05 eV and 0.277/0.155 at 0.20 eV. The legacy
metadata field `publication_grade_kmesh` describes a non-Gamma grid in the
frozen code; it is **not** a convergence certification. New runs use
`kmesh_non_gamma` instead. The raw 12 and 14-cubed metadata came from the
earlier frozen script and retain the old key; interpret it only as a non-Gamma
flag. The historical `.gpw` paths in metadata refer to the isolated local
worktree used for these runs; they are not included here. The 0.20 eV curves
can be regenerated from those local restarts with
`analysis/reprocess_dielectric_restart.py`.
