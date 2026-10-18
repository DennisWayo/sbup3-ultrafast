# GaAs propagation and dielectric sensitivity record (17 October 2026)

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
self-consistent pulse prediction. The exploratory 2 micrometre Gaussian test
remained box dependent: changing a 10 to 20 micrometre transverse window at
fixed spacing changed the central field by relative RMS `0.920`.
`gaas_gaussian_box_diagnostic.json` records that failure. No finite-beam GaAs
field curve is claimed as converged.

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

At fixed 6-cubed ground state, the response-grid changes 6 to 8 and 8 to 10
give absorption differences of 0.176 and 0.196 in that normalization. With an
8-cubed response grid, the ground-state grid change 6 to 8 gives 0.0057.
Thus the response k-grid is not converged on the tested range. The legacy
metadata field `publication_grade_kmesh` describes a non-Gamma grid in the
frozen code; it is **not** a convergence certification. New runs use
`kmesh_non_gamma` instead. The historical `.gpw` paths in metadata refer to
the isolated local worktree used for these runs; they are not included here.
