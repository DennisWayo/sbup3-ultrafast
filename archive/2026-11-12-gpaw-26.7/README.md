# GPAW 26.7.0 dielectric integration check (12 November 2026)

This archive accompanies the numerical-limit discussion in the SBUP3 study
and [GPAW investigation #1665](https://gitlab.com/gpaw/gpaw/-/work_items/1665).
It is not evidence of a confirmed GPAW defect.

The isolated Mac Studio environment used GPAW 26.7.0, ASE 3.29.0,
NumPy 2.4.6, SciPy 1.17.1 and libxc 7.1.2. The response settings were
PBE GaAs, 48 bands, 150 eV cutoff, 0.05 eV broadening, no intraband term
and a nonlinear frequency grid with `domega0=0.005`, `omega2=5.0`,
and `omegamax=8.0`.

The relative L2 norm is computed on 1.5716--2.9777 eV from column 5
(local-field-corrected imaginary dielectric function), using the full-zone
curve as the denominator. There are 199 exported frequency samples in this
window; the 84 Jellison points used elsewhere in the paper are interpolation
targets, not independent model frequencies.

| Mesh | GPAW input | Point, symmetry on/off | Tetrahedron, symmetry on/off |
| --- | --- | ---: | ---: |
| 8³ | Fresh 26.7.0 ground state and response wavefunctions | 3.919967063e-6 | 0.3378577480 |
| 16³ | Same archived 25.7 wavefunctions re-evaluated on 26.7.0 | 2.733336628e-6 | 0.03512303915 |
| 24³ | Prior 25.7 check, not rerun here | 1.462902554e-6 | 0.01110344633 |

All four 16³ spectra under 26.7.0 are numerically **exactly equal** to
the corresponding archived 25.7.0 CSVs in this directory (964 rows,
five columns each). The 8³ response was generated independently on 26.7.0.
The fresh-run script used on the calculation host and its status file are in
`fresh_k8/`; a path-portable variant is also provided but was not rerun.
The same-input script and status are in `same_input_k16/`. Status files
record source, GPW, and output hashes; copied source and CSV hashes were
checked against those records. The GPW restart files are retained on the
calculation host and are not included in this branch. All copied CSVs are
finite, 964 by 5, and share a common frequency axis.

The tetrahedron discrepancy decreases from the coarse to finer meshes.
Its cause could be a finite-grid tessellation effect. The paper therefore
uses point integration for its optical-sensitivity audit and does not claim
that the 0.05 eV dielectric response has converged or that GPAW has a bug.
