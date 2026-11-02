# GaAs 32³–36³ response-grid sweep

The bounded Mac Studio runner completed both point-integration dielectric
response calculations on 24 October 2026 and stopped. Both use the same 6³
PBE ground-state restart, 48 response bands, 150 eV response cutoff,
0.05 eV broadening, and nonlinear frequency grid as the archived 28³ run.
The four CSV/metadata files here match the SHA-256 digests in the remote
`response_grid_sweep_32_36.json` status record. The Mac Studio retains the
7.7 GB and 11 GB GPAW response restarts; they are not in Git.

`dft_k28_k32_k36_report.json` compares the curves at the same 84 Jellison
energies (1.5715537–2.9776807 eV), without fitting or scaling. Each RMS
difference is normalized by the maximum absolute measured value in its
channel. The report uses the same comparator as the earlier high-grid archive.

| Response-grid change | Refractive-index difference | Absorption difference |
| --- | ---: | ---: |
| 24³→28³ | 0.0247 | 0.0511 |
| 28³→32³ | 0.0178 | 0.0367 |
| 32³→36³ | 0.0139 | 0.0293 |

At 36³, direct PBE-to-Jellison normalized errors are 0.1656 for refractive
index and 0.3189 for absorption. The shrinking grid differences are useful
finite-grid sensitivity evidence. They do **not** establish convergence at
0.05 eV, convergence of the ground state, bands or cutoff, or agreement with
experiment. No reduced transition set, SBE calibration, or long-distance
nonlinear propagation was recomputed from these curves.

The source status record, calculation logs, CSVs, metadata, report, and
`manifest.json` preserve the inputs and outputs of this completed sweep.
