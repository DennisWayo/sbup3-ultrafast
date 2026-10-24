# GaAs 28³ response-grid result

The Mac Studio completed the 28³ point-integration dielectric response on
22 October 2026. It uses the same 6³ PBE ground-state restart, 48 response
bands, 150 eV response cutoff, 0.05 eV broadening, and nonlinear frequency
grid as the archived 16³–24³ runs. The CSV and metadata here have the same
SHA-256 digests as the Mac Studio originals. The 5.2 GB GPAW restart remains
on the Mac Studio and is not included in Git.

`dft_k28_report.json` compares response curves on the same 84 Jellison
energies from 1.5715537 to 2.9776807 eV. Each RMS difference is divided by
the maximum absolute measured value in its channel; no scaling or fitting is
applied.

| Response-grid change | Index difference | Absorption difference |
| --- | ---: | ---: |
| 16³→18³ | 0.0399 | 0.0854 |
| 18³→20³ | 0.0334 | 0.0692 |
| 20³→24³ | 0.0374 | 0.0793 |
| 24³→28³ | 0.0247 | 0.0511 |

The direct 28³ PBE-to-Jellison normalized errors are 0.1659 for refractive
index and 0.3193 for absorption. The latest grid change is smaller, but this
does not establish convergence at 0.05 eV or agreement with measured optics.
No reduced transition set or SBE calibration was recomputed from this curve.

The serial 32³, 36³ follow-up sweep was launched on the Mac Studio on
24 October 2026 with `analysis/gaas_response_grid_sweep.py`. It records live
state in `/Users/dela/sbup3-validation-20261019/response_grid_sweep_32_36.json`.
Neither follow-up point is included in this archive until its calculation and
output validation finish.
