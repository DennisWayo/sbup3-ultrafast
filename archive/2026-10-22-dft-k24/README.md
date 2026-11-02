# GaAs response-grid extension to 24³

The 24³ point-integration dielectric response finished on the Mac Studio on
19 October 2026. It uses the same 6³ PBE ground-state restart, 48 response
bands, 150 eV response cutoff, 0.05 eV broadening, and nonlinear frequency
grid with a 0.02 eV seed as the archived 14³–20³ responses. The exported CSV
has 242 data rows; its metadata records the settings. The CSV and metadata
were copied from `/Users/dela/sbup3-validation-20261019/dft/` and their
SHA-256 digests match the source files. Large GPAW restart files remain on
the Mac Studio and are not in Git.

`dft_k24_report.json` compares all five grids on the same 84 Jellison
energies, 1.5715537–2.9776807 eV. Each difference is the RMS curve
difference divided by the maximum absolute measured value of that channel.

| Response grid change | Index difference | Absorption difference |
| --- | ---: | ---: |
| 14³→16³ | 0.054 | 0.128 |
| 16³→18³ | 0.040 | 0.085 |
| 18³→20³ | 0.033 | 0.069 |
| 20³→24³ | 0.037 | 0.079 |

The 24³ direct PBE-to-Jellison normalized errors are 0.167 in index and
0.323 in absorption. The last grid change grows slightly rather than
decreasing, so the 0.05 eV response is **not converged** on this window.
The mismatch with measured optics remains substantial. No reduced
transition set or SBE calibration was recomputed from this curve. These
tests do not validate long-distance nonlinear GaAs propagation.

Regenerate the report from the repository root with:

```sh
python3 -m analysis.dft_highgrid_report \
  --case 14=archive/2026-10-17-gaas-validation/dft/gaas_dielectric_gs6_k14_b48_e150.csv \
  --case 16=archive/2026-10-19-dft-highgrid/dft/gaas_dielectric_gs6_k16_b48_e150.csv \
  --case 18=archive/2026-10-19-dft-highgrid/dft/gaas_dielectric_gs6_k18_b48_e150.csv \
  --case 20=archive/2026-10-19-dft-highgrid/dft/gaas_dielectric_gs6_k20_b48_e150.csv \
  --case 24=archive/2026-10-22-dft-k24/dft/gaas_dielectric_gs6_k24_b48_e150.csv \
  --out-json archive/2026-10-22-dft-k24/dft_k24_report.json
```

`manifest.json` records sizes and SHA-256 hashes for the files in this
archive other than itself.
