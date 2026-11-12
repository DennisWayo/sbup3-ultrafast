# GaAs 0.05 eV response-grid extension (19 October 2026)

These point-integration response calculations extend the prior 6³ to 14³
study to 16³, 18³, and 20³ using the same 6³ PBE plane-wave ground state,
48 response bands, 150 eV response cutoff, 0.05 eV broadening, and 0.02 eV
nonlinear frequency-grid seed. GPAW 25.7.0 ran on a Mac Studio with 96 GB
memory. All three runs used the same ground-state `.gpw` file, whose SHA-256
was `46d7c289795a9713797bceaaeb4b2dd743bcc2f0f125ec28c276e9db60f515e1`.
That restart and the large response restarts are retained outside Git.
`dft/` contains the exported CSVs and settings metadata. The copied CSVs
and metadata were checked against their remote SHA-256 digests.

`dft_highgrid_report.json` compares 14³, 16³, 18³, and 20³ on the same
84 Jellison energies, 1.5715537–2.9776807 eV. Each difference is an RMS
curve difference divided by the maximum absolute measured value of that
channel. The 14³ input is in
`archive/2026-10-17-gaas-validation/dft/gaas_dielectric_gs6_k14_b48_e150.csv`.

| Response grid change | Index difference | Absorption difference |
| --- | ---: | ---: |
| 14³→16³ | 0.054 | 0.128 |
| 16³→18³ | 0.040 | 0.085 |
| 18³→20³ | 0.033 | 0.069 |

The 20³ direct PBE-to-Jellison normalized errors are 0.170 in index and
0.325 in absorption. These exported curves are **not converged at the
reported 0.05 eV linewidth**, and the direct experimental discrepancy
remains appreciable. No reduced transition set or SBE calibration was
recomputed from the denser curves. The 0.20 eV broadening sensitivity in the
earlier archive changes the response model and cannot certify convergence of
the 0.05 eV result.

Regenerate the report from the repository root with:

```sh
python -m analysis.dft_highgrid_report \
  --case 14=archive/2026-10-17-gaas-validation/dft/gaas_dielectric_gs6_k14_b48_e150.csv \
  --case 16=archive/2026-10-19-dft-highgrid/dft/gaas_dielectric_gs6_k16_b48_e150.csv \
  --case 18=archive/2026-10-19-dft-highgrid/dft/gaas_dielectric_gs6_k18_b48_e150.csv \
  --case 20=archive/2026-10-19-dft-highgrid/dft/gaas_dielectric_gs6_k20_b48_e150.csv \
  --out-json OUTPUT.json
```

The metadata records the exact response settings and relative local restart
paths. `manifest.json` records sizes and SHA-256 hashes for this archive.
