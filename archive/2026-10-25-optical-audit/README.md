# GaAs optical-reference audit

The completed 36³ PBE dielectric response was compared without fitting to
two archived optical references. `reference_spread_14pt.json` uses the same
14 Aspnes energies within 1.5716–2.9777 eV for all three pairwise curves;
the Jellison and PBE curves are interpolated onto those energies. The input
paths and SHA-256 digests are recorded in the JSON. Reproduce it with:

```sh
python -m analysis.gaas_reference_spread \
  --dft-csv archive/2026-10-25-dft-k32-k36/dft/gaas_dielectric_gs6_k36_b48_e150.csv \
  --out-json archive/2026-10-25-optical-audit/reference_spread_14pt.json
```

| Common 14-point comparison | Index normalized RMSE | Absorption normalized RMSE |
| --- | ---: | ---: |
| Jellison versus Aspnes | 0.00388 | 0.0290 |
| 36³ PBE versus Aspnes | 0.1594 | 0.3955 |
| 36³ PBE versus Jellison | 0.1607 | 0.4469 |

Each RMSE is divided by the maximum absolute reference value in its pair
and channel. On the separate, denser **84-point Jellison grid**, the 36³
PBE-to-Jellison errors are 0.1656 for index and 0.3189 for absorption;
the 32³→36³ computed-curve changes are 0.0139 and 0.0293. The 14-point
and 84-point absorption scores must not be compared as if they used the same
sampling.

The Jellison and Aspnes studies used different GaAs samples and protocols;
their spread is **not** a measurement-uncertainty bound. The response-grid
difference, cross-study spread, and PBE residual are not independent or
additive error terms. At 36³ the numerical response grid still changes the
curve, while the measured mismatch remains substantial. These results do
not isolate how much of that mismatch comes from PBE, broadening, the
ground-state calculation, the optical data, or the reduced SBE model.
