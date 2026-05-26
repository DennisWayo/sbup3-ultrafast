# Reference Data Format

Provide a CSV with a header row and at least one of the supported columns:

- `energy_eV` (required)
- `chi_real` (optional)
- `chi_imag` (optional)
- `alpha` (optional, `alpha_m_inv` is also accepted)
- `n` (optional)

Example:

```
energy_eV,alpha
0.5,1.2e4
0.6,1.8e4
0.7,2.5e4
```

Units should be consistent across the file. For baseline comparisons, the script
fits a scale factor to best match the simulation in the overlapping energy range.

For absolute-scale calibration against experiment, prefer measured `alpha` and/or
`n` columns and run:

```
python analysis/calibrate_polarization_scale.py \
  --ref-csv analysis/data/<your_reference>.csv \
  --fit-channels alpha,n
```

Proxy/oscillator-only references are intentionally unsupported in this path.

## Included Published GaAs References

These reference files are now checked in:

- `analysis/data/gaas_exp_aspnes_1986.csv`
- `analysis/data/gaas_exp_jellison_1992.csv`

Source YAML snapshots (public-domain refractiveindex.info database) are stored in:

- `analysis/data/sources/gaas/Aspnes_1986.yml`
- `analysis/data/sources/gaas/Jellison_1992.yml`
- `analysis/data/sources/gaas/Skauli_2003.yml`

To regenerate CSV files from the source YAML:

```bash
python analysis/data/import_refractiveindex_nk.py \
  --in-yaml analysis/data/sources/gaas/Aspnes_1986.yml \
  --out-csv analysis/data/gaas_exp_aspnes_1986.csv \
  --source-label Aspnes1986_GaAs \
  --source-url "https://refractiveindex.info/?shelf=main&book=GaAs&page=Aspnes"
```
