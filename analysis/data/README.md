# Reference Data Format

Provide a CSV with a header row and at least one of the supported columns:

- `energy_eV` (required)
- `chi_real` (optional)
- `chi_imag` (optional)
- `alpha` (optional)
- `n` (optional)
- `osc` (optional, oscillator strength or proxy)

Example:

```
energy_eV,alpha
0.5,1.2e4
0.6,1.8e4
0.7,2.5e4
```

Units should be consistent across the file. For baseline comparisons, the script
fits a scale factor to best match the simulation in the overlapping energy range.

If no CSV is available, `analysis/validate_sbup3.py` will fall back to parsing
`dft/lrtddft.log` or `dft/gaas_lrtddft.log` and interpret `|me|` as an oscillator proxy.
