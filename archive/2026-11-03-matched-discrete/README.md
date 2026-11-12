# Matched-discrete local SBE--UPPE checks (3 November 2026)

This archive records bounded numerical sensitivity checks of the calibrated
reduced model. The original analytic-subtraction calculations and their failures
remain in `archive/2026-10-29-convergence-check`; they are not replaced by
these results.

The candidate uses a population-frozen SBE response evaluated on the **same
RK4 time grid** as the nonlinear SBE update. Subtracting this matched discrete
linear response from the full response avoids the temporal-discretization
residual produced by subtracting a separately sampled analytic susceptibility.
The residual polarization is recomputed during propagation, with a
predictor--corrector average across each propagation step.

| Length | Step change | Time change | Box change | Outer-fifth energy fraction |
| --- | ---: | ---: | ---: | ---: |
| 10 µm | 6.891864366e-7 | 1.148515811e-6 | 9.977537074e-5 | 1.714971233e-8 |
| 100 µm | 2.821112044e-7 | 4.547093062e-6 | 8.967282460e-4 | 2.213311693e-6 |

Values are relative RMS differences in the central -5 to 5 µm output field.
The step comparison uses the base and half-step fields; time and box
comparisons use the half-step field as their reference. The outer-fifth energy
fraction is measured in the doubled-width field.
The 10 µm base uses a 40 µm transverse box and a 0.5 µm propagation step.
The 100 µm base uses a 160 µm transverse box, 156.25 nm spacing, 2 µm
input waist, 0.25 µm propagation step, and 0.05 fs sampled time step.
The step comparison uses 0.25 versus 0.125 µm; the time comparison uses
0.05 versus 0.025 fs; the box comparison uses 160 versus 320 µm at fixed
spatial spacing. The 10 µm controls are specified in its status report.

The source hashes recorded by the 100 µm run are:

- `analysis/gaas_local_sbe_uppe.py`: `4235143c864bbc76a87d2353590891f5f9e457e37fb86584dc71e8060b1aa4ef`
- `analysis/gaas_linear_material_window.py`: `093ea9c022bfd680c68db468cb78727e77f8c97d842cb0e1554be00c52d8b186`
- `sbe/local_response.py`: `9203e8512a51ed9870f09da4dedbc115db9ea88bb10a1b488715af84b129618e`
- `uppe/dispersive_residual.py`: `b080fa8ebce88a3fa896c69ff4f8eac29017851de16ee826bcd3ed7101d0043e`

The copied source files match these hashes. The status files contain input and
output hashes and completion state; the sensitivity files contain the reported
comparisons. Full field arrays remain on the calculation host and are not in
this Git branch. The archived checks found finite arrays and verified their
hashes, settings, and logs. In this temporary checkout, three targeted
local-coupling tests and all 11 discovered repository tests passed. The
repository test that launches `.venv/bin/python` needed a temporary local
symlink to the available Python executable; that symlink is not committed.

These results establish numerical consistency of this reduced model at the
tested controls. They do not establish convergence of the 0.05 eV GPAW
dielectric input, quantitative agreement with GaAs optical measurements, or
physical validation of finite-beam propagation.
