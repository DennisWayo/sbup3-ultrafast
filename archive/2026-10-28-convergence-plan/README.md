# GaAs response and local material-field checks (28 October 2026)

The existing `gs6` PBE response at 0.05 eV broadening still changes between
32³ and 36³ response grids: measured-maximum-normalized RMS differences are
0.0139 in refractive index and 0.0293 in absorption on the 84-point Jellison
window. This record does not claim dielectric convergence.

## Mac Studio calculations

1. `analysis/gaas_response_grid_sweep.py` is running serial 40³ and 44³
   point-integration response calculations in
   `/Users/dela/sbup3-validation-20261019/`. The ground-state restart,
   48 bands, 150 eV response cutoff, 0.05 eV broadening, and frequency grid
   match the 36³ baseline. Each case has an 18-hour stop limit. Progress is in
   `response_grid_sweep_40_44.json`; failed or timed-out runs remain recorded.
2. `analysis/gaas_coupled_sweep.py` is queued after that sweep to avoid
   concurrent heavy calculations. It tests 10 µm propagation of the existing
   calibrated 16-transition local-SBE model with a 2 µm Gaussian waist.
   Four cases separately refine propagation step (0.5→0.25 µm), time sampling
   (0.05→0.025 fs), and transverse box (40→80 µm). Progress is in
   `/Users/dela/sbup3-coupled-10um-20261028/coupled_status.json`.

The queued local-SBE calculation recomputes nonlinear residual polarization
from the propagated field at each longitudinal step. Its parameters remain
effective model parameters derived from the archived dielectric input, so a
numerically stable run would not establish an ab-initio GaAs transmission
prediction or agreement with experiment.

## Decision gates

- Compare 36³→40³ and 40³→44³ on identical measured energies. A working
  numerical target is below 1% measured-maximum-normalized RMS in both optical
  channels on successive refinements. Also inspect spectral shape; a single
  scalar threshold cannot certify convergence. Band, response-cutoff, and
  ground-state sensitivity remain separate checks before a converged DFT
  optical input can be claimed.
- For the coupled 10 µm pilot, compare central ±5 µm fields under all three
  refinements and the outer-box energy fraction. If stable at a stated
  tolerance, extend the same local-update calculation to the intended 100 µm
  distance; otherwise refine before extending it.
- Keep the paper's current limitations until completed results support narrower
  claims. The six authors must approve the final changed draft before
  submission.

All raw `.gpw` response restarts and coupled field `.npz` files remain on Mac
Studio. Archive small reports and their checksums after the sweeps finish.
