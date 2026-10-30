# GaAs response and local material-field checks (29 October 2026)

This record reports completed calculations and preserves the boundary between numerical refinement and physical validation. The large field arrays remain on the Mac Studio.

## Dielectric response

The 36³, 40³, and 44³ point-integration responses share a 6³ ground state, 48 bands, 150 eV cutoff, 0.05 eV broadening, and the same 242-point frequency grid through 8.024 eV. The two new cases exited with code 0. CSV and metadata SHA-256 digests match the Mac Studio status file; the ground-state, metadata, and response-script digests were checked against their recorded values. The 84-point comparison interpolates model curves to the Jellison energy grid (1.5715537–2.9776807 eV) without fitting or extrapolation. Errors below are RMS differences divided by the measured maximum in each channel.

| Comparison | Index n | Absorption alpha |
|---|---:|---:|
| 36³ to 40³ | 0.010786 | 0.022713 |
| 40³ to 44³ | 0.008609 | 0.017890 |
| 44³ versus Jellison | 0.166184 | 0.319378 |

The successive grid differences decrease but remain finite. This does not establish convergence at 0.05 eV or agreement with experiment. The optical residual is much larger than the latest grid step.

## Local material-field pilot

The 10 µm four-case calculation completed with source and input hashes recorded in `coupled_10um_status.json`. Each report and NPZ checksum matched the recorded digest, and the runner checked array dimensions, finite values, z, box, step, time stride, and input hashes. In the central ±5 µm field, relative RMS differences were 0.000889 on propagation-step halving, 0.001109 on temporal-sampling refinement, and 0.0000997 on box doubling. The outer fifth of the 80 µm box contained 1.71e-8 of output field energy. Peak modeled population was about 4.41e-7. This supports local numerical stability for the tested reduced model over 10 µm; it is not experimental GaAs validation.

The bounded 100 µm four-case sweep completed with all case reports, NPZ hashes, settings, finite-value and shape checks passing. On a 40 µm transverse box, central-field relative RMS changes were 0.10744 under propagation-step halving, 0.07059 under temporal refinement, and 0.20614 when the box doubled to 80 µm. The outer fifth of the 80 µm field held 0.02363 of total energy. The 40–80 µm domain is too narrow for a reliable 100 µm result, and step/time sensitivity is also substantial. The full arrays remain on Mac Studio; status and sensitivity summaries are archived here. These results preserve the failed numerical-convergence gate.

An earlier 100 µm sweep with the same hashed source and inputs found a 0.000916 central-field difference between 160 and 320 µm boxes and outer-fifth energy fraction 2.35e-6 in the 320 µm box, while step and time differences remained about 0.1095 and 0.0720. This points to a domain-size problem in the 40–80 µm run rather than resolving step and time convergence.

A new bounded four-case refinement began on 30 October with 160 and 320 µm boxes, 0.25→0.125 µm propagation steps, and stride 10→5 temporal sampling. Each case has a 12-hour limit and writes to `/Users/dela/sbup3-coupled-100um-wide-20261030/`. Its outcome is pending. The manuscript and journal submission remain unchanged.
