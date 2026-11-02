# Additional 100 µm propagation-step refinement (1 November 2026)

This check compares the existing 160 µm transverse-box runs at propagation steps 0.25 and 0.125 µm with a new 0.0625 µm run. All three propagate to 100 µm with 156.25 nm transverse sampling, temporal stride 10 (0.05 fs), a 150 fs zero-field tail, the same effective local SBE inputs, and the same source code. It is a numerical sensitivity check of the reduced model, not a GaAs experiment or a dielectric-convergence result.

| Step pair (µm) | Relative RMS change in the central ±5 µm field |
|---|---:|
| 0.25 → 0.125 | 0.1052888637 |
| 0.125 → 0.0625 | 0.0940550242 |

The second difference is 89.3% of the first. This additional halving does not establish propagation-step convergence. The central-field energy changes by 1.36% and the peak magnitude by 1.61% from 0.125 to 0.0625 µm; these scalar measures do not replace the field comparison.

## Integrity checks

- New output: `/Users/dela/sbup3-coupled-100um-quarter-20261101/quarter_dz.npz`, SHA-256 `28b51f083dbc52599ed742ff2053a6e2dbae578495877221ac2709c2ecc7d298`.
- New JSON report: SHA-256 `d57539d8451146735c9777c0dd6582be17bfe80a11e0ccdb8ad349f1b79ba0a2`. Its saved NPZ checksum matches the file.
- Existing 0.25 and 0.125 µm NPZ checksums: `6ea104269258f1b7e923075a914443700646eece805a3d8b2d6effa798eebb54` and `62e5201766253b11b771781b3a715476b500a8b2ce24dfa6924adecf86287d34`.
- Source and input SHA-256 hashes in the new local `plan.json` match the code and inputs on the calculation host and the new report. The run log ends successfully. The report records 1,600 steps, 4,000 temporal samples, and peak excited population `4.406767082094821e-7`.
- The three field arrays are finite, each has shape 1024 × 4000, and their spatial and temporal coordinates match exactly. The full arrays and detailed step log remain on the Mac Studio; this small summary preserves the result without committing large data.

## Claim boundary

The 100 µm local material–field calculation remains sensitive to propagation and temporal resolution. The 44³ dielectric calculation also remains unconverged at the reported 0.05 eV linewidth. Do not use these numerical outputs as a converged physical GaAs transmission prediction or submit the manuscript on their basis. Diagnose accumulated field error and refine the coupled solver before any manuscript claim change.
