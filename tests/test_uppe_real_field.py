"""Manufactured real-field and SI-source checks for the UPPE operator."""

import unittest

import numpy as np

from uppe.uppe_2d import c0, eps0, propagate_uppe_2d


class TestRealFieldUppe(unittest.TestCase):
    def setUp(self):
        self.wavelength = 1.55e-6
        self.n = 3.6
        self.period = self.wavelength / c0
        self.time = np.arange(256) * self.period / 64
        self.x = np.arange(8) * 0.5e-6
        self.wave = np.broadcast_to(np.cos(2 * np.pi * self.time / self.period),
                                    (len(self.x), len(self.time))).copy()
        self.zeros = np.zeros_like(self.wave)
        self.quarter_wave = self.wavelength / (4 * self.n)

    def test_quarter_wave_preserves_amplitude_and_shifts_phase(self):
        result = propagate_uppe_2d(self.x, self.time, self.wave, self.zeros,
                                   self.quarter_wave, 1, background_index=self.n)
        expected = np.broadcast_to(
            np.cos(2 * np.pi * self.time / self.period - np.pi / 2), result.shape)
        np.testing.assert_allclose(result, expected, rtol=0, atol=2e-13)
        self.assertAlmostEqual(np.sqrt(np.mean(result**2)) /
                               np.sqrt(np.mean(self.wave**2)), 1.0, places=12)

    def test_si_source_and_step_refinement(self):
        # Frozen, prescribed polarization P = eps0 * chi * E0 cos(omega t).
        chi = 0.02
        p = eps0 * chi * self.wave
        length = self.wavelength / 8
        one = propagate_uppe_2d(self.x, self.time, self.wave, p, length, 1,
                                background_index=1.0, polarization_kind="full")
        many = propagate_uppe_2d(self.x, self.time, self.wave, p, length / 16, 16,
                                 background_index=1.0, polarization_kind="full")
        phase = 2 * np.pi * length / self.wavelength
        expected = ((1 + chi / 2) * np.cos(2 * np.pi * self.time / self.period - phase)
                    - chi / 2 * np.cos(2 * np.pi * self.time / self.period))
        np.testing.assert_allclose(one[0], expected, rtol=0, atol=2e-13)
        np.testing.assert_allclose(many, one, rtol=0, atol=2e-13)

    def test_transverse_mode_uses_longitudinal_wavenumber(self):
        x = np.arange(64) * 0.1e-6
        kx = 2 * np.pi / (len(x) * (x[1] - x[0]))
        transverse = np.cos(kx * x)[:, None]
        field = transverse * self.wave[0][None, :]
        omega = 2 * np.pi / self.period
        q = np.sqrt((self.n * omega / c0)**2 - kx**2)
        distance = 1.2e-6
        out = propagate_uppe_2d(x, self.time, field, np.zeros_like(field),
                                distance, 1, background_index=self.n)
        expected = transverse * np.cos(omega * self.time[None, :] - q * distance)
        np.testing.assert_allclose(out, expected, rtol=0, atol=2e-13)

    def test_full_response_cannot_overlap_background_index(self):
        with self.assertRaisesRegex(ValueError, "double counting"):
            propagate_uppe_2d(self.x, self.time, self.wave, eps0 * self.wave,
                               self.quarter_wave, 1, background_index=self.n,
                               polarization_kind="full")

    def test_nonzero_source_requires_explicit_material_convention(self):
        with self.assertRaisesRegex(ValueError, "Specify full or residual"):
            propagate_uppe_2d(self.x, self.time, self.wave, eps0 * self.wave,
                               self.quarter_wave, 1, background_index=self.n)


if __name__ == "__main__":
    unittest.main()
