"""Physical checks for the local SBE and dispersive-background handoff."""

import unittest

import numpy as np

from analysis.gaas_linear_material_window import linear_sbe_chi
from analysis.gaas_local_sbe_uppe import local_sbe_uppe
from sbe.local_response import local_polarization
from uppe.dispersive_residual import DispersiveResidualStep
from uppe.uppe_2d import c0, propagate_uppe_2d


class TestLocalMaterialCoupling(unittest.TestCase):
    def test_harmonic_sbe_response_matches_causal_susceptibility(self):
        elementary_charge = 1.602176634e-19
        hbar = 1.054571817e-34
        eps0 = 8.8541878128e-12
        omega = 2.0 * elementary_charge / hbar
        time = np.arange(0.0, 200e-15, 0.02e-15)
        amplitude = 1.0e3
        field = amplitude * np.cos(omega * time)
        transition = np.array([omega])
        dipole = np.array([2.0e-30])
        weight = np.array([1.0])
        density = 1.0e26
        gamma = 5.0e13
        polarization, peak_population = local_polarization(
            field[None, :], time, transition, dipole, weight,
            density_m3=density, polarization_scale=1.0, gamma_s=gamma,
        )
        late = time > 100e-15
        basis = np.column_stack((np.cos(omega * time[late]),
                                 np.sin(omega * time[late])))
        cosine, sine = np.linalg.lstsq(basis, polarization[0, late], rcond=None)[0]
        measured = (cosine - 1j * sine) / (eps0 * amplitude)
        analytic = linear_sbe_chi(
            np.array([omega]), transition, dipole, weight,
            density_m3=density, polarization_scale=1.0, gamma_s=gamma,
        )[0]
        self.assertLess(abs(measured - analytic) / abs(analytic), 0.01)
        self.assertLess(analytic.imag, 0.0)
        self.assertLess(peak_population, 1e-4)

    def test_zero_material_matches_existing_vacuum_propagator(self):
        wavelength = 1.55e-6
        period = wavelength / c0
        time = np.arange(256) * period / 64
        x = np.arange(8) * 0.25e-6
        spatial = 1 + 0.2 * np.cos(2 * np.pi * np.arange(8) / 8)
        field = spatial[:, None] * np.cos(2 * np.pi * time[None, :] / period)
        dz = wavelength / 8
        step = DispersiveResidualStep(x, time, np.zeros(129, complex), dz)
        result = step.advance(field, np.zeros_like(field))
        reference = propagate_uppe_2d(
            x, time, field, np.zeros_like(field), dz, 1,
            background_index=1.0,
        )
        np.testing.assert_allclose(result, reference, atol=5e-13, rtol=0)

    def test_weak_local_feedback_agrees_with_linear_material(self):
        elementary_charge = 1.602176634e-19
        hbar = 1.054571817e-34
        omega0 = 2.0 * elementary_charge / hbar
        time = np.arange(1024) * 0.05e-15
        x = (np.arange(8) - 4) * 0.5e-6
        pulse = (1e3 * np.exp(-((time - 20e-15) / (5e-15)) ** 2)
                 * np.cos(0.8 * omega0 * (time - 20e-15)))
        initial = np.exp(-(x[:, None] / 2e-6) ** 2) * pulse[None, :]
        transition = np.array([omega0])
        dipole = np.array([2e-30])
        weights = np.array([1.0])
        freq = 2 * np.pi * np.fft.rfftfreq(len(time), time[1] - time[0])
        chi = linear_sbe_chi(
            freq, transition, dipole, weights,
            density_m3=1e26, polarization_scale=1.0, gamma_s=5e13,
        )
        dz = 0.1e-6
        result, report = local_sbe_uppe(
            x, time, initial, transition, dipole, weights,
            density_m3=1e26, polarization_scale=1.0, gamma_s=5e13,
            chi_positive=chi, dz_m=dz, z_steps=2, max_field_vpm=1e6,
        )
        operator = DispersiveResidualStep(x, time, chi, dz)
        expected = operator.advance(operator.advance(initial, np.zeros_like(initial)),
                                    np.zeros_like(initial))
        error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
        self.assertLess(error, 0.001)
        self.assertLess(report["peak_population"], 1e-5)


if __name__ == "__main__":
    unittest.main()
