"""Forward real-field step in a passive dispersive linear background.

The complex background index contains the full linear material response.
Only polarization beyond that response enters the explicit source term.
"""

from __future__ import annotations

import numpy as np


C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.0 / (EPS0 * C0**2)


class DispersiveResidualStep:
    def __init__(self, x_m: np.ndarray, time_s: np.ndarray,
                 chi_positive: np.ndarray, dz_m: float):
        x = np.asarray(x_m, dtype=float)
        time = np.asarray(time_s, dtype=float)
        chi = np.asarray(chi_positive, dtype=complex)
        if x.ndim != 1 or time.ndim != 1 or len(x) < 2 or len(time) < 2:
            raise ValueError("Need one-dimensional x and time grids")
        dx = float(x[1] - x[0])
        dt = float(time[1] - time[0])
        if dx <= 0 or dt <= 0 or dz_m <= 0:
            raise ValueError("Spatial, temporal, and propagation steps must be positive")
        if (not np.allclose(np.diff(x), dx, rtol=1e-10, atol=0)
                or not np.allclose(np.diff(time), dt, rtol=1e-10, atol=0)):
            raise ValueError("FFT coordinate grids must be uniform")
        self.nx = len(x)
        self.nt = len(time)
        self.omega = 2 * np.pi * np.fft.rfftfreq(self.nt, dt)
        if chi.shape != self.omega.shape or not np.isfinite(chi).all():
            raise ValueError("Susceptibility must cover all nonnegative temporal FFT bins")
        n = np.sqrt(1.0 + chi)
        if np.any(n.real <= 0) or np.any(n.imag > 1e-10):
            raise ValueError("Complex background index must be passive")
        self.chi = chi
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, dx)[:, None]
        om = self.omega[None, :]
        q = np.sqrt((n[None, :] * om / C0) ** 2 - kx**2)
        q = np.where(q.imag > 0, -q, q)
        self.phase = np.exp(-1j * q * dz_m)
        self.phase[:, 0] = np.exp(-np.abs(kx[:, 0]) * dz_m)
        self.source_step = np.zeros_like(q)
        propagating = (q.real > 0) & (om > 0)
        omega_b = np.broadcast_to(om, q.shape)
        self.source_step[propagating] = (
            -1j * MU0 * omega_b[propagating] ** 2 / (2 * q[propagating])
            * dz_m * np.exp(-0.5j * q[propagating] * dz_m)
            * np.sinc(q[propagating] * dz_m / (2 * np.pi))
        )
        if self.nt % 2 == 0:
            nyquist = self.nt // 2
            self.phase[:, nyquist] = (np.exp(q[:, nyquist].imag * dz_m)
                                      * np.cos(q[:, nyquist].real * dz_m))
            self.source_step[:, nyquist] = 0.0

    def linear_polarization(self, field_xt: np.ndarray) -> np.ndarray:
        field = np.asarray(field_xt, dtype=float)
        if field.shape != (self.nx, self.nt):
            raise ValueError("Field shape does not match propagator")
        return np.fft.irfft(
            EPS0 * self.chi[None, :] * np.fft.rfft(field, axis=1),
            n=self.nt, axis=1,
        )

    def advance(self, field_xt: np.ndarray, residual_xt: np.ndarray) -> np.ndarray:
        field = np.asarray(field_xt, dtype=float)
        residual = np.asarray(residual_xt, dtype=float)
        if field.shape != (self.nx, self.nt) or residual.shape != field.shape:
            raise ValueError("Field and residual must match the propagator grid")
        if not np.isfinite(field).all() or not np.isfinite(residual).all():
            raise ValueError("Field and residual must be finite")
        field_kw = np.fft.fft(np.fft.rfft(field, axis=1), axis=0)
        residual_kw = np.fft.fft(np.fft.rfft(residual, axis=1), axis=0)
        out_kw = self.phase * field_kw + self.source_step * residual_kw
        return np.fft.irfft(np.fft.ifft(out_kw, axis=0), n=self.nt, axis=1)
