# sbup3/uppe/dispersion.py
import numpy as np

c0 = 2.99792458e8  # m/s

def refractive_index_gaas(omega):
    """
    Simple constant-n model (near band edge).
    You can later replace with Sellmeier.
    """
    n0 = 3.6
    return n0 * np.ones_like(omega)

def propagation_constant(omega):
    n = refractive_index_gaas(omega)
    return n * omega / c0