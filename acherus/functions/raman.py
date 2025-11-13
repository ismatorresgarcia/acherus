"""Helper module for stimulated Raman scattering (SRS) contribution."""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_raman(inten_a, ram_a, ram_x_a, ram_c1_a, ram_c2_a):
    """
    Compute Raman contribution delayed response for all time steps
    using the trapezoidal rule.

    Parameters
    ----------
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    ram_a : (M, N) array_like
        Raman response at all time slices.
    ram_x_a : (M, N) array_like
        Complex Raman response at all time slices.
    ram_c1_a : float
        Raman frequency coefficient for the first term.
    ram_c2_a : float
        Raman frequency coefficient for the second term.

    """
    nr, nt = inten_a.shape()
    for ii in prange(nr):
        for kk in range(nt - 1):
            ram_x_a[ii, kk + 1] = ram_c1_a * ram_x_a[ii, kk] + ram_c2_a * (inten_a[ii, kk + 1] + ram_c1_a * inten_a[ii, kk])

    ram_a[:] = np.imag(ram_x_a)
