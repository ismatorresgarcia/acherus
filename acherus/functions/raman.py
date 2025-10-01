"""Raman scattering module."""

from numba import njit, prange


@njit(parallel=True)
def compute_raman(int_a, ram_a, t_a, ram_c1_a, ram_c2_a):
    """
    Compute molecular Raman scattering delayed response for all time steps
    using trapezoidal integration.

    Parameters
    ----------
    int_a : function
        Intensity function at current propagation step.
    ram_a : (M, N) array_like
        Raman response at all time slices.
    t_a : (N,) array_like
        Time coordinates grid.
    ram_c1_a : float
        Raman frequency coefficient for the first term.
    ram_c2_a : float
        Raman frequency coefficient for the second term.

    """
    for kk in prange(len(t_a) - 1):
        ram_a[:, kk + 1] = ram_c1_a * ram_a[:, kk] + ram_c2_a * (int_a[:, kk + 1] + ram_c1_a * int_a[:, kk])
