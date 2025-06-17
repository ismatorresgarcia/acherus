"""Density evolution module."""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_density(inten_a, dens_a, ion_a, t_a, dens_n_a, dens_0_a, ava_c_a):
    """
    Compute electron density evolution for all time steps.

    Parameters
    ----------
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    t_a : (N,) array_like
        Time coordinates grid.
    dens_n_a : float
        Neutral atom density of the chosen medium.
    dens_0_a : float
        Initial electron density of the chosen medium.
    ava_c_a : float
        Avalanche ionization coefficient.

    """
    n_t_a = len(t_a)
    dt = t_a[1] - t_a[0]
    dens_a[:, 0] = dens_0_a
    for ll in prange(n_t_a - 1):
        int_s = inten_a[:, ll]
        dens_s = dens_a[:, ll]
        ion_s = ion_a[:, ll]

        dens_s_rk4 = _rk4_density_step(int_s, dens_s, ion_s, dens_n_a, ava_c_a, dt)

        dens_a[:, ll + 1] = dens_s_rk4


@njit
def _rk4_density_step(int_s_a, dens_s_a, ion_s_a, dens_n_a, ava_c_a, dt):
    """
    Compute one time step of the RK4 integration for electron
    density evolution.

    Parameters
    ----------
    int_s_a: (M,) array_like
        Intensity at current time slice.
    dens_s_a: (M,) array_like
        Density at current time slice.
    ion_s_a: (M,) array_like
        Ionization rate at current time slice.
    dens_n_a : float
        Neutral atom density of the chosen medium.
    ava_c_a : float
        Avalanche ionization coefficient.
    dt : float
        Time step.

    Returns
    -------
    dens_s_rk4 : (M,) float ndarray.
        Electron density solution at next time slice.
        M is the number of radial nodes.

    """
    k1_dens = _set_density(dens_s_a, int_s_a, ion_s_a, dens_n_a, ava_c_a)
    dens_1 = dens_s_a + 0.5 * dt * k1_dens

    k2_dens = _set_density(dens_1, int_s_a, ion_s_a, dens_n_a, ava_c_a)
    dens_2 = dens_s_a + 0.5 * dt * k2_dens

    k3_dens = _set_density(dens_2, int_s_a, ion_s_a, dens_n_a, ava_c_a)
    dens_3 = dens_s_a + dt * k3_dens

    k4_dens = _set_density(dens_3, int_s_a, ion_s_a, dens_n_a, ava_c_a)

    dens_s_rk4 = dens_s_a + dt * (k1_dens + 2 * k2_dens + 2 * k3_dens + k4_dens) / 6

    return dens_s_rk4


@njit
def _set_density(dens_s_a, int_s_a, ion_s_a, dens_n_a, ava_c_a):
    """
    Compute the electron density evolution terms.

    Parameters
    ----------
    dens_s_a : (M,) array_like
        Density at current time slice.
    int_s_a : (M,) array_like
        Intensity at current time slice.
    ion_s_a : array_like
        Ionization rate at current time slice.
    dens_n_a : float
        Neutral density of the medium chosen.
    ava_c_a : float
        Avalanche ionization coefficient.

    Returns
    -------
    rhs : (M,) float ndarray.
        Electron density evolution RHS at current time slice.
        M is the number of radial nodes.

    """
    rate_ofi = ion_s_a * (dens_n_a - dens_s_a)
    rate_ava = ava_c_a * dens_s_a * int_s_a

    return rate_ofi + rate_ava
