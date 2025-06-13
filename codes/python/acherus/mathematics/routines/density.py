"""Density evolution module."""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_density(env_a, dens_a, ion_a, n_t_a, dens_n_a, dens_0_a, ava_c_a, dt):
    """
    Compute electron density evolution for all time steps.

    Parameters
    ----------
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    n_t_a : integer
        Number of time nodes.
    dens_n_a : float
        Neutral atom density of the chosen medium.
    dens_0_a : float
        Initial electron density of the chosen medium.
    ava_c_a : float
        Avalanche ionization coefficient.
    dt : float
        Time step.

    """
    dens_a[:, 0] = dens_0_a
    for ll in prange(n_t_a - 1):
        env_s = env_a[:, ll]
        dens_s = dens_a[:, ll]
        ion_s = ion_a[:, ll]

        dens_s_rk4 = _rk4_density_step(env_s, dens_s, ion_s, dens_n_a, ava_c_a, dt)

        dens_a[:, ll + 1] = dens_s_rk4


@njit
def _rk4_density_step(env_s_a, dens_s_a, ion_s_a, dens_n_a, ava_c_a, dt):
    """
    Compute one time step of the RK4 integration for electron
    density evolution.

    Parameters
    ----------
    env_s_a: (M,) array_like
        Complex envelope at current time slice.
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
    k1_dens = _set_density_operator(dens_s_a, env_s_a, ion_s_a, dens_n_a, ava_c_a)
    dens_1 = dens_s_a + 0.5 * dt * k1_dens

    k2_dens = _set_density_operator(dens_1, env_s_a, ion_s_a, dens_n_a, ava_c_a)
    dens_2 = dens_s_a + 0.5 * dt * k2_dens

    k3_dens = _set_density_operator(dens_2, env_s_a, ion_s_a, dens_n_a, ava_c_a)
    dens_3 = dens_s_a + dt * k3_dens

    k4_dens = _set_density_operator(dens_3, env_s_a, ion_s_a, dens_n_a, ava_c_a)

    dens_s_rk4 = dens_s_a + dt * (k1_dens + 2 * k2_dens + 2 * k3_dens + k4_dens) / 6

    return dens_s_rk4


@njit
def _set_density_operator(dens_s_a, env_s_a, ion_s_a, dens_n_a, ava_c_a):
    """
    Compute the electron density evolution terms.

    Parameters
    ----------
    dens_s_a : (M,) array_like
        Density at current time slice.
    env_s_a : (M,) array_like
        Complex envelope at current time slice.
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
    rate_ava = ava_c_a * dens_s_a * np.abs(env_s_a) ** 2

    return rate_ofi + rate_ava
