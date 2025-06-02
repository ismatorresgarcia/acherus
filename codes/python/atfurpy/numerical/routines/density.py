"""Density evolution module."""

import numba as nb
import numpy as np


@nb.njit(parallel=True)
def compute_density(env, dens, ion_rate, dens_rk4, n_t, dens_n, coef_ava, dt):
    """
    Compute electron density evolution for all time steps.

    Parameters
    ----------
    env : (M, N) array_like
        Complex envelope at current propagation step.
    dens : (M, N) array_like
        Density at current propagation step.
    ion_rate : (M, N) array_like
        Ionization rate at current propagation step.
    dens_rk4 : (M,) array_like
        Auxiliary density array for RK4 integration.
    n_t : integer
        Number of time nodes.
    dens_n : float
        Neutral atom density of the chosen medium.
    coef_ava : float
        Avalanche/cascade ionization coefficient.
    dt : float
        Time step.

    """
    # Solve the electron density evolution
    # pylint: disable=not-an-iterable
    for ll in nb.prange(n_t - 1):
        env_s = env[:, ll]
        dens_s = dens[:, ll]
        ion_rate_s = ion_rate[:, ll]

        dens_s_rk4 = _rk4_density_step(
            env_s, dens_s, ion_rate_s, dens_rk4, dens_n, coef_ava, dt
        )

        dens[:, ll + 1] = dens_s_rk4


@nb.njit
def _rk4_density_step(env_s, dens_s, ion_rate_s, dens_rk4, dens_n, coef_ava, dt):
    """
    Compute one time step of the RK4 integration for electron
    density evolution.

    Parameters
    ----------
    env_s: (M,) array_like
        Complex envelope at current time slice.
    dens_s: (M,) array_like
        Density at current time slice.
    ion_rate_s: (M,) array_like
        Ionization rate at current time slice.
    dens_rk4: (M,) array_like
        Auxiliary density array for RK4 integration.
    dens_n : float
        Neutral atom density of the chosen medium.
    coef_ava : float
        Avalanche/cascade ionization coefficient.
    dt : float
        Time step.

    Returns
    -------
    dens_s_rk4 : (M,) float ndarray.
        Electron density solution at next time slice.
        M is the number of radial nodes.

    """
    k1_dens = _set_density_operator(dens_s, env_s, ion_rate_s, dens_n, coef_ava)
    dens_rk4 = dens_s + 0.5 * dt * k1_dens

    k2_dens = _set_density_operator(dens_rk4, env_s, ion_rate_s, dens_n, coef_ava)
    dens_rk4 = dens_s + 0.5 * dt * k2_dens

    k3_dens = _set_density_operator(dens_rk4, env_s, ion_rate_s, dens_n, coef_ava)
    dens_rk4 = dens_s + dt * k3_dens

    k4_dens = _set_density_operator(dens_rk4, env_s, ion_rate_s, dens_n, coef_ava)

    dens_s_rk4 = dens_s + dt * (k1_dens + 2 * k2_dens + 2 * k3_dens + k4_dens) / 6

    return dens_s_rk4


@nb.njit
def _set_density_operator(dens_s, env_s, ion_rate_s, dens_n, coef_ava):
    """
    Compute the electron density evolution terms.

    Parameters
    ----------
    dens_s : (M,) array_like
        Density at current time slice.
    env_s : (M,) array_like
        Complex envelope at current time slice.
    ion_rate_s : array_like
        Ionization rate at current time slice.
    dens_n : float
        Neutral density of the medium chosen.
    coef_ava : float
        Avalanche/cascade ionization coefficient.

    Returns
    -------
    rhs : (M,) float ndarray.
        Electron density evolution RHS at current time slice.
        M is the number of radial nodes.

    """
    intensity_s = np.abs(env_s) ** 2

    rate_ofi = ion_rate_s * (dens_n - dens_s)
    rate_ava = coef_ava * dens_s * intensity_s

    return rate_ofi + rate_ava
