"""Density evolution module."""

import numba as nb
import numpy as np


@nb.njit(parallel=True)
def solve_density(env, dens, ion_rate, dens_rk4, n_t, dens_args, dt, dt_2, dt_6):
    """
    Solve electron density evolution for all time steps.

    Parameters:
    - env: envelope at all time slices
    - dens: density at all time slices
    - ion_rate: ionization rate at all time slices
    - dens_rk4: auxiliary density array for RK4 integration
    - n_t: number of time nodes
    - dens_args: arguments for the density operator
    - dt: time step
    - dt_2: half time step
    - dt_6: time step divided by 6
    """
    # Set the initial condition
    dens[:, 0] = 0

    # Solve the electron density evolution
    # pylint: disable=not-an-iterable
    for ll in nb.prange(n_t - 1):
        env_s = env[:, ll]
        dens_s = dens[:, ll]
        ion_rate_s = ion_rate[:, ll]

        dens_s_rk4 = _rk4_density_step(
            env_s, dens_s, ion_rate_s, dens_rk4, dens_args, dt, dt_2, dt_6
        )

        dens[:, ll + 1] = dens_s_rk4


@nb.njit
def _rk4_density_step(env_s, dens_s, ion_rate_s, dens_rk4, dens_args, dt, dt_2, dt_6):
    """
    Compute one time step of the RK4 integration for electron
    density evolution.

    Parameters:
    - env_s: envelope at current time slice
    - dens_s: density at current time slice
    - ion_rate_s: ionization rate at current time slice
    - dens_rk4: auxiliary density array for RK4 integration
    - dens_args: arguments for the density operator
    - dt: time step
    - dt_2: half time step
    - dt_6: time step divided by 6

    Returns:
    - float 1D-array: Electron density at next time slice
    """
    k1_dens = _set_density_operator(dens_s, env_s, ion_rate_s, *dens_args)
    dens_rk4 = dens_s + dt_2 * k1_dens

    k2_dens = _set_density_operator(dens_rk4, env_s, ion_rate_s, *dens_args)
    dens_rk4 = dens_s + dt_2 * k2_dens

    k3_dens = _set_density_operator(dens_rk4, env_s, ion_rate_s, *dens_args)
    dens_rk4 = dens_s + dt * k3_dens

    k4_dens = _set_density_operator(dens_rk4, env_s, ion_rate_s, *dens_args)

    dens_s_rk4 = dens_s + dt_6 * (k1_dens + 2 * k2_dens + 2 * k3_dens + k4_dens)

    return dens_s_rk4


@nb.njit
def _set_density_operator(dens_s, env_s, ion_rate_s, dens_n, coef_ava):
    """Set up the electron density evolution terms.

    Parameters:
    - dens_s: density at current time slice
    - env_s: envelope at current time slice
    - ion_rate_s: ionization rate at current time slice
    - dens_n: neutral density of the medium
    - coef_ava: avalanche/cascade coefficient

    Returns:
    - float 1D-array: Electron density operator
    """
    env_s_2 = np.abs(env_s) ** 2

    term_ofi = ion_rate_s * (dens_n - dens_s)
    term_ava = coef_ava * dens_s * env_s_2

    return term_ofi + term_ava
