"""Raman scattering module."""

import numpy as np
from numba import njit


@njit
def compute_raman(ram_a, dram_a, env_a, n_t_a, ram_c1_a, ram_c2_a, dt):
    """
    Compute molecular Raman scattering delayed response for all time steps.

    Parameters
    ----------
    ram_a : (M, N) array_like
        Raman response at all time slices.
    dram_a : (M, N) array_like
        Raman response time derivative at all time slices.
    env_a : (M, N) array_like
        Complex envelope at all time slices.
    n_t_a : integer
        Number of time nodes.
    ram_c1_a : float
        Raman frequency coefficient for the first ODE term.
    ram_c2_a : float
        Raman frequency coefficient for the second ODE term.
    dt : float
        Time step.

    """
    ram_a[:, 0], dram_a[:, 0] = 0, 0
    for ll in range(n_t_a - 1):
        ram_s = ram_a[:, ll]
        dram_s = dram_a[:, ll]
        env_s = env_a[:, ll]

        ram_s_rk4, dram_s_rk4 = _rk4_raman_step(
            ram_s,
            dram_s,
            env_s,
            ram_c1_a,
            ram_c2_a,
            dt,
        )

        ram_a[:, ll + 1] = ram_s_rk4
        dram_a[:, ll + 1] = dram_s_rk4


@njit
def _rk4_raman_step(ram_s_a, dram_s_a, env_s_a, ram_c1_a, ram_c2_a, dt):
    """
    Compute one time step of the RK4 integration for Raman
    scattering evolution.

    Parameters
    ----------
    ram_s_a : (M,) array_like
        Raman response at current time slice
    dram_s_a : (M,) array_like
        Time derivative raman response at current time slice
    env_s_a : (M,) array_like
        Complex envelope at current time slice
    ram_c1_a : float
        Raman frequency coefficient for the first ODE term
    ram_c2_a : float
        Raman frequency coefficient for the second ODE term
    dt: float
        Time step

    Returns
    -------
    ram_s_rk4 : (M,) ndarray
        Raman response and at next time slice
    dram_s_rk4 : (M,) ndarray
        Raman response derivative and at next time slice

    """
    k1_ram, k1_dram = _set_raman_operator(
        ram_s_a, dram_s_a, env_s_a, ram_c1_a, ram_c2_a
    )
    ram_1 = ram_s_a + 0.5 * dt * k1_ram
    dram_1 = dram_s_a + 0.5 * dt * k1_dram

    k2_ram, k2_dram = _set_raman_operator(ram_1, dram_1, env_s_a, ram_c1_a, ram_c2_a)
    ram_2 = ram_s_a + 0.5 * dt * k2_ram
    dram_2 = dram_s_a + 0.5 * dt * k2_dram

    k3_ram, k3_dram = _set_raman_operator(ram_2, dram_2, env_s_a, ram_c1_a, ram_c2_a)
    ram_3 = ram_s_a + dt * k3_ram
    dram_3 = dram_s_a + dt * k3_dram

    k4_ram, k4_dram = _set_raman_operator(ram_3, dram_3, env_s_a, ram_c1_a, ram_c2_a)

    ram_s_rk4 = ram_s_a + dt * (k1_ram + 2 * k2_ram + 2 * k3_ram + k4_ram) / 6
    dram_s_rk4 = dram_s_a + dt * (k1_dram + 2 * k2_dram + 2 * k3_dram + k4_dram) / 6

    return ram_s_rk4, dram_s_rk4


@njit
def _set_raman_operator(ram_s_a, dram_s_a, env_s_a, ram_c1_a, ram_c2_a):
    """
    Compute the Raman scattering evolution terms.

    Parameters
    ----------
    ram_s_a : (M,) array_like
        Raman response at current time slice.
    dram_s_a : (M,) array_like
        Raman response time derivative at current time slice.
    env_s_a : (M,) array_like
        Complex envelope at current time slice.
    ram_c1_a : float
        Raman frequency coefficient for the first ODE term.
    ram_c2_a : float
        Raman frequency coefficient for the second ODE term.

    Returns
    -------
    rhs : (M,) ndarray
        Raman scattering RHS at current time slice.
    drhs : (M,) ndarray
        Raman scattering derivative RHS at current time slice.

    """
    return dram_s_a, ram_c1_a * (np.abs(env_s_a) ** 2 - ram_s_a) + ram_c2_a * dram_s_a
