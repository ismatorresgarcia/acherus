"""Raman scattering module."""

import numba as nb


@nb.njit(parallel=True)
def compute_raman(ram, dram, env, ram_rk4, dram_rk4, n_t, coef_ode1, coef_ode2, dt):
    """
    Compute molecular Raman scattering delayed response for all time steps.

    Parameters
    ----------
    ram : (M, N) array_like
        Raman response at all time slices.
    dram : (M, N) array_like
        Raman response time derivative at all time slices.
    env : (M, N) array_like
        Complex envelope at all time slices.
    ram_rk4 : (M,) array_like
        Auxiliary raman response array.
    dram_rk4 : (M,) array_like
        Auxiliary raman response time derivative array.
    n_t : integer
        Number of time nodes.
    coef_ode1 : float
        Raman frequency coefficient for the first ODE term.
    coef_ode2 : float
        Raman frequency coefficient for the second ODE term.
    dt : float
        Time step.

    """
    # Set the initial conditions
    ram[:, 0], dram[:, 0] = 0, 0

    # Solve the raman scattering response
    # pylint: disable=not-an-iterable
    for ll in nb.prange(n_t - 1):
        ram_s = ram[:, ll]
        dram_s = dram[:, ll]
        env_s = env[:, ll]

        ram_s_rk4, dram_s_rk4 = _rk4_raman_step(
            ram_s,
            dram_s,
            env_s,
            ram_rk4,
            dram_rk4,
            coef_ode1,
            coef_ode2,
            dt,
        )

        ram[:, ll + 1] = ram_s_rk4
        dram[:, ll + 1] = dram_s_rk4


@nb.njit
def _rk4_raman_step(ram_s, dram_s, env_s, ram_rk4, dram_rk4, coef_ode1, coef_ode2, dt):
    """
    Compute one time step of the RK4 integration for Raman
    scattering evolution.

    Parameters
    ----------
    ram_s : (M,) array_like
        Raman response at current time slice
    dram_s : (M,) array_like
        Time derivative raman response at current time slice
    env_s : (M,) array_like
        Complex envelope at current time slice
    ram_rk4: (M,) array_like
        Auxiliary raman response array for RK4 integration
    dram_rk4: (M,) array_like
        Auxiliary raman response time derivative array for RK4 integration
    coef_ode1: float
        Raman frequency coefficient for the first ODE term
    coef_ode2: float
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
    k1_ram, k1_dram = _set_raman_operator(ram_s, dram_s, env_s, coef_ode1, coef_ode2)
    ram_rk4 = ram_s + 0.5 * dt * k1_ram
    dram_rk4 = dram_s + 0.5 * dt * k1_dram

    k2_ram, k2_dram = _set_raman_operator(
        ram_rk4, dram_rk4, env_s, coef_ode1, coef_ode2
    )
    ram_rk4 = ram_s + 0.5 * dt * k2_ram
    dram_rk4 = dram_s + 0.5 * dt * k2_dram

    k3_ram, k3_dram = _set_raman_operator(
        ram_rk4, dram_rk4, env_s, coef_ode1, coef_ode2
    )
    ram_rk4 = ram_s + dt * k3_ram
    dram_rk4 = dram_s + dt * k3_dram

    k4_ram, k4_dram = _set_raman_operator(
        ram_rk4, dram_rk4, env_s, coef_ode1, coef_ode2
    )

    ram_s_rk4 = ram_s + dt * (k1_ram + 2 * k2_ram + 2 * k3_ram + k4_ram) / 6
    dram_s_rk4 = dram_s + dt * (k1_dram + 2 * k2_dram + 2 * k3_dram + k4_dram) / 6

    return ram_s_rk4, dram_s_rk4


@nb.njit
def _set_raman_operator(ram_s, dram_s, env_s, coef_ode1, coef_ode2):
    """
    Compute the Raman scattering evolution terms.

    Parameters
    ----------
    ram_s : (M,) array_like
        Raman response at current time slice.
    dram_s : (M,) array_like
        Raman response time derivative at current time slice.
    env_s : (M,) array_like
        Complex envelope at current time slice.
    coef_ode1 : float
        Raman frequency coefficient for the first ODE term.
    coef_ode2 : float
        Raman frequency coefficient for the second ODE term.

    Returns
    -------
    drhs : (M,) ndarray
        Raman scattering RHS at current time slice.
    rhs : (M,) ndarray
        Raman scattering derivative RHS at current time slice.

    """
    diff_s = env_s - ram_s
    return dram_s, coef_ode1 * diff_s + coef_ode2 * dram_s
