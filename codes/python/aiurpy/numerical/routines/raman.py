"""Raman scattering module."""

import numba as nb


@nb.njit(parallel=True)
def solve_scattering(
    ram, dram, env, ram_rk4, dram_rk4, n_t, coef_ode1, coef_ode2, dt, dt_2, dt_6
):
    """
    Solve molecular Raman scattering delayed response for all time steps.

    Parameters:
    -> ram: raman response at all time slices
    -> dram: raman response time derivative at all time slices
    -> env: envelope at all time slices
    -> ram_rk4: auxiliary raman response array
    -> dram_rk4: auxiliary raman response time derivative array
    -> n_t: number of time nodes
    -> coef_ode1: Raman frequency coefficient for the first ODE term
    -> coef_ode2: Raman frequency coefficient for the second ODE term
    -> dt: time step
    -> dt_2: half time step
    -> dt_6: time step divided by 6
    """
    # Set the initial conditions
    ram[:, 0], dram[:, 0] = 0, 0

    # Solve the raman scattering response
    # pylint: disable=not-an-iterable
    for ll in nb.prange(n_t - 1):
        ram_s = ram[:, ll]
        dram_s = dram[:, ll]
        env_s = env[:, ll]

        ram_s_rk4, dram_s_rk4 = _rk4_scattering_step(
            ram_s,
            dram_s,
            env_s,
            ram_rk4,
            dram_rk4,
            coef_ode1,
            coef_ode2,
            dt,
            dt_2,
            dt_6,
        )

        ram[:, ll + 1] = ram_s_rk4
        dram[:, ll + 1] = dram_s_rk4


@nb.njit
def _rk4_scattering_step(
    ram_s, dram_s, env_s, ram_rk4, dram_rk4, coef_ode1, coef_ode2, dt, dt_2, dt_6
):
    """
    Compute one time step of the RK4 integration for Raman
    scattering evolution.

    Parameters:
    -> ram_s: raman response at current time slice
    -> dram_s: time derivative raman response at current time slice
    -> env_s: envelope at current time slice
    -> ram_rk4: auxiliary raman response array for RK4 integration
    -> dram_rk4: auxiliary raman response time derivative array for RK4 integration
    -> coef_ode1: Raman frequency coefficient for the first ODE term
    -> coef_ode2: Raman frequency coefficient for the second ODE term
    -> dt: time step
    -> dt_2: half time step
    -> dt_6: time step divided by 6

    Returns:
    -> float 1D-array: Raman response at next time slice
    """
    k1_ram, k1_dram = _set_scattering_operator(
        ram_s, dram_s, env_s, coef_ode1, coef_ode2
    )
    ram_rk4 = ram_s + dt_2 * k1_ram
    dram_rk4 = dram_s + dt_2 * k1_dram

    k2_ram, k2_dram = _set_scattering_operator(
        ram_rk4, dram_rk4, env_s, coef_ode1, coef_ode2
    )
    ram_rk4 = ram_s + dt_2 * k2_ram
    dram_rk4 = dram_s + dt_2 * k2_dram

    k3_ram, k3_dram = _set_scattering_operator(
        ram_rk4, dram_rk4, env_s, coef_ode1, coef_ode2
    )
    ram_rk4 = ram_s + dt * k3_ram
    dram_rk4 = dram_s + dt * k3_dram

    k4_ram, k4_dram = _set_scattering_operator(
        ram_rk4, dram_rk4, env_s, coef_ode1, coef_ode2
    )

    ram_s_rk4 = ram_s + dt_6 * (k1_ram + 2 * k2_ram + 2 * k3_ram + k4_ram)
    dram_s_rk4 = dram_s + dt_6 * (k1_dram + 2 * k2_dram + 2 * k3_dram + k4_dram)

    return ram_s_rk4, dram_s_rk4


@nb.njit
def _set_scattering_operator(ram_s, dram_s, env_s, coef_ode1, coef_ode2):
    """Set up the Raman scattering evolution terms.

    Parameters:
    -> ram_s: Raman response at current time slice
    -> dram_s: Raman response time derivative at current time slice
    -> env_s: envelope at current time slice
    -> coef_ode1: Raman frequency coefficient for the first ODE term
    -> coef_ode2: Raman frequency coefficient for the second ODE term

    Returns:
    -> float 1D-array: Raman scattering operators
    """
    diff_s = env_s - ram_s
    return dram_s, coef_ode1 * diff_s + coef_ode2 * dram_s
