"""Raman scattering module."""

import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp


@njit(parallel=True)
def compute_raman_rk4(ram_a, dram_a, int_a, t_a, ram_c1_a, ram_c2_a):
    """
    Compute molecular Raman scattering delayed response for all time steps
    using RK4.

    Parameters
    ----------
    ram_a : (M, N) array_like
        Raman response at all time slices.
    dram_a : (M, N) array_like
        Raman response time derivative at all time slices.
    int_a : (M, N) array_like
        Intensity at all time slices.
    t_a : (N,) array_like
        Time coordinates grid.
    ram_c1_a : float
        Raman frequency coefficient for the first ODE term.
    ram_c2_a : float
        Raman frequency coefficient for the second ODE term.

    """
    n_t_a = len(t_a)
    dt = t_a[1] - t_a[0]
    ram_a[:, 0], dram_a[:, 0] = 0, 0
    for ll in prange(n_t_a - 1):  # pylint: disable=not-an-iterable
        ram_s = ram_a[:, ll]
        dram_s = dram_a[:, ll]
        int_s = int_a[:, ll]
        int_s_n = int_a[:, ll + 1]
        int_trp_s = 0.5 * (int_s + int_s_n)  # linear interpolation

        k1_ram, k1_dram = _set_raman_rk4(ram_s, dram_s, int_s, ram_c1_a, ram_c2_a)
        ram_1 = ram_s + 0.5 * dt * k1_ram
        dram_1 = dram_s + 0.5 * dt * k1_dram

        k2_ram, k2_dram = _set_raman_rk4(ram_1, dram_1, int_trp_s, ram_c1_a, ram_c2_a)
        ram_2 = ram_s + 0.5 * dt * k2_ram
        dram_2 = dram_s + 0.5 * dt * k2_dram

        k3_ram, k3_dram = _set_raman_rk4(ram_2, dram_2, int_trp_s, ram_c1_a, ram_c2_a)
        ram_3 = ram_s + dt * k3_ram
        dram_3 = dram_s + dt * k3_dram

        k4_ram, k4_dram = _set_raman_rk4(ram_3, dram_3, int_s_n, ram_c1_a, ram_c2_a)

        ram_s_rk4 = ram_s + dt * (k1_ram + 2 * k2_ram + 2 * k3_ram + k4_ram) / 6
        dram_s_rk4 = dram_s + dt * (k1_dram + 2 * k2_dram + 2 * k3_dram + k4_dram) / 6

        ram_a[:, ll + 1] = ram_s_rk4
        dram_a[:, ll + 1] = dram_s_rk4


@njit
def _set_raman_rk4(ram_s_a, dram_s_a, int_s_a, ram_c1_a, ram_c2_a):
    """
    Compute the Raman scattering evolution terms using RK4.

    Parameters
    ----------
    ram_s_a : (M,) array_like
        Raman response at current time slice.
    dram_s_a : (M,) array_like
        Raman response time derivative at current time slice.
    int_s_a : (M,) array_like
        Intensity at current time slice.
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
    return dram_s_a, ram_c1_a * (int_s_a - ram_s_a) + ram_c2_a * dram_s_a


def compute_raman(int_a, ram_a, r_a, t_a, ram_c1_a, ram_c2_a, method_a):
    """
    Compute molecular Raman scattering delayed response for all time steps
    using ODE solver.

    Parameters
    ----------
    int_a : function
        Intensity function at current propagation step.
    ram_a : (M, N) array_like
        Raman response at all time slices.
    r_a : (M,) array_like
        Radial coordinates grid.
    t_a : (N,) array_like
        Time coordinates grid.
    ram_c1_a : float
        Raman frequency coefficient for the first ODE term.
    ram_c2_a : float
        Raman frequency coefficient for the second ODE term.
    method_a : str
        Method for computing raman response.

    """
    sol = solve_ivp(
        _set_raman,
        (t_a[0], t_a[-1]),
        np.zeros((2, len(r_a))).flatten(),
        t_eval=t_a,
        method=method_a,
        args=(ram_c1_a, ram_c2_a, int_a, r_a),
        vectorized=True,
        rtol=1e-3,
        atol=1e-6,
    )
    ram_a[:] = sol.y.reshape((2, len(r_a), len(t_a)))[1]


def _set_raman(t, y, a, b, c, r):
    """
    Compute the Raman response terms for ODE solver.

    Parameters
    ----------
    t : float
        Time value.
    y : (2, M) array_like
        Raman response vector at t.
    a : float
        Raman frequency coefficient for the first ODE term.
    b : float
        Raman frequency coefficient for the second ODE term.
    c : function
        Interpolated function for intensity.
    r : (M,) array_like
        Radial coordinates grid.

    """
    k = len(r)
    m = y.shape[1]
    raman_s_a = y.reshape(2, k, m)
    y0, y1 = raman_s_a
    int_s_a = c((r, t)).reshape(k, 1)
    dy0, dy1 = a * (int_s_a - y1) + b * y0, y0

    return np.concatenate([dy0, dy1], axis=0)
