"""Density evolution module."""

import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


@njit(parallel=True)
def compute_density_rk4(int_a, dens_a, ion_a, t_a, dens_n_a, dens_0_a, ava_c_a):
    """
    Compute electron density evolution for all time steps using RK4.

    Parameters
    ----------
    int_a : (M, N) array_like
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
    for ll in prange(n_t_a - 1):  # pylint: disable=not-an-iterable
        int_s = int_a[:, ll]
        int_s_n = int_a[:, ll + 1]
        dens_s = dens_a[:, ll]
        ion_s = ion_a[:, ll]
        ion_s_n = ion_a[:, ll + 1]
        int_trp_s = 0.5 * (int_s + int_s_n)  # linear interpolation
        ion_trp_s = 0.5 * (ion_s + ion_s_n)

        k1_dens = _set_density_rk4(dens_s, int_s, ion_s, dens_n_a, ava_c_a)
        dens_1 = dens_s + 0.5 * dt * k1_dens

        k2_dens = _set_density_rk4(dens_1, int_trp_s, ion_trp_s, dens_n_a, ava_c_a)
        dens_2 = dens_s + 0.5 * dt * k2_dens

        k3_dens = _set_density_rk4(dens_2, int_trp_s, ion_trp_s, dens_n_a, ava_c_a)
        dens_3 = dens_s + dt * k3_dens

        k4_dens = _set_density_rk4(dens_3, int_s_n, ion_s_n, dens_n_a, ava_c_a)

        dens_s_rk4 = dens_s + dt * (k1_dens + 2 * k2_dens + 2 * k3_dens + k4_dens) / 6

        dens_a[:, ll + 1] = dens_s_rk4


@njit
def _set_density_rk4(dens_s_a, int_s_a, ion_s_a, dens_n_a, ava_c_a):
    """
    Compute the electron density evolution terms using RK4.

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


def compute_density(
    int_a, dens_a, ion_a, r_a, t_a, dens_n_a, dens_0_a, ava_c_a, method_a
):
    """
    Compute electron density evolution for all time steps using ODE solver.

    Parameters
    ----------
    int_a : function
        Intensity function at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    r_a : (M,) array_like
        Radial coordinates grid.
    t_a : (N,) array_like
        Time coordinates grid.
    dens_n_a : float
        Neutral atom density of the chosen medium.
    dens_0_a : float
        Initial electron density of the chosen medium.
    ava_c_a : float
        Avalanche ionization coefficient.
    method_a : str
        Method for computing density evolution.

    """
    ion_trp = RegularGridInterpolator(
        (r_a, t_a), ion_a, method="linear", bounds_error=False, fill_value=None
    )

    sol = solve_ivp(
        _set_density,
        (t_a[0], t_a[-1]),
        np.full(len(r_a), dens_0_a),
        t_eval=t_a,
        method=method_a,
        args=(ion_trp, dens_n_a, int_a, ava_c_a, r_a),
        vectorized=True,
        rtol=1e-12,
        atol=1e-6,
    )
    dens_a[:] = sol.y.reshape((len(r_a), len(t_a)))


def _set_density(t, y, a, b, c, d, r):
    """
    Compute the electron density evolution terms using ODE solver.

    Parameters
    ----------
    t : float
        Time value.
    y : (M,) array_like
        Density at t.
    a : function
        Interpolated function for ionization rate.
    b : float
        Neutral density of the medium chosen.
    c : function
        Interpolated function for intensity.
    d : float
        Avalanche ionization coefficient.
    r : (M,) array_like
        Radial coordinates grid.

    """
    dens_s_a = y.reshape(len(r))
    ion_s_a = a((r, t))
    int_s_a = c((r, t))

    rate_ofi = ion_s_a * (b - dens_s_a)
    rate_ava = d * dens_s_a * int_s_a

    return rate_ofi + rate_ava
