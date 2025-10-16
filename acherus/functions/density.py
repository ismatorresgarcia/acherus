"""Density evolution module."""

import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags_array


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
    int_a, dens_a, ion_a, r_a, t_a, dens_n_a, dens_0_a, ava_c_a, method_a, first_step_a, rtol_a, atol_a
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
    first_step_a : float
        Initial step size for the ODE solver.
    rtol_a : float
        Relative tolerance for the ODE solver.
    atol_a : float
        Absolute tolerance for the ODE solver.

    """
    ion_f_a = RegularGridInterpolator(
        (r_a, t_a), ion_a, method="linear", bounds_error=False, fill_value=None
    )

    k = len(r_a)
    sol = solve_ivp(
        _set_density,
        (t_a[0], t_a[-1]),
        np.full(k, dens_0_a),
        method=method_a,
        t_eval=t_a,
        args=(ion_f_a, dens_n_a, int_a, ava_c_a, r_a),
        first_step=first_step_a,
        rtol=rtol_a,
        atol=atol_a,
        jac=_set_jacobian
    )
    dens_a[:] = sol.y.reshape((k, len(t_a)))


def _set_density(t, dens, ion_f, dens_n, intens_f, ava_c, r):
    """
    Compute the electron density evolution terms using ODE solver.

    Parameters
    ----------
    t : float
        Time value.
    dens : (M,) array_like
        Density at t.
    ion_f : function
        Interpolated function for ionization rate.
    dens_n : float
        Neutral density of the medium chosen.
    intens_f : function
        Interpolated function for intensity.
    ava_c : float
        Avalanche ionization coefficient.
    r : (M,) array_like
        Radial coordinates grid.

    """
    ion_s = ion_f((r, t))
    int_s = intens_f((r, t))

    ofi_rate = ion_s * (dens_n - dens)
    ava_rate = ava_c * dens * int_s

    return ofi_rate + ava_rate


def _set_jacobian(t, dens, ion_f, dens_n, intens_f, ava_c, r):
    """
    Compute the electron density evolution Jacobian matrix for
    Radau and BDF implicit methods. Better in comparison with
    the vectorized approach, which relies on finite-difference
    approximations.

    Parameters
    ----------
    t : float
        Time value.
    dens : (M,) array_like
        Density at t.
    ion_f : function
        Interpolated function for ionization rate.
    dens_n : float
        Neutral density of the medium chosen.
    intens_f : function
        Interpolated function for intensity.
    ava_c : float
        Avalanche ionization coefficient.
    r : (M,) array_like
        Radial coordinates grid.

    """
    ion_s = ion_f((r, t))
    int_s = intens_f((r, t))

    diag = -ion_s + ava_c * int_s

    return diags_array(diag, offsets=0)
