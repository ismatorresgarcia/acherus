"""Density evolution module."""

import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
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

    int_trp = 0.5 * (int_a[:, :-1] + int_a[:, 1:])
    ion_trp = 0.5 * (ion_a[:, :-1] + ion_a[:, 1:])

    for ll in prange(n_t_a - 1):  # pylint: disable=not-an-iterable
        dens_s = dens_a[:, ll]

        k1_dens = _set_density_rk4(dens_s, int_a[:, ll], ion_a[:, ll], dens_n_a, ava_c_a)
        dens_1 = dens_s + 0.5 * dt * k1_dens

        k2_dens = _set_density_rk4(dens_1, int_trp[:, ll], ion_trp[:, ll], dens_n_a, ava_c_a)
        dens_2 = dens_s + 0.5 * dt * k2_dens

        k3_dens = _set_density_rk4(dens_2, int_trp[:, ll], ion_trp[:, ll], dens_n_a, ava_c_a)
        dens_3 = dens_s + dt * k3_dens

        k4_dens = _set_density_rk4(dens_3, int_a[:, ll + 1], ion_a[:, ll + 1], dens_n_a, ava_c_a)

        dens_a[:, ll + 1] = dens_s + dt * (k1_dens + 2 * k2_dens + 2 * k3_dens + k4_dens) / 6


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
    inten_a, dens_a, ion_a, tmp_a, t_a, dens_n_a, dens_0_a, ava_c_a, method_a, first_step_a, rtol_a, atol_a
):
    """
    Compute electron density evolution for all time steps using ODE solver.

    Parameters
    ----------
    inten_a : function
        Intensity function at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    tmp_a : (M,) array_like
        Temporary array for intermediate results.
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
    ion_to_t = PchipInterpolator(t_a, ion_a, axis=1, extrapolate=True)
    inten_to_t = PchipInterpolator(t_a, inten_a, axis=1, extrapolate=True)

    k = len(tmp_a)
    sol = solve_ivp(
        _set_density,
        (t_a[0], t_a[-1]),
        np.full(k, dens_0_a),
        method=method_a,
        t_eval=t_a,
        args=(tmp_a, ion_to_t, inten_to_t, dens_n_a, ava_c_a),
        first_step=first_step_a,
        rtol=rtol_a,
        atol=atol_a,
        jac=_set_jacobian
    )
    dens_a[:] = sol.y


def _set_density(t, dens, tmp, ion_f, inten_f, dens_n, ava_c):
    """
    Compute the electron density evolution terms using ODE solver.

    Parameters
    ----------
    t : float
        Time value.
    dens : (M,) array_like
        Density at t.
    tmp : (M,) array_like
        Temporary array for intermediate results.
    ion_f : function
        Interpolated function for ionization rate.
    inten_f : function
        Interpolated function for intensity.
    dens_n : float
        Neutral density of the medium chosen.
    ava_c : float
        Avalanche ionization coefficient.

    """
    ion_s = ion_f(t)
    inten_s = inten_f(t)

    np.subtract(dens_n, dens, out=tmp)
    tmp *= ion_s
    tmp += ava_c * dens * inten_s

    return tmp


def _set_jacobian(t, dens, tmp, ion_f, inten_f, dens_n, ava_c):
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
    tmp : (M,) array_like
        Temporary array for intermediate results.
    ion_f : function
        Interpolated function for ionization rate.
    inten_f : function
        Interpolated function for intensity.
    dens_n : float
        Neutral density of the medium chosen.
    ava_c : float
        Avalanche ionization coefficient.

    """
    ion_s = ion_f(t)
    inten_s = inten_f(t)

    np.multiply(ava_c, inten_s, out=tmp)
    tmp -= ion_s

    return diags_array(tmp, offsets=0)
