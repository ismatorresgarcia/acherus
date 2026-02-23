"""Helper module for electron density evolution ODE solution."""

import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline


@njit(parallel=True)
def compute_density_rk4(inten_a, dens_a, ion_a, t_a, dens_n_a, dens_0_a, ava_c_a):
    """
    Compute electron density evolution for all time steps using RK4.

    Parameters
    ----------
    inten_a : (M, N) array_like
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
    n_r, n_t = dens_a.shape
    dt = np.diff(t_a)
    dens_a[:, 0] = dens_0_a

    for nn in prange(n_r):  # pylint: disable=not-an-iterable
        inten_nn = inten_a[nn]
        ion_nn = ion_a[nn]
        dens_nn = dens_a[nn]

        dens_p = dens_0_a

        for ll in range(n_t - 1):
            dt_ll = dt[ll]

            inten_0 = inten_nn[ll]
            inten_1 = inten_nn[ll + 1]
            ion_0 = ion_nn[ll]
            ion_1 = ion_nn[ll + 1]

            inten_mid = 0.5 * (inten_0 + inten_1)
            ion_mid = 0.5 * (ion_0 + ion_1)

            k1 = _rhs_rk4(dens_p, inten_0, ion_0, dens_n_a, ava_c_a)
            dens_1 = dens_p + 0.5 * dt_ll * k1

            k2 = _rhs_rk4(dens_1, inten_mid, ion_mid, dens_n_a, ava_c_a)
            dens_2 = dens_p + 0.5 * dt_ll * k2

            k3 = _rhs_rk4(dens_2, inten_mid, ion_mid, dens_n_a, ava_c_a)
            dens_3 = dens_p + dt_ll * k3

            k4 = _rhs_rk4(dens_3, inten_1, ion_1, dens_n_a, ava_c_a)

            dens_p += dt_ll * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            dens_nn[ll + 1] = dens_p


@njit(inline="always")
def _rhs_rk4(dens_s_a, int_s_a, ion_s_a, dens_n_a, ava_c_a):
    """RHS of the electron density evolution ODE for RK4."""
    return ion_s_a * (dens_n_a - dens_s_a) + ava_c_a * dens_s_a * int_s_a


def compute_density_nr(
    inten_a,
    dens_a,
    ion_a,
    t_a,
    dens_n_a,
    dens_0_a,
    ava_c_a,
    method_a,
    rtol_a,
    atol_a,
    rhs_buf,
    tmp_buf,
):
    """
    Compute electron density evolution ODE for all time steps with SciPy's 'solve_ivp'.
    No recombination term is included in the ODE.

    Parameters
    ----------
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    t_a : (N,) array_like
        Time coordinates grid.
    dens_n_a : float
        Neutral atom density of the chosen medium.
    dens_0_a : (M,) array_like
        Initial electron density of the chosen medium.
    ava_c_a : float
        Avalanche ionization coefficient.
    method_a : str
        Method for computing density evolution.
    rtol_a : float
        Relative tolerance for the ODE solver.
    atol_a : float
        Absolute tolerance for the ODE solver.
    rhs_buf : (M,) array_like
        Buffer array for RHS computations.
    tmp_buf : (M,) array_like
        Temporary buffer array for RHS computations.

    """
    ion_to_t = make_interp_spline(t_a, ion_a, k=1, axis=1)
    inten_to_t = make_interp_spline(t_a, inten_a, k=1, axis=1)

    def _set_density(t, dens):
        """RHS of the electron density evolution ODE for SciPy."""
        ion_s = ion_to_t(t)
        inten_s = inten_to_t(t)

        np.subtract(dens_n_a, dens, out=tmp_buf)
        np.multiply(ion_s, tmp_buf, out=rhs_buf)
        np.multiply(dens, inten_s, out=tmp_buf)
        np.multiply(ava_c_a, tmp_buf, out=tmp_buf)
        np.add(rhs_buf, tmp_buf, out=rhs_buf)
        return rhs_buf

    sol = solve_ivp(
        _set_density,
        t_span=(t_a[0], t_a[-1]),
        y0=dens_0_a,
        method=method_a,
        t_eval=t_a,
        rtol=rtol_a,
        atol=atol_a,
    )
    dens_a[:] = sol.y


def compute_density_r(
    inten_a,
    dens_a,
    ion_a,
    t_a,
    dens_n_a,
    dens_0_a,
    ava_c_a,
    rec_c_a,
    method_a,
    rtol_a,
    atol_a,
    rhs_buf,
    tmp_buf,
):
    """
    Compute electron density evolution ODE for all time steps with SciPy's 'solve_ivp'.
    Recombination is included as: rec_c_a * dens_a**2.
    """
    ion_to_t = make_interp_spline(t_a, ion_a, k=1, axis=1)
    inten_to_t = make_interp_spline(t_a, inten_a, k=1, axis=1)

    def _set_density(t, dens):
        """RHS of the electron density evolution ODE with recombination."""
        ion_s = ion_to_t(t)
        inten_s = inten_to_t(t)

        np.subtract(dens_n_a, dens, out=tmp_buf)
        np.multiply(ion_s, tmp_buf, out=rhs_buf)
        np.multiply(dens, inten_s, out=tmp_buf)
        np.multiply(ava_c_a, tmp_buf, out=tmp_buf)
        np.add(rhs_buf, tmp_buf, out=rhs_buf)
        np.power(dens, 2, out=tmp_buf)
        np.multiply(rec_c_a, tmp_buf, out=tmp_buf)
        np.subtract(rhs_buf, tmp_buf, out=rhs_buf)
        return rhs_buf

    sol = solve_ivp(
        _set_density,
        t_span=(t_a[0], t_a[-1]),
        y0=dens_0_a,
        method=method_a,
        t_eval=t_a,
        rtol=rtol_a,
        atol=atol_a,
    )
    dens_a[:] = sol.y
