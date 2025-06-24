"""Envelope evolution module."""

import numpy as np
from numba import njit, prange

from ..shared.fourier import compute_fft, compute_ifft


@njit(parallel=True)
def compute_nonlinear_rk4(
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    n_t_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
    dz,
):
    """
    Compute envelope propagation nonlinearities for current
    propagation step using RK4 for FSS solver.

    Parameters
    ----------
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ram_a : (M, N) array_like
        Raman response at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    nlin_a : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    n_t_a : integer
        Number of time nodes.
    dens_n_a : float
        Neutral density of the medium chosen.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    ram_c_a : float
        Raman coefficient.
    dz : float
        Axial step.

    """
    for ll in prange(n_t_a):  # pylint: disable=not-an-iterable
        env_s = env_a[:, ll]
        dens_s = dens_a[:, ll]
        ram_s = ram_a[:, ll]
        ion_s = ion_a[:, ll]

        k1_env = _set_nlin(
            env_s,
            dens_s,
            ram_s,
            ion_s,
            dens_n_a,
            pls_c_a,
            mpa_c_a,
            kerr_c_a,
            ram_c_a,
        )
        env_1 = env_s + 0.5 * dz * k1_env

        k2_env = _set_nlin(
            env_1,
            dens_s,
            ram_s,
            ion_s,
            dens_n_a,
            pls_c_a,
            mpa_c_a,
            kerr_c_a,
            ram_c_a,
        )
        env_2 = env_s + 0.5 * dz * k2_env

        k3_env = _set_nlin(
            env_2,
            dens_s,
            ram_s,
            ion_s,
            dens_n_a,
            pls_c_a,
            mpa_c_a,
            kerr_c_a,
            ram_c_a,
        )
        env_3 = env_s + dz * k3_env

        k4_env = _set_nlin(
            env_3,
            dens_s,
            ram_s,
            ion_s,
            dens_n_a,
            pls_c_a,
            mpa_c_a,
            kerr_c_a,
            ram_c_a,
        )

        nlin_s_rk4 = dz * (k1_env + 2 * k2_env + 2 * k3_env + k4_env) / 6

        nlin_a[:, ll] = nlin_s_rk4


@njit
def _set_nlin(
    env_s_a,
    dens_s_a,
    ram_s_a,
    ion_s_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
):
    """
    Compute envelope propagation nonlinear terms for FSS solver.

    Parameters
    ----------
    env_s_a : (M,) array_like
        Complex envelope at current time slice.
    dens_s_a : (M,) array_like
        Electron density at current time slice.
    ram_s_a : (M,) array_like
        Raman response at current time slice.
    ion_s_a : (M,) array_like
        Ionization rate at current time slice.
    dens_n_a : float
        Neutral density of the medium.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    ram_c_a : float
        Raman coefficient.

    Returns
    -------
    rhs : (M,) ndarray
        Complex envelope nonlinearities at next time slice.
        M is the number of radial nodes.

    """
    intensity_s = np.abs(env_s_a) ** 2

    # Compute nonlinear terms in the space domain
    nlin_p_s = pls_c_a * dens_s_a
    nlin_m_s = mpa_c_a * ion_s_a * (dens_n_a - dens_s_a) / intensity_s
    nlin_k_s = kerr_c_a * intensity_s
    nlin_r_s = ram_c_a * ram_s_a
    nlin_s = (nlin_p_s + nlin_m_s + nlin_k_s + nlin_r_s) * env_s_a

    return nlin_s


def compute_nonlinear_w_rk4(
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    shock_op_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
    dz,
):
    """
    Compute envelope propagation nonlinearities for current
    propagation step using RK4 for FCN solver.

    Parameters
    ----------
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ram_a : (M, N) array_like
        Raman response at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    nlin_a : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    shock_op_a : (N,) array_like
        Self-steepening operator acting on nonlinearities.
    dens_n_a : float
        Neutral density of the medium.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    ram_c_a : float
        Raman coefficient.
    dz : float
        Axial step

    """
    k1_env = _set_nlin_w(
        env_a,
        dens_a,
        ram_a,
        ion_a,
        shock_op_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    env_1 = env_a + 0.5 * dz * compute_ifft(k1_env)

    k2_env = _set_nlin_w(
        env_1,
        dens_a,
        ram_a,
        ion_a,
        shock_op_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    env_2 = env_a + 0.5 * dz * compute_ifft(k2_env)

    k3_env = _set_nlin_w(
        env_2,
        dens_a,
        ram_a,
        ion_a,
        shock_op_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    env_3 = env_a + dz * compute_ifft(k3_env)

    k4_env = _set_nlin_w(
        env_3,
        dens_a,
        ram_a,
        ion_a,
        shock_op_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )

    nlin_rk4 = dz * (k1_env + 2 * k2_env + 2 * k3_env + k4_env) / 6

    nlin_a[:] = nlin_rk4


def _set_nlin_w(
    env_a,
    dens_a,
    ram_a,
    ion_a,
    shock_op_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
):
    """
    Compute envelope propagation nonlinear terms for FCN solver.

    Parameters
    ----------
    env : (M, N) array_like
        Complex envelope at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ram_a : (M, N) array_like
        Raman response at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    shock_op_a : (N,)
        Self-steepening operator acting on nonlinearities.
    dens_n_a : float
        Neutral density of the medium.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    ram_c_a : float
        Raman coefficient.

    Returns
    -------
    nlin : (M, N) ndarray
        Complex envelope nonlinearities. M is the number
        of radial nodes and N the number of time nodes.

    """
    intensity_s = np.abs(env_a) ** 2

    # Compute nonlinear terms in the frequency domain
    nlin_p = pls_c_a * compute_fft(dens_a * env_a) / shock_op_a
    nlin_m = mpa_c_a * compute_fft(ion_a * (dens_n_a - dens_a) * env_a / intensity_s)
    nlin_k = shock_op_a * kerr_c_a * compute_fft(env_a * intensity_s)
    nlin_r = shock_op_a * ram_c_a * compute_fft(env_a * ram_a)
    nlin = nlin_p + nlin_m + nlin_k + nlin_r

    return nlin
