"""Envelope evolution module."""

import numpy as np
from numba import njit, prange

from ..shared.fourier import compute_fft, compute_ifft


@njit(parallel=True)
def compute_nlin_rk4(
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
    for ll in prange(n_t_a):
        env_s = env_a[:, ll]
        dens_s = dens_a[:, ll]
        ram_s = ram_a[:, ll]
        ion_s = ion_a[:, ll]

        nlin_s_rk4 = _rk4_envelope_step(
            env_s,
            dens_s,
            ram_s,
            ion_s,
            dens_n_a,
            pls_c_a,
            mpa_c_a,
            kerr_c_a,
            ram_c_a,
            dz,
        )
        nlin_a[:, ll] = nlin_s_rk4


@njit
def _rk4_envelope_step(
    env_s_a,
    dens_s_a,
    ram_s_a,
    ion_s_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
    dz,
):
    """
    Compute one RK4 integration step for FSS envelope propagation.

    Parameters
    ----------
    env_s_a : (M,) array_like
        Complex envelope at current time slice.
    dens_s_a : (M,) array_like
        Density at current time slice.
    ram_s_a : (M,) array_like
        Raman response at current time slice.
    ion_s_a : (M,) array_like
        Ionization rate at current time slice.
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
    dz : Axial step.

    Returns
    -------
    nlin_s_rk4 : (M,) ndarray
        Complex envelope nonlinearities RK4 integration
        at next time slice. M is the number of radial nodes.

    """
    k1_env = _set_envelope_operator(
        env_s_a,
        dens_s_a,
        ram_s_a,
        ion_s_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    env_1 = env_s_a + 0.5 * dz * k1_env

    k2_env = _set_envelope_operator(
        env_1,
        dens_s_a,
        ram_s_a,
        ion_s_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    env_2 = env_s_a + 0.5 * dz * k2_env

    k3_env = _set_envelope_operator(
        env_2,
        dens_s_a,
        ram_s_a,
        ion_s_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    env_3 = env_s_a + dz * k3_env

    k4_env = _set_envelope_operator(
        env_3,
        dens_s_a,
        ram_s_a,
        ion_s_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )

    nlin_s_rk4 = dz * (k1_env + 2 * k2_env + 2 * k3_env + k4_env) / 6

    return nlin_s_rk4


@njit
def _set_envelope_operator(
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

    nlin_s = env_s_a * (
        pls_c_a * dens_s_a
        + mpa_c_a * ion_s_a * (dens_n_a - dens_s_a) / intensity_s
        + kerr_c_a * intensity_s
        + ram_c_a * ram_s_a
    )

    return nlin_s


def compute_nlin_rk4_w(
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
    nlin_a[:] = _rk4_envelope_step_w(
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
        dz,
    )


def _rk4_envelope_step_w(
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
    dz,
):
    """
    Compute one RK4 integration step for FCN envelope propagation.

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
    dz : float
        Axial step.

    Returns
    -------
    nlin_rk4 : (M, N) ndarray
        Complex envelope nonlinearities RK4 integration
        at current propagation step. M is the number of radial nodes
        and N the number of time nodes.

    """
    k1_env = _set_envelope_operator_w(
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

    k2_env = _set_envelope_operator_w(
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

    k3_env = _set_envelope_operator_w(
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

    k4_env = _set_envelope_operator_w(
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

    return nlin_rk4


def _set_envelope_operator_w(
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
    # Compute nonlinear terms in the frequency domain
    nlin_p = pls_c_a * compute_fft(dens_a * env_a) / shock_op_a
    nlin_m = mpa_c_a * compute_fft(
        ion_a * (dens_n_a - dens_a) * env_a / np.abs(env_a) ** 2
    )
    nlin_k = shock_op_a * kerr_c_a * compute_fft(env_a * np.abs(env_a) ** 2)
    nlin_r = shock_op_a * ram_c_a * compute_fft(env_a * ram_a)

    # Add nonlinear terms
    nlin = nlin_p + nlin_m + nlin_k + nlin_r

    return nlin
