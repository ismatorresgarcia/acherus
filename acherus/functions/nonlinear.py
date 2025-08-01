"""Envelope evolution module."""

import numpy as np

from .fourier import compute_fft, compute_ifft


def compute_nonlinear_rk4(
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
    dz,
):
    """
    Compute envelope propagation nonlinearities in FSS scheme
    for current propagation step using RK4.

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
    k1_env = _set_nlin(
        env_a,
        dens_a,
        ram_a,
        ion_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    env_1 = env_a + 0.5 * dz * k1_env

    k2_env = _set_nlin(
        env_1,
        dens_a,
        ram_a,
        ion_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    env_2 = env_a + 0.5 * dz * k2_env

    k3_env = _set_nlin(
        env_2,
        dens_a,
        ram_a,
        ion_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    env_3 = env_a + dz * k3_env

    k4_env = _set_nlin(
        env_3,
        dens_a,
        ram_a,
        ion_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )

    nlin_rk4 = dz * (k1_env + 2 * k2_env + 2 * k3_env + k4_env) / 6

    nlin_a[:] = nlin_rk4


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
    Compute envelope propagation nonlinearities in FCN scheme
    for current propagation step using RK4.

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


def _set_nlin(
    env_a,
    dens_a,
    ram_a,
    ion_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
):
    """
    Compute envelope propagation nonlinear terms in FSS scheme.

    Parameters
    ----------
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    dens_a : (M, N) array_like
        Electron density at current propagation step.
    ram_a : (M, N) array_like
        Raman response at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
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
    intensity = np.abs(env_a) ** 2

    # Compute nonlinear terms in the time domain
    nlin_p = pls_c_a * dens_a
    nlin_m = mpa_c_a * ion_a * (dens_n_a - dens_a) / intensity
    nlin_k = kerr_c_a * intensity
    nlin_r = ram_c_a * ram_a
    nlin = (nlin_p + nlin_m + nlin_k + nlin_r) * env_a

    return nlin


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
    Compute envelope propagation nonlinear terms in FCN scheme.

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
    intensity = np.abs(env_a) ** 2

    # Compute nonlinear terms in the frequency domain
    nlin_p = pls_c_a * compute_fft(dens_a * env_a)
    nlin_m = mpa_c_a * compute_fft(ion_a * (dens_n_a - dens_a) * env_a / intensity)
    nlin_k = kerr_c_a * compute_fft(env_a * intensity)
    nlin_r = ram_c_a * compute_fft(env_a * ram_a)
    nlin = nlin_p / shock_op_a + nlin_m + shock_op_a * (nlin_k + nlin_r)

    return nlin
