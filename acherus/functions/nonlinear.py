"""Envelope evolution module."""

import numpy as np

from .fft_backend import fft, ifft


def compute_nonlinear_ab2(
    stp_a,
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    nlin_p,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
    dz,
):
    """
    Compute envelope propagation nonlinearities in FSS scheme
    for current propagation step using AB2.

    Parameters
    ----------
    stp_a: integer
        Current propagation step index.
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
    nlin_p : (M, N) array_like
        Previous step nonlinear terms.
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
    nlin_1 = _set_nlin(
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
    if stp_a == 1:
        nlin_ab2 = dz * nlin_1
    else:
        nlin_ab2 = dz * (1.5 * nlin_1 - 0.5 * nlin_p)

    nlin_a[:] = nlin_ab2

def compute_nonlinear_w_ab2(
    stp_a,
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    nlin_p,
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
    for current propagation step using AB2.

    Parameters
    ----------
    stp_a: integer
        Current propagation step.
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
    nlin_p : (M, N) array_like
        Previous step nonlinear terms.
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
    nlin_w_1 = _set_nlin_w(
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
    if stp_a == 1:
        nlin_ab2 = dz * nlin_w_1
    else:
        nlin_ab2 = dz * (1.5 * nlin_w_1 - 0.5 * nlin_p)

        nlin_a[:] = nlin_ab2

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
    nlin_1 = _set_nlin(
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
    env_1 = env_a + 0.5 * dz * nlin_1

    nlin_2 = _set_nlin(
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
    env_2 = env_a + 0.5 * dz * nlin_2

    nlin_3 = _set_nlin(
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
    env_3 = env_a + dz * nlin_3

    nlin_4 = _set_nlin(
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

    nlin_rk4 = dz * (nlin_1 + 2 * nlin_2 + 2 * nlin_3 + nlin_4) / 6

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
    nlin_1 = _set_nlin_w(
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
    env_1 = env_a + 0.5 * dz * ifft(nlin_1)

    nlin_2 = _set_nlin_w(
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
    env_2 = env_a + 0.5 * dz * ifft(nlin_2)

    nlin_3 = _set_nlin_w(
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
    env_3 = env_a + dz * ifft(nlin_3)

    nlin_4 = _set_nlin_w(
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

    nlin_rk4 = dz * (nlin_1 + 2 * nlin_2 + 2 * nlin_3 + nlin_4) / 6

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
    nlin_p = pls_c_a * fft(dens_a * env_a)
    nlin_m = mpa_c_a * fft(ion_a * (dens_n_a - dens_a) * env_a / intensity)
    nlin_k = kerr_c_a * fft(env_a * intensity)
    nlin_r = ram_c_a * fft(env_a * ram_a)
    nlin = nlin_p / shock_op_a + nlin_m + shock_op_a * (nlin_k + nlin_r)

    return nlin
