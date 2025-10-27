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
    tmp_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
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
    tmp_a : (M, N) array_like
        Pre-allocated array for intermediate results.
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

    """
    nlin_1 = _set_nlin(
        env_a,
        dens_a,
        ram_a,
        ion_a,
        nlin_a,
        tmp_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    if stp_a == 1:
        nlin_a[:] = nlin_1
    else:
        nlin_a[:] = nlin_1
        nlin_a *= 3
        nlin_a -= nlin_p
        nlin_a *= 0.5

def compute_nonlinear_w_ab2(
    stp_a,
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    nlin_p,
    tmp_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
):
    """
    Compute envelope propagation nonlinearities in FCN scheme
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
    tmp_a : (M, N) array_like
        Pre-allocated array for intermediate results.
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

    """
    nlin_1 = _set_nlin_w(
        env_a,
        dens_a,
        ram_a,
        ion_a,
        nlin_a,
        tmp_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )
    if stp_a == 1:
        nlin_a[:] = nlin_1
    else:
        nlin_a[:] = nlin_1
        nlin_a *= 3
        nlin_a -= nlin_p
        nlin_a *= 0.5

def compute_nonlinear_rk4(
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    tmp_a,
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
    tmp_a : (M, N) array_like
        Pre-allocated array for intermediate results.
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
        nlin_a,
        tmp_a,
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
        nlin_a,
        tmp_a,
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
        nlin_a,
        tmp_a,
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
        nlin_a,
        tmp_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )

    np.add(nlin_1, nlin_4, out=nlin_a)
    nlin_a += nlin_2
    nlin_a += nlin_2
    nlin_a += nlin_3
    nlin_a += nlin_3
    nlin_a *= (1/6)


def compute_nonlinear_w_rk4(
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    tmp_a,
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
    tmp_a : (M, N) array_like
        Pre-allocated array for intermediate results.
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
        nlin_a,
        tmp_a,
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
        nlin_a,
        tmp_a,
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
        nlin_a,
        tmp_a,
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
        nlin_a,
        tmp_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        ram_c_a,
    )

    np.add(nlin_1, nlin_4, out=nlin_a)
    nlin_a += 2 * nlin_2
    nlin_a += 2 * nlin_3
    nlin_a *= (1/6)


def _set_nlin(
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    tmp_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    ram_c_a,
):
    """
    Compute envelope propagation nonlinear terms in SSCN scheme.

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
    nlin_a : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    tmp_a : (M, N) array_like
        Pre-allocated array for intermediate results.
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
    nlin_a.fill(0.0)

    inten = np.abs(env_a) ** 2

    np.multiply(pls_c_a, dens_a, out=tmp_a)
    nlin_a += tmp_a

    np.subtract(dens_n_a, dens_a, out=tmp_a)
    np.multiply(ion_a, tmp_a, out=tmp_a)
    np.divide(tmp_a, inten, out=tmp_a)
    tmp_a *= mpa_c_a
    nlin_a += tmp_a

    np.multiply(kerr_c_a, inten, out=tmp_a)
    nlin_a += tmp_a

    np.multiply(ram_c_a, ram_a, out=tmp_a)
    nlin_a += tmp_a

    np.multiply(nlin_a, env_a, out=nlin_a)

    return nlin_a


def _set_nlin_w(
    env_a,
    dens_a,
    ram_a,
    ion_a,
    nlin_a,
    tmp_a,
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
    nlin_a : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    tmp_a : (M, N) array_like
        Pre-allocated array for intermediate results.
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
    nlin_a.fill(0.0)

    inten = np.abs(env_a) ** 2

    np.multiply(dens_a, env_a, out=tmp_a)
    tmp_a[:] = fft(tmp_a)
    tmp_a *= pls_c_a
    nlin_a += tmp_a

    np.subtract(dens_n_a, dens_a, out=tmp_a)
    np.multiply(ion_a, tmp_a, out=tmp_a)
    np.multiply(env_a, tmp_a, out=tmp_a)
    np.divide(tmp_a, inten, out=tmp_a)
    tmp_a[:] = fft(tmp_a)
    tmp_a *= mpa_c_a
    nlin_a += tmp_a

    np.multiply(inten, env_a, out=tmp_a)
    tmp_a[:] = fft(tmp_a)
    tmp_a *= kerr_c_a
    nlin_a += tmp_a

    np.multiply(ram_a, env_a, out=tmp_a)
    tmp_a[:] = fft(tmp_a)
    tmp_a *= ram_c_a
    nlin_a += tmp_a

    return nlin_a
