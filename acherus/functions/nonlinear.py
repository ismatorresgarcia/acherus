"""Helper module for computing nonlinear terms during propagation."""

import numpy as np

from .fft_backend import compute_fft


def compute_nonlinear_rsscn(
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
    inten_a,
    tmp_buf_t=None,
):
    """
    Compute envelope propagation nonlinearities in SSCN scheme
    for current propagation step using AB2 with Raman delayed
    response.

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
        Neutral density of the medium.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    ram_c_a : float
        Raman coefficient.
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    tmp_buf_t : (M, N) array_like, optional
        Temporary buffer in time domain.

    """
    _set_nlin_rsccn(
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
        inten_a,
        tmp_buf_t,
    )
    if stp_a != 1:
        nlin_a[:] = 1.5 * nlin_a - 0.5 * nlin_p

def _set_nlin_rsccn(
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
    inten_a,
    tmp_buf_t=None,
):
    """
    Compute RHS in rSSCN.

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
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    tmp_buf_t : (M, N) array_like, optional
        Temporary buffer in time domain.

    """
    if tmp_buf_t is None:
        tmp_buf_t = np.empty_like(env_a)

    np.multiply(dens_a, env_a, out=nlin_a)
    np.multiply(pls_c_a, nlin_a, out=nlin_a)

    np.subtract(dens_n_a, dens_a, out=tmp_buf_t)
    np.multiply(tmp_buf_t, ion_a, out=tmp_buf_t)
    np.divide(tmp_buf_t, inten_a + 1.0e-30, out=tmp_buf_t)
    np.multiply(tmp_buf_t, env_a, out=tmp_buf_t)
    np.multiply(mpa_c_a, tmp_buf_t, out=tmp_buf_t)
    np.add(nlin_a, tmp_buf_t, out=nlin_a)

    np.multiply(inten_a, env_a, out=tmp_buf_t)
    np.multiply(kerr_c_a, tmp_buf_t, out=tmp_buf_t)
    np.add(nlin_a, tmp_buf_t, out=nlin_a)

    np.multiply(ram_a, env_a, out=tmp_buf_t)
    np.multiply(ram_c_a, tmp_buf_t, out=tmp_buf_t)
    np.add(nlin_a, tmp_buf_t, out=nlin_a)

def compute_nonlinear_nrsscn(
    stp_a,
    env_a,
    dens_a,
    ion_a,
    nlin_a,
    nlin_p,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    inten_a,
    tmp_buf_t=None,
):
    """
    Compute envelope propagation nonlinearities in SSCN scheme
    for current propagation step using AB2 without Raman delayed
    response.

    Parameters
    ----------
    stp_a: integer
        Current propagation step index.
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    nlin_a : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    nlin_p : (M, N) array_like
        Previous step nonlinear terms.
    dens_n_a : float
        Neutral density of the medium.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    tmp_buf_t : (M, N) array_like, optional
        Temporary buffer in time domain.

    """
    _set_nlin_nrsccn(
        env_a,
        dens_a,
        ion_a,
        nlin_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        inten_a,
        tmp_buf_t,
    )
    if stp_a != 1:
        nlin_a[:] = 1.5 * nlin_a - 0.5 * nlin_p

def _set_nlin_nrsccn(
    env_a,
    dens_a,
    ion_a,
    nlin_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    inten_a,
    tmp_buf_t=None,
):
    """
    Compute RHS in nrSSCN.

    Parameters
    ----------
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    dens_a : (M, N) array_like
        Electron density at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    nlin_a : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    dens_n_a : float
        Neutral density of the medium.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    tmp_buf_t : (M, N) array_like, optional
        Temporary buffer in time domain.

    """
    if tmp_buf_t is None:
        tmp_buf_t = np.empty_like(env_a)

    np.multiply(dens_a, env_a, out=nlin_a)
    np.multiply(pls_c_a, nlin_a, out=nlin_a)

    np.subtract(dens_n_a, dens_a, out=tmp_buf_t)
    np.multiply(tmp_buf_t, ion_a, out=tmp_buf_t)
    np.divide(tmp_buf_t, inten_a + 1.0e-30, out=tmp_buf_t)
    np.multiply(tmp_buf_t, env_a, out=tmp_buf_t)
    np.multiply(mpa_c_a, tmp_buf_t, out=tmp_buf_t)
    np.add(nlin_a, tmp_buf_t, out=nlin_a)

    np.multiply(inten_a, env_a, out=tmp_buf_t)
    np.multiply(kerr_c_a, tmp_buf_t, out=tmp_buf_t)
    np.add(nlin_a, tmp_buf_t, out=nlin_a)

def compute_nonlinear_rfcn(
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
    inten_a,
    tmp_buf_t=None,
    tmp_buf_w=None,
):
    """
    Compute envelope propagation nonlinearities in FCN scheme
    for current propagation step using AB2 with Raman delayed
    response.

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
        Neutral density of the medium.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    ram_c_a : float
        Raman coefficient.
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    tmp_buf_t : (M, N) array_like, optional
        Temporary buffer in time domain.
    tmp_buf_w : (M, N) array_like, optional
        Temporary buffer in frequency domain.

    """
    _set_nlin_rfcn(
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
        inten_a,
        tmp_buf_t,
        tmp_buf_w,
    )
    if stp_a != 1:
        nlin_a[:] = 1.5 * nlin_a - 0.5 * nlin_p

def _set_nlin_rfcn(
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
    inten_a,
    tmp_buf_t=None,
    tmp_buf_w=None,
):
    """
    Compute RHS in rFCN.

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
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    tmp_buf_t : (M, N) array_like, optional
        Temporary buffer in time domain.
    tmp_buf_w : (M, N) array_like, optional
        Temporary buffer in frequency domain.

    """
    if tmp_buf_t is None:
        tmp_buf_t = np.empty_like(env_a)
    if tmp_buf_w is None:
        tmp_buf_w = np.empty_like(env_a)

    np.multiply(dens_a, env_a, out=tmp_buf_t)
    tmp_buf_w[:] = compute_fft(tmp_buf_t)
    np.multiply(pls_c_a, tmp_buf_w, out=nlin_a)

    np.subtract(dens_n_a, dens_a, out=tmp_buf_t)
    np.multiply(tmp_buf_t, ion_a, out=tmp_buf_t)
    np.multiply(tmp_buf_t, env_a, out=tmp_buf_t)
    np.divide(tmp_buf_t, inten_a + 1.0e-30, out=tmp_buf_t)
    tmp_buf_w[:] = compute_fft(tmp_buf_t)
    np.multiply(mpa_c_a, tmp_buf_w, out=tmp_buf_w)
    np.add(nlin_a, tmp_buf_w, out=nlin_a)

    np.multiply(inten_a, env_a, out=tmp_buf_t)
    tmp_buf_w[:] = compute_fft(tmp_buf_t)
    np.multiply(kerr_c_a, tmp_buf_w, out=tmp_buf_w)
    np.add(nlin_a, tmp_buf_w, out=nlin_a)

    np.multiply(ram_a, env_a, out=tmp_buf_t)
    tmp_buf_w[:] = compute_fft(tmp_buf_t)
    np.multiply(ram_c_a, tmp_buf_w, out=tmp_buf_w)
    np.add(nlin_a, tmp_buf_w, out=nlin_a)

def compute_nonlinear_nrfcn(
    stp_a,
    env_a,
    dens_a,
    ion_a,
    nlin_a,
    nlin_p,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    inten_a,
    tmp_buf_t=None,
    tmp_buf_w=None,
):
    """
    Compute envelope propagation nonlinearities in FCN scheme
    for current propagation step using AB2 without Raman delayed
    response.

    Parameters
    ----------
    stp_a: integer
        Current propagation step index.
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    nlin_a : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    nlin_p : (M, N) array_like
        Previous step nonlinear terms.
    dens_n_a : float
        Neutral density of the medium.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    tmp_buf_t : (M, N) array_like, optional
        Temporary buffer in time domain.
    tmp_buf_w : (M, N) array_like, optional
        Temporary buffer in frequency domain.

    """
    _set_nlin_nrfcn(
        env_a,
        dens_a,
        ion_a,
        nlin_a,
        dens_n_a,
        pls_c_a,
        mpa_c_a,
        kerr_c_a,
        inten_a,
        tmp_buf_t,
        tmp_buf_w,
    )
    if stp_a != 1:
        nlin_a[:] = 1.5 * nlin_a - 0.5 * nlin_p

def _set_nlin_nrfcn(
    env_a,
    dens_a,
    ion_a,
    nlin_a,
    dens_n_a,
    pls_c_a,
    mpa_c_a,
    kerr_c_a,
    inten_a,
    tmp_buf_t=None,
    tmp_buf_w=None,
):
    """
    Compute RHS in nrFCN.

    Parameters
    ----------
    env : (M, N) array_like
        Complex envelope at current propagation step.
    dens_a : (M, N) array_like
        Density at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    nlin_a : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    dens_n_a : float
        Neutral density of the medium.
    pls_c_a : float
        Plasma coefficient.
    mpa_c_a : float
        MPA coefficient.
    kerr_c_a : float
        Kerr coefficient.
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    tmp_buf_t : (M, N) array_like, optional
        Temporary buffer in time domain.
    tmp_buf_w : (M, N) array_like, optional
        Temporary buffer in frequency domain.

    """
    if tmp_buf_t is None:
        tmp_buf_t = np.empty_like(env_a)
    if tmp_buf_w is None:
        tmp_buf_w = np.empty_like(env_a)

    np.multiply(dens_a, env_a, out=tmp_buf_t)
    tmp_buf_w[:] = compute_fft(tmp_buf_t)
    np.multiply(pls_c_a, tmp_buf_w, out=nlin_a)

    np.subtract(dens_n_a, dens_a, out=tmp_buf_t)
    np.multiply(tmp_buf_t, ion_a, out=tmp_buf_t)
    np.multiply(tmp_buf_t, env_a, out=tmp_buf_t)
    np.divide(tmp_buf_t, inten_a + 1.0e-30, out=tmp_buf_t)
    tmp_buf_w[:] = compute_fft(tmp_buf_t)
    np.multiply(mpa_c_a, tmp_buf_w, out=tmp_buf_w)
    np.add(nlin_a, tmp_buf_w, out=nlin_a)

    np.multiply(inten_a, env_a, out=tmp_buf_t)
    tmp_buf_w[:] = compute_fft(tmp_buf_t)
    np.multiply(kerr_c_a, tmp_buf_w, out=tmp_buf_w)
    np.add(nlin_a, tmp_buf_w, out=nlin_a)
