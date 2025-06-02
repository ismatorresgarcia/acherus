"""Envelope evolution module."""

import numba as nb
import numpy as np
from scipy.fft import fft, ifft


def compute_fft(data):
    """Transform data from time domain to frequency domain using FFT."""
    return fft(data, axis=1, workers=-1)


def compute_ifft(data):
    """Transform data from frequency domain to time domain using IFFT."""
    return ifft(data, axis=1, workers=-1)


@nb.njit(parallel=True)
def compute_nlin_rk4(
    env,
    dens,
    ram,
    ion_rate,
    nlin,
    env_rk4,
    n_t,
    dens_n,
    coef_p,
    coef_m,
    coef_k,
    coef_r,
    dz,
):
    """
    Compute envelope propagation nonlinearities for current
    propagation step using RK4 for FSS solver.

    Parameters
    ----------
    env : (M, N) array_like
        Complex envelope at current propagation step.
    dens : (M, N) array_like
        Density at current propagation step.
    ram : (M, N) array_like
        Raman response at current propagation step.
    ion_rate : (M, N) array_like
        Ionization rate at current propagation step.
    nlin : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    env_rk4 : (N,) array_like
        Auxiliary envelope array for RK4 integration.
    n_t : integer
        Number of time nodes.
    dens_n : float
        Neutral density of the medium chosen.
    coef_p : float
        Plasma coefficient.
    coef_m : float
        MPA coefficient.
    coef_k : float
        Kerr coefficient.
    coef_r : float
        Raman coefficient.
    dz : float
        Axial step.

    """
    # pylint: disable=not-an-iterable
    for ll in nb.prange(n_t):
        env_s = env[:, ll]
        dens_s = dens[:, ll]
        ram_s = ram[:, ll]
        ion_rate_s = ion_rate[:, ll]

        nlin_s_rk4 = _rk4_envelope_step(
            env_s,
            dens_s,
            ram_s,
            ion_rate_s,
            env_rk4,
            dens_n,
            coef_p,
            coef_m,
            coef_k,
            coef_r,
            dz,
        )

        nlin[:, ll] = nlin_s_rk4


@nb.njit
def _rk4_envelope_step(
    env_s,
    dens_s,
    ram_s,
    ion_rate_s,
    env_rk4,
    dens_n,
    coef_p,
    coef_m,
    coef_k,
    coef_r,
    dz,
):
    """
    Compute one RK4 integration step for FSS envelope propagation.

    Parameters
    ----------
    env_s : (M,) array_like
        Complex envelope at current time slice.
    dens_s : (M,) array_like
        Density at current time slice.
    ram_s : (M,) array_like
        Raman response at current time slice.
    ion_rate_s : (M,) array_like
        Ionization rate at current time slice.
    env_rk4 : (N,) array_like
        Auxiliary envelope array for RK4 integration.
    dens_n : float
        Neutral density of the medium chosen.
    coef_p : float
        Plasma coefficient.
    coef_m : float
        MPA coefficient.
    coef_k : float
        Kerr coefficient.
    coef_r : float
        Raman coefficient.
    dz : Axial step.

    Returns
    -------
    nlin_s_rk4 : (M,) ndarray
        Complex envelope nonlinearities RK4 integration
        at next time slice. M is the number of radial nodes.

    """
    k1_env = _set_envelope_operator(
        env_s, dens_s, ram_s, ion_rate_s, dens_n, coef_p, coef_m, coef_k, coef_r
    )
    env_rk4 = env_s + 0.5 * dz * k1_env

    k2_env = _set_envelope_operator(
        env_rk4, dens_s, ram_s, ion_rate_s, dens_n, coef_p, coef_m, coef_k, coef_r
    )
    env_rk4 = env_s + 0.5 * dz * k2_env

    k3_env = _set_envelope_operator(
        env_rk4, dens_s, ram_s, ion_rate_s, dens_n, coef_p, coef_m, coef_k, coef_r
    )
    env_rk4 = env_s + dz * k3_env

    k4_env = _set_envelope_operator(
        env_rk4, dens_s, ram_s, ion_rate_s, dens_n, coef_p, coef_m, coef_k, coef_r
    )

    nlin_s_rk4 = dz * (k1_env + 2 * k2_env + 2 * k3_env + k4_env) / 6

    return nlin_s_rk4


@nb.njit
def _set_envelope_operator(
    env_s, dens_s, ram_s, ion_rate_s, dens_n, coef_p, coef_m, coef_k, coef_r
):
    """
    Compute envelope propagation nonlinear terms for FSS solver.

    Parameters
    ----------
    env_s : (M,) array_like
        Complex envelope at current time slice.
    dens_s : (M,) array_like
        Electron density at current time slice.
    ram_s : (M,) array_like
        Raman response at current time slice.
    ion_rate_s : (M,) array_like
        Ionization rate at current time slice.
    dens_n : float
        Neutral density of the medium.
    coef_p : float
        Plasma coefficient.
    coef_m : float
        MPA coefficient.
    coef_k : float
        Kerr coefficient.
    coef_r : float
        Raman coefficient.

    Returns
    -------
    rhs : (M,) ndarray
        Complex envelope nonlinearities at next time slice.
        M is the number of radial nodes.

    """
    intensity_s = np.abs(env_s) ** 2
    dens_s_sat = dens_n - dens_s

    nlin_s = env_s * (
        coef_p * dens_s
        + coef_m * ion_rate_s * dens_s_sat / intensity_s
        + coef_k * intensity_s
        + coef_r * ram_s
    )

    return nlin_s


def compute_nlin_rk4_frequency(
    env,
    dens,
    ram,
    ion_rate,
    nlin,
    env_rk4,
    steep_op,
    dens_n,
    coef_p,
    coef_m,
    coef_k,
    coef_r,
    dz,
):
    """
    Compute envelope propagation nonlinearities for current
    propagation step using RK4 for FCN solver.

    Parameters
    ----------
    env : (M, N) array_like
        Complex envelope at current propagation step.
    dens : (M, N) array_like
        Density at current propagation step.
    ram : (M, N) array_like
        Raman response at current propagation step.
    ion_rate : (M, N) array_like
        Ionization rate at current propagation step.
    env_rk4 : (N,) array_like
        Auxiliary envelope array for RK4 integration.
    nlin : (M, N) array_like
        Pre-allocated array for the nonlinear terms.
    steep_op : (N,) array_like
        Self-steepening operator for diffraction.
    dens_n : float
        Neutral density of the medium.
    coef_p : float
        Plasma coefficient.
    coef_m : float
        MPA coefficient.
    coef_k : float
        Kerr coefficient.
    coef_r : float
        Raman coefficient.
    dz : float
        Axial step

    """
    nlin[:] = _rk4_envelope_step_frequency(
        env,
        dens,
        ram,
        ion_rate,
        steep_op,
        env_rk4,
        dens_n,
        coef_p,
        coef_m,
        coef_k,
        coef_r,
        dz,
    )


def _rk4_envelope_step_frequency(
    env,
    dens,
    ram,
    ion_rate,
    steep_op,
    env_rk4,
    dens_n,
    coef_p,
    coef_m,
    coef_k,
    coef_r,
    dz,
):
    """
    Compute one RK4 integration step for FCN envelope propagation.

    Parameters
    ----------
    env : (M, N) array_like
        Complex envelope at current propagation step.
    dens: (M, N) array_like
        Density at current propagation step.
    ram : (M, N) array_like
        Raman response at current propagation step.
    ion_rate : (M, N) array_like
        Ionization rate at current propagation step.
    steep_op : (N,)
        Self-steepening operator for diffraction.
    env_rk4 : (N,) array_like
        Auxiliary envelope array for RK4 integration.
    dens_n : float
        Neutral density of the medium.
    coef_p : float
        Plasma coefficient.
    coef_m : float
        MPA coefficient.
    coef_k : float
        Kerr coefficient.
    coef_r : float
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
    k1_env = _set_envelope_operator_frequency(
        env, dens, ram, ion_rate, steep_op, dens_n, coef_p, coef_m, coef_k, coef_r
    )
    env_rk4 = env + 0.5 * dz * compute_ifft(k1_env)

    k2_env = _set_envelope_operator_frequency(
        env_rk4, dens, ram, ion_rate, steep_op, dens_n, coef_p, coef_m, coef_k, coef_r
    )
    env_rk4 = env + 0.5 * dz * compute_ifft(k2_env)

    k3_env = _set_envelope_operator_frequency(
        env_rk4, dens, ram, ion_rate, steep_op, dens_n, coef_p, coef_m, coef_k, coef_r
    )
    env_rk4 = env + dz * compute_ifft(k3_env)

    k4_env = _set_envelope_operator_frequency(
        env_rk4, dens, ram, ion_rate, steep_op, dens_n, coef_p, coef_m, coef_k, coef_r
    )

    nlin_rk4 = dz * (k1_env + 2 * k2_env + 2 * k3_env + k4_env) / 6

    return nlin_rk4


def _set_envelope_operator_frequency(
    env, dens, ram, ion_rate, steep_op, dens_n, coef_p, coef_m, coef_k, coef_r
):
    """
    Compute envelope propagation nonlinear terms for FCN solver.

    Parameters
    ----------
    env : (M, N) array_like
        Complex envelope at current propagation step.
    dens: (M, N) array_like
        Density at current propagation step.
    ram : (M, N) array_like
        Raman response at current propagation step.
    ion_rate : (M, N) array_like
        Ionization rate at current propagation step.
    steep_op : (N,)
        Self-steepening operator for diffraction.
    dens_n : float
        Neutral density of the medium.
    coef_p : float
        Plasma coefficient.
    coef_m : float
        MPA coefficient.
    coef_k : float
        Kerr coefficient.
    coef_r : float
        Raman coefficient.

    Returns
    -------
    nlin : (M, N) ndarray
        Complex envelope nonlinearities. M is the number
        of radial nodes and N the number of time nodes.

    """
    # Calculate shared quantities
    intensity = np.abs(env) ** 2
    dens_sat = dens_n - dens

    # Plasma term
    nlin_p = coef_p * compute_fft(dens * env) / steep_op

    # MPA term
    nlin_m = coef_m * compute_fft(ion_rate * dens_sat * env / intensity)

    # Kerr term
    nlin_k = steep_op * coef_k * compute_fft(env * intensity)

    # Raman term
    nlin_r = steep_op * coef_r * compute_fft(env * ram)

    nlin = nlin_p + nlin_m + nlin_k + nlin_r

    # Return combined terms
    return nlin
