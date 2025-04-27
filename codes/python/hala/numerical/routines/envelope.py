"""Envelope evolution module."""

import numba as nb
import numpy as np
from scipy.fft import fft, ifft


def frequency_domain(data):
    """Transform data to frequency domain using FFT."""
    return fft(data, axis=1, workers=-1)


def time_domain(data):
    """Transform data from frequency domain to time domain using IFFT."""
    return ifft(data, axis=1, workers=-1)


@nb.njit
def _set_envelope_operator(
    env_s, dens_s, ram_s, n_k, dens_n, coef_p, coef_m, coef_k, coef_r
):
    """Set up the envelope propagation nonlinear terms for FSS solver.

    Parameters:
    - env_s: envelope at current time slice
    - dens_s: electron density at current time slice
    - ram_s: Raman response at current time slice
    - n_k: number of photons for MPI
    - dens_n: neutral density of the medium
    - coef_p: plasma coefficient
    - coef_m: MPA coefficient
    - coef_k: Kerr coefficient
    - coef_r: Raman coefficient

    Returns:
    - complex 1D-array: Nonlinear operator
    """
    env_s_2 = np.abs(env_s) ** 2
    env_s_2k2 = env_s_2 ** (n_k - 1)
    dens_s_sat = 1 - (dens_s / dens_n)

    nlin_s = env_s * (
        coef_p * dens_s
        + coef_m * dens_s_sat * env_s_2k2
        + coef_k * env_s_2
        + coef_r * ram_s
    )

    return nlin_s


@nb.njit
def _rk4_envelope_step(env_s, dens_s, ram_s, env_rk4, env_args, dz, dz_2, dz_6):
    """
    Compute one step of the RK4 integration for envelope propagation for FSS solver.

    Parameters:
    - env_s: envelope at current time slice
    - dens_s: density at current time slice
    - ram_s: raman response at current time slice
    - env_rk4: auxiliary envelope array for RK4 integration
    - env_args: arguments for the envelope operator
    - dz: z step
    - dz_2: z step divided by 2
    - dz_6: z step divided by 6

    Returns:
    - complex 1D-array: RK4 integration for one time slice
    """
    k1_env = _set_envelope_operator(env_s, dens_s, ram_s, *env_args)
    env_rk4 = env_s + dz_2 * k1_env

    k2_env = _set_envelope_operator(env_rk4, dens_s, ram_s, *env_args)
    env_rk4 = env_s + dz_2 * k2_env

    k3_env = _set_envelope_operator(env_rk4, dens_s, ram_s, *env_args)
    env_rk4 = env_s + dz * k3_env

    k4_env = _set_envelope_operator(env_rk4, dens_s, ram_s, *env_args)

    nlin_s_rk4 = dz_6 * (k1_env + 2 * k2_env + 2 * k3_env + k4_env)

    return nlin_s_rk4


@nb.njit(parallel=True)
def solve_nonlinear_rk4(env, dens, ram, env_rk4, nlin, n_t, env_args, dz, dz_2, dz_6):
    """
    Solve envelope propagation nonlinearities for all
    time steps using RK4 for FSS solver.

    Parameters:
    - env: envelope at current propagation step
    - dens: density at current propagation step
    - ram: raman response at current propagation step
    - env_rk4: auxiliary envelope array for RK4 integration
    - nlin: pre-allocated array for the nonlinear terms
    - n_t: number of time nodes
    - env_args: arguments for the envelope operator
    - dz: z step
    - dz_2: z step divided by 2
    - dz_6: z step divided by 6
    """
    # pylint: disable=not-an-iterable
    for ll in nb.prange(n_t):
        env_s = env[:, ll]
        dens_s = dens[:, ll]
        ram_s = ram[:, ll]

        nlin_s_rk4 = _rk4_envelope_step(
            env_s, dens_s, ram_s, env_rk4, env_args, dz, dz_2, dz_6
        )

        nlin[:, ll] = nlin_s_rk4


def _set_envelope_operator_freq(
    env, dens, ram, steep_op, n_k, dens_n, coef_p, coef_m, coef_k, coef_r
):
    """Set up all nonlinear operators together for FCN solver.

    Parameters:
    - env: envelope at current propagation step
    - dens: density at current propagation step
    - ram: raman response at current propagation step
    - steep_op: self-steepening operator
    - n_k: number of photons for MPI
    - dens_n: neutral density of the medium
    - coef_p: plasma coefficient
    - coef_m: MPA coefficient
    - coef_k: Kerr coefficient
    - coef_r: Raman coefficient

    Returns:
    - complex 2D-array: Combined nonlinear terms in frequency domain
    """
    # Calculate shared quantities
    env_2 = np.abs(env) ** 2
    env_2k2 = env_2 ** (n_k - 1)
    dens_sat = 1 - (dens / dens_n)

    # Plasma term
    nlin_p = coef_p * frequency_domain(dens * env) / steep_op

    # MPA term
    nlin_m = coef_m * frequency_domain(dens_sat * env * env_2k2)

    # Kerr term
    nlin_k = steep_op * coef_k * frequency_domain(env * env_2)

    # Raman term
    nlin_r = steep_op * coef_r * frequency_domain(env * ram)

    nlin = nlin_p + nlin_m + nlin_k + nlin_r

    # Return combined terms
    return nlin


def _rk4_envelope_step_freq(
    env, dens, ram, steep_op, env_rk4, env_args, dz, dz_2, dz_6
):
    """Combined RK4 step for all nonlinear terms for FCN solver.

    Parameters:
    - env: envelope at current propagation step
    - dens: density at current propagation step
    - ram: raman response at current propagation step
    - steep_op: self-steepening operator
    - env_rk4: auxiliary envelope array for RK4 integration
    - env_args: arguments for the envelope operator
    - dz: z step
    - dz_2: z step divided by 2
    - dz_6: z step divided by 6

    Returns:
    - complex 2D-array: RK4 integration for all nonlinear terms
    """
    k1_env = _set_envelope_operator_freq(env, dens, ram, steep_op, *env_args)
    env_rk4 = env + dz_2 * time_domain(k1_env)

    k2_env = _set_envelope_operator_freq(env_rk4, dens, ram, steep_op, *env_args)
    env_rk4 = env + dz_2 * time_domain(k2_env)

    k3_env = _set_envelope_operator_freq(env_rk4, dens, ram, steep_op, *env_args)
    env_rk4 = env + dz * time_domain(k3_env)

    k4_env = _set_envelope_operator_freq(env_rk4, dens, ram, steep_op, *env_args)

    nlin_rk4 = dz_6 * (k1_env + 2 * k2_env + 2 * k3_env + k4_env)

    return nlin_rk4


def solve_nonlinear_rk4_freq(
    env, dens, ram, steep_op, env_rk4, nlin, env_args, dz, dz_2, dz_6
):
    """
    Solve envelope propagation nonlinearities for all
    time steps using RK4 for FCN solver.

    Parameters:
    - env: envelope at current propagation step
    - dens: density at current propagation step
    - ram: raman response at current propagation step
    - steep_op: self-steepening operator
    - env_rk4: auxiliary envelope array for RK4 integration
    - nlin: pre-allocated array for the nonlinear terms
    - env_args: arguments for the envelope operator
    - dz: z step
    - dz_2: z step divided by 2
    - dz_6: z step divided by 6
    """
    nlin[:] = _rk4_envelope_step_freq(
        env, dens, ram, steep_op, env_rk4, env_args, dz, dz_2, dz_6
    )
