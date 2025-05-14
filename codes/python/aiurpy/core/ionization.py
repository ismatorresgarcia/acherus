"""
Peremolov, Popov, and Terent'ev (PPT) ionization rate module for atoms.

This module uses Talebpour et al. (1999) electron charge shielding correction
for fitting the effective Coulomb barrier felt by electrons tunneling out
of the atom.
"""

import numpy as np
from scipy.integrate import quad


def calculate_ionization(
    env,
    ion_rate,
    ion_sum,
    n_k,
    n_r,
    n_t,
    coef_f0,
    coef_ns,
    coef_ga,
    coef_nu,
    coef_ion,
    coef_ofi,
    ion_model="mpi",
    tol=1e-2,
):
    """
    Calculate ionization rate from PPT model.

    Parameters:
    -> env: envelope at current propagation step
    -> ion_rate: pre-allocated ionization rate array
    -> ion_sum: pre-allocated series term array
    -> n_k: number of photons for MPI
    -> n_r: number of radial nodes
    -> n_t: number of time nodes
    -> coef_f0: electric field intensity constant in atomic units
    -> coef_ns: principal quantum number corrected for the chosen material
    -> coef_ga: Keldysh adiabaticity coefficient constant term
    -> coef_nu: Keldysh number of photons index constant term
    -> coef_ion: frequency conversion constant in atomic units
    -> coef_ofi: OFI coefficient
    -> model: ionization model to use ("mpi" or "ppt")

    Returns:
    -> float 2D-array: Ionization rate for envelope
    """
    env_mod = np.abs(env)  # Peak field strength

    if ion_model == "mpi":
        ion_rate[:] = coef_ofi * env_mod ** (2 * n_k)

    elif ion_model == "ppt":
        # Calculate Keldysh adiabaticity coefficient
        gamma_ppt = coef_ga / env_mod

        # Calculate gamma dependent terms
        asinh_ppt = calculate_asinh_gamma(gamma_ppt)
        idx_ppt = calculate_idx_gamma(gamma_ppt)
        beta_ppt = calculate_b_gamma(gamma_ppt)
        alpha_ppt = calculate_a_gamma(asinh_ppt, beta_ppt)
        gfunc_ppt = calculate_g_gamma(gamma_ppt, asinh_ppt, idx_ppt, beta_ppt)

        # Calculate power term 2n_star - 3/2
        ns_term = (2 * coef_f0 / (env_mod * np.sqrt(1 + gamma_ppt**2))) ** (
            2 * coef_ns - 1.5
        )

        # Calculate exponential g function term
        g_term = np.exp(-2 * coef_f0 * gfunc_ppt / (3 * env_mod))

        # Calculate gamma squared quotient term
        g_term_2 = (0.5 * beta_ppt) ** 2

        # Calculate ionization rate for each field strength point
        for i in range(n_r):
            for j in range(n_t):
                # Skip points with extremely low field
                if env_mod[i, j] < 1e-12:
                    continue

                alpha_ij = alpha_ppt[i, j]
                beta_ij = beta_ppt[i, j]
                gamma_ij = gamma_ppt[i, j]

                # Calculate summation term
                ion_sum[i, j] = calculate_sum(alpha_ij, beta_ij, gamma_ij, coef_nu, tol)

        # Calculate ionization rate
        ion_rate[:] = coef_ion * ns_term * g_term * g_term_2 * ion_sum

    else:
        raise ValueError(
            f"Not available ionization model: '{ion_model}'. "
            f"Available models are: 'ppt' or 'mpi'."
        )


def calculate_sum(alpha_s, beta_s, idx_ppt_s, coef_nu, tol):
    """
    Calculate the exponential series term.

    Parameters:
    -> alpha_s: alpha function values for each field strength
    -> beta_s: beta function values for each field strength
    -> idx_ppt_s: gamma summation index for each field strength
    -> coef_nu: Keldysh number of photons index constant term
    -> tol: Tolerance for partial sum convergence checking

    Returns:
    -> complex: The partial sum from the exponential series
    """
    nu_thr = coef_nu * idx_ppt_s

    # Initialize the summation index
    max_idx = 200
    idx_min = int(np.ceil(nu_thr))

    # Initialize partial sum
    sum_value = 0.0

    # Sum until convergence is achieved
    idx = idx_min
    while True:
        sum_term = np.exp(-alpha_s * (idx - nu_thr)) * phi_integral(
            np.sqrt(beta_s * (idx - nu_thr))
        )
        sum_value += sum_term

        if sum_term < tol * sum_value:
            break

        idx += 1

        if idx > idx_min + max_idx:
            break

    return sum_value


def phi_integral(upper_l):
    """
    Calculate Φ_0(x) using numerical integration.

    Parameters:
    -> int_l: upper integration limit
    """

    def integrand(y):
        return np.exp(y**2)

    # Check if int_l is valid for integration range
    if upper_l <= 0:
        return 0.0

    int_result, _ = quad(integrand, 0, upper_l)
    return np.exp(-(upper_l**2)) * int_result


def calculate_a_gamma(asinh, beta):
    """
    Calculate the alpha function evaluated at
    every gamma coefficient value.
    """
    return 2 * asinh - beta


def calculate_b_gamma(gamma):
    """
    Calculate the beta function evaluated at
    every gamma coefficient value.
    """
    return 2 * gamma / np.sqrt(1 + gamma**2)


def calculate_g_gamma(gamma, asinh, idx, beta):
    """
    Calculate the g function evaluated at
    every gamma coefficient value.
    """
    return (1.5 / gamma) * (idx * asinh - 1 / beta)


def calculate_asinh_gamma(gamma):
    """
    Calculate the sinh function evaluated at
    every gamma coefficient value.
    """
    return np.arcsinh(gamma)


def calculate_idx_gamma(gamma):
    """
    Calculate the sum index gamma part evaluated at
    every gamma coefficient value.
    """
    return 1 + 0.5 / gamma**2
