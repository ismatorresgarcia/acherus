"""
Peremolov, Popov, and Terent'ev (PPT) ionization rate module for atoms.

This module uses Talebpour et al. (1999) electron charge shielding correction
for fitting the effective Coulomb barrier felt by electrons tunneling out
of the atom.
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.special import dawsn


def compute_ionization(
    env,
    ion_rate,
    ion_sum,
    n_k,
    n_r,
    n_t,
    coef_f0,
    coef_nc,
    coef_ga,
    coef_nu,
    coef_ion,
    coef_ofi,
    ion_model="mpi",
    tol=1e-2,
):
    """
    Compute the ionization rates from the generalised "PPT" model.

    Parameters
    ----------
    env : (M, N) array_like
        Complex envelope at current propagation step.
    ion_rate : (M, N) array_like
        Pre-allocated ionization rate array.
    ion_sum : (M, N) array_like
        Pre-allocated summation term array.
    n_k : integer
        Number of photons required for multiphoton ionization (MPI).
    n_r : integer
        Number of radial nodes.
    n_t : integer
        Number of time nodes.
    coef_f0 : float
        Electric field intensity constant in atomic units.
    coef_nc : float
        Principal quantum number corrected for the chosen material.
    coef_ga : float
        Keldysh adiabaticity coefficient constant term.
    coef_nu : float
        Keldysh number of photons index constant term.
    coef_ion : float
        Frequency conversion constant in atomic units.
    coef_ofi : float
        Optical field ionization (OFI) coefficient.
    model : str, default: "mpi"
        Ionization model to use, "mpi" for multiphotonic
        limit or "ppt" for general PPT model.
    tol : float, default: 1e-2
        Tolerance for partial sum convergence checking.

    Returns
    -------
    rate : (M, N) ndarray.
        Ionization rate for current propagation step.
        M is the number of radial nodes and N the number of
        time nodes.

    """
    env_mod = np.abs(env)  # Peak field strength

    if ion_model == "mpi":
        ion_rate[:] = coef_ofi * env_mod ** (2 * n_k)

    elif ion_model == "ppt":
        env_mod = np.abs(env) / np.sqrt(0.5 * c_light * eps_0)  # Peak field strength
        zero_mask = env_mod == 0
        if np.any(zero_mask):
            env_mod[zero_mask] = 1e-25

        # Compute Keldysh adiabaticity coefficient
        gamma_ppt = coef_ga / env_mod

        # Compute gamma dependent terms
        asinh_ppt = compute_asinh_gamma(gamma_ppt)
        idx_ppt = compute_idx_gamma(gamma_ppt)
        beta_ppt = compute_b_gamma(gamma_ppt)
        alpha_ppt = compute_a_gamma(asinh_ppt, beta_ppt)
        g_ppt = compute_g_gamma(gamma_ppt, asinh_ppt, idx_ppt, beta_ppt)

        # Compute power term 2n_c - 1.5
        nc_term = (2 * coef_f0 / (env_mod * np.sqrt(1 + gamma_ppt**2))) ** (
            2 * coef_nc - 1.5
        )

        # Compute exponential g function term
        g_term = np.exp(-2 * coef_f0 * g_ppt / (3 * env_mod))

        # Compute gamma squared quotient terms
        g_term_2 = (0.5 * beta_ppt) ** 2
        # g_term_3 = (2 * gamma_ppt**2 + 3) / (1 + gamma_ppt**2)  # Mishima term

        # Compute ionization rate for each field strength point
        for ii in range(n_r):
            for jj in range(n_t):
                alpha_ij = alpha_ppt[ii, jj]
                beta_ij = beta_ppt[ii, jj]
                idx_ij = idx_ppt[ii, jj]

                ion_sum[ii, jj] = compute_sum(alpha_ij, beta_ij, idx_ij, coef_nu, tol)

        # Compute ionization rate
        ion_rate[:] = coef_ion * nc_term * g_term * g_term_2 * ion_sum

    else:
        raise ValueError(
            f"Not available ionization model: '{ion_model}'. "
            f"Available models are: 'ppt' or 'mpi'."
        )


def compute_sum(alpha_s, beta_s, idx_ppt_s, coef_nu, tol):
    """
    Compute the exponential series term.

    Parameters
    ----------
    alpha_s : float
        Alpha function values for each field strength.
    beta_s : float
        Beta function values for each field strength.
    idx_ppt_s : float
        Gamma summation index for each field strength.
    coef_nu : float
        Keldysh number of photons index constant term.
    tol : float, default: 1e-2
        Tolerance for partial sum convergence checking

    Returns
    -------
    sum : float
        The partial sum from the exponential series term after
        convergence checking.

    """
    nu_thr = coef_nu * idx_ppt_s

    # Initialize the summation index
    idx_min = int(np.ceil(nu_thr))

    # Initialize partial sum
    sum_value = 0.0

    # Sum until convergence is achieved
    idx = idx_min
    while True:
        arg = idx - nu_thr
        sum_term = np.exp(-alpha_s * arg) * dawsn(np.sqrt(beta_s * arg))
        sum_value += sum_term

        if sum_term < tol * sum_value:
            break

        idx += 1

    return sum_value


def compute_a_gamma(asinh, beta):
    """
    Compute the alpha function evaluated at
    every gamma coefficient value.
    """
    return 2 * asinh - beta


def compute_b_gamma(gamma):
    """
    Compute the beta function evaluated at
    every gamma coefficient value.
    """
    return 2 * gamma / np.sqrt(1 + gamma**2)


def compute_g_gamma(gamma, asinh, idx, beta):
    """
    Compute the g function evaluated
    at every gamma coefficient value.
    """
    return 1.5 * (idx * asinh - 1 / beta) / gamma


def compute_asinh_gamma(gamma):
    """
    Compute the sinh function evaluated at
    every gamma coefficient value.
    """
    return np.arcsinh(gamma)


def compute_idx_gamma(gamma):
    """
    Compute the sum index gamma part evaluated at
    every gamma coefficient value.
    """
    return 1 + 0.5 / gamma**2
