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
    coef_f0,
    coef_nc,
    coef_ga,
    coef_nu,
    coef_ion,
    coef_ofi,
    ion_model="mpi",
    tol=1e-2,
    max_iter=200,
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
    max_iter : int, default: 200
        Maximum number of iterations for the partial sum.

    Returns
    -------
    rate : (M, N) ndarray.
        Ionization rate for current propagation step.
        M is the number of radial nodes and N the number of
        time nodes.

    """
    env_mod = np.abs(env)  # Peak field strength inner values

    if ion_model == "mpi":
        ion_rate[:] = coef_ofi * env_mod ** (2 * n_k)

    elif ion_model == "ppt":
        env_mod = env_mod / np.sqrt(0.5 * c_light * eps_0)  # Peak field strength

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
        alpha_flat = alpha_ppt.ravel()
        beta_flat = beta_ppt.ravel()
        idx_flat = idx_ppt.ravel()

        ion_sum_flat = compute_sum(
            alpha_flat, beta_flat, idx_flat, coef_nu, tol, max_iter
        )

        ion_sum = ion_sum_flat.reshape(alpha_ppt.shape)

        # Compute ionization rate
        ion_rate[:] = coef_ion * nc_term * g_term * g_term_2 * ion_sum

    else:
        raise ValueError(
            f"Not available ionization model: '{ion_model}'. "
            f"Available models are: 'ppt' or 'mpi'."
        )


def compute_sum(alpha_a, beta_a, idx_a, coef_nu, tol, max_iter):
    """
    Compute the exponential series term.

    Parameters
    ----------
    alpha_a : float
        Alpha function values for each field strength.
    beta_a : float
        Beta function values for each field strength.
    idx_a : float
        Gamma summation index for each field strength.
    coef_nu : float
        Keldysh number of photons index constant term.
    tol : float, default: 1e-2
        Tolerance for partial sum convergence checking
    max_iter : int, default: 200
        Maximum number of iterations for the partial sum.

    Returns
    -------
    sum : float
        The partial sum from the exponential series term after
        convergence checking.

    """
    # Initialize the summation index
    n = idx_a.size
    nu_thr = coef_nu * idx_a
    idx_min = np.ceil(nu_thr).astype(int)

    # Create the 2D array of indices
    ids = idx_min[:, None] + np.arange(max_iter)[None, :]
    args = ids - nu_thr[:, None]

    # Compute partial sums terms
    sum_terms = np.exp(-alpha_a[:, None] * args) * dawsn(
        np.sqrt(beta_a[:, None] * args)
    )
    # Compute the cumulative sum for each row
    sum_values = np.cumsum(sum_terms, axis=1)

    # Find the indices where convergence is fulfilled
    stop = sum_terms < tol * sum_values
    first_stop = np.argmax(stop, axis=1)

    # Gather the sum at the stopping index for each row
    ion_sums_flat = sum_values[np.arange(n), first_stop]

    return ion_sums_flat


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
