"""
Peremolov, Popov, and Terent'ev (PPT) ionization rate module for atoms.

This module uses Talebpour et al. (1999) electron charge shielding correction
for fitting the effective Coulomb barrier felt by electrons tunneling out
of the atom.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.special import dawsn


def compute_ionization(
    env_a,
    ionz_rate_a,
    ionz_sum_a,
    n_k_a,
    hyd_f_a,
    hyd_nc_a,
    kel_a,
    idx_c_a,
    ppt_c_a,
    mpi_a,
    ion_model="mpi",
    tol=1e-2,
):
    """
    Compute the ionization rates from the generalised "PPT" model.

    Parameters
    ----------
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    ionz_rate_a : (M, N) array_like
        Pre-allocated ionization rate array.
    ionz_sum_a : (M, N) array_like
        Pre-allocated summation term array.
    n_k_a : integer
        Number of photons required for multiphoton ionization (MPI).
    hyd_f_a : float
        Hydrogenic electric field intensity correction.
    hyd_nc_a : float
        Hydrogenic principal quantum number correction.
    kel_a : float
        Keldysh adiabaticity coefficient.
    idx_c_a : float
        Gamma summation index coeffcient.
    ppt_c_a : float
        Frequency conversion constant to SI units.
    mpi_a : float
        Multiphoton ionization (MPI) coefficient.
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
    if ion_model == "mpi":
        ionz_rate_a[:] = mpi_a * np.abs(env_a) ** (2 * n_k_a)

    elif ion_model == "ppt":
        int_sqrt = np.sqrt(np.abs(env_a) ** 2 / (0.5 * c_light * eps_0))  # in SI

        # Compute Keldysh adiabaticity coefficient
        gamma_ppt = kel_a / int_sqrt

        # Compute gamma dependent terms
        asinh_ppt = compute_asinh_gamma(gamma_ppt)
        idx_ppt = compute_idx_gamma(gamma_ppt)
        beta_ppt = compute_b_gamma(gamma_ppt)
        alpha_ppt = compute_a_gamma(asinh_ppt, beta_ppt)
        g_ppt = compute_g_gamma(gamma_ppt, asinh_ppt, idx_ppt, beta_ppt)

        # Compute power term 2n_c - 1.5
        nc_term = (2 * hyd_f_a / (int_sqrt * np.sqrt(1 + gamma_ppt**2))) ** (
            2 * hyd_nc_a - 1.5
        )

        # Compute exponential g function term
        g_term = np.exp(-2 * hyd_f_a * g_ppt / (3 * int_sqrt))

        # Compute gamma squared quotient terms
        g_term_2 = 0.25 * beta_ppt**2
        # g_term_3 = (2 * gamma_ppt**2 + 3) / (1 + gamma_ppt**2)  # Mishima term

        # Compute ionization rate for each field strength point
        flat_results = np.empty(alpha_ppt.size, dtype=np.float64)
        batch_size = 400
        indices = list(range(alpha_ppt.size))
        batches = [
            indices[ii * ii + batch_size] for ii in range(0, len(indices), batch_size)
        ]

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    compute_sum_batch, batch, alpha_ppt, beta_ppt, idx_ppt, idx_c_a, tol
                ): ii
                for ii, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                batch_indices = batches[batch_idx]
                results = future.result()
                flat_results[batch_indices[0] : batch_indices[0] + len(results)] = (
                    results
                )

        # Flatten the list of results in the correct order
        ionz_sum_a.flat[:] = flat_results

        # Compute ionization rate
        ionz_rate_a[:] = ppt_c_a * nc_term * g_term * g_term_2 * ionz_sum_a

    else:
        raise ValueError(
            f"Not available ionization model: '{ion_model}'. "
            f"Available models are: 'ppt' or 'mpi'."
        )


def compute_sum(alpha_a, beta_a, idx_a, idx_c_a, tol):
    """
    Compute the exponential series term.

    Parameters
    ----------
    alpha_a : float
        Alpha function values for each field strength.
    beta_a : float
        Beta function values for each field strength.
    idx_a : float
        Gamma summation index parenthesis for each field strength.
    idx_c_a : float
        Gamma summation index coefficient.
    tol : float, default: 1e-2
        Tolerance for partial sum convergence checking

    Returns
    -------
    sum : float
        The partial sum from the exponential series term after
        convergence checking.

    """
    # Initialize the summation index
    nu_thr = idx_c_a * idx_a
    idx_min = np.ceil(nu_thr).astype(int)

    # Initialize the sum value
    sum_value = 0.0

    # Sum until convergence is reached
    idx = idx_min
    while True:
        arg = idx - nu_thr
        sum_term = np.exp(-alpha_a * arg) * dawsn(np.sqrt(beta_a * arg))
        sum_value += sum_term

        if sum_term < tol * sum_value:
            break

        idx += 1

    return sum_value


def compute_sum_batch(indices, alpha_a, beta_a, idx_a, nu_a, tol):
    """
    Wrapper for batch parallelization of the compute_sum function.
    """
    return [
        compute_sum(alpha_a.flat[ii], beta_a.flat[ii], idx_a.flat[ii], nu_a, tol)
        for ii in indices
    ]


def compute_a_gamma(asinh, beta):
    """
    Compute the alpha function evaluated at
    every gamma value.
    """
    return 2 * asinh - beta


def compute_b_gamma(gamma):
    """
    Compute the beta function evaluated at
    every gamma value.
    """
    return 2 * gamma / np.sqrt(1 + gamma**2)


def compute_g_gamma(gamma, asinh, idx, beta):
    """
    Compute the g function evaluated
    at every gamma value.
    """
    return 1.5 * (idx * asinh - 1 / beta) / gamma


def compute_asinh_gamma(gamma):
    """
    Compute the arcsinh function evaluated at
    every gamma value.
    """
    return np.arcsinh(gamma)


def compute_idx_gamma(gamma):
    """
    Compute the gamma summation index
    parenthesis part evaluated at every
    gamma value.
    """
    return 1 + 0.5 / gamma**2
