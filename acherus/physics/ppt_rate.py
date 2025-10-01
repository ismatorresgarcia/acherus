"""
Peremolov, Popov, and Terent'ev (PPT) ionization rate module.

Includes the Talebpour et al. (1999) electron charge shielding
correction for fitting the effective Coulomb barrier felt by
electrons tunneling out of the O2 and N2 molecules.

How this works
--------------

1. The module goal is generating the PPT ionization rate
for a given medium and laser central frequency. This is
done by computing a peak intensity array within a desired
range of values, and then used to compute their corresponding
ionization rates.

2. For the chosen peak intensity values, the Keldysh parameter
`gamma` is computed, followed by the computation of its
dependencies. This values are needed for the most important
part of the module, which is the computation of the series
summation done by `compute_sum`.

3. The series summation represents the contribution of the
different multiphoton processes to the ionization rate. The
series must be truncated to a certain number of terms, which is
done by computing the first terms of the series until the
desired convergence given by the `tol` parameter is achieved.

4. The series summation starts with the minimum threshold value
for which the multiphoton ionization occurs, whose index is
represented by `idx_min`. The fastest way for computing the
series is to calculate the terms for all the indices at once,
generating a 2D array of indices.

The first column of this 2D array represent all the values the
starting `idx_min` index takes for the different peak intensities,
chosen, and the suceeding columns add 1 to the previous index
value, until some expected maximum number of iterations
`max_iter` is reached. This `max_iter` must be large enough to
ensure the series converges for all the peak intensities with
the desired tolerance.

5. Then, another 2D array with the expression for the series
terms is computed, which is then summed along each row in a
cumulative fashion (every term for a given row is added to
the previous one) to obtain the final value of the series.

6. Here comes the tricky part. For each row in the 2D array
`sum_values`, the convergence condition is checked for every
entry. The variable `stop` is just another 2D array made of
0s in every starting row, until the 1s (the entries where the
convergence condition is met) appear.

7. Since we are only interested in the first occurrence,
the `first_stop` variable is computed, which contains the
index (position) of the first 1 in each row of the `stop`
array. This array is then passed to the `sum_values`
array to extract the corresponding values, which are the
truncated series summation for each peak intensity.

"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar, m_e, physical_constants
from scipy.special import dawsn  # pylint: disable=no-name-in-module
from scipy.special import gamma as g_eu  # pylint: disable=no-name-in-module


def compute_ppt_rate(medium, laser):
    """
    Compute and return PPT ionization rates.
    """
    w_au = 1 / physical_constants["atomic unit of time"][0]
    f_au = physical_constants["atomic unit of electric field"][0]
    e_au = physical_constants["Hartree energy in eV"][0]
    w_len = laser.wavelength
    z_eff = medium.z_effective
    n_0 = medium.refraction_index_linear
    e_gap = medium.ionization_energy

    def compute_rate(f, w_a, n_q, n_c, g, e_h, sum_a, f_a, g_a, z_a, u_a):
        """Final PPT ionization rate."""
        const = (16 / 3) * (4 * np.sqrt(2) / np.pi)
        units = w_a * u_a / e_h
        c_nl = 2 ** (2 * n_c) / (n_c * g_eu(1 + n_c * (2 - z_a**2)) * g_eu(n_q))
        a_m = g**2 * sum_a / (1 + g**2)
        p_pw = 2 * f_a / (f * np.sqrt(1 + g**2)) ** (2 * n_c - 1.5)
        g_ex = np.exp(-2 * f_a * g_a / (3 * f))

        return const * units * c_nl * a_m * p_pw * g_ex

    def compute_sum(alpha_a, beta_a, idx_a, nu_a, tol, max_iter):
        """
        Compute the PPT series truncated term.
        """
        n = len(alpha_a)
        nu_thr = (nu_a * idx_a).astype(np.float32)
        idx_min = np.ceil(nu_thr).astype(np.int16)

        ids = (idx_min[:, None] + np.arange(max_iter)[None, :]).astype(np.int16)
        args = (ids - nu_thr[:, None]).astype(np.float32)

        sum_terms = (
            np.exp(-alpha_a[:, None] * args) * dawsn(np.sqrt(beta_a[:, None] * args))
        ).astype(np.float32)

        sum_values = (np.cumsum(sum_terms, axis=1)).astype(np.float32)

        stop = sum_terms < tol * sum_values
        first_stop = np.argmax(stop, axis=1)

        ion_sums = (sum_values[np.arange(n), first_stop]).astype(np.float32)

        return ion_sums

    w_0 = 2 * np.pi * c_light / w_len
    nu_0 = e_gap * e_charge / (hbar * w_0)
    f_0 = f_au * np.sqrt((2 * e_gap / e_au) ** 3)
    n_corr = 1 / np.sqrt(2 * e_gap / e_au)
    n_quan = z_eff**2 * n_corr

    field_str = np.sqrt(np.linspace(1e15, 1e19, 10000)).astype(np.float32)

    i_fact = np.sqrt(0.5 * c_light * eps_0 * n_0)
    gamma = w_0 * np.sqrt(2 * m_e * e_gap * e_charge) / (e_charge * field_str / i_fact)

    asinh = np.arcsinh(gamma).astype(np.float32)
    idx = (1 + 0.5 / gamma**2).astype(np.float32)
    beta = (2 * gamma / np.sqrt(1 + gamma**2)).astype(np.float32)
    alpha = (2 * asinh - beta).astype(np.float32)
    g_g = (1.5 * (idx * asinh - 1 / beta) / gamma).astype(np.float32)

    ion_sum = compute_sum(alpha, beta, idx, nu_0, tol=1e-4, max_iter=250)
    ion_ppt_rate = compute_rate(
        field_str / i_fact,
        w_au,
        n_quan,
        n_corr,
        gamma,
        e_au,
        ion_sum,
        f_0,
        g_g,
        z_eff,
        e_gap,
    )

    return field_str**2, ion_ppt_rate
