"""
Peremolov, Popov, and Terent'ev (PPT) ionization rate module.

Mishima et al. (2002) molecular corrections for O2 and N2
molecules are included in the PPT rate.

How this works
--------------

1. The module goal is generating the Keldysh ionization rate
for a given medium and laser central frequency. This is
done by computing a peak intensity array within a desired
range of values, and then used to compute their corresponding
ionization rates. In the end, an interpolating object or
function is provided in the end.

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
chosen, and the succeeding columns add 1 to the previous index
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

NOTE: It can be checked the created `field` uses the SI
convention, which implies the intensity array provided to
the function would have to follow the same SI units. This
would require to convert the intensity at each propagation step
to SI units, in order to match the interpolating object.
Instead, the interpolating function returned has a built-in
unit conversion for the intensity array, which is meant to
be provided in units of I = E**2, as expected inside Acherus.

"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar, m_e, physical_constants
from scipy.interpolate import PchipInterpolator
from scipy.special import dawsn  # pylint: disable=no-name-in-module
from scipy.special import gamma as g_euler  # pylint: disable=no-name-in-module


def compute_ppt_rate(medium, laser):
    """
    Compute and return PPT ionization rates.
    """
    w_au = 1 / physical_constants["atomic unit of time"][0]
    f_au = physical_constants["atomic unit of electric field"][0]
    u_h = physical_constants["Rydberg constant times hc in eV"][0]
    l_0 = laser.wavelength
    n_0 = medium.refraction_index_linear
    u_i = medium.ionization_energy

    def compute_rate(f, w_a, n_q, g, u_h, sum_a, f_0, g_a, u_i):
        """Final PPT ionization rate."""
        mishima = 4 * (16 / 3) * (2 * g**2 + 3) / (1 + g**2)
        units = 0.5 * w_a * u_i / u_h
        c_nl = 2 ** (2 * n_q) / (g_euler(n_q + 1))**2
        a_m = (4 * np.sqrt(2) / np.pi) * (g**2 / (1 + g**2)) * sum_a
        b_1 = (2 * f_0 / (f * np.sqrt(1 + g**2))) ** (2 * n_q - 1.5)
        b_2 = np.exp(-2 * f_0 * g_a / (3 * f))

        return mishima * units * c_nl * a_m * b_1 * b_2

    def compute_sum(alpha_a, beta_a, nu_a, tol, max_iter):
        """
        Compute the PPT series truncated term.
        """
        n = len(alpha_a)
        idx_min = np.ceil(nu_a)

        ids = idx_min[:, None] + np.arange(max_iter)[None, :]
        args = ids - nu_a[:, None]

        sum_terms = (
            np.exp(-alpha_a[:, None] * args) * dawsn(np.sqrt(beta_a[:, None] * args))
        )

        sum_values = np.cumsum(sum_terms, axis=1)

        stop = sum_terms < tol * sum_values
        first_stop = np.argmax(stop, axis=1)

        ion_sums = sum_values[np.arange(n), first_stop]

        return ion_sums

    w_0 = 2 * np.pi * c_light / l_0
    nu_0 = u_i * e_charge / (hbar * w_0)
    f_0 = f_au * np.sqrt(u_i / u_h) ** 3
    n_quantum = 1 / np.sqrt(u_i / u_h)

    inten_c = 0.5 * c_light * eps_0 * n_0
    field = np.sqrt(np.linspace(1e-1, 1e19, 10000) / inten_c)

    gamma = w_0 * np.sqrt(2 * m_e * u_i / e_charge) / field

    asinh = np.arcsinh(gamma)
    idx = 1 + 0.5 / gamma**2
    nu = nu_0 * idx
    beta = 2 * gamma / np.sqrt(1 + gamma**2)
    alpha = 2 * asinh - beta
    g_gamma = 1.5 * (idx * asinh - 1 / beta) / gamma

    ion_sum = compute_sum(alpha, beta, nu, tol=1e-4, max_iter=250)
    ion_rate = compute_rate(
        field,
        w_au,
        n_quantum,
        gamma,
        u_h,
        ion_sum,
        f_0,
        g_gamma,
        u_i,
    )

    ion_inten = PchipInterpolator(inten_c * field**2, ion_rate, extrapolate=True)

    return ion_inten
