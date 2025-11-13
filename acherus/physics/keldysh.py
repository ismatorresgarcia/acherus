"""
Peremolov, Popov, and Terent'ev (PPT) ionization rate module for gaseous media.

Keldysh ionization rate module for condensed media.

Mishima et al. (2002) molecular corrections for O2 and N2
molecules are included in the PPT rate.

How this works
--------------

1. The module goal is generating the Keldysh ionization rate
for a given medium and laser central frequency. This is
done by computing a peak intensity array within a desired
range of values, and then used to compute their corresponding
ionization rates. In the end, an interpolating object or
function is provided.

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
from scipy.constants import hbar
from scipy.interpolate import interp1d

from ..functions.keldysh_rates import keldysh_condensed_rate, keldysh_gas_rate, mpi_rate


class KeldyshIonization:
    """
    MPI limit or general-Keldysh ionization models for gaseous and condensed media.
    """

    def __init__(self, dispersion, model_name, params):
        """
        Initialize the Keldysh ionization calculator with the given parameters.

        Parameters
        ----------
        dispersion : object
            Class containing the dispersion properties function.
        model_name : str
            Ionization model.
        params : object
            Dataclass containing all required parameters.

        Raises
        ------
        ValueError
            If parameter combinations are invalid for the chosen model
        """
        self._dispersion = dispersion
        self._model = model_name
        self._parameters = params
        self.interpolator = None
        self._check_parameters(params)

    def _check_parameters(self, params) -> None:
        """Validate inputs for the chosen model and medium."""
        general_checks = [
            (params.wavelength is None, "wavelength value must be given"),
            (params.energy_gap is None, "energy_gap value must be given"),
            (params.wavelength <= 0, "wavelength must be positive"),
            (params.energy_gap <= 0, "energy_gap must be positive"),
            (
                params.intensity_range[0] <= 0,
                "intensity_range lower bound must be positive",
            ),
            (
                params.intensity_range[1] <= 0,
                "intensity_range upper bound must be positive",
            ),
            (
                params.intensity_range[1] <= params.intensity_range[0],
                "intensity_range upper bound must be greater than lower bound",
            ),
        ]
        for condition, message in general_checks:
            if condition:
                raise ValueError(message)

        checks = {
            "MPI": self._validate_mpi_params,
            "PPTG": self._validate_pptg_params,
            "PPTC": self._validate_pptc_params,
        }
        check = checks.get(params.model)
        if check is None:
            raise ValueError("model must be: 'MPI', 'PPTG', or 'PPTC'")
        check(params)

    def _validate_mpi_params(self, params) -> None:
        if params.cross_section is None:
            raise ValueError("cross_section must be given for 'MPI'")

    def _validate_pptg_params(self, params) -> None:
        pass

    def _validate_pptc_params(self, params) -> None:
        if params.neutral_density is None:
            raise ValueError("neutral_density must be given for 'PPTC'")
        if params.reduced_mass is None:
            raise ValueError("reduced_mass must be given for 'PPTC'")

    def _ionization_rate(self):
        """Calculate ionization rates based on the chosen model."""
        d = self._dispersion
        m = self._model
        p = self._parameters
        inten = np.linspace(*p.intensity_range, p.num_points)
        energy_gap = p.energy_gap * e_charge
        omega_0 = 2 * np.pi * c_light / p.wavelength
        index_0, _, _ = d.properties(omega_0)
        photons = energy_gap / (hbar * omega_0)
        field_factor = 0.5 * index_0 * c_light * eps_0

        if m == "MPI":
            return inten, mpi_rate(inten, photons, p.cross_section)
        if m == "PPTC":
            return inten, keldysh_gas_rate(
                inten,
                field_factor,
                omega_0,
                energy_gap,
                photons,
                p.tolerance,
                p.max_iterations,
            )
        return inten, keldysh_condensed_rate(
            inten,
            field_factor,
            omega_0,
            energy_gap,
            photons,
            p.reduced_mass,
            p.neutral_density,
            p.tolerance,
            p.max_iterations,
        )

    @property
    def intensity_to_rate(self):
        """Interpolation function for ionization rate vs intensity."""
        if self.interpolator is None:
            inten, rate = self._ionization_rate()
            self.interpolator = interp1d(inten, rate, bounds_error=False, fill_value="extrapolate")
        return self.interpolator
