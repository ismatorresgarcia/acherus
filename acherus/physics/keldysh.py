"""
Perelomov et al. (PPT, 1966) ionization rate module for gaseous
media. Mishima et al. (2002) revised atomic and molecular corrections are
included in the PPT. The Coulomb-corrected formula for long-range action
appears, for example, in Nikishov and Ritus (1967) or in
Perelomov and Popov (1967) future works.

Keldysh (1965) ionization rate module for condensed media.

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
from scipy.interpolate import PchipInterpolator

from ..functions.keldysh_rates import keldysh_condensed_rate, keldysh_gas_rate, mpi_rate


class KeldyshIonization:
    """
    MPI limit or general-Keldysh ionization models for gaseous and condensed media.
    """

    def __init__(self, medium, laser, model_name, params):
        """
        Initialize the Keldysh ionization calculator with the given parameters.

        Parameters
        ----------
        medium : object
            Instance of the "Medium" class containing medium properties.
        laser : object
            Instance of the "Laser" class containing laser properties.
        model_name : str
            Ionization model.
        params : object
            Dataclass containing all required parameters.

        Raises
        ------
        ValueError
            If parameter combinations are invalid for the chosen model
        """
        self._medium = medium
        self._laser = laser
        self._model = model_name
        self._parameters = params
        self.interpolator = None
        self._check_parameters(params)

    def _check_parameters(self, params) -> None:
        """Validate inputs for the chosen model and medium."""
        p = self._parameters

        if params.intensity_range[0] <= 0:
            raise ValueError("intensity_range lower bound must be positive")
        if params.intensity_range[1] <= 0:
            raise ValueError("intensity_range upper bound must be positive")
        if params.intensity_range[1] <= params.intensity_range[0]:
            raise ValueError("intensity_range upper bound must be greater than lower bound")

        if self._model == "mpi":
            if not hasattr(p, "cross_section"):
                raise ValueError("cross_section must be given for multiphoton limit")
        elif self._model == "keldysh":
            if not hasattr(p, "tolerance") or not hasattr(p, "max_iterations"):
                raise ValueError(
                    "tolerance and max_iterations must be given for keldysh model"
                )
            if self._medium.name in ["water", "silica"]:
                if not hasattr(p, "reduced_mass"):
                    raise ValueError("reduced_mass must be given for condensed media")
        else:
            raise ValueError(f"Invalid ionization model: '{self._model}'.")

    def _ionization_rate(self):
        """Calculate ionization rates based on the chosen model."""
        m = self._model
        p = self._parameters
        medium = self._medium
        laser = self._laser

        inten = np.linspace(*p.intensity_range, p.num_points)
        omega_0 = laser.frequency_0
        energy_gap = medium.energy_gap * e_charge
        n0 = laser.index_0
        photons = energy_gap / (hbar * omega_0)
        field_factor = 0.5 * n0 * c_light * eps_0

        if m == "mpi":
            return inten, mpi_rate(inten, photons, p.cross_section)
        elif m == "keldysh":
            if medium.name == "air":
                return inten, keldysh_gas_rate(
                    inten,
                    field_factor,
                    omega_0,
                    energy_gap,
                    photons,
                    p.tolerance,
                    p.max_iterations,
                )
            elif medium.name in ["water", "silica"]:
                return inten, keldysh_condensed_rate(
                    inten,
                    field_factor,
                    omega_0,
                    energy_gap,
                    photons,
                    p.reduced_mass,
                    medium.neutral_density,
                    p.tolerance,
                    p.max_iterations,
                )

    @property
    def intensity_to_rate(self):
        """Interpolation function for ionization rate vs intensity."""
        if self.interpolator is None:
            inten, rate = self._ionization_rate()
            self.interpolator = PchipInterpolator(inten, rate, extrapolate=True)
        return self.interpolator
