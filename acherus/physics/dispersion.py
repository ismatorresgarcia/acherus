"""
Full dispersion properties module for a given medium.

How this works
--------------

1. The module goal is computing the dispersion properties
for a given medium and frequency values. This is done by
using different semi-empirical formulas for the refraction
index. In the end, a wrapper function `properties` is
generated as an attribute for the class.

2. For the chosen medium, the module checks first if this
option belongs to any of the available media. At the moment,
only air, water, and silica are available options.

3. Then, it creates a wrapper function `properties` which
can be summoned from subsequent Acherus modules for computing
the refraction index, wavenumber, or its derivative at any
frequency values, when required.

"""

from ..functions.dispersion_models import (
    sellmeier_air,
    sellmeier_silica,
    sellmeier_water,
)


class MediumDispersion:
    """
    Dispersion class properties for a given medium.
    """

    _MEDIA = {
        "air": sellmeier_air,
        "water": sellmeier_water,
        "silica": sellmeier_silica,
    }

    def __init__(self, medium_name: str):
        medium_name = medium_name.lower()
        if medium_name not in self._MEDIA:
            raise ValueError(f"Unknown medium '{medium_name}'")

        self.sellmeier_f = self._MEDIA[medium_name]

    def properties(self, omega):
        """
        Wrapper function for computing the dispersion properties

        Parameters
        ----------
        omega : float or np.ndarray
            Angular frequency variable.
        """
        return self.sellmeier_f(omega)
