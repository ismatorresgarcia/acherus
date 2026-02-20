"""
Full medium properties module for a given option.

How this works
--------------

1. The module goal is storing the medium properties
for a given option, together with the dispersive behavior.
This is done by taking the configuration options for the 
chosen medium parameters combined with different semi-empirical
formulas for the refraction index. In the end, a wrapper function
`dispersion_properties` is generated as an attribute for the class,
together with the remaining medium properties.

2. For the chosen medium, the module checks first if this
option belongs to any of the available media. At the moment,
only air, water, and silica are available options.

3. Then, it creates the wrapper `dispersion_properties` which
can be summoned from subsequent Acherus modules for computing
the refraction index, wavenumber, or its derivative at any
frequency values, when required. It also stores the remaining
medium properties as direct attributes of the class.

"""

from ..functions.sellmeier import (
    sellmeier_air,
    sellmeier_silica,
    sellmeier_water,
)


class Medium:
    """Chosen medium properties class."""

    _MEDIA = {
        "air": sellmeier_air,
        "water": sellmeier_water,
        "silica": sellmeier_silica,
    }

    def __init__(self, medium_name: str, medium_par):
        medium_name = medium_name.lower()
        if medium_name not in self._MEDIA:
            raise ValueError(
                f"Invalid medium: '{medium_name}'. "
                f"Available media are: {', '.join(self._MEDIA.keys())}"
            )
        self.name = medium_name
        self.dispersion = self._MEDIA[medium_name]

        # Save all medium properties as direct attributes
        for key, value in medium_par.__dict__.items():
            setattr(self, key, value)

    def dispersion_properties(self, omega):
        """
        Wrapper function for computing the dispersion properties

        Parameters
        ----------
        omega : float or np.ndarray
            Angular frequency variable.
        """
        return self.dispersion(omega)
