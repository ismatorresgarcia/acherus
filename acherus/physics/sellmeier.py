"""
Semi-empirical full dispersion equation module.

Air dispersion uses a two-term dispersion formula, with only four parameters,
from E. R. Peck and K. Reeder (1972). The wavelength validity range goes from
0.23 microns (far-UV) to 1.69 microns (near-IR) at 15 degrees Celsius.

Water dispersion uses a four-term dispersion formula, with five parameters,
from K. D. Mielenz (1978). The wavelength validity range goes from
0.235 microns (far-UV) to 1.028 microns (near-IR) at 20 degrees Celsius.

Silica dispersion uses a three-term dispersion formula, with six parameters,
from I. H. Malitson (1965). The wavelength validity range goes from
0.21 microns (far-UV) to 3.71 microns (mid-IR) at 20 degrees Celsius.

How this works
--------------
Both functions have the angular frequency (rad/s) as argument and return the
refraction index (non-dimensional) at that frequency.

Peck's 'x' variable is the reciprocal of the vacuum wavelength squared, with
the wavelength in units of microns. Thus, conversion from meters to microns
is needed.

Mielenz's 'x' variable is the vacuum wavelength in units of microns.
Hence, the same unit conversion is computed. The same goes for Malitson's 
formula.

"""

import numpy as np
from scipy.constants import c


def sellmeier_air(omega):
    wavelen = 2 * np.pi * c / omega
    x = 1 / (wavelen * 1e6) ** 2

    two_term = 5791817 / (238.0185 - x) + 167909 / (57.362 - x)

    return 1 + 1e-8 * two_term

def sellmeier_water(omega):
    x = 1e6 * 2 * np.pi * c / omega
    x2 = x ** 2

    four_term = 1.7604457 + 4.03368e-3 * x - 1.54182e-2 * x2 + 6.44277e-3 / (x2 - 1.49119e-2)

    return np.sqrt(four_term)

def sellmeier_silica(omega):
    x = 1e6 * 2 * np.pi * c / omega
    x2 = x ** 2

    six_term = x2 * (
        0.6961663 / (x2 - 0.0684043**2) + 0.4079426 / (x2 - 0.1162414**2) + 0.8974794 / (x2 - 9.896161**2)
    )

    return np.sqrt(1 + six_term)
