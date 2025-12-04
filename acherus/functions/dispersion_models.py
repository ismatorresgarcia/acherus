"""
Helper module injected into "dispersion.py" for computing dispersion properties
of a given medium. This is done using Sellmeier semi-empirical equations for
the refractive index.

Air dispersion uses a three-term dispersion formula, with five parameters,
from E. R. Peck and K. Reeder (1972). The wavelength validity range goes from
0.185 microns (far-UV) to 1.69 microns (near-IR) at 15 degrees Celsius.

Water dispersion uses a four-term dispersion formula, with five parameters,
from K. D. Mielenz (1978). The wavelength validity range goes from
0.235 microns (far-UV) to 1.028 microns (near-IR) at 20 degrees Celsius.

Silica dispersion uses a three-term dispersion formula, with six parameters,
from I. H. Malitson (1965). The wavelength validity range goes from
0.21 microns (far-UV) to 3.71 microns (mid-IR) at 20 degrees Celsius.

How this works
--------------
Both functions have the angular frequency (rad/s) as argument and return the
refraction index, wavenumber, and its derivative at that frequency.

Peck's 'sigma' variable is the vacuum (spectroscopic) number, in units of
reciprocal microns (inverse wavelength). Thus, conversion from meters to
microns is needed.

Mielenz's 'lambda' variable is the vacuum wavelength in units of microns.
Hence, the same unit conversion is computed.

Malitson's 'lambda' variable is the vacuum wavelength in units of microns.
Hence, the same unit conversion is computed.

"""
import numpy as np
from scipy.constants import c


def sellmeier_air(omega):
    B1, B2 = 5791817, 167909
    C1, C2 = 238.0185, 57.362

    k = 1e-6 * omega / (2 * np.pi * c)
    x = k**2

    d1 = C1 - x
    d2 = C2 - x
    d1_2 = d1**2
    d2_2 = d2**2

    p1 = B1 / d1 + B2 / d2
    p2 = B1 / d1_2 + B2 / d2_2

    n = 1 + 1e-8 * p1
    ng = n - 2e-8 * x * p2
    k_w = n * omega / c
    dk = ng / c

    return n, k_w, dk


def sellmeier_water(omega):
    B1, B2, B3, B4 = 1.7604457, 4.03368e-3, -1.54182e-2, 6.44277e-3
    C1 = 1.49119e-2

    l = 1e6 * 2 * np.pi * c / omega
    x = l**2

    d1 = x - C1
    d1_2 = d1**2

    p1 = B1 + B2 * l + B3 * x + B4 / d1
    p2 = B2 + 2 * l * (B3 - B4 / d1_2)

    n = np.sqrt(p1)
    ng = n - 0.5 * l * p2 / n
    k_w = n * omega / c
    dk = ng / c

    return n, k_w, dk


def sellmeier_silica(omega):
    B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
    C1, C2, C3 = 0.0684043, 0.1162414, 9.896161

    l = 1e6 * 2 * np.pi * c / omega
    x = l**2

    d1 = x - C1**2
    d2 = x - C2**2
    d3 = x - C3**2
    d1_2 = d1**2
    d2_2 = d2**2
    d3_2 = d3**2

    p1 = x * (B1 / d1 + B2 / d2 + B3 / d3)
    p2 = -2 * l * (B1 * C1**2 / d1_2 + B2 * C2**2 / d2_2 + B3 * C3**2 / d3_2)

    n = np.sqrt(1 + p1)
    ng = n - 0.5 * l * p2 / n
    k_w = n * omega / c
    dk = ng / c

    return n, k_w, dk
