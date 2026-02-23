"""
Helper module injected into "medium.py" for computing dispersion properties
of a given material. This is done using Sellmeier semi-empirical equations for
the refractive index.

Air dispersion uses a three-term dispersion formula, with five parameters,
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
    """Return air refractive n(w), k(w), and dk/dω from Peck's model.

    Validity range: 0.23-1.69 µm (15 °C).
    """
    coeff_a0 = 1e-8
    coeff_b1, coeff_b2 = 5791817, 167909
    coeff_c1, coeff_c2 = 238.0185, 57.362

    wavenumber = 1e-6 * omega / (2 * np.pi * c)
    k2 = wavenumber**2

    d1 = coeff_c1 - k2
    d2 = coeff_c2 - k2

    p1 = coeff_b1 / d1 + coeff_b2 / d2
    p2 = coeff_b1 / d1**2 + coeff_b2 / d2**2

    n = 1 + coeff_a0 * p1
    ng = n + 2 * coeff_a0 * k2 * p2
    k_w = n * omega / c
    dk = ng / c

    return n, k_w, dk


def sellmeier_water(omega):
    """Return water n(w), k(w), and dk/dω from Mielenz's model.

    Validity range: 0.235-1.028 µm (20 °C).
    """
    coeff_b1, coeff_b2, coeff_b3, coeff_b4 = (
        1.7604457,
        4.03368e-3,
        1.54182e-2,
        6.44277e-3,
    )
    coeff_c1 = 1.49119e-2

    wavelength = 1e6 * 2 * np.pi * c / omega
    l2 = wavelength**2

    d1 = l2 - coeff_c1

    p1 = coeff_b1 + coeff_b2 * wavelength - coeff_b3 * l2 + coeff_b4 / d1
    p2 = coeff_b2 - 2 * wavelength * (coeff_b3 + coeff_b4 / d1**2)

    n = np.sqrt(p1)
    ng = n - 0.5 * wavelength * p2 / n
    k_w = n * omega / c
    dk = ng / c

    return n, k_w, dk


def sellmeier_silica(omega):
    """Return silica n(w), k(w), and dk/dω from Malitson's model.

    Validity range: 0.21-3.71 µm (20 °C).
    """
    coeff_b1, coeff_b2, coeff_b3 = 0.6961663, 0.4079426, 0.8974794
    coeff_c1, coeff_c2, coeff_c3 = 0.0684043**2, 0.1162414**2, 9.896161**2

    wavelength = 1e6 * 2 * np.pi * c / omega
    l2 = wavelength**2

    d1 = l2 - coeff_c1
    d2 = l2 - coeff_c2
    d3 = l2 - coeff_c3

    p1 = l2 * (coeff_b1 / d1 + coeff_b2 / d2 + coeff_b3 / d3)
    p2 = (
        coeff_b1 * coeff_c1 / d1**2
        + coeff_b2 * coeff_c2 / d2**2
        + coeff_b3 * coeff_c3 / d3**2
    )

    n = np.sqrt(1 + p1)
    ng = n + l2 * p2 / n
    k_w = n * omega / c
    dk = ng / c

    return n, k_w, dk
