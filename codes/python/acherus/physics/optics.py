"""Laser pulse parameters for a Gaussian beam in cylindrical coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy.constants import c as c_light
from scipy.special import gamma

from .materials import MaterialParameters


@dataclass
class LaserParameters:
    """Laser pulse optical parameters."""

    # Input initial parameters
    material: MaterialParameters
    wavelength: float = 800e-9
    waist: float = 3.5e-3  # half-width at 1/e^2
    duration: float = 85e-15  # half-width at 1/e^2
    energy: float = 100e-3
    chirp: float = 0
    focal_length: float = 0
    pulse_opt: str = "gaussian"
    gauss_opt: int = 2

    def __post_init__(self):
        """Post-initialization after defining basic pulse parameters."""
        # Compute derived laser optical properties
        self.wavenumber_0 = 2 * np.pi / self.wavelength
        self.wavenumber = self.wavenumber_0 * self.material.refraction_index_linear
        self.frequency_0 = self.wavenumber_0 * c_light
        self.ini_power = self.energy / (self.duration * np.sqrt(0.5 * np.pi))
        self.ini_cr_power = (
            3.77
            * self.wavelength**2
            / (
                8
                * np.pi
                * self.material.refraction_index_linear
                * self.material.refraction_index_nonlinear
            )
        )
        if self.pulse_opt == "gaussian":
            n = self.gauss_opt
            self.ini_intensity = (
                n
                * self.ini_power
                * 2 ** (2 / n)
                / (2 * np.pi * self.waist**2 * gamma(2 / n))
            )
        else:
            raise ValueError(
                f"Invalid pulse type: {self.pulse_opt}. "
                f"Choose 'gaussian' or 'to_be_defined'."
            )
