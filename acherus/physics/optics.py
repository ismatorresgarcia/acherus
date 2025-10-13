"""Laser pulse parameters for a Gaussian beam in cylindrical coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy.constants import c as c_light
from scipy.special import gamma as g_eu

from .media import MediumParameters


@dataclass
class LaserParameters:
    """Laser pulse optical parameters."""

    # Input initial parameters
    medium: MediumParameters
    wavelength: float = 800e-9
    waist: float = 3.57e-3  # half-width at 1/e^2
    duration: float = 85e-15  # half-width at 1/e^2
    energy: float = 6.7e-3
    chirp: float = 0
    focal_length: float = 0
    pulse_opt: str = "gaussian"
    gauss_opt: int = 2

    def __post_init__(self):
        """Post-initialization after defining basic pulse parameters."""
        # Compute derived laser optical properties
        self.wavenumber_0 = (
            2 * np.pi * self.medium.refraction_index_linear / self.wavelength
        )
        self.frequency_0 = 2 * np.pi * c_light / self.wavelength
        self.ini_power = self.energy / (self.duration * np.sqrt(0.5 * np.pi))
        if self.pulse_opt == "gaussian":
            self.ini_intensity = (
                self.gauss_opt
                * self.ini_power
                * 2 ** (2 / self.gauss_opt)
                / (2 * np.pi * self.waist**2 * g_eu(2 / self.gauss_opt))
            )
        else:
            raise ValueError(
                f"Invalid pulse type: {self.pulse_opt}. "
                f"Choose 'gaussian' or 'to_be_defined'."
            )
