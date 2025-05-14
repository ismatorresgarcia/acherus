"""Laser pulse parameters for a Gaussian beam in cylindrical coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy.constants import c as c_light
from scipy.special import gamma as eu_gamma


@dataclass
class LaserInputParameters:
    """Initial laser pulse physical properties."""

    wavelength: float = 775e-9
    waist: float = 7e-4  # half-width at 1/e^2
    duration: float = 85e-15  # half-width at 1/e^2
    energy: float = 0.995e-3
    chirp: float = 0
    focal_length: float = 0


class LaserPulseParameters:
    """Laser pulse physical parameters derived from the
    initial input parameters.

    Parameters:
    -> material: MaterialParameters object with material properties
    -> pulse_opt: Pulse type chosen ("gaussian" or "supergaussian")
    -> gauss_opt: Gaussian order chosen (2 "gaussian" or n > 2 for "supergaussian")
    """

    def __init__(self, material, pulse_opt="gaussian", gauss_opt=2):
        """Initialization function."""
        self.params = LaserInputParameters()
        self.material = material
        self.pulse_opt = pulse_opt

        if self.pulse_opt not in ["gaussian", "to_be_defined"]:
            raise ValueError(
                f"Invalid pulse type: {pulse_opt}. "
                f"Choose 'gaussian' or 'to_be_defined'."
            )

        if self.pulse_opt == "gaussian":
            self.gaussian_order = gauss_opt
        elif self.pulse_opt == "to_be_defined":
            print("Other pulse types are not implemented yet.")

        # Calculate derived laser pulse properties
        self.wavenumber_0 = 2 * np.pi / self.params.wavelength
        self.wavenumber = self.wavenumber_0 * material.refraction_index_linear
        self.frequency_0 = self.wavenumber_0 * c_light

        self.power = self.calculate_power()
        self.power_cr = self.calculate_power_cr()
        self.intensity = self.calculate_intensity()
        self.amplitude = self.calculate_amplitude()

    def calculate_power(self):
        """Calculate initial laser pulse power."""
        return self.params.energy / (self.params.duration * np.sqrt(0.5 * np.pi))

    def calculate_power_cr(self):
        """Calculate critical power for self-focusing (valid for Gaussian beams)."""
        return (
            3.77
            * self.params.wavelength**2
            / (
                8
                * np.pi
                * self.material.refraction_index_linear
                * self.material.refraction_index_nonlinear
            )
        )

    def calculate_intensity(self):
        """Calculate initial laser pulse intensity for any initial pulse."""
        order = self.gaussian_order
        return (
            order
            * self.power
            * 2 ** (2 / order)
            / (2 * np.pi * self.params.waist**2 * eu_gamma(2 / order))
        )

    def calculate_amplitude(self):
        """Calculate initial laser pulse envelope amplitude."""
        return np.sqrt(self.intensity)

    @property
    def power_ratio(self):
        """Calculate the initial ratio of the laser power to the critical power."""
        return self.power / self.power_cr
