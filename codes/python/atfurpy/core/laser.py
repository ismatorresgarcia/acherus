"""Laser pulse parameters for a Gaussian beam in cylindrical coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy.constants import c as c_light
from scipy.special import gamma as eu_gamma


@dataclass
class LaserInputParameters:
    """Initial laser pulse physical parameters."""

    wavelength: float = 775e-9
    waist: float = 7e-4  # half-width at 1/e^2
    duration: float = 85e-15  # half-width at 1/e^2
    energy: float = 0.995e-3
    chirp: float = 0
    focal_length: float = 0


class LaserPulseParameters:
    """Laser pulse physical parameters derived from the
    initial input parameters."""

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

        # Compute derived laser pulse properties
        self.wavenumber_0 = 2 * np.pi / self.params.wavelength
        self.wavenumber = self.wavenumber_0 * material.refraction_index_linear
        self.frequency_0 = self.wavenumber_0 * c_light

        self.power = self.compute_power()
        self.power_cr = self.compute_power_cr()
        self.intensity = self.compute_intensity()
        self.amplitude = self.compute_amplitude()

    def compute_power(self):
        """Compute initial laser pulse power."""
        return np.float64(
            self.params.energy / (self.params.duration * np.sqrt(0.5 * np.pi))
        )

    def compute_power_cr(self):
        """Compute critical power for self-focusing (valid for Gaussian beams)."""
        return np.float64(
            3.77
            * self.params.wavelength**2
            / (
                8
                * np.pi
                * self.material.refraction_index_linear
                * self.material.refraction_index_nonlinear
            )
        )

    def compute_intensity(self):
        """Compute initial laser pulse intensity for any initial pulse."""
        order = self.gaussian_order
        return np.float64(
            order
            * self.power
            * 2 ** (2 / order)
            / (2 * np.pi * self.params.waist**2 * eu_gamma(2 / order))
        )

    def compute_amplitude(self):
        """Compute initial laser pulse envelope amplitude."""
        return np.float64(np.sqrt(self.intensity))
