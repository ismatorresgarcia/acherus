"""Laser pulse parameters for a Gaussian beam in cylindrical coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma


@dataclass
class LaserPulseParameters:
    """Laser pulse physical parameters and derived properties.

    Parameters:
       - const: Constants object containing physical constants
       - medium: MediumParameters object containing medium properties
       - pulse_opt: Pulse type ("gauss" or other options to be defined)
       - gauss_opt: Gaussian order parameter (2 for regular Gaussian, >2 for super-Gaussian)
    """

    def __init__(self, const, medium, pulse_opt="gauss", gauss_opt=2):
        self.input_wavelength = 775e-9
        self.input_waist = 7e-4  # half-width at 1/e^2
        self.input_duration = 85e-15  # half-width at 1/e^2
        self.input_energy = 0.995e-3
        self.input_chirp = 0
        self.input_focal_length = 0

        self.pulse_type = pulse_opt.upper()

        if self.pulse_type == "GAUSS":
            self.input_gauss_order = gauss_opt
        else:  # to be defined in the future
            pass

        # Compute laser pulse properties
        self.input_wavenumber_0 = 2 * const.pi / self.input_wavelength
        self.input_wavenumber = self.input_wavenumber_0 * medium.refraction_index_linear
        self.input_frequency_0 = self.input_wavenumber_0 * const.light_speed_0
        self.input_power = self.input_energy / (
            self.input_duration * np.sqrt(0.5 * const.pi)
        )
        self.critical_power = (  # For regular Gaussian beams
            3.77
            * self.input_wavelength**2
            / (
                8
                * const.pi
                * medium.refraction_index_linear
                * medium.refraction_index_nonlinear
            )
        )
        self.input_intensity = (
            self.input_gauss_order
            * self.input_power
            * 2 ** (2 / self.input_gauss_order)
            / (2 * const.pi * self.input_waist**2 * gamma(2 / self.input_gauss_order))
        )
        self.input_amplitude = np.sqrt(self.input_intensity)
