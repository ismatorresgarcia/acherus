"""
Beam Configuration Module.

This module defines the configuration parameters for an ultra-intense and ultra-short
laser pulse. It handles both basic beam parameters and derived quantities needed for
the simulation, including:
    - Fundamental beam parameters (wavelength, waist, energy, etc.)
    - Media properties (refractive index, permittivity, etc.)
    - Derived parameters (wavenumber, intensity, amplitude, etc.)
"""

from dataclasses import dataclass

import numpy as np

PI = np.pi
SPEED_OF_LIGHT = 299792458
VACUUM_PERMITTIVITY = 8.8541878128e-12


@dataclass
class BeamConfig:
    """
    Configuration class for laser beam parameters.

    This class handles both the initialization of fundamental beam parameters
    and the calculation of derived quantities needed for beam propagation.

    Attributes:
        wavelength (float): Central wavelength in meters. Defaults to 800nm
        waist (float): Initial beam waist in meters. Defaults to 9mm
        peak_time (float): Peak time in seconds. Defaults to 130fs
        energy (float): Pulse energy in Joules. Defaults to 4mJ
        focal_length (float): Focal length in meters. Defaults to 10m
        media (dict): Dictionary containing media properties
        wavenumber_0 (float): Vacuum wavenumber
        wavenumber (float): Wavenumber in medium
        power (float): Peak power
        intensity (float): Peak intensity
        amplitude (float): Field amplitude
    """

    wavelength: float = 800e-9
    waist: float = 9e-3
    peak_time: float = 130e-15
    energy: float = 4e-3
    focal_length: float = 10

    def __post_init__(self):
        """Initialize derived parameters after instance creation."""
        self.media = self._create_media_constants()
        self._calculate_derived_parameters()

    def _create_media_constants(self):
        """
        Create dictionary of media constants.

        Returns:
            dict: Dictionary containing physical constants and derived parameters
                for different media (water and vacuum)
        """
        # Variable that depends on wavelength
        lin_ref_ind_water = 1.334  # This could be replaced with a dispersion formula

        return {
            "WATER": {
                "LIN_REF_IND": lin_ref_ind_water,
                "INT_FACTOR": 0.5
                * SPEED_OF_LIGHT
                * VACUUM_PERMITTIVITY
                * lin_ref_ind_water,
            },
            "VACUUM": {
                "LIGHT_SPEED": SPEED_OF_LIGHT,
                "PERMITTIVITY": VACUUM_PERMITTIVITY,
            },
        }

    def _calculate_derived_parameters(self):
        """
        Calculate derived beam parameters.

        Computes:
            - Wavenumbers (vacuum and medium)
            - Peak power
            - Peak intensity
            - Field amplitude
        """
        self.wavenumber_0 = 2 * PI / self.wavelength
        self.wavenumber = 2 * PI * self.media["WATER"]["LIN_REF_IND"] / self.wavelength
        self.power = self.energy / (self.peak_time * np.sqrt(0.5 * PI))
        self.intensity = 2 * self.power / (PI * self.waist**2)
        self.amplitude = np.sqrt(self.intensity / self.media["WATER"]["INT_FACTOR"])
