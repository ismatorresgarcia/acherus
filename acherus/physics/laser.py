"""Laser pulse properties for a Gaussian beam in cylindrical coordinates."""

import numpy as np
from scipy.constants import c as c_light
from scipy.special import gamma as g_euler


class Laser:
    """Laser pulse."""

    def __init__(self, medium, grid, pulse_typ, pulse_par):
        # Initialize class attributes
        self.medium = medium
        self.shape = pulse_typ

        # Initialize parameters
        self.wavelength = pulse_par.wavelength
        self.waist = pulse_par.waist
        self.duration = pulse_par.duration
        self.energy = pulse_par.energy
        self.gauss_order = pulse_par.gauss_order
        self.focal_length = pulse_par.focal_length
        self.chirp = pulse_par.chirp
        self.r_grid = grid.r_grid
        self.t_grid = grid.t_grid

        # Initialize functions
        self._init_parameters()
        self._init_envelope()

    def _init_parameters(self):
        """Initialize derived laser optical properties"""
        self.wavenumber_0 = (
            2 * np.pi * self.medium.refraction_index_linear / self.wavelength
        )
        self.frequency_0 = 2 * np.pi * c_light / self.wavelength
        self.initial_power = self.energy / (self.duration * np.sqrt(0.5 * np.pi))
        if self.shape == "gaussian":
            self.ini_intensity = (
                self.gauss_order
                * self.initial_power
                * 2 ** (2 / self.gauss_order)
                / (2 * np.pi * self.waist**2 * g_euler(2 / self.gauss_order))
            )

    def _init_envelope(self):
        """
        Compute the initial condition for an envelope at z = 0.

        Returns
        -------
        out : (M, N) ndarray
            The initial complex envelope. M is the number of radial
            nodes and N the number of time nodes.
        """
        r_grid_2d, t_grid_2d = np.meshgrid(self.r_grid, self.t_grid, indexing="ij")
        exp_r2 = -((r_grid_2d / self.waist) ** self.gauss_order).astype(np.complex128)
        exp_t2 = -((t_grid_2d / self.duration) ** 2).astype(np.complex128)

        if self.focal_length != 0:
            exp_r2 -= 0.5j * self.wavenumber_0 * r_grid_2d**2 / self.focal_length
        if self.chirp != 0:
            exp_t2 -= 1j * self.chirp * (t_grid_2d / self.duration) ** 2

        return np.sqrt(self.ini_intensity) * np.exp(exp_r2 + exp_t2)
