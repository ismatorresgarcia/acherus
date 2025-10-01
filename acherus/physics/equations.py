"""
Nonlinear envelope equation and density evolution parameters.
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar, m_e


class EquationParameters:
    """Pulse propagation and electron density evolution
    parameters for the final numerical scheme."""

    def __init__(self, medium, laser, grid):
        # Initialize class attributes
        self.medium = medium
        self.laser = laser
        self.grid = grid

        # Initialize parameters
        self.alpha = self.medium.raman_partition
        self.e_gap = self.medium.ionization_energy
        self.k_0 = self.laser.wavenumber_0
        self.n_0 = self.medium.refraction_index_linear
        self.n_2 = self.medium.refraction_index_nonlinear
        self.w_r = self.medium.raman_rotational_frequency
        self.w_0 = self.laser.frequency_0
        self.tau = self.medium.drude_time
        self.z_eff = self.medium.effective_charge
        self.dt = self.grid.t_res

        # Initialize functions
        self._init_densities()
        self._init_coefficients()
        self._init_operators()

    def _init_densities(self):
        """Initialize density parameters."""
        rho_c = eps_0 * m_e * (self.w_0 / q_e) ** 2
        self.n_k = np.ceil(self.e_gap * q_e / (hbar * self.w_0))
        self.sigma_0 = (self.w_0**2 * self.tau) / (
            (self.n_0 * c_light * rho_c) * (1 + (self.w_0 * self.tau) ** 2)
        )

    def _init_coefficients(self):
        """Initialize equations coefficients."""
        # Density equation coefficients
        self.mpi_c = self.medium.constant_mpi
        self.ava_c = self.sigma_0 / (self.e_gap * q_e)

        # Raman equation coefficients
        if self.medium.has_raman:
            raman_damping = 1 / self.medium.raman_response_time
            raman_r0 = (raman_damping**2 + self.w_r**2) / self.w_r
            self.raman_c1 = np.exp(-raman_damping + 1j * self.w_r) * self.dt
            self.raman_c2 = 0.5 * raman_r0 * self.dt
        else:
            self.raman_c1 = 0.0
            self.raman_c2 = 0.0

    def _init_operators(self):
        """Initialize equation operators."""
        self.plasma_c = -0.5 * self.sigma_0 * (1 + 1j * self.w_0 * self.tau)
        self.mpa_c = -0.5 * self.n_k * hbar * self.w_0

        if self.medium.has_raman:
            self.kerr_c = 1j * self.w_0 * (1 - self.alpha) * self.n_2 / c_light
            self.raman_c = 1j * self.w_0 * self.alpha * self.n_2 / c_light
        else:
            self.kerr_c = 1j * self.w_0 * self.n_2 / c_light
            self.raman_c = 0.0
