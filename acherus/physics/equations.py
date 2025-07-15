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

    def __init__(self, material, laser, ion_model="MPI"):
        # Initialize class attributes
        self.material = material
        self.laser = laser
        self.ion_model = ion_model

        # Initialize parameters
        self.alpha = self.material.raman_partition
        self.e_gap = self.material.ionization_energy
        self.k_0 = self.laser.wavenumber_0
        self.n_0 = self.material.refraction_index_linear
        self.n_2 = self.material.refraction_index_nonlinear
        self.w_r = self.material.raman_rotational_frequency
        self.w_0 = self.laser.frequency_0
        self.tau = self.material.drude_time
        self.z_eff = self.material.effective_charge

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
        self.mpi_c = self.material.constant_mpi
        self.ava_c = self.sigma_0 / (self.e_gap * q_e)

        # Raman equation coefficients
        if self.material.has_raman:
            raman_damping = 1 / self.material.raman_response_time
            self.raman_c1 = raman_damping**2 + self.w_r**2
            self.raman_c2 = -2 * raman_damping
        else:
            self.raman_c1 = 0.0
            self.raman_c2 = 0.0

    def _init_operators(self):
        """Initialize equation operators."""
        self.plasma_c = -0.5 * self.sigma_0 * (1 + 1j * self.w_0 * self.tau)
        self.mpa_c = -0.5 * self.n_k * hbar * self.w_0

        if self.material.has_raman:
            self.kerr_c = 1j * self.w_0 * (1 - self.alpha) * self.n_2 / c_light
            self.raman_c = 1j * self.w_0 * self.alpha * self.n_2 / c_light
        else:
            self.kerr_c = 1j * self.w_0 * self.n_2 / c_light
            self.raman_c = 0.0
