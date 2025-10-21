"""
Equation coefficients.
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar, m_e


class Equation:
    """Pulse propagation and electron density evolution
    coefficients of the numerical scheme."""

    def __init__(self, medium, laser, grid):
        # Initialize class attributes
        self.medium = medium
        self.laser = laser
        self.grid = grid

        # Initialize parameters
        self.alpha = self.medium.raman_partition
        self.u_i = self.medium.ionization_energy
        self.n_0 = self.medium.refraction_index_linear
        self.n_2 = self.medium.refraction_index_nonlinear
        self.w_r = self.medium.raman_rotational_frequency
        self.w_0 = self.laser.frequency_0
        self.tau = self.medium.drude_time
        self.dt = self.grid.t_res

        # Initialize functions
        self.init_densities()
        self.init_coefficients()
        self.init_operators()

    def init_densities(self):
        """Initialize density parameters."""
        rho_c = eps_0 * m_e * (self.w_0 / q_e) ** 2
        self.n_k = np.ceil(self.u_i * q_e / (hbar * self.w_0))
        self.sigma_0 = (self.w_0**2 * self.tau) / (
            (self.n_0 * c_light * rho_c) * (1 + (self.w_0 * self.tau) ** 2)
        )

    def init_coefficients(self):
        """Initialize equations coefficients."""
        self.mpi_c = self.medium.constant_mpi
        self.ava_c = self.sigma_0 / (self.u_i * q_e)

        if self.medium.has_raman:
            raman_damping = 1 / self.medium.raman_response_time
            raman_r0 = (raman_damping**2 + self.w_r**2) / self.w_r
            self.raman_ode1 = np.exp((-raman_damping + 1j * self.w_r) * self.dt)
            self.raman_ode2 = 0.5 * raman_r0 * self.dt
        else:
            self.raman_ode1 = 0.0
            self.raman_ode2 = 0.0

    def init_operators(self):
        """Initialize equation operators."""
        k_vacuum = self.w_0 / c_light
        self.plasma_c = -0.5 * self.sigma_0 * (1 + 1j * self.w_0 * self.tau)
        self.mpa_c = -0.5 * self.n_k * hbar * self.w_0

        if self.medium.has_raman:
            self.kerr_c = 1j * k_vacuum * (1 - self.alpha) * self.n_2
            self.raman_c = 1j * k_vacuum * self.alpha * self.n_2
        else:
            self.kerr_c = 1j * k_vacuum * self.n_2
            self.raman_c = 0.0
