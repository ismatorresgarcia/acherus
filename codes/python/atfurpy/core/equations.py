"""
Nonlinear envelope equation and density evolution parameters.
"""

import numpy as np
from scipy.constants import e as q_electron
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar
from scipy.constants import m_e as m_electron
from scipy.constants import physical_constants
from scipy.special import gamma as euler_gamma


class EquationParameters:
    """Pulse propagation and electron density evolution
    parameters for the final numerical scheme."""

    def __init__(self, material, laser, ion_model="mpi"):
        # Initialize class attributes
        self.material = material
        self.laser = laser
        self.ion_model = ion_model

        # Initialize typical parameters
        self.wavenumber_0 = self.laser.wavenumber_0
        self.frequency_0 = self.laser.frequency_0
        self.frequency_tau = self.frequency_0 * self.material.drude_collision_time
        self.ion_energy = self.material.ionization_energy

        # Initialize main function parameters
        self._init_densities()
        self._init_coefficients()
        self._init_operators()

    def _init_densities(self):
        """Initialize density parameters."""
        self.density_critical = (
            eps_0 * m_electron * (self.frequency_0 / q_electron) ** 2
        )
        self.cross_section_0 = (self.wavenumber_0 * self.frequency_tau) / (
            (self.material.refraction_index_linear * self.density_critical)
            * (1 + self.frequency_tau**2)
        )

    def _init_coefficients(self):
        """Initialize equations coefficients."""
        # PPT ionization rate coefficients
        w_au = np.float64(1 / physical_constants["atomic unit of time"][0])
        f_au = np.float64(physical_constants["atomic unit of electric field"][0])
        e_au = np.float64(physical_constants["Hartree energy in eV"][0])
        self.coefficient_f0 = f_au * np.sqrt((2 * self.ion_energy / e_au) ** 3)
        self.coefficient_nc = 1 / np.sqrt(2 * self.ion_energy / e_au)
        coefficient_nq = self.material.effective_charge**2 * self.coefficient_nc
        self.coefficient_gamma = self.frequency_0 * np.sqrt(
            2 * m_electron * self.ion_energy / q_electron
        )
        self.coefficient_nu = self.ion_energy * q_electron / (hbar * self.frequency_0)
        c_effective = 2 ** (2 * self.coefficient_nc) / (
            self.coefficient_nc
            * euler_gamma(
                1 + (2 - self.material.effective_charge**2) * self.coefficient_nc
            )
            * euler_gamma(coefficient_nq)
        )
        self.coefficient_ion = (
            w_au
            * (16 / 3)
            * (4 * np.sqrt(2) / np.pi)
            * c_effective
            * self.ion_energy
            / e_au
        )

        # Density equation coefficients
        self.coefficient_ofi = self.material.constant_mpi
        self.coefficient_ava = self.cross_section_0 / (self.ion_energy * q_electron)

        # Raman equation coefficients
        if self.material.has_raman:
            self.raman_response_frequency = 1 / self.material.raman_response_time
            self.raman_coefficient_1 = (
                self.raman_response_frequency**2
                + self.material.raman_rotational_frequency**2
            )
            self.raman_coefficient_2 = -2 * self.raman_response_frequency
        else:
            self.raman_coefficient_1 = 0.0
            self.raman_coefficient_2 = 0.0

    def _init_operators(self):
        """Initialize equation operators."""
        self.coefficient_plasma = (
            -0.5 * self.cross_section_0 * (1 + 1j * self.frequency_tau)
        )
        self.coefficient_mpa = (
            -0.5 * self.material.number_photons * hbar * self.wavenumber_0
        )

        if self.material.has_raman:
            self.coefficient_kerr = (
                1j
                * self.wavenumber_0
                * (1 - self.material.raman_partition)
                * self.material.refraction_index_nonlinear
            )
            self.coefficient_raman = (
                1j
                * self.wavenumber_0
                * self.material.raman_partition
                * self.material.refraction_index_nonlinear
            )
        else:
            self.coefficient_kerr = (
                1j * self.wavenumber_0 * self.material.refraction_index_nonlinear
            )
            self.coefficient_raman = 0.0
