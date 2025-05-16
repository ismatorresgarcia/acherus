"""
Nonlinear envelope equation and density evolution parameters.
"""

import numpy as np
from scipy.constants import e as q_electron
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar
from scipy.constants import m_e as m_electron
from scipy.constants import physical_constants
from scipy.special import gamma as eu_gamma


class EquationParameters:
    """Pulse propagation and electron density evolution
    parameters for the final numerical scheme."""

    def __init__(self, material, laser, ion_model="mpi"):
        # Initialize typical parameters
        self.frequency_0 = laser.frequency_0
        self.frequency_tau = self.frequency_0 * material.drude_collision_time
        self.ion_energy = material.ionization_energy

        # Initialize ionization model
        self.ion_model = ion_model

        # Initialize main function parameters
        self._init_densities(material, laser)
        self._init_coefficients(material)
        self._init_operators(material, laser)

    def _init_densities(self, material, laser):
        """Initialize density parameters."""
        self.density_critical = (
            eps_0 * m_electron * (self.frequency_0 / q_electron) ** 2
        )
        self.bremsstrahlung_cross_section_0 = (
            laser.wavenumber_0 * self.frequency_tau
        ) / (
            (material.refraction_index_linear * self.density_critical)
            * (1 + self.frequency_tau**2)
        )

    def _init_coefficients(self, material):
        """Initialize equations coefficients."""
        # PPT ionization rate coefficients
        w_atomic_u = 1 / physical_constants["atomic unit of time"][0]
        f_atomic_u = physical_constants["atomic unit of electric field"][0]
        hartree_u = physical_constants["Hartree energy"][0]
        self.coefficient_f0 = f_atomic_u * (2 * self.ion_energy / hartree_u) ** 1.5
        self.coefficient_gamma = (
            self.frequency_0 * np.sqrt(2 * m_electron * self.ion_energy) / q_electron
        )
        self.coefficient_nu = self.ion_energy / (hbar * self.frequency_0)
        self.coefficient_ns = material.effective_charge / np.sqrt(
            2 * self.ion_energy / hartree_u
        )
        c_effective = 2 ** (2 * self.coefficient_ns) / (
            self.coefficient_ns * eu_gamma(2 * self.coefficient_ns)
        )
        self.coefficient_ion = (
            w_atomic_u
            * 4
            * np.sqrt(2)
            * c_effective
            * self.ion_energy
            / (np.pi * hartree_u)
        )

        # Density equation coefficients
        self.coefficient_ofi = material.constant_mpi
        self.coefficient_ava = self.bremsstrahlung_cross_section_0 / self.ion_energy

        # Raman equation coefficients
        if material.has_raman:
            self.raman_response_frequency = 1 / material.raman_response_time
            self.raman_coefficient_1 = (
                self.raman_response_frequency**2
                + material.raman_rotational_frequency**2
            )
            self.raman_coefficient_2 = -2 * self.raman_response_frequency
        else:
            self.raman_coefficient_1 = 0
            self.raman_coefficient_2 = 0

    def _init_operators(self, material, laser):
        """Initialize equation operators."""
        # Plasma coefficient calculation
        self.coefficient_plasma = (
            -0.5 * self.bremsstrahlung_cross_section_0 * (1 + 1j * self.frequency_tau)
        )

        # MPA coefficient calculation
        self.coefficient_mpa = (
            -0.5 * material.number_photons * hbar * laser.wavenumber_0
        )

        # Kerr coefficient calculation
        if material.has_raman:
            self.coefficient_kerr = (
                1j
                * laser.wavenumber_0
                * (1 - material.raman_partition)
                * material.refraction_index_nonlinear
            )

            # Raman coefficient calculation
            self.coefficient_raman = (
                1j
                * laser.wavenumber_0
                * material.raman_partition
                * material.refraction_index_nonlinear
            )
        else:
            self.coefficient_kerr = (
                1j * laser.wavenumber_0 * material.refraction_index_nonlinear
            )
            self.coefficient_raman = 0
