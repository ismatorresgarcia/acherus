"""
Nonlinear envelope equation and density evolution parameters.
"""

from scipy.constants import e as q_electron
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import m_e as m_electron


class EquationParameters:
    """Pulse propagation and electron density evolution
    parameters for the final numerical scheme."""

    def __init__(self, material, laser):
        # Initialize typical parameters
        self.frequency_0 = laser.frequency_0
        self.frequency_tau = self.frequency_0 * material.drude_collision_time

        # Initialize main function parameters
        self._init_densities(material, laser)
        self._init_coefficients(material)
        self._init_operators(material, laser)

    def _init_densities(self, material, laser):
        "Initialize density parameters."
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
        "Initialize equation coefficients."
        self.coefficient_ofi = material.constant_mpi
        self.coefficient_ava = (
            self.bremsstrahlung_cross_section_0 / material.ionization_energy
        )

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
        "Initialize equation operators."
        # Plasma coefficient calculation
        self.coefficient_plasma = (
            -0.5 * self.bremsstrahlung_cross_section_0 * (1 + 1j * self.frequency_tau)
        )

        # MPA coefficient calculation
        self.coefficient_mpa = -0.5 * material.constant_mpa

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
