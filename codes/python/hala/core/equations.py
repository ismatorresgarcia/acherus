"""
Nonlinear envelope equation and density evolution parameters.
"""


class NEEParameters:
    """Pulse propagation and electron density evolution
    parameters for the final numerical scheme."""

    def __init__(self, const, medium, laser):
        # Initialize typical parameters
        self.frequency_0 = laser.input_frequency_0
        self.frequency_tau = self.frequency_0 * medium.drude_collision_time

        # Initialize main function parameters
        self._init_densities(const, medium, laser)
        self._init_coefficients(medium)
        self._init_operators(const, medium, laser)

    def _init_densities(self, const, medium, laser):
        "Initialize density parameters."
        self.density_critical = (
            const.electric_permittivity_0
            * const.electron_mass
            * (self.frequency_0 / const.electron_charge) ** 2
        )
        self.bremsstrahlung_cross_section_0 = (
            laser.input_wavenumber_0 * self.frequency_tau
        ) / (
            (medium.refraction_index_linear * self.density_critical)
            * (1 + self.frequency_tau**2)
        )

    def _init_coefficients(self, medium):
        "Initialize equation coefficients."
        self.coefficient_ofi = medium.constant_mpi
        self.coefficient_ava = (
            self.bremsstrahlung_cross_section_0 / medium.ionization_energy
        )

        if medium.has_raman:
            self.raman_response_frequency = 1 / medium.raman_response_time
            self.raman_coefficient_1 = (
                self.raman_response_frequency**2 + medium.raman_rotational_frequency**2
            )
            self.raman_coefficient_2 = -2 * self.raman_response_frequency
        else:
            self.raman_coefficient_1 = 0
            self.raman_coefficient_2 = 0

    def _init_operators(self, const, medium, laser):
        "Initialize equation operators."
        # Plasma coefficient calculation
        self.coefficient_plasma = (
            -0.5
            * self.bremsstrahlung_cross_section_0
            * (1 + const.imaginary_unit * self.frequency_tau)
        )

        # MPA coefficient calculation
        self.coefficient_mpa = -0.5 * medium.constant_mpa

        # Kerr coefficient calculation
        if medium.has_raman:
            self.coefficient_kerr = (
                const.imaginary_unit
                * laser.input_wavenumber_0
                * (1 - medium.raman_partition)
                * medium.refraction_index_nonlinear
            )

            # Raman coefficient calculation
            self.coefficient_raman = (
                const.imaginary_unit
                * laser.input_wavenumber_0
                * medium.raman_partition
                * medium.refraction_index_nonlinear
            )
        else:
            self.coefficient_kerr = (
                const.imaginary_unit
                * laser.input_wavenumber_0
                * medium.refraction_index_nonlinear
            )
            self.coefficient_raman = 0
