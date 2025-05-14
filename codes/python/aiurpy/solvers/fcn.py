"""Fourier Crank-Nicolson (FCN) solver module."""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags_array

from ..core.ionization import calculate_ionization
from ..numerical.routines.density import solve_density
from ..numerical.routines.envelope import (
    frequency_domain,
    solve_nonlinear_rk4_frequency,
    time_domain,
)
from ..numerical.routines.raman import solve_scattering
from ..numerical.shared.fluence import calculate_fluence
from ..numerical.shared.radius import calculate_radius
from .base import SolverBase


class SolverFCN(SolverBase):
    """Fourier Crank-Nicolson class implementation."""

    def __init__(self, material, laser, grid, eqn, method_opt="rk4", ion_model="mpi"):
        """Initialize FCN solver.

        Parameters:
        -> material: MediumParameters object with medium properties
        -> laser: LaserPulseParameters object with laser properties
        -> grid: GridParameters object with grid definition
        -> eqn: EquationParameters object with equation parameters
        -> method_opt: Nonlinear solver method (default: "rk4")
        -> ion_model: Ionization model to use (default: "mpi")
        """
        # Initialize base class
        super().__init__(material, laser, grid, eqn, method_opt, ion_model)

        # Initialize FCN-specific arrays
        self.envelope_fourier_rt = np.empty_like(self.envelope_rt)
        self.envelope_fourier_next_rt = np.empty_like(self.envelope_rt)

        # Setup operators and initial condition
        self.setup_operators()
        self.setup_initial_condition()

    def create_matrix(self, n_r, m_p, r_low, r_up, coef_m_s, coef_o_s):
        """
        Set the three diagonals for the Crank-Nicolson array with centered differences.

        Parameters:
        -> n_r: number of radial nodes
        -> m_p: position of the Crank-Nicolson array (left or right)
        -> r_low: lower diagonal coefficients
        -> r_up: upper diagonal coefficients
        -> coef_m_s: main diagonal coefficient
        -> coef_o_s: off-diagonal coefficient

        Returns:
        -> For left: banded matrix array for solving a large tridiagonal system
        -> For right: sparse matrix in CSR format for optimal matrix-vector product
        """
        diag_lower = -coef_o_s * r_low
        diag_main = np.full(n_r, coef_m_s)
        diag_upper = -coef_o_s * r_up

        diag_lower = np.append(diag_lower, [0])
        diag_upper = np.insert(diag_upper, 0, [0])
        if m_p == "left":
            # Boundary conditions for the left matrix
            diag_main[0], diag_main[-1] = coef_m_s, 1
            diag_upper[0] = -2 * coef_o_s

            band_matrix = np.zeros((3, n_r), dtype=complex)
            band_matrix[0, 1:] = diag_upper
            band_matrix[1, :] = diag_main
            band_matrix[2, :-1] = diag_lower

            # For the left hand side matrix, which will be used for
            # solving a large tridiagonal system of linear equations, return the
            # diagonals for latter usage in the banded solver
            return band_matrix

        if m_p == "right":
            # Boundary conditions for the right matrix
            diag_main[0], diag_main[-1] = coef_m_s, 0
            diag_upper[0] = -2 * coef_o_s

        diags = [diag_lower, diag_main, diag_upper]
        diags_ind = [-1, 0, 1]

        # For the right hand side matrix, which will be used for
        # computing a matrix-vector product, return the 'DIA' format
        # for tridiagonal matrices which is more efficient
        return diags_array(diags, offsets=diags_ind, format="dia")

    def setup_operators(self):
        """Setup FCN operators."""
        self.self_steepening = 1 + self.grid.w_grid / self.laser.frequency_0
        coefficient_diffraction = (
            0.25
            * self.grid.del_z
            / (self.laser.wavenumber * self.grid.del_r**2 * self.self_steepening)
        )
        coefficient_dispersion = (
            0.25 * self.grid.del_z * self.material.constant_gvd * self.grid.w_grid**2
        )

        # Setup FCN coefficients
        self.diff_operator = 1j * coefficient_diffraction
        self.disp_operator = 1j * coefficient_dispersion
        self.matrix_cnt_left = 1 + 2 * self.diff_operator - self.disp_operator
        self.matrix_cnt_right = 1 - 2 * self.diff_operator + self.disp_operator

        # Setup CN outer diagonals radial index dependence
        self.diag_down = 1 - 0.5 / np.arange(1, self.grid.r_nodes - 1)
        self.diag_up = 1 + 0.5 / np.arange(1, self.grid.r_nodes - 1)

    def solve_envelope(self):
        """
        Solve one step of the generalized Crank-Nicolson scheme for envelope
        propagation.
        """
        self.envelope_fourier_rt[:] = frequency_domain(self.envelope_rt)

        for ll in range(self.grid.td.t_nodes):
            matrix_cn_left = self.create_matrix(
                self.grid.r_nodes,
                "left",
                self.diag_down,
                self.diag_up,
                self.matrix_cnt_left[ll],
                self.diff_operator[ll],
            )
            matrix_cn_right = self.create_matrix(
                self.grid.r_nodes,
                "right",
                self.diag_down,
                self.diag_up,
                self.matrix_cnt_right[ll],
                -self.diff_operator[ll],
            )

            # Solve matrix-vector product using "DIA" sparse format
            rhs_linear = matrix_cn_right @ self.envelope_fourier_rt[:, ll]

            # Compute the left-hand side of the equation
            rhs = rhs_linear + self.nonlinear_rt[:, ll]

            # Solve the tridiagonal system using the banded solver
            self.envelope_fourier_next_rt[:, ll] = solve_banded(
                (1, 1), matrix_cn_left, rhs
            )

        self.envelope_next_rt[:] = time_domain(self.envelope_fourier_next_rt)

    def solve_step(self):
        """Perform one propagation step."""
        # Calculate ionization rate
        calculate_ionization(
            self.envelope_rt,
            self.ionization_rate,
            self.ionization_sum,
            self.material.number_photons,
            self.grid.r_nodes,
            self.grid.td.t_nodes,
            self.eqn.coefficient_f0,
            self.eqn.coefficient_ns,
            self.eqn.coefficient_gamma,
            self.eqn.coefficient_nu,
            self.eqn.coefficient_ion,
            self.eqn.coefficient_ofi,
            ion_model=self.ion_model,
            tol=1e-2,
        )

        # Solve density evolution
        solve_density(
            self.envelope_rt,
            self.density_rt,
            self.ionization_rate,
            self.density_rk4_stage,
            self.grid.td.t_nodes,
            self.density_arguments,
            self.del_t,
            self.del_t_2,
            self.del_t_6,
        )

        # Solve Raman response if requested
        if self.use_raman:
            solve_scattering(
                self.raman_rt,
                self.draman_rt,
                self.envelope_rt,
                self.raman_rk4_stage,
                self.draman_rk4_stage,
                self.grid.td.t_nodes,
                self.eqn.raman_coefficient_1,
                self.eqn.raman_coefficient_2,
                self.del_t,
                self.del_t_2,
                self.del_t_6,
            )
        else:
            self.raman_rt.fill(0)

        # Solve nonlinear part using RK4
        if self.method == "rk4":
            solve_nonlinear_rk4_frequency(
                self.envelope_rt,
                self.density_rt,
                self.raman_rt,
                self.ionization_rate,
                self.self_steepening,
                self.envelope_rk4_stage,
                self.nonlinear_rt,
                self.envelope_arguments,
                self.del_z,
                self.del_z_2,
                self.del_z_6,
            )
        else:  # to be defined in the future
            pass

        # Solve envelope equation
        self.solve_envelope()

        # Calculate beam fluence and radius
        calculate_fluence(self.envelope_next_rt, self.fluence_r, self.grid.del_t)
        calculate_radius(self.fluence_r, self.radius, self.grid.r_grid)

        # Update arrays for next step
        self.envelope_rt[:], self.envelope_next_rt[:] = (
            self.envelope_next_rt,
            self.envelope_rt,
        )
