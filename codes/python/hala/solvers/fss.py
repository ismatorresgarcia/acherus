"""Fourier Split-Step (FSS) solver module."""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags_array

from ..numerical.routines.density import solve_density
from ..numerical.routines.envelope import (
    frequency_domain,
    solve_nonlinear_rk4,
    time_domain,
)
from ..numerical.routines.raman import solve_scattering
from ..numerical.shared.fluence import calculate_fluence
from ..numerical.shared.radius import calculate_radius
from .base import SolverBase


class SolverFSS(SolverBase):
    """Fourier Split-Step class implementation for cylindrical coordinates."""

    def __init__(self, const, medium, laser, grid, nee, method_opt="rk4"):
        """Initialize FSS class.

        Parameters:
        - const: Constants object with physical constants
        - medium: MediumParameters object with medium properties
        - laser: LaserPulseParameters object with laser properties
        - grid: GridParameters object with grid definition
        - nee: NEEParameters object with equation parameters
        - method_opt: Nonlinear solver method (default: "rk4")
        """
        # Initialize base class
        super().__init__(const, medium, laser, grid, nee, method_opt)

        # Initialize FSS-specific arrays
        self.envelope_split_rt = np.empty_like(self.envelope_rt)

        # Setup operators and initial condition
        self.setup_operators()
        self.setup_initial_condition()

    def create_crank_nicolson_matrix(self, n_r, m_p, coef_d):
        """
        Set the three diagonals for the Crank-Nicolson array with centered differences.

        Parameters:
        - n_r: number of radial nodes
        - m_p: position of the Crank-Nicolson array (left or right)
        - coef_d: coefficient for the diagonal elements

        Returns:
        - For LEFT: banded matrix array for solving a large tridiagonal system
        - For RIGHT: sparse matrix in CSR format for optimal matrix-vector product
        """
        coef_main = 1 + 2 * coef_d
        r_ind = np.arange(1, n_r - 1)

        diag_lower = -coef_d * (1 - 0.5 / r_ind)
        diag_main = np.full(n_r, coef_main)
        diag_upper = -coef_d * (1 + 0.5 / r_ind)

        diag_lower = np.append(diag_lower, [0])
        diag_upper = np.insert(diag_upper, 0, [0])
        if m_p.upper() == "LEFT":
            # Boundary conditions for the left matrix
            diag_main[0], diag_main[-1] = coef_main, 1
            diag_upper[0] = -2 * coef_d

            band_matrix = np.zeros((3, n_r), dtype=complex)
            band_matrix[0, 1:] = diag_upper
            band_matrix[1, :] = diag_main
            band_matrix[2, :-1] = diag_lower

            # For the left hand side matrix, which will be used for
            # solving a large tridiagonal system of linear equations, return the
            # diagonals for latter usage in the banded solver
            return band_matrix

        if m_p.upper() == "RIGHT":
            # Boundary conditions for the right matrix
            diag_main[0], diag_main[-1] = coef_main, 0
            diag_upper[0] = -2 * coef_d

        diags = [diag_lower, diag_main, diag_upper]
        diags_ind = [-1, 0, 1]

        # For the right hand side matrix, which will be used for
        # computing a matrix-vector product, return the 'CSR' format
        # for sparse matrices which is more efficient
        return diags_array(diags, offsets=diags_ind, format="csr")

    def setup_operators(self):
        """Setup FSS operators."""
        coefficient_diffraction = (
            0.25 * self.grid.del_z / (self.laser.input_wavenumber * self.grid.del_r**2)
        )
        coefficient_dispersion = (
            -0.25 * self.grid.del_z * self.medium.constant_gvd / self.grid.del_t**2
        )

        # Setup Fourier propagator for dispersion
        self.propagator_fft = np.exp(
            -2
            * self.const.imaginary_unit
            * coefficient_dispersion
            * (self.grid.w_grid * self.grid.del_t) ** 2
        )

        # Setup CN operators for diffraction
        matrix_constant = self.const.imaginary_unit * coefficient_diffraction
        self.matrix_cn_left = self.create_crank_nicolson_matrix(
            self.grid.nodes_r, "left", matrix_constant
        )
        self.matrix_cn_right = self.create_crank_nicolson_matrix(
            self.grid.nodes_r, "right", -matrix_constant
        )

    def solve_dispersion(self):
        """
        Solve one step of the FFT propagation scheme for dispersion.
        """
        self.envelope_split_rt[:] = time_domain(
            self.propagator_fft * frequency_domain(self.envelope_rt),
        )

    def solve_envelope(self):
        """
        Solve one step of the generalized Crank-Nicolson scheme
        for envelope propagation.
        """
        for ll in range(self.grid.nodes_t):
            # Solve matrix-vector product using CSR sparse format
            rhs_linear = self.matrix_cn_right @ self.envelope_split_rt[:, ll]

            # Compute the left-hand side of the equation
            lhs = rhs_linear + self.nonlinear_rt[:, ll]

            # Solve the tridiagonal system using the banded solver
            self.envelope_next_rt[:, ll] = solve_banded(
                (1, 1), self.matrix_cn_left, lhs
            )

    def solve_step(self):
        """Perform one propagation step."""
        # Solve density evolution
        solve_density(
            self.envelope_rt,
            self.density_rt,
            self.density_rk4_stage,
            self.grid.nodes_t,
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
                self.grid.nodes_t,
                self.nee.raman_coefficient_1,
                self.nee.raman_coefficient_2,
                self.del_t,
                self.del_t_2,
                self.del_t_6,
            )
        else:
            self.raman_rt.fill(0)

        # Solve dispersion part using FFT
        self.solve_dispersion()

        # Solve nonlinear part using RK4
        if self.method.upper() == "RK4":
            solve_nonlinear_rk4(
                self.envelope_split_rt,
                self.density_rt,
                self.raman_rt,
                self.envelope_rk4_stage,
                self.nonlinear_rt,
                self.grid.nodes_t,
                self.envelope_arguments,
                self.del_z,
                self.del_z_2,
                self.del_z_6,
            )
        else:  # to be defined in the future!
            pass

        # Solve envelope equation
        self.solve_envelope()

        # Calculate beam characteristics
        calculate_fluence(self.envelope_next_rt, self.fluence_r, self.grid.del_t)
        calculate_radius(self.fluence_r, self.radius, self.grid.r_grid)

        # Update arrays for next step
        self.envelope_rt[:], self.envelope_next_rt[:] = (
            self.envelope_next_rt,
            self.envelope_rt,
        )
