"""Fourier Crank-Nicolson (FCN) solver module."""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags_array

from ..core.ionization import compute_ionization
from ..numerical.routines.density import compute_density
from ..numerical.routines.envelope import (
    compute_fft,
    compute_ifft,
    compute_nlin_rk4_frequency,
)
from ..numerical.routines.raman import compute_raman
from ..numerical.shared.fluence import compute_fluence
from ..numerical.shared.radius import compute_radius
from .base import SolverBase


class SolverFCN(SolverBase):
    """Fourier Crank-Nicolson class implementation."""

    def __init__(self, material, laser, grid, eqn, method_opt="rk4", ion_model="mpi"):
        """Initialize FCN solver.

        Parameters
        ----------
        material : object
            Contains the chosen medium parameters.
        laser : object
            Contains the laser input parameters.
        grid : object
            Contains the grid input parameters.
        eqn : object
            Contains the equation parameters.
        method_opt : str, default: "rk4"
            Nonlinear solver method chosen.
        ion_model : str, default: "mpi"
            Ionization model chosen.

        """
        # Initialize base class
        super().__init__(material, laser, grid, eqn, method_opt, ion_model)

        # Initialize FCN-specific arrays
        self.envelope_fourier_rt = np.zeros_like(self.envelope_rt)
        self.envelope_fourier_next_rt = np.zeros_like(self.envelope_rt)

        # Setup operators and initial condition
        self.setup_operators()
        self.setup_initial_condition()

    def compute_matrix(self, n_r, m_p, r_low, r_up, coef_m_s, coef_o_s):
        """
        Compute the three diagonals for the Crank-Nicolson array
        with centered differences.

        Parameters
        ----------
        n_r : integer
            Number of radial nodes.
        m_p : str
            Position of the Crank-Nicolson array ("left" or "right").
        r_low : (M-1,) array_like
            Lower diagonal elements.
        r_up : (M-1,) array_like
            Upper diagonal elements.
        coef_m_s : complex
            Main diagonal coefficient.
        coef_o_s : complex
            Off-diagonal coefficient.

        Returns
        -------
        lres : (3, M) ndarray
            Banded array for solving a large tridiagonal system.
        rres : sparse array
            Sparse array in CSR format for optimal matrix-vector product.

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

    def compute_envelope(self):
        """
        Compute one step of the generalized Crank-Nicolson scheme
        for envelope propagation.
        """
        self.envelope_fourier_rt[1:-1, :] = compute_fft(self.envelope_rt[1:-1, :])

        for ll in range(self.grid.td.t_nodes):
            matrix_cn_left = self.compute_matrix(
                self.grid.r_nodes,
                "left",
                self.diag_down,
                self.diag_up,
                self.matrix_cnt_left[ll],
                self.diff_operator[ll],
            )
            matrix_cn_right = self.compute_matrix(
                self.grid.r_nodes,
                "right",
                self.diag_down,
                self.diag_up,
                self.matrix_cnt_right[ll],
                -self.diff_operator[ll],
            )

            # Compute matrix-vector product using "DIA" sparse format
            rhs_linear = matrix_cn_right @ self.envelope_fourier_rt[:, ll]

            # Compute the left-hand side of the equation
            rhs = rhs_linear + self.nonlinear_rt[:, ll]

            # Solve the tridiagonal system using the banded solver
            self.envelope_fourier_next_rt[:, ll] = solve_banded(
                (1, 1), matrix_cn_left, rhs
            )

        self.envelope_next_rt[1:-1, :] = compute_ifft(
            self.envelope_fourier_next_rt[1:-1, :]
        )

    def solve_step(self):
        """Perform one propagation step."""
        # Compute ionization rate
        compute_ionization(
            self.envelope_rt[1:-1, :],
            self.ionization_rate[1:-1, :],
            self.ionization_sum[1:-1, :],
            self.material.number_photons,
            self.eqn.coefficient_f0,
            self.eqn.coefficient_nc,
            self.eqn.coefficient_gamma,
            self.eqn.coefficient_nu,
            self.eqn.coefficient_ion,
            self.eqn.coefficient_ofi,
            ion_model=self.ion_model,
            tol=1e-3,
            max_iter=250,
        )

        # Compute density evolution
        compute_density(
            self.envelope_rt[1:-1, :],
            self.density_rt[1:-1, :],
            self.ionization_rate[1:-1, :],
            self.density_rk4_stage[1:-1],
            self.grid.td.t_nodes,
            self.density_neutral,
            self.coefficient_ava,
            self.del_t,
        )

        # Compute Raman response if requested
        if self.use_raman:
            compute_raman(
                self.raman_rt[1:-1, :],
                self.draman_rt[1:-1, :],
                self.envelope_rt[1:-1, :],
                self.raman_rk4_stage[1:-1],
                self.draman_rk4_stage[1:-1],
                self.grid.td.t_nodes,
                self.eqn.raman_coefficient_1,
                self.eqn.raman_coefficient_2,
                self.del_t,
            )

        # Compute nonlinear part using RK4
        if self.method == "rk4":
            compute_nlin_rk4_frequency(
                self.envelope_rt[1:-1, :],
                self.density_rt[1:-1, :],
                self.raman_rt[1:-1, :],
                self.ionization_rate[1:-1, :],
                self.nonlinear_rt[1:-1, :],
                self.envelope_rk4_stage[1:-1],
                self.self_steepening,
                self.density_neutral,
                self.coefficient_plasma,
                self.coefficient_mpa,
                self.coefficient_kerr,
                self.coefficient_raman,
                self.del_z,
            )

        # Compute envelope equation
        self.compute_envelope()

        # Compute beam fluence and radius
        compute_fluence(
            self.envelope_next_rt[1:-1, :], self.fluence_r[1:-1], self.grid.del_t
        )
        compute_radius(self.fluence_r[1:-1], self.radius, self.grid.r_grid)

        # Update arrays for next step
        self.envelope_rt[:], self.envelope_next_rt[:] = (
            self.envelope_next_rt,
            self.envelope_rt,
        )
