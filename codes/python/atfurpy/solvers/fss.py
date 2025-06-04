"""Fourier Split-Step (FSS) solver module."""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags_array

from ..core.ionization import compute_ionization
from ..numerical.routines.density import compute_density
from ..numerical.routines.envelope import compute_fft, compute_ifft, compute_nlin_rk4
from ..numerical.routines.raman import compute_raman
from ..numerical.shared.fluence import compute_fluence
from ..numerical.shared.radius import compute_radius
from .base import SolverBase


class SolverFSS(SolverBase):
    """Fourier Split-Step class implementation for cylindrical coordinates."""

    def __init__(self, material, laser, grid, eqn, method_opt="rk4", ion_model="mpi"):
        """Initialize FSS class.

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

        # Initialize FSS-specific arrays
        self.envelope_split_rt = np.zeros_like(self.envelope_rt)

        # Set operators and initial condition
        self.set_operators()
        self.set_initial_condition()

    def compute_matrix(self, n_r, m_p, coef_d):
        """
        Compute the three diagonals for the Crank-Nicolson array
        with centered differences.

        Parameters
        ----------
        n_r : integer
            Number of radial nodes.
        m_p : str
            Position of the Crank-Nicolson array ("left" or "right").
        coef_d : complex
            Coefficient for the diagonal elements.

        Returns
        -------
        lres : (3, M) ndarray
            Banded array for solving a large tridiagonal system.
        rres : sparse array
            Sparse array in CSR format for optimal matrix-vector product.
        """
        coef_main = 1 + 2 * coef_d
        r_ind = np.arange(1, n_r - 1)

        diag_lower = -coef_d * (1 - 0.5 / r_ind)
        diag_main = np.full(n_r, coef_main)
        diag_upper = -coef_d * (1 + 0.5 / r_ind)

        diag_lower = np.append(diag_lower, [0])
        diag_upper = np.insert(diag_upper, 0, [0])
        if m_p == "left":
            # Boundary conditions for the left matrix
            diag_main[0], diag_main[-1] = coef_main, 1
            diag_upper[0] = -2 * coef_d

            band_matrix = np.zeros((3, n_r), dtype=np.complex128)
            band_matrix[0, 1:] = diag_upper
            band_matrix[1, :] = diag_main
            band_matrix[2, :-1] = diag_lower

            # For the left hand side matrix, which will be used for
            # solving a large tridiagonal system of linear equations, return the
            # diagonals for latter usage in the banded solver
            return band_matrix

        if m_p == "right":
            # Boundary conditions for the right matrix
            diag_main[0], diag_main[-1] = coef_main, 0
            diag_upper[0] = -2 * coef_d

        diags = [diag_lower, diag_main, diag_upper]
        diags_ind = [-1, 0, 1]

        # For the right hand side matrix, which will be used for
        # computing a matrix-vector product, return the 'DIA' format
        # for tridiagonal matrices which is more efficient
        return diags_array(diags, offsets=diags_ind, format="dia")

    def set_operators(self):
        """Set FSS operators."""
        coefficient_diffraction = (
            0.25 * self.z_res / (self.laser.wavenumber * self.r_res**2)
        )
        coefficient_dispersion = (
            -0.25 * self.r_res * self.material.constant_gvd / self.t_res**2
        )

        # Set Fourier propagator for dispersion
        self.propagator_fft = np.exp(
            -2j * coefficient_dispersion * (self.w_grid * self.t_res) ** 2
        )

        # Set CN operators for diffraction
        self.matrix_cn_left = self.compute_matrix(
            self.r_nodes, "left", 1j * coefficient_diffraction
        )
        self.matrix_cn_right = self.compute_matrix(
            self.r_nodes, "right", -1j * coefficient_diffraction
        )

    def compute_dispersion(self):
        """
        Compute one step of the FFT propagation scheme for dispersion.
        """
        self.envelope_split_rt[1:-1, :] = compute_fft(
            self.propagator_fft * compute_ifft(self.envelope_rt[1:-1, :]),
        )

    def compute_envelope(self):
        """
        Compute one step of the generalized Crank-Nicolson scheme
        for envelope propagation.
        """
        # Compute matrix-vector product using "DIA" sparse format
        rhs_linear = self.matrix_cn_right @ self.envelope_split_rt

        # Compute the left-hand side of the equation
        rhs = rhs_linear + self.nonlinear_rt

        # Solve the tridiagonal system using the banded solver
        self.envelope_next_rt[:] = solve_banded((1, 1), self.matrix_cn_left, rhs)

    def solve_step(self):
        """Perform one propagation step."""
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
            tol=1e-4,
            max_iter=250,
        )
        compute_density(
            self.envelope_rt[1:-1, :],
            self.density_rt[1:-1, :],
            self.ionization_rate[1:-1, :],
            self.density_rk4_stage[1:-1],
            self.t_nodes,
            self.density_neutral,
            self.coefficient_ava,
            self.t_res,
        )
        if self.use_raman:
            compute_raman(
                self.raman_rt[1:-1, :],
                self.draman_rt[1:-1, :],
                self.envelope_rt[1:-1, :],
                self.raman_rk4_stage[1:-1],
                self.draman_rk4_stage[1:-1],
                self.t_nodes,
                self.eqn.raman_coefficient_1,
                self.eqn.raman_coefficient_2,
                self.t_res,
            )
        self.compute_dispersion()
        if self.method == "rk4":
            compute_nlin_rk4(
                self.envelope_split_rt[1:-1, :],
                self.density_rt[1:-1, :],
                self.raman_rt[1:-1, :],
                self.ionization_rate[1:-1, :],
                self.nonlinear_rt[1:-1, :],
                self.envelope_rk4_stage[1:-1],
                self.t_nodes,
                self.density_neutral,
                self.coefficient_plasma,
                self.coefficient_mpa,
                self.coefficient_kerr,
                self.coefficient_raman,
                self.z_res,
            )
        self.compute_envelope()
        compute_fluence(
            self.envelope_next_rt[1:-1, :], self.fluence_r[1:-1], self.t_res
        )
        compute_radius(self.fluence_r[1:-1], self.radius, self.r_grid)

        self.envelope_rt[:], self.envelope_next_rt[:] = (
            self.envelope_next_rt,
            self.envelope_rt,
        )
