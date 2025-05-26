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
        self.envelope_split_rt = np.empty_like(self.envelope_rt)

        # Setup operators and initial condition
        self.setup_operators()
        self.setup_initial_condition()

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
            diag_main[0], diag_main[-1] = coef_main, 0
            diag_upper[0] = -2 * coef_d

        diags = [diag_lower, diag_main, diag_upper]
        diags_ind = [-1, 0, 1]

        # For the right hand side matrix, which will be used for
        # computing a matrix-vector product, return the 'DIA' format
        # for tridiagonal matrices which is more efficient
        return diags_array(diags, offsets=diags_ind, format="dia")

    def setup_operators(self):
        """Setup FSS operators."""
        coefficient_diffraction = (
            0.25 * self.grid.del_z / (self.laser.wavenumber * self.grid.del_r**2)
        )
        coefficient_dispersion = (
            -0.25 * self.grid.del_z * self.material.constant_gvd / self.grid.del_t**2
        )

        # Setup Fourier propagator for dispersion
        self.propagator_fft = np.exp(
            -2j * coefficient_dispersion * (self.grid.w_grid * self.grid.del_t) ** 2
        )

        # Setup CN operators for diffraction
        matrix_constant = 1j * coefficient_diffraction
        self.matrix_cn_left = self.compute_matrix(
            self.grid.r_nodes, "left", matrix_constant
        )
        self.matrix_cn_right = self.compute_matrix(
            self.grid.r_nodes, "right", -matrix_constant
        )

    def compute_dispersion(self):
        """
        Compute one step of the FFT propagation scheme for dispersion.
        """
        self.envelope_split_rt[:] = compute_fft(
            self.propagator_fft * compute_ifft(self.envelope_rt),
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
        # Compute ionization rate
        compute_ionization(
            self.envelope_rt,
            self.ionization_rate,
            self.ionization_sum,
            self.material.number_photons,
            self.grid.r_nodes,
            self.grid.td.t_nodes,
            self.eqn.coefficient_f0,
            self.eqn.coefficient_nc,
            self.eqn.coefficient_gamma,
            self.eqn.coefficient_nu,
            self.eqn.coefficient_ion,
            self.eqn.coefficient_ofi,
            ion_model=self.ion_model,
            tol=1e-4,
        )

        # Compute density evolution
        compute_density(
            self.envelope_rt,
            self.density_rt,
            self.ionization_rate,
            self.density_rk4_stage,
            self.grid.td.t_nodes,
            self.density_neutral,
            self.coefficient_ava,
            self.del_t,
        )

        # Compute Raman response if requested
        if self.use_raman:
            compute_raman(
                self.raman_rt,
                self.draman_rt,
                self.envelope_rt,
                self.raman_rk4_stage,
                self.draman_rk4_stage,
                self.grid.td.t_nodes,
                self.eqn.raman_coefficient_1,
                self.eqn.raman_coefficient_2,
                self.del_t,
            )
        else:
            self.raman_rt.fill(0)

        # Compute dispersion part using FFT
        self.compute_dispersion()

        # Compute nonlinear part using RK4
        if self.method == "rk4":
            compute_nlin_rk4(
                self.envelope_split_rt,
                self.density_rt,
                self.raman_rt,
                self.ionization_rate,
                self.envelope_rk4_stage,
                self.nonlinear_rt,
                self.grid.td.t_nodes,
                self.density_neutral,
                self.coefficient_plasma,
                self.coefficient_mpa,
                self.coefficient_kerr,
                self.coefficient_raman,
                self.del_z,
            )
        else:  # to be defined in the future!
            pass

        # Compute envelope equation
        self.compute_envelope()

        # Compute beam fluence and radius
        compute_fluence(self.envelope_next_rt, self.fluence_r, self.grid.del_t)
        compute_radius(self.fluence_r, self.radius, self.grid.r_grid)

        # Update arrays for next step
        self.envelope_rt[:], self.envelope_next_rt[:] = (
            self.envelope_next_rt,
            self.envelope_rt,
        )
