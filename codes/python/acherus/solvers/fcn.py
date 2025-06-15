"""Fourier Crank-Nicolson (FCN) solver module."""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.fft import fftfreq
from scipy.linalg import solve_banded
from scipy.sparse import diags_array

from ..mathematics.routines.density import compute_density
from ..mathematics.routines.envelope import compute_nlin_rk4_w
from ..mathematics.routines.raman import compute_raman
from ..mathematics.shared.fluence import compute_fluence
from ..mathematics.shared.fourier import compute_fft, compute_ifft
from ..mathematics.shared.intensity import compute_intensity
from ..mathematics.shared.radius import compute_radius
from ..physics.ionization import compute_ionization
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
        self.envelope_fourier_rt = np.empty_like(self.envelope_rt)
        self.envelope_fourier_next_rt = np.empty_like(self.envelope_rt)

        # Set initial conditions, equation operators and batches
        self.set_initial_conditions()
        self.set_operators()
        self.set_frequency_batches(batch_size=270)

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
            diag_main[0], diag_main[-1] = coef_m_s, 0
            diag_upper[0] = -2 * coef_o_s

        diags = [diag_lower, diag_main, diag_upper]
        diags_ind = [-1, 0, 1]

        # For the right hand side matrix, which will be used for
        # computing a matrix-vector product, return the 'DIA' format
        # for tridiagonal matrices which is more efficient
        return diags_array(diags, offsets=diags_ind, format="dia")

    def set_operators(self):
        """Set FCN operators."""
        w_grid = 2 * np.pi * fftfreq(self.t_nodes, self.t_res)
        diff_c = 0.25 * self.z_res / (self.k_n * self.r_res**2 * self.shock_op)
        disp_c = 0.25 * self.z_res * self.k_pp * w_grid**2
        self.shock_op = 1 + (w_grid / self.w_0)

        # Set FCN coefficients
        self.diff_op = 1j * diff_c
        self.disp_op = 1j * disp_c
        self.matrix_cnt_left = 1 + 2 * self.diff_op - self.disp_op
        self.matrix_cnt_right = 1 - 2 * self.diff_op + self.disp_op

        # Set CN outer diagonals radial index dependence
        self.diag_down = 1 - 0.5 / np.arange(1, self.r_nodes - 1)
        self.diag_up = 1 + 0.5 / np.arange(1, self.r_nodes - 1)

    def set_frequency_batches(self, batch_size=200):
        """
        Set frequency batches for parallel processing
        the Crank-Nicolson propagation scheme in Fourier space.
        """
        t_nodes = self.t_nodes
        omega_indices = list(range(t_nodes))
        self.batches = [
            omega_indices[ll : ll + batch_size] for ll in range(0, t_nodes, batch_size)
        ]

    def compute_frequency_batch(self, omega_batch):
        results = []
        for omega in omega_batch:
            matrix_left = self.compute_matrix(
                self.r_nodes,
                "left",
                self.diag_down,
                self.diag_up,
                self.matrix_cnt_left[omega],
                self.diff_op[omega],
            )
            matrix_right = self.compute_matrix(
                self.r_nodes,
                "right",
                self.diag_down,
                self.diag_up,
                self.matrix_cnt_right[omega],
                -self.diff_op[omega],
            )

            # Compute matrix-vector product using "DIA" sparse format
            rhs_linear = matrix_right @ self.envelope_fourier_rt[:, omega]

            # Compute the left-hand side of the equation
            rhs = rhs_linear + self.nonlinear_rt[:, omega]

            # Solve the tridiagonal system using the banded solver
            fourier_next = solve_banded((1, 1), matrix_left, rhs)
            results.append((omega, fourier_next))

        return results

    def compute_envelope(self):
        """
        Compute one step of the generalized Crank-Nicolson scheme
        for envelope propagation.
        """
        self.envelope_fourier_rt[:-1, :] = compute_fft(self.envelope_rt[:-1, :])

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.compute_frequency_batch, batch)
                for batch in self.batches
            ]
            for future in as_completed(futures):
                for omega, fourier_next in future.result():
                    self.envelope_fourier_next_rt[:, omega] = fourier_next

        self.envelope_next_rt[:-1, :] = compute_ifft(
            self.envelope_fourier_next_rt[:-1, :]
        )

    def solve_step(self):
        """Perform one propagation step."""
        compute_intensity(self.envelope_rt[:-1, :], self.intensity_rt[:-1, :])
        compute_ionization(
            self.intensity_rt[:-1, :],
            self.ionization_rate[:-1, :],
            self.ionization_sum[:-1, :],
            self.number_photons,
            self.hydrogen_f0,
            self.hydrogen_nc,
            self.keldysh_c,
            self.index_c,
            self.ppt_c,
            self.mpi_c,
            ion_model=self.ion_model,
            tol=1e-4,
        )
        compute_density(
            self.intensity_rt[:-1, :],
            self.density_rt[:-1, :],
            self.ionization_rate[:-1, :],
            self.t_nodes,
            self.density_n,
            self.density_ini,
            self.avalanche_c,
            self.t_res,
        )
        if self.use_raman:
            compute_raman(
                self.raman_rt[:-1, :],
                self.draman_rt[:-1, :],
                self.intensity_rt[:-1, :],
                self.t_nodes,
                self.raman_c1,
                self.raman_c2,
                self.t_res,
            )
        else:
            self.raman_rt.fill(0.0)
        if self.method == "rk4":
            compute_nlin_rk4_w(
                self.envelope_rt[:-1, :],
                self.density_rt[:-1, :],
                self.raman_rt[:-1, :],
                self.ionization_rate[:-1, :],
                self.nonlinear_rt[:-1, :],
                self.shock_op,
                self.density_n,
                self.plasma_c,
                self.mpa_c,
                self.kerr_c,
                self.raman_c,
                self.z_res,
            )
        self.compute_envelope()
        compute_fluence(self.envelope_next_rt[:-1, :], self.fluence_r[:-1], self.t_grid)
        compute_radius(self.fluence_r[:-1], self.radius, self.r_grid)

        self.envelope_rt[:], self.envelope_next_rt[:] = (
            self.envelope_next_rt,
            self.envelope_rt,
        )
