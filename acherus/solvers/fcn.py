"""Fourier-Crank-Nicolson (FCN) solver module."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.fft import fftfreq
from scipy.linalg import solve_banded
from scipy.sparse import diags_array

from ..mathematics.routines.density import compute_density, compute_density_rk4
from ..mathematics.routines.nonlinear import compute_nonlinear_w_rk4
from ..mathematics.routines.raman import compute_raman, compute_raman_rk4
from ..mathematics.shared.fluence import compute_fluence
from ..mathematics.shared.fourier import compute_fft, compute_ifft
from ..mathematics.shared.intensity import compute_intensity
from ..mathematics.shared.ionization import compute_ionization
from ..mathematics.shared.radius import compute_radius
from .base import SolverBase


class SolverFCN(SolverBase):
    """Fourier Crank-Nicolson class implementation."""

    def __init__(
        self,
        material,
        laser,
        grid,
        eqn,
        method_d_opt="RK4",
        method_r_opt="RK4",
        method_nl_opt="RK4",
        ion_model="MPI",
    ):
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
        method_d_opt : str, default: "RK4"
            Density solver method chosen.
        method_r_opt : str, default: "RK4"
            Raman solver method chosen.
        method_nl_opt : str, default: "RK4"
            Nonlinear solver method chosen.
        ion_model : str, default: "MPI"
            Ionization model chosen.

        """
        # Initialize base class
        super().__init__(
            material,
            laser,
            grid,
            eqn,
            method_d_opt,
            method_r_opt,
            method_nl_opt,
            ion_model,
        )

        # Initialize FCN-specific arrays
        self.envelope_fourier = np.zeros_like(self.envelope_rt)
        self.envelope_fourier_next = np.zeros_like(self.envelope_rt)

        # Set initial conditions and operators
        self.set_initial_conditions()
        self.set_operators()

    def compute_matrices(self, n, coef_d, coef_p):
        """
        Compute the three diagonals for the Crank-Nicolson matrices
        with centered differences.

        Parameters
        ----------
        n : integer
            Number of radial nodes.
        coef_d : complex
            Complex diffraction coefficient.
        coef_p : complex
            Complex dispersion coefficient.

        Returns
        -------
        lres : (3, M) ndarray
            Banded array for solving a large tridiagonal system.
        rres : sparse array
            Sparse array in DIA format for optimal matrix-vector product.

        """
        r_ind = np.arange(1, n - 1)

        dl_left = np.zeros(n - 1, dtype=np.complex128)
        d_left = np.full(n, 1 + 2j * coef_d - 1j * coef_p, dtype=np.complex128)
        du_left = np.zeros(n - 1, dtype=np.complex128)

        dl_left[:-1] = -1j * coef_d * (1 - 0.5 / r_ind)
        du_left[1:] = -1j * coef_d * (1 + 0.5 / r_ind)

        # Boundary conditions for left matrix
        d_left[0], d_left[-1] = 1 + 4j * coef_d - 1j * coef_p, 1
        du_left[0], dl_left[-1] = -4j * coef_d, 0

        matrix_band = np.zeros((3, n), dtype=np.complex128)
        matrix_band[0, 1:] = du_left
        matrix_band[1, :] = d_left
        matrix_band[2, :-1] = dl_left

        dl_right = np.zeros(n - 1, dtype=np.complex128)
        d_right = np.full(n, 1 - 2j * coef_d + 1j * coef_p, dtype=np.complex128)
        du_right = np.zeros(n - 1, dtype=np.complex128)

        dl_right[:-1] = 1j * coef_d * (1 - 0.5 / r_ind)
        du_right[1:] = 1j * coef_d * (1 + 0.5 / r_ind)

        # Boundary conditions for right matrix
        d_right[0], d_right[-1] = 1 - 4j * coef_d + 1j * coef_p, 0
        du_right[0], dl_right[-1] = 4j * coef_d, 0

        diags = [dl_right, d_right, du_right]

        matrix_right = diags_array(diags, offsets=[-1, 0, 1], format="dia")
        return matrix_band, matrix_right

    def set_operators(self):
        """Set FCN operators."""
        w_grid = 2 * np.pi * fftfreq(self.t_nodes, self.t_res)
        self.shock_op = 1 + w_grid / self.w_0
        diff_c = 0.25 * self.z_res / (self.k_0 * self.r_res**2 * self.shock_op)
        disp_c = 0.25 * self.z_res * self.k_2 * w_grid**2

        self.mats_left = [None] * self.t_nodes
        self.mats_right = [None] * self.t_nodes

        def matrix_wrapper(ww):
            """Wrapper for parallel computation and storage of matrices."""
            return self.compute_matrices(self.r_nodes, diff_c[ww], disp_c[ww])

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(matrix_wrapper, range(self.t_nodes)))

        for ww, (mat_left, mat_right) in enumerate(results):
            self.mats_left[ww] = mat_left
            self.mats_right[ww] = mat_right

    def compute_envelope(self):
        """
        Compute one step of the generalized Fourier-Crank-Nicolson
        scheme for envelope propagation.
        """
        self.envelope_fourier[:-1, :] = compute_fft(self.envelope_rt[:-1, :])

        def slice_wrapper(ww):
            """Wrapper for parallel computation of each slice."""
            # Compute matrix-vector product using "DIA" sparse format
            rhs_linear = self.mats_right[ww] @ self.envelope_fourier[:, ww]

            # Compute the left-hand side of the equation
            rhs = rhs_linear + self.nonlinear_rt[:, ww]

            # Solve the tridiagonal system using the banded solver
            return solve_banded((1, 1), self.mats_left[ww], rhs)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(slice_wrapper, range(self.t_nodes)))

        for ww, result in enumerate(results):
            self.envelope_fourier_next[:, ww] = result

        self.envelope_next_rt[:-1, :] = compute_ifft(self.envelope_fourier_next[:-1, :])

    def solve_step(self):
        """Perform one propagation step."""
        intensity_f = compute_intensity(
            self.envelope_rt[:-1, :],
            self.intensity_rt[:-1, :],
            self.r_grid[:-1],
            self.t_grid,
        )
        if self.ion_model == "PPT":
            compute_ionization(
                self.intensity_rt[:-1, :],
                self.ionization_rate[:-1, :],
                self.number_photons,
                self.mpi_c,
                self.ion_model,
                self.peak_intensity,
                self.ppt_rate,
            )
        else:
            compute_ionization(
                self.intensity_rt[:-1, :],
                self.ionization_rate[:-1, :],
                self.number_photons,
                self.mpi_c,
                self.ion_model,
            )
        if self.method_d == "RK4":
            compute_density_rk4(
                self.intensity_rt[:-1, :],
                self.density_rt[:-1, :],
                self.ionization_rate[:-1, :],
                self.t_grid,
                self.density_n,
                self.density_ini,
                self.avalanche_c,
            )
        else:
            compute_density(
                intensity_f,
                self.density_rt[:-1, :],
                self.ionization_rate[:-1, :],
                self.r_grid[:-1],
                self.t_grid,
                self.density_n,
                self.density_ini,
                self.avalanche_c,
                self.method_d,
            )
        if self.use_raman:
            if self.method_r == "RK4":
                compute_raman_rk4(
                    self.raman_rt[:-1, :],
                    self.draman_rt[:-1, :],
                    self.intensity_rt[:-1, :],
                    self.t_grid,
                    self.raman_c1,
                    self.raman_c2,
                )
            else:
                compute_raman(
                    intensity_f,
                    self.raman_rt[:-1, :],
                    self.r_grid[:-1],
                    self.t_grid,
                    self.raman_c1,
                    self.raman_c2,
                    self.method_r,
                )
        else:
            self.raman_rt.fill(0.0)
        if self.method_nl == "RK4":
            compute_nonlinear_w_rk4(
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
        compute_fluence(self.envelope_next_rt, self.t_grid, self.fluence_r)
        compute_radius(self.fluence_r, self.r_grid, self.radius)

        self.envelope_rt[:], self.envelope_next_rt[:] = (
            self.envelope_next_rt,
            self.envelope_rt,
        )
