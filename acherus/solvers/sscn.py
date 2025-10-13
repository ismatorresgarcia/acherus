"""Split-Step Crank-Nicolson (SSCN) solver module."""

import numpy as np
from scipy.fft import fftfreq
from scipy.linalg import solve_banded
from scipy.sparse import diags_array

from ..config import ConfigOptions
from ..functions.density import compute_density, compute_density_rk4
from ..functions.fluence import compute_fluence
from ..functions.fourier import compute_fft, compute_ifft
from ..functions.intensity import compute_intensity
from ..functions.interp_w import compute_ionization
from ..functions.nonlinear import compute_nonlinear_ab2
from ..functions.radius import compute_radius
from ..functions.raman import compute_raman
from ..mesh.grid import GridParameters
from ..physics.equations import EquationParameters
from ..physics.media import MediumParameters
from ..physics.optics import LaserParameters
from .base import SolverBase


class SolverSSCN(SolverBase):
    """Fourier Split-Step class implementation for cylindrical coordinates."""

    def __init__(
        self,
        config: ConfigOptions,
        medium: MediumParameters,
        laser: LaserParameters,
        grid: GridParameters,
        eqn: EquationParameters,
    ):
        """Initialize SSCN solver.

        Parameters
        ----------
        config: object
            Contains the simulation options.
        medium : object
            Contains the chosen medium parameters.
        laser : object
            Contains the laser input parameters.
        grid : object
            Contains the grid input parameters.
        eqn : object
            Contains the equation parameters.
        """

        # Initialize base class
        super().__init__(
            config,
            medium,
            laser,
            grid,
            eqn,
        )

        # Initialize SSCN-specific arrays
        self.envelope_split_rt = np.zeros_like(self.envelope_rt)
        self.nonlinear_next_rt = np.zeros_like(self.nonlinear_rt)

        # Set initial conditions and operators
        self.set_initial_conditions()
        self.set_operators()

    def compute_matrices(self, n, coef_d):
        """
        Compute the three diagonals for the Crank-Nicolson matrices
        with centered differences.

        Parameters
        ----------
        n : integer
            Number of radial nodes.
        coef_d : complex
            Complex diffraction coefficient.

        Returns
        -------
        lres : (3, M) ndarray
            Banded array for solving a large tridiagonal system.
        rres : sparse array
            Sparse array in DIA format for optimal matrix-vector product.
        """
        r_ind = np.arange(1, n - 1)

        dl_left = np.zeros(n - 1, dtype=np.complex128)
        d_left = np.full(n, 1 + 2j * coef_d, dtype=np.complex128)
        du_left = np.zeros(n - 1, dtype=np.complex128)

        dl_left[:-1] = -1j * coef_d * (1 - 0.5 / r_ind)
        du_left[1:] = -1j * coef_d * (1 + 0.5 / r_ind)

        # Boundary conditions for left matrix
        d_left[0], d_left[-1] = 1 + 4j * coef_d, 1
        du_left[0], dl_left[-1] = -4j * coef_d, 0

        matrix_band = np.zeros((3, n), dtype=np.complex128)
        matrix_band[0, 1:] = du_left
        matrix_band[1, :] = d_left
        matrix_band[2, :-1] = dl_left

        dl_right = np.zeros(n - 1, dtype=np.complex128)
        d_right = np.full(n, 1 - 2j * coef_d, dtype=np.complex128)
        du_right = np.zeros(n - 1, dtype=np.complex128)

        dl_right[:-1] = 1j * coef_d * (1 - 0.5 / r_ind)
        du_right[1:] = 1j * coef_d * (1 + 0.5 / r_ind)

        # Boundary conditions for right matrix
        d_right[0], d_right[-1] = 1 - 4j * coef_d, 0
        du_right[0], dl_right[-1] = 4j * coef_d, 0

        diags = [dl_right, d_right, du_right]

        matrix_right = diags_array(diags, offsets=[-1, 0, 1], format="dia")
        return matrix_band, matrix_right

    def set_operators(self):
        """Set SSCN operators."""
        w_grid = 2 * np.pi * fftfreq(self.t_nodes, self.t_res)
        diff_c = 0.25 * self.z_res / (self.k_0 * self.r_res**2)
        disp_c = 0.25 * self.z_res * self.k_2 * w_grid**2
        self.disp_exp = np.exp(2j * disp_c)
        self.mat_left, self.mat_right = self.compute_matrices(self.r_nodes, diff_c)

    def compute_dispersion(self):
        """
        Compute one step of the FFT propagation scheme for dispersion.
        """
        self.envelope_split_rt[:-1, :] = compute_fft(
            self.disp_exp * compute_ifft(self.envelope_rt[:-1, :]),
        )

    def compute_envelope(self):
        """
        Compute one step of the generalized Crank-Nicolson scheme
        for envelope propagation.
        """
        # Compute matrix-vector product using "DIA" sparse format
        rhs_linear = self.mat_right @ self.envelope_split_rt

        # Compute the left-hand side of the equation
        rhs = rhs_linear + self.nonlinear_rt

        # Solve the tridiagonal system using the banded solver
        self.envelope_next_rt[:] = solve_banded((1, 1), self.mat_left, rhs)

    def solve_step(self, step):
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
                self.i_const
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
            compute_raman(
                self.intensity_rt[:-1, :],
                self.raman_rt[:-1, :],
                self.raman_aux[:-1, :],
                self.t_grid,
                self.raman_c1,
                self.raman_c2,
            )
        else:
            self.raman_rt.fill(0.0)
        self.compute_dispersion()
        compute_nonlinear_ab2(
            step,
            self.envelope_rt[:-1, :],
            self.density_rt[:-1, :],
            self.raman_rt[:-1, :],
            self.ionization_rate[:-1, :],
            self.nonlinear_next_rt[:-1, :],
            self.nonlinear_rt[:-1, :],
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
