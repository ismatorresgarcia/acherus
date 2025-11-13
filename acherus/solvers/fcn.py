"""Solver module for Fourier-Crank-Nicolson (FCN) scheme."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.constants import c as c_light
from scipy.linalg import solve_banded
from scipy.sparse import diags_array

from ..functions.density import compute_density, compute_density_rk4
from ..functions.fft_backend import compute_fft, compute_ifft
from ..functions.fluence import compute_fluence
from ..functions.intensity import compute_intensity
from ..functions.ionization import compute_ion_rate
from ..functions.nonlinear import compute_nonlinear_w_ab2
from ..functions.radius import compute_radius
from ..functions.raman import compute_raman
from .base import SolverBase


class SolverFCN(SolverBase):
    """Fourier-Crank-Nicolson class implementation."""

    def __init__(
        self,
        config,
        dispersion,
        medium,
        laser,
        grid,
        eqn,
        ion,
        output
    ):
        """Initialize FCN solver.

        Parameters
        ----------
        config: object
            Contains the simulation options.
        dispersion : object
            Contains the dispersion medium properties.
        medium : object
            Contains the chosen medium parameters.
        laser : object
            Contains the laser input parameters.
        grid : object
            Contains the grid input parameters.
        eqn : object
            Contains the equation parameters.
        ion : object
            Contains the ionization model parameters.
        output : object
            Contains the output manager methods for saving propagation results.
        """

        # Initialize base class
        super().__init__(
            config,
            dispersion,
            medium,
            laser,
            grid,
            eqn,
            ion,
            output
        )

        # Initialize FCN-specific arrays
        self.envelope_fourier = np.zeros_like(self.envelope_rt)
        self._nlin_tmp_w = np.empty_like(self.envelope_rt)

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

    def compute_dispersion_function(self, w_det, w_0):
        """
        Compute the dispersion operator using the full dispersion
        relation with Sellmeier formulas.

        Parameters
        ----------
        w_det : (N,) array_like
            Angular frequency detuning.
        w_0 : float
            Beam central frequency.

        Returns
        -------
        disp : (N,) ndarray
            Dispersion function for each frequency.

        """
        w = w_det + w_0

        _, k_w, _ = self.dispersion.properties(w)

        return k_w - self.k_0 - self.k_1 * w_det

    def set_operators(self):
        """Set Fourier-Crank-Nicolson operators."""
        self.steep_op = 1 + self.w_grid / self.w_0
        diff_op = 0.25 * self.z_res / (self.k_0 * self.r_res**2 * self.steep_op)
        disp_op = 0.5 * self.z_res * self.compute_dispersion_function(self.w_grid, self.w_0)
        self.plasma_op = self.z_res * self.plasma_c / self.steep_op
        self.mpa_op = self.z_res * self.mpa_c
        self.kerr_op = self.z_res * self.kerr_c * self.steep_op
        self.raman_op = self.z_res * self.raman_c * self.steep_op

        self.mats_left = [None] * self.t_nodes
        self.mats_right = [None] * self.t_nodes

        def matrix_wrapper(ww):
            """Wrapper for parallel computation and storage of matrices."""
            return self.compute_matrices(self.r_nodes, diff_op[ww], disp_op[ww])

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
            rhs_linear = self.mats_right[ww] @ self.envelope_fourier[:, ww]
            rhs = rhs_linear + self.nonlinear_rt[:, ww]
            return solve_banded((1, 1), self.mats_left[ww], rhs, overwrite_b=True, check_finite=False)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(slice_wrapper, range(self.t_nodes)))

        for ww, result in enumerate(results):
            self.envelope_fourier[:, ww] = result

        self.envelope_rt[:-1, :] = compute_ifft(self.envelope_fourier[:-1, :])

    def solve_step(self, step):
        """Perform one propagation step."""
        compute_intensity(
            self.envelope_rt[:-1, :],
            self.intensity_rt[:-1, :],
        )
        compute_ion_rate(
            self.intensity_rt[:-1, :],
            self.ionization_rate[:-1, :],
            self.intensity_to_rate,
        )
        if self.dens_meth == "RK4":
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
                self.intensity_rt[:-1, :],
                self.density_rt[:-1, :],
                self.ionization_rate[:-1, :],
                self.t_grid,
                self.density_n,
                self._dens_init_buf[:-1],
                self.avalanche_c,
                self.dens_meth,
                self.dens_meth_ini_step,
                self.dens_meth_atol,
                self.dens_meth_rtol,
                self._dens_rhs_buf[:-1],
                self._dens_tmp_buf[:-1],
            )
        if self.use_raman:
            compute_raman(
                self.intensity_rt[:-1, :],
                self.raman_rt[:-1, :],
                self.raman_aux[:-1, :],
                self.raman_ode1,
                self.raman_ode2,
            )
        else:
            self.raman_rt.fill(0.0)
        compute_nonlinear_w_ab2(
            step,
            self.envelope_rt[:-1, :],
            self.density_rt[:-1, :],
            self.raman_rt[:-1, :],
            self.ionization_rate[:-1, :],
            self.nonlinear_next_rt[:-1, :],
            self.nonlinear_rt[:-1, :],
            self.density_n,
            self.plasma_op,
            self.mpa_op,
            self.kerr_op,
            self.raman_op,
            self.intensity_rt[:-1, :],
            self._nlin_tmp_t[:-1, :],
            self._nlin_tmp_w[:-1, :],
        )
        self.compute_envelope()
        compute_fluence(self.envelope_rt, self.t_grid, self.fluence_r)
        compute_radius(self.fluence_r, self.r_grid, self.radius)

        self.nonlinear_rt[:] = self.nonlinear_next_rt
