"""Solver module for Split-Step Crank-Nicolson (SSCN) scheme."""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags_array

from ..functions.fft_backend import compute_fft, compute_ifft
from ..functions.fluence import compute_fluence
from ..functions.intensity import compute_intensity
from ..functions.ionization import compute_ion_rate
from ..functions.nonlinear import compute_nonlinear_nrsscn, compute_nonlinear_rsscn
from ..functions.raman import compute_raman
from .shared import Shared


class SSCN(Shared):
    """Split-Step Crank-Nicolson (SSCN) solver in cylindrical coordinates."""

    def __init__(self, config, medium, laser, grid, eqn, ion, output):
        """Initialize SSCN solver."""
        super().__init__(config, medium, laser, grid, eqn, ion, output)

        self.has_raman = config.medium_par.raman_partition is not None

        if self.has_raman:
            self.raman_c = eqn.raman_c
            self.raman_ode1 = eqn.raman_ode1
            self.raman_ode2 = eqn.raman_ode2
            self.raman_rt = np.zeros(self.shape_rt, dtype=np.float64)
            self.raman_aux = np.zeros(self.shape_rt, dtype=np.complex128)
            self._update_raman = self._update_raman_yes
            self._compute_nonlinear = self._compute_nonlinear_raman
        else:
            self._update_raman = self._update_raman_no
            self._compute_nonlinear = self._compute_nonlinear_no_raman

        self.nonlinear_next_rt = np.zeros_like(self.nonlinear_rt)
        self._nlin_tmp_t = np.empty(self.shape_rt, dtype=np.complex128)

        self.set_initial_conditions()
        self.set_operators()

    def compute_matrices(self, n, coef_d):
        """Compute diagonals for Crank-Nicolson matrices."""
        r_ind = np.arange(1, n - 1)

        dl_left = np.zeros(n - 1, dtype=np.complex128)
        d_left = np.full(n, 1 + 2j * coef_d, dtype=np.complex128)
        du_left = np.zeros(n - 1, dtype=np.complex128)

        dl_left[:-1] = -1j * coef_d * (1 - 0.5 / r_ind)
        du_left[1:] = -1j * coef_d * (1 + 0.5 / r_ind)

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

        d_right[0], d_right[-1] = 1 - 4j * coef_d, 0
        du_right[0], dl_right[-1] = 4j * coef_d, 0

        matrix_right = diags_array(
            [dl_right, d_right, du_right], offsets=[-1, 0, 1], format="dia"
        )
        return matrix_band, matrix_right

    def compute_dispersion(self, w_det, w_0):
        """Compute dispersion function using Sellmeier formulas."""
        w = w_det + w_0
        _, k_w, _ = self.medium.dispersion_properties(w)
        return k_w - self.k_0 - self.k_1 * w_det

    def set_operators(self):
        """Set SSCN operators."""
        diff_op = 0.25 * self.z_res / (self.k_0 * self.r_res**2)
        disp_op = 0.5 * self.z_res * self.compute_dispersion(self.w_grid, self.w_0)
        self.plasma_op = self.z_res * self.plasma_c
        self.mpa_op = self.z_res * self.mpa_c
        self.kerr_op = self.z_res * self.kerr_c
        if self.has_raman:
            self.raman_op = self.z_res * self.raman_c

        self.disp_exp = np.exp(2j * disp_op)
        self.mat_left, self.mat_right = self.compute_matrices(self.r_nodes, diff_op)

    def compute_dispersion_step(self):
        """Compute one step of the FFT propagation scheme for dispersion."""
        self.envelope_rt[:-1, :] = compute_fft(
            self.disp_exp * compute_ifft(self.envelope_rt[:-1, :]),
        )

    def compute_envelope(self):
        """Compute one step of generalized Crank-Nicolson envelope propagation."""
        rhs_linear = self.mat_right @ self.envelope_rt
        rhs = rhs_linear + self.nonlinear_rt
        self.envelope_rt[:] = solve_banded((1, 1), self.mat_left, rhs)

    def _update_raman_no(self):
        """No-op Raman update for non-Raman runs."""

    def _update_raman_yes(self):
        """Update Raman delayed response for Raman runs."""
        compute_raman(
            self.intensity_rt[:-1, :],
            self.raman_rt[:-1, :],
            self.raman_aux[:-1, :],
            self.raman_ode1,
            self.raman_ode2,
        )

    def _compute_nonlinear_no_raman(self, step):
        """Compute nonlinear term for non-Raman propagation."""
        compute_nonlinear_nrsscn(
            step,
            self.envelope_rt[:-1, :],
            self.density_rt[:-1, :],
            self.ionization_rate[:-1, :],
            self.nonlinear_next_rt[:-1, :],
            self.nonlinear_rt[:-1, :],
            self.density_n,
            self.plasma_op,
            self.mpa_op,
            self.kerr_op,
            self.intensity_rt[:-1, :],
            self._nlin_tmp_t[:-1, :],
        )

    def _compute_nonlinear_raman(self, step):
        """Compute nonlinear term for Raman propagation."""
        compute_nonlinear_rsscn(
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
        )

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
        self._compute_density()
        self._update_raman()
        self.compute_dispersion_step()
        self._compute_nonlinear(step)
        self.compute_envelope()

        self.nonlinear_rt[:] = self.nonlinear_next_rt
        compute_fluence(self.envelope_rt, self.t_grid, self.fluence_r)
