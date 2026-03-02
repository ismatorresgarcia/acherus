"""Solver module for Fourier-Crank-Nicolson (FCN) scheme."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

from ..functions.fft_backend import compute_fft, compute_ifft
from ..functions.fluence import compute_fluence
from ..functions.intensity import compute_intensity
from ..functions.ionization import compute_ion_rate
from ..functions.nonlinear import compute_nonlinear_nrfcn, compute_nonlinear_rfcn
from ..functions.raman import compute_raman
from .shared import Shared


class FCN(Shared):
    """Fourier-Crank-Nicolson (FCN) solver."""

    def __init__(self, config, medium, laser, grid, eqn, ion, output):
        """Initialize FCN solver."""
        super().__init__(config, medium, laser, grid, eqn, ion, output)
        self.pml_par = config.pml_par
        self._executor = ThreadPoolExecutor()
        self._gttrf, self._gttrs = get_lapack_funcs(
            ("gttrf", "gttrs"), dtype=np.complex128
        )

        self.has_raman = config.medium_par.raman_partition is not None

        if self.has_raman:
            self.raman_c = eqn.raman_c
            self.raman_ode1 = eqn.raman_ode1
            self.raman_ode2 = eqn.raman_ode2
            self.raman_rt = np.zeros(self.shape_rt, dtype=np.float64)
            self.raman_aux = np.zeros(self.shape_rt, dtype=np.complex128)
            self._compute_raman = self._compute_raman_yes
            self._compute_nonlinear = self._compute_nonlinear_raman
        else:
            self._compute_raman = self._compute_raman_no
            self._compute_nonlinear = self._compute_nonlinear_no_raman

        self.envelope_fourier = np.zeros_like(self.envelope_rt)
        self.nonlinear_next_rt = np.zeros(self.shape_rt, dtype=np.complex128)
        self._lin_rhs = np.empty(self.shape_rt, dtype=np.complex128)
        self._lin_tmp = np.empty(self.shape_rt, dtype=np.complex128)
        self._nlin_tmp_t = np.empty(self.shape_rt, dtype=np.complex128)
        self._nlin_tmp_w = np.empty_like(self.envelope_rt)

        self.set_initial_conditions()
        self.set_operators()

    def compute_matrices(self, n, coef_d, coef_p):
        """Compute tridiagonal coefficients for Crank-Nicolson matrices."""
        dr = self.r_res
        r = self.r_grid

        if self.pml_par is None:
            r_ind = np.arange(1, n - 1)
            radial_term = 0.5 / r_ind

            dl_left = np.zeros(n - 1, dtype=np.complex128)
            d_left = np.full(n, 1 + 2j * coef_d - 1j * coef_p, dtype=np.complex128)
            du_left = np.zeros(n - 1, dtype=np.complex128)

            lower_diag = -1j * coef_d * (1 - radial_term)
            upper_diag = -1j * coef_d * (1 + radial_term)
            dl_left[:-1] = lower_diag
            du_left[1:] = upper_diag

            d_left[0], d_left[-1] = 1 + 4j * coef_d - 1j * coef_p, 1
            du_left[0], dl_left[-1] = -4j * coef_d, 0

            dl_right = np.conjugate(dl_left)
            d_right = np.conjugate(d_left)
            du_right = np.conjugate(du_left)

            d_right[0], d_right[-1] = np.conjugate(d_left[0]), 0
            du_right[0], dl_right[-1] = np.conjugate(du_left[0]), 0

            return dl_left, d_left, du_left, dl_right, d_right, du_right

        r_max = r[-1]
        pml_rotation = 0.5 * np.sqrt(2) * (1 + 1j)
        pml_damping = self.pml_par.pml_damping

        pml_width_01 = self.pml_par.pml_width
        if pml_width_01 <= 0 or pml_width_01 >= 1:
            raise ValueError(
                "PML requires pml_width (delta) in the open interval (0, 1)."
            )

        r_boundary = r_max * (1.0 - pml_width_01)
        pml_width = r_max - r_boundary

        f_r = r.astype(np.complex128)
        p_r = np.zeros_like(f_r)
        dp_dr = np.zeros_like(f_r)

        mask = (r >= r_boundary) & (r < r_max)

        p_r[mask] = pml_damping * ((f_r[mask] - r_boundary) / pml_width) ** 2
        dp_dr[mask] = 2 * pml_damping * (f_r[mask] - r_boundary) / pml_width

        f_r[mask] += (
            pml_rotation
            * pml_damping
            * ((f_r[mask] - r_boundary) ** 3 - r_max**3)
            / (3 * pml_width**2)
        )

        s_r = 1 + pml_rotation * p_r
        ds_dr = pml_rotation * dp_dr

        a_r = 1 / s_r**2
        a_r_inner = a_r[1:-1]
        b_r = 1 / f_r[1:-1] - ds_dr[1:-1] / s_r[1:-1] ** 3

        dl_left = np.zeros(n - 1, dtype=np.complex128)
        d_left = 1 + 2j * coef_d * a_r - 1j * coef_p
        du_left = np.zeros(n - 1, dtype=np.complex128)

        dl_left[:-1] = -1j * coef_d * (a_r_inner - 0.5 * dr * b_r)
        du_left[1:] = -1j * coef_d * (a_r_inner + 0.5 * dr * b_r)

        d_left[0], d_left[-1] = 1 + 4j * coef_d - 1j * coef_p, 1
        du_left[0], dl_left[-1] = -4j * coef_d, 0

        dl_right = np.conjugate(dl_left)
        d_right = np.conjugate(d_left)
        du_right = np.conjugate(du_left)

        d_right[0], d_right[-1] = np.conjugate(d_left[0]), 0
        du_right[0], dl_right[-1] = np.conjugate(du_left[0]), 0

        return dl_left, d_left, du_left, dl_right, d_right, du_right

    def compute_dispersion(self, w_det, w_0):
        """Compute the dispersion function using Sellmeier formulas."""
        if self.dispersion_name == "full":
            w = w_det + w_0
            _, k_w, _ = self.medium.dispersion_properties(w)
            return k_w - self.k_0 - self.k_1 * w_det

        if self.dispersion_name == "partial":
            disp = 0.5 * self.dispersion_par.k2 * w_det**2
            if self.dispersion_par.k3 is not None:
                disp += (1.0 / 6.0) * self.dispersion_par.k3 * w_det**3
            if self.dispersion_par.k4 is not None:
                disp += (1.0 / 24.0) * self.dispersion_par.k4 * w_det**4
            if self.dispersion_par.k5 is not None:
                disp += (1.0 / 120.0) * self.dispersion_par.k5 * w_det**5
            return disp

        raise ValueError(f"Invalid dispersion model: '{self.dispersion_name}'.")

    def set_operators(self):
        """Set Fourier-Crank-Nicolson operators."""
        self.steep_op = 1 + self.w_grid / self.w_0
        self.focus_op = 1 + self.k_1 * self.w_grid / self.k_0
        diff_op = 0.25 * self.z_res / (self.k_0 * self.r_res**2 * self.focus_op)
        disp_op = 0.5 * self.z_res * self.compute_dispersion(self.w_grid, self.w_0)
        self.plasma_op = self.z_res * self.plasma_c / self.focus_op
        self.mpa_op = self.z_res * self.mpa_c * self.steep_op / self.focus_op
        self.kerr_op = self.z_res * self.kerr_c * self.steep_op**2 / self.focus_op

        if self.has_raman:
            self.raman_op = self.z_res * self.raman_c * self.steep_op**2 / self.focus_op

        self.mats_left = [None] * self.t_nodes
        self.mats_right = [None] * self.t_nodes

        def matrix_wrapper(ww):
            return self.compute_matrices(self.r_nodes, diff_op[ww], disp_op[ww])

        for ww, (dl_left, d_left, du_left, dl_right, d_right, du_right) in enumerate(
            self._executor.map(matrix_wrapper, range(self.t_nodes))
        ):
            dl_fac, d_fac, du_fac, du2_fac, ipiv_fac, _ = self._gttrf(
                dl_left,
                d_left,
                du_left,
                overwrite_dl=True,
                overwrite_d=True,
                overwrite_du=True,
            )
            self.mats_left[ww] = (dl_fac, d_fac, du_fac, du2_fac, ipiv_fac)
            self.mats_right[ww] = (dl_right, d_right, du_right)

    def compute_envelope(self):
        """Compute one generalized Fourier-Crank-Nicolson propagation step."""
        self.envelope_fourier[:-1, :] = compute_fft(self.envelope_rt[:-1, :])

        def slice_wrapper(ww):
            dl_right, d_right, du_right = self.mats_right[ww]
            env_w = self.envelope_fourier[:, ww]
            rhs = self._lin_rhs[:, ww]
            tmp = self._lin_tmp[:, ww]

            np.multiply(d_right, env_w, out=rhs)
            np.multiply(dl_right, env_w[:-1], out=tmp[1:])
            rhs[1:] += tmp[1:]
            np.multiply(du_right, env_w[1:], out=tmp[:-1])
            rhs[:-1] += tmp[:-1]
            rhs += self.nonlinear_rt[:, ww]

            dl_fac, d_fac, du_fac, du2_fac, ipiv_fac = self.mats_left[ww]
            sol, _ = self._gttrs(
                dl_fac,
                d_fac,
                du_fac,
                du2_fac,
                ipiv_fac,
                rhs,
                overwrite_b=True,
            )
            return sol

        for ww, result in enumerate(
            self._executor.map(slice_wrapper, range(self.t_nodes))
        ):
            self.envelope_fourier[:, ww] = result

        self.envelope_rt[:-1, :] = compute_ifft(self.envelope_fourier[:-1, :])

    def _compute_raman_no(self):
        """Do not update Raman delayed response for non-Raman runs."""

    def _compute_raman_yes(self):
        """Update Raman delayed response for Raman runs."""
        compute_raman(
            self.intensity_rt[:-1, :],
            self.raman_rt[:-1, :],
            self.raman_aux[:-1, :],
            self.raman_ode1,
            self.raman_ode2,
        )

    def _compute_nonlinear_no_raman(self, step):
        """Compute nonlinear term for non-Raman FCN propagation."""
        compute_nonlinear_nrfcn(
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
            self._nlin_tmp_w[:-1, :],
        )

    def _compute_nonlinear_raman(self, step):
        """Compute nonlinear term for Raman FCN propagation."""
        compute_nonlinear_rfcn(
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
        self._compute_raman()
        self._compute_nonlinear(step)
        self.compute_envelope()
        self.envelope_rt[-1, :] = 0

        self.nonlinear_rt[:] = self.nonlinear_next_rt
        compute_fluence(self.envelope_rt, self.t_grid, self.fluence_r)

    def __del__(self):
        """Ensure worker threads are released when solver is garbage-collected."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
