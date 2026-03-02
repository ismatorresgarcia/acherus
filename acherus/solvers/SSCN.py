"""Solver module for Split-Step Crank-Nicolson (SSCN) scheme."""

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

from ..functions.fft_backend import compute_fft, compute_ifft
from ..functions.fluence import compute_fluence
from ..functions.intensity import compute_intensity
from ..functions.ionization import compute_ion_rate
from ..functions.nonlinear import compute_nonlinear_nrsscn, compute_nonlinear_rsscn
from ..functions.raman import compute_raman
from .shared import Shared


class SSCN(Shared):
    """Split-Step Crank-Nicolson (SSCN) solver."""

    def __init__(self, config, medium, laser, grid, eqn, ion, output):
        """Initialize SSCN solver."""
        super().__init__(config, medium, laser, grid, eqn, ion, output)
        self.pml_par = config.pml_par
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

        self.nonlinear_next_rt = np.zeros_like(self.nonlinear_rt)
        self._nlin_tmp_t = np.empty(self.shape_rt, dtype=np.complex128)
        self._lin_rhs = np.empty(self.shape_rt, dtype=np.complex128)
        self._lin_tmp = np.empty(self.shape_rt, dtype=np.complex128)

        self.set_initial_conditions()
        self.set_operators()

    def compute_matrices(self, n, coef_d):
        """Compute diagonals for Crank-Nicolson matrices."""
        dr = self.r_res
        r = self.r_grid

        if self.pml_par is None:
            r_ind = np.arange(1, n - 1)
            radial_term = 0.5 / r_ind

            dl_left = np.zeros(n - 1, dtype=np.complex128)
            d_left = np.full(n, 1 + 2j * coef_d, dtype=np.complex128)
            du_left = np.zeros(n - 1, dtype=np.complex128)

            lower_diag = -1j * coef_d * (1 - radial_term)
            upper_diag = -1j * coef_d * (1 + radial_term)
            dl_left[:-1] = lower_diag
            du_left[1:] = upper_diag

            d_left[0], d_left[-1] = 1 + 4j * coef_d, 1
            du_left[0], dl_left[-1] = -4j * coef_d, 0

            dl_right = np.conjugate(dl_left)
            d_right = np.conjugate(d_left)
            du_right = np.conjugate(du_left)

            d_right[0], d_right[-1] = np.conjugate(d_left[0]), 0
            du_right[0], dl_right[-1] = np.conjugate(du_left[0]), 0
        else:
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
            d_left = 1 + 2j * coef_d * a_r
            du_left = np.zeros(n - 1, dtype=np.complex128)

            dl_left[:-1] = -1j * coef_d * (a_r_inner - 0.5 * dr * b_r)
            du_left[1:] = -1j * coef_d * (a_r_inner + 0.5 * dr * b_r)

            d_left[0], d_left[-1] = 1 + 4j * coef_d, 1
            du_left[0], dl_left[-1] = -4j * coef_d, 0

            dl_right = np.conjugate(dl_left)
            d_right = np.conjugate(d_left)
            du_right = np.conjugate(du_left)

            d_right[0], d_right[-1] = np.conjugate(d_left[0]), 0
            du_right[0], dl_right[-1] = np.conjugate(du_left[0]), 0

        return dl_left, d_left, du_left, dl_right, d_right, du_right

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
        dl_left, d_left, du_left, dl_right, d_right, du_right = self.compute_matrices(
            self.r_nodes, diff_op
        )

        dl_fac, d_fac, du_fac, du2_fac, ipiv_fac, _ = self._gttrf(
            dl_left,
            d_left,
            du_left,
            overwrite_dl=True,
            overwrite_d=True,
            overwrite_du=True,
        )
        self.mat_left = (dl_fac, d_fac, du_fac, du2_fac, ipiv_fac)
        self.mat_right = (dl_right, d_right, du_right)

    def compute_dispersion_step(self):
        """Compute one step of the FFT propagation scheme for dispersion."""
        self.envelope_rt[:-1, :] = compute_fft(
            self.disp_exp * compute_ifft(self.envelope_rt[:-1, :]),
        )

    def compute_envelope(self):
        """Compute one step of generalized Crank-Nicolson envelope propagation."""
        dl_right, d_right, du_right = self.mat_right
        rhs = self._lin_rhs
        tmp = self._lin_tmp

        np.multiply(d_right[:, None], self.envelope_rt, out=rhs)
        np.multiply(dl_right[:, None], self.envelope_rt[:-1, :], out=tmp[1:, :])
        rhs[1:, :] += tmp[1:, :]
        np.multiply(du_right[:, None], self.envelope_rt[1:, :], out=tmp[:-1, :])
        rhs[:-1, :] += tmp[:-1, :]
        rhs += self.nonlinear_rt

        dl_fac, d_fac, du_fac, du2_fac, ipiv_fac = self.mat_left
        self.envelope_rt[:], _ = self._gttrs(
            dl_fac,
            d_fac,
            du_fac,
            du2_fac,
            ipiv_fac,
            rhs,
            overwrite_b=True,
        )

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
        self._compute_raman()
        self.compute_dispersion_step()
        self._compute_nonlinear(step)
        self.compute_envelope()
        self.envelope_rt[-1, :] = 0

        self.nonlinear_rt[:] = self.nonlinear_next_rt
        compute_fluence(self.envelope_rt, self.t_grid, self.fluence_r)
