"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse in cylindrical coordinates with radial symmetry.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Spectral (in frequency) Crank-Nicolson (CN) scheme.
    - Initial condition: Gaussian.
    - Boundary conditions: Neumann-Dirichlet (radial) and Periodic (temporal).

UPPE:          ∂E/∂z = i/(2k) ∇²E - ik''/2 ∂²E/∂t²


E: envelope.
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
t: time coordinate.
k: wavenumber (in the interacting media).
k'': GVD coefficient of 2nd order.
∇²: laplace operator (for the transverse direction).
"""

from dataclasses import dataclass

import numpy as np
from numpy.fft import fft, ifft
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def initial_condition(r, t, im, beam):
    """
    Set the post-lens chirped Gaussian beam.
    """
    return beam.amplitude * np.exp(
        -((r / beam.waist_0) ** 2)
        - 0.5 * im * beam.wavenumber * r**2 / beam.focal_length
        - (1 + im * beam.chirp) * (t / beam.peak_time) ** 2
    )


def crank_nicolson_matrix(n, lr, c):
    """
    Set the Crank-Nicolson sparse array in CSR format using the diagonals.
    """
    ind = np.arange(1, n - 1)

    diag_m1 = -c * (1 - 0.5 / ind)
    diag_0 = np.ones(n)
    diag_p1 = -c * (1 + 0.5 / ind)

    diag_m1 = np.append(diag_m1, [0])
    diag_p1 = np.insert(diag_p1, 0, [0])

    if lr == "left":
        diag_p1[0] = -2 * c
    else:
        diag_p1[0] = -2 * c

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]

    return diags_array(diags, offsets=offset, format="csr")


def calculate_nonlinear(e_c, w_c, equation):
    """
    Compute one step of the Adam-Bashforth scheme for the nonlinear terms.

    Parameters:
    - e_c: envelope at step k
    - w_c: pre-allocated array for Adam-Bashforth terms
    - media: dictionary with media parameters
    Compute one step of the Adam-Bashforth scheme for the nonlinear terms.
    """
    e_c_2 = np.abs(e_c) ** 2
    e_c_2k2 = np.abs(e_c) ** equation.mpa_exp
    w_c[:] = e_c * (equation.kerr_coef * e_c_2 + equation.mpa_coef * e_c_2k2)


def frequency_domain(e_c, fe_c, w_c, w_n, b):
    """
    Compute the FFT of the envelope and Adam-Bashforth terms.

    Parameters:
    - e_c: envelope at step k
    - fe_c: pre-allocated array for Fourier envelope at step k
    - w_c: current step nonlinear terms
    - w_n: previous step nonlinear terms
    - b: pre-allocated array for temporary results
    """
    fe_c[:] = fft(e_c, axis=1)
    b = fft(w_c[:, :], axis=1)
    w_c[:, :] = b
    b = fft(w_n[:, :], axis=1)
    w_n[:, :] = b


def solve_envelope(mats, arr, cffs):
    """
    Update Crank-Nicolson arrays for one frequency step.
    """
    lm, rm = mats
    fe_c, fe_n, w_c, w_n, b, c = arr
    lc, rc = cffs
    for l in range(fe_c.shape[1]):
        # Update matrices for current frequency
        lm.setdiag(lc[l])
        rm.setdiag(rc[l])
        # Set boundary conditions
        lm.data[-1] = 1
        rm.data[-1] = 0
        # Solve with Crank-Nicolson for current frequency
        b = rm @ fe_c[:, l]
        c = b + 1.5 * w_c[:, l] - 0.5 * w_n[:, l]
        fe_n[:, l] = spsolve(lm, c)


def time_domain(fe_c, e_c):
    """
    Compute the IFFT of the Fourier envelope at step k.
    """
    e_c[:] = ifft(fe_c, axis=1)


@dataclass
class UniversalConstants:
    "UniversalConstants."

    def __init__(self):
        self.light_speed = 299792458
        self.permittivity = 8.8541878128e-12
        self.pi = np.pi
        self.im_unit = 1j
        self.re_unit = 1


@dataclass
class MediaParameters:
    "Media parameters."

    def __init__(self, const):
        self.lin_ref_ind_water = 1.328
        self.nlin_ref_ind_water = 1.6e-20
        self.gvd_coef_water = 241e-28
        self.n_photons_water = 5
        self.mpa_cnt_water = 8e-64
        # self.int_factor = (
        #    0.5 * const.light_speed * const.permittivity * self.lin_ref_ind_water
        # )
        self.int_factor = const.re_unit


@dataclass
class BeamParameters:
    "Beam parameters."

    def __init__(self, const, media):
        # Basic parameters
        self.wavelength_0 = 800e-9
        self.waist_0 = 100e-6
        self.peak_time = 50e-15
        self.energy = 2.83e-6
        self.focal_length = 20
        self.chirp = -1

        # Derived parameters
        self.wavenumber_0 = 2 * const.pi / self.wavelength_0
        self.wavenumber = 2 * const.pi * media.lin_ref_ind_water / self.wavelength_0
        self.power = self.energy / (self.peak_time * np.sqrt(0.5 * const.pi))
        self.intensity = 2 * self.power / (const.pi * self.waist_0**2)
        self.amplitude = np.sqrt(self.intensity / media.int_factor)


class DomainParameters:
    "Spatial and temporal domain parameters."

    def __init__(self):
        # Radial domain
        self.ini_radi_coor = 0
        self.fin_radi_coor = 25e-4
        self.i_radi_nodes = 1500
        self.n_radi_nodes = self.i_radi_nodes + 2
        self.ini_radi_coor = 0

        # Distance domain
        self.ini_dist_coor = 0
        self.fin_dist_coor = 2e-2
        self.n_steps = 1000
        self.dist_index = 0
        self.dist_limit = 5

        # Time domain
        self.ini_time_coor = -200e-15
        self.fin_time_coor = 200e-15
        self.n_time_nodes = 4096

        self.setup_domain()

    def setup_domain(self):
        "Setup domain parameters."
        # Calculate steps
        self.radi_step_len = (self.fin_radi_coor - self.ini_radi_coor) / (
            self.n_radi_nodes - 1
        )
        self.dist_step_len = (self.fin_dist_coor - self.ini_dist_coor) / self.n_steps
        self.time_step_len = (self.fin_time_coor - self.ini_time_coor) / (
            self.n_time_nodes - 1
        )

        self.axis_node = int(-self.ini_radi_coor / self.radi_step_len)
        self.peak_node = self.n_time_nodes // 2

        self.frq_step_len = 2 * np.pi / (self.n_time_nodes * self.time_step_len)

        # Create arrays
        self.radi_array = np.linspace(
            self.ini_radi_coor, self.fin_radi_coor, self.n_radi_nodes
        )
        self.dist_array = np.linspace(
            self.ini_dist_coor, self.fin_dist_coor, self.n_steps + 1
        )
        self.time_array = np.linspace(
            self.ini_time_coor, self.fin_time_coor, self.n_time_nodes
        )
        w1 = np.linspace(
            0,
            np.pi / self.time_step_len - self.frq_step_len,
            self.n_time_nodes // 2,
        )
        w2 = np.linspace(
            -np.pi / self.time_step_len,
            -self.frq_step_len,
            self.n_time_nodes // 2,
        )
        self.frq_array = np.append(w1, w2)

        # Create 2D arrays
        self.radi_2d_array, self.time_2d_array = np.meshgrid(
            self.radi_array, self.time_array, indexing="ij"
        )


@dataclass
class EquationParameters:
    """Parameters for the final equation."""

    def __init__(self, const, media, beam, domain):
        self.mpa_exp = 2 * media.n_photons_water - 2
        self.kerr_coef = (
            const.im_unit
            * beam.wavenumber_0
            * media.nlin_ref_ind_water
            * domain.dist_step_len
            * media.int_factor
        )
        self.mpa_coef = (
            -0.5
            * media.mpa_cnt_water
            * domain.dist_step_len
            * media.int_factor ** (media.n_photons_water - 1)
        )


class SCNSolver:
    "Solver class."

    def __init__(self, const, media, beam, domain):
        self.const = const
        self.media = media
        self.beam = beam
        self.domain = domain
        self.equation = EquationParameters(const, media, beam, domain)

        # Initialize arrays and operators
        shape = (self.domain.n_radi_nodes, self.domain.n_time_nodes)
        self.envelope = np.empty(shape, dtype=complex)
        self.next_envelope = np.empty_like(self.envelope)
        self.dist_envelope = np.empty(
            [
                self.domain.n_radi_nodes,
                self.domain.dist_limit + 1,
                self.domain.n_time_nodes,
            ],
            dtype=complex,
        )
        self.axis_envelope = np.empty(
            [self.domain.n_steps + 1, self.domain.n_time_nodes], dtype=complex
        )
        self.peak_envelope = np.empty(
            [self.domain.n_radi_nodes, self.domain.n_steps + 1], dtype=complex
        )
        self.fourier_envelope = np.empty_like(self.envelope)
        self.next_fourier_envelope = np.empty_like(self.envelope)
        self.w_array = np.empty_like(self.envelope)
        self.next_w_array = np.empty_like(self.envelope)
        self.b_array = np.empty(self.domain.n_radi_nodes, dtype=complex)
        self.c_array = np.empty_like(self.b_array)
        self.d_array = np.empty_like(self.envelope)
        self.k_array = np.empty(self.domain.dist_limit + 1, dtype=int)

        # Setup operators and initial condition
        self.setup_operators()
        self.set_initial_condition()

    def setup_operators(self):
        """Setup operators."""
        # Setup CN parameters
        self.delta_r = (
            0.25
            * self.domain.dist_step_len
            / (self.beam.wavenumber * self.domain.radi_step_len**2)
        )
        self.delta_t = 0.25 * self.domain.dist_step_len * self.media.gvd_coef_water
        self.mat_cnt_1 = self.const.im_unit * self.delta_r

        # Setup matrices
        fourier_coeff = self.const.im_unit * self.delta_t * self.domain.frq_array**2
        self.matrix_cnt_2 = 1 - 2 * self.mat_cnt_1 + fourier_coeff
        self.matrix_cnt_3 = 1 + 2 * self.mat_cnt_1 - fourier_coeff

        self.operators = (
            crank_nicolson_matrix(self.domain.n_radi_nodes, "left", self.mat_cnt_1),
            crank_nicolson_matrix(self.domain.n_radi_nodes, "right", -self.mat_cnt_1),
        )
        self.vectors = (
            self.fourier_envelope,
            self.next_fourier_envelope,
            self.w_array,
            self.next_w_array,
            self.b_array,
            self.c_array,
        )
        self.entries = (self.matrix_cnt_3, self.matrix_cnt_2)

    def set_initial_condition(self):
        "Set the initial condition for the solver."
        self.envelope = initial_condition(
            self.domain.radi_2d_array,
            self.domain.time_2d_array,
            self.const.im_unit,
            self.beam,
        )
        self.axis_envelope[0, :] = self.envelope[self.domain.axis_node, :]
        self.peak_envelope[:, 0] = self.envelope[:, self.domain.peak_node]

    def solve_step(self, step):
        "Perform one propagation step."
        calculate_nonlinear(self.envelope, self.w_array, self.equation)

        # For k = 0, initialize Adam_Bashforth second condition
        if step == 0:
            self.next_w_array = self.w_array.copy()
            self.axis_envelope[step + 1, :] = self.envelope[self.domain.axis_node, :]
            self.peak_envelope[:, 1] = self.envelope[:, self.domain.peak_node]
        frequency_domain(
            self.envelope,
            self.fourier_envelope,
            self.w_array,
            self.next_w_array,
            self.d_array,
        )
        solve_envelope(self.operators, self.vectors, self.entries)
        time_domain(self.next_fourier_envelope, self.next_envelope)

        # Update arrays
        self.envelope, self.next_envelope = self.next_envelope, self.envelope
        self.next_w_array = self.w_array

    def save_diagnostics(self, step):
        """Save diagnostics data for current step."""
        if (
            (step % (self.domain.n_steps // self.domain.dist_limit) == 0)
            or (step == self.domain.n_steps - 1)
        ) and self.domain.dist_index <= self.domain.dist_limit:

            self.dist_envelope[:, self.domain.dist_index, :] = self.envelope
            self.k_array[self.domain.dist_index] = step
            self.domain.dist_index += 1

        # Store axis data
        if step > 0:
            self.axis_envelope[step + 1, :] = self.envelope[self.domain.axis_node, :]
            self.peak_envelope[:, step + 1] = self.envelope[:, self.domain.peak_node]

    def propagate(self):
        """Propagate beam through all steps."""
        for k in tqdm(range(self.domain.n_steps)):
            self.solve_step(k)
            self.save_diagnostics(k)


def main():
    "Main function."
    # Initialize classes
    const = UniversalConstants()
    domain = DomainParameters()
    media = MediaParameters(const)
    beam = BeamParameters(const, media)

    # Create and run solver
    solver = SCNSolver(const, media, beam, domain)
    solver.propagate()

    # Save to file
    np.savez(
        "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/ffdmk_scn_1",
        e_dist=solver.dist_envelope,
        e_axis=solver.axis_envelope,
        e_peak=solver.peak_envelope,
        k_array=solver.k_array,
        INI_RADI_COOR=domain.ini_radi_coor,
        FIN_RADI_COOR=domain.fin_radi_coor,
        INI_DIST_COOR=domain.ini_dist_coor,
        FIN_DIST_COOR=domain.fin_dist_coor,
        INI_TIME_COOR=domain.ini_time_coor,
        FIN_TIME_COOR=domain.fin_time_coor,
        AXIS_NODE=domain.axis_node,
        PEAK_NODE=domain.peak_node,
        LIN_REF_IND=media.lin_ref_ind_water,
    )


if __name__ == "__main__":
    main()
