"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse in cylindrical coordinates with radial symmetry.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Alternating Direction Implicit (ADI) scheme.
    - Initial condition: Gaussian.
    - Boundary conditions: Neumann-Dirichlet (radial) and homogeneous Dirichlet (temporal).

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


def crank_nicolson_radial_diags(n, lr, c):
    """
    Set the three diagonals for the Crank-Nicolson radial array with centered differences.

    Parameters:
    - n (int): number of radial nodes
    - lr (str): position of the Crank-Nicolson array (left or right)
    - c (float): coefficient for the diagonal elements

    Returns:
    - tuple: upper, main, and lower diagonals
    """
    dc = 1 + 2 * c
    ind = np.arange(1, n - 1)

    diag_m1 = -c * (1 - 0.5 / ind)
    diag_0 = np.full(n, dc)
    diag_p1 = -c * (1 + 0.5 / ind)

    diag_m1 = np.append(diag_m1, [0])
    diag_p1 = np.insert(diag_p1, 0, [0])
    if lr == "left":
        diag_0[0], diag_0[-1] = dc, 1
        diag_p1[0] = -2 * c
    else:
        diag_0[0], diag_0[-1] = dc, 0
        diag_p1[0] = -2 * c

    return diag_m1, diag_0, diag_p1


def crank_nicolson_time_diags(n, lr, c):
    """
    Set the three diagonals for a Crank-Nicolson time array with centered differences.

    Parameters:
    - n (int): number of time nodes
    - lr (str): position of the Crank-Nicolson array (left or right)
    - c (float): coefficient for the diagonal elements

    Returns:
    - tuple: upper, main, and lower diagonals
    """
    dc = 1 + 2 * c

    diag_m1 = np.full(n - 1, -c)
    diag_0 = np.full(n, dc)
    diag_p1 = np.full(n - 1, -c)

    diag_p1[0], diag_m1[-1] = 0, 0
    if lr == "left":
        diag_0[0], diag_0[-1] = 1, 1
    else:
        diag_0[0], diag_0[-1] = 0, 0

    return diag_m1, diag_0, diag_p1


def crank_nicolson_radial_matrix(n, lr, c):
    """
    Set the Crank-Nicolson radial sparse array in CSR format using the diagonals.

    Parameters:
    - n (int): number of radial nodes
    - lr (str): position of the Crank-Nicolson array (left or right)
    - c (float): coefficient for the diagonal elements

    Returns:
    - array: Crank-Nicolson sparse array in CSR format
    """
    diag_m1, diag_0, diag_p1 = crank_nicolson_radial_diags(n, lr, c)

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]
    return diags_array(diags, offsets=offset, format="csr")


def crank_nicolson_time_matrix(n, lr, c):
    """
    Set the Crank-Nicolson sparse time array in CSR format using the diagonals.

    Parameters:
    - n (int): number of time nodes
    - lr (str): position of the Crank-Nicolson array (left or right)
    - c (float): coefficient for the diagonal elements

    Returns:
    - array: Crank-Nicolson sparse array in CSR format
    """
    diag_m1, diag_0, diag_p1 = crank_nicolson_time_diags(n, lr, c)

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]
    return diags_array(diags, offsets=offset, format="csr")


def solve_radial_adi(lmr, rmt, e_c, b, e_n):
    """
    Compute first half-step (ADI radial direction).

    Parameters:
    - lmr: sparse array for left-hand side (radial)
    - rmt: sparse array for right-hand side (time)
    - e_c: envelope at step k
    - b: pre-allocated array for intermediate results
    - e_n: pre-allocated array for envelope at step k + 1
    """
    # Compute right-hand side matrix product row by row
    for i in range(e_c.shape[0]):
        b[i, :] = rmt @ e_c[i, :]

    # Compute first half-step solution
    for l in range(e_c.shape[1]):
        e_n[:, l] = spsolve(lmr, b[:, l])


def solve_time_adi(lmt, rmr, e_c, b, e_n):
    """
    Compute second half-step (ADI time direction).

    Parameters:
    - lmt: sparse array for left-hand side (time)
    - rmr: sparse array for right-hand side (radial)
    - e_c: envelope at step k
    - b: preallocated array for intermediate results
    - e_n: pre-allocated array for envelope at step k + 1
    """
    # Compute right-hand side matrix product column by column
    for l in range(e_c.shape[1]):
        b[:, l] = rmr @ e_c[:, l]

    # Compute second half-step solution
    for i in range(e_c.shape[0]):
        e_n[i, :] = spsolve(lmt, b[i, :])


def solve_nonlinear(e_c, e_n, w_c, w_n, equation):
    """
    Compute one step of the Adam-Bashforth scheme for the nonlinear terms.

    Parameters:
    - e_c: envelope at step k
    - e_n: envelope at step k + 1
    - w_c: pre-allocated array for Adam-Bashforth terms at step k
    - w_n: pre-allocated array for Adam-Bashforth terms at step k + 1
    - media: dictionary with media parameters
    Compute one step of the Adam-Bashforth scheme for the nonlinear terms.
    """
    e_c_2 = np.abs(e_c) ** 2
    e_c_2k2 = np.abs(e_c) ** equation.mpa_exp
    w_c[:] = e_c * (equation.kerr_coef * e_c_2 + equation.mpa_coef * e_c_2k2)
    for l in range(e_c.shape[1]):
        e_n[:, l] = e_c[:, l] + 1.5 * w_c[:, l] - 0.5 * w_n[:, l]


@dataclass
class UniversalConstants:
    "UniversalConstants."

    def __init__(self):
        self.light_speed = 299792458
        self.permittivity = 8.8541878128e-12
        self.pi = np.pi
        self.im_unit = 1j


@dataclass
class MediaParameters:
    "Media parameters."

    def __init__(self):
        self.lin_ref_ind_water = 1.328
        self.nlin_ref_ind_water = 1.6e-20
        self.gvd_coef_water = 241e-28
        self.n_photons_water = 5
        self.mpa_cnt_water = 8e-64
        # self.int_factor = (
        #    0.5 * const.light_speed * const.permittivity * self.lin_ref_ind_water
        # )
        self.int_factor = 1


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

        # Distance domain
        self.ini_dist_coor = 0
        self.fin_dist_coor = 2e-2
        self.n_steps = 1000
        self.dist_index = 0
        self.dist_limit = 5

        # Time domain
        self.ini_time_coor = -300e-15
        self.fin_time_coor = 300e-15
        self.i_time_nodes = 1000
        self.n_time_nodes = self.i_time_nodes + 2

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

        # Create arrays
        self.radi_array = np.linspace(
            self.ini_radi_coor, self.fin_radi_coor, self.n_radi_nodes
        )
        self.dist_array = np.linspace(
            self.ini_dist_coor, self.fin_dist_coor, self.n_steps + 1
        )
        self.time_matrix = np.linspace(
            self.ini_time_coor, self.fin_time_coor, self.n_time_nodes
        )

        # Create 2D arrays
        self.radi_2d_array, self.time_2d_array = np.meshgrid(
            self.radi_array, self.time_matrix, indexing="ij"
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


class ADISolver:
    """ADI solver class for beam propagation."""

    def __init__(self, const, media, beam, domain):
        self.const = const
        self.media = media
        self.beam = beam
        self.domain = domain
        self.equation = EquationParameters(const, media, beam, domain)

        # Initialize field arrays
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
        self.w_array = np.empty_like(self.envelope)
        self.next_w_array = np.empty_like(self.envelope)
        self.b_array = np.empty_like(self.envelope)
        self.c_array = np.empty_like(self.envelope)
        self.d_array = np.empty_like(self.envelope)
        self.k_array = np.empty(self.domain.dist_limit + 1, dtype=int)

        # Setup operators and initial condition
        self.setup_operators()
        self.set_initial_condition()

    def setup_operators(self):
        """Setup ADI operators."""
        self.delta_r = (
            0.25
            * self.domain.dist_step_len
            / (self.beam.wavenumber * self.domain.radi_step_len**2)
        )
        self.delta_t = (
            -0.25
            * self.domain.dist_step_len
            * self.media.gvd_coef_water
            / self.domain.time_step_len**2
        )

        self.mat_cnt_1r = self.const.im_unit * self.delta_r
        self.mat_cnt_1t = self.const.im_unit * self.delta_t

        self.left_operator_r = crank_nicolson_radial_matrix(
            self.domain.n_radi_nodes, "left", self.mat_cnt_1r
        )
        self.right_operator_r = crank_nicolson_radial_matrix(
            self.domain.n_radi_nodes, "right", -self.mat_cnt_1r
        )
        self.left_operator_t = crank_nicolson_time_matrix(
            self.domain.n_time_nodes, "left", self.mat_cnt_1t
        )
        self.right_operator_t = crank_nicolson_time_matrix(
            self.domain.n_time_nodes, "right", -self.mat_cnt_1t
        )

    def set_initial_condition(self):
        """Set initial Gaussian beam."""
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
        solve_radial_adi(
            self.left_operator_r,
            self.right_operator_t,
            self.envelope,
            self.b_array,
            self.c_array,
        )
        solve_time_adi(
            self.left_operator_t,
            self.right_operator_r,
            self.c_array,
            self.b_array,
            self.d_array,
        )

        # For k = 0, initialize Adam_Bashforth second condition
        if step == 0:
            self.next_w_array = self.w_array.copy()
            self.axis_envelope[1, :] = self.envelope[self.domain.axis_node, :]
            self.peak_envelope[:, 1] = self.envelope[:, self.domain.peak_node]
        solve_nonlinear(
            self.d_array,
            self.next_envelope,
            self.w_array,
            self.next_w_array,
            self.equation,
        )

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
    media = MediaParameters()
    beam = BeamParameters(const, media)

    # Create and run solver
    solver = ADISolver(const, media, beam, domain)
    solver.propagate()

    # Save to file
    np.savez(
        "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/ffdmk_adi_1",
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
