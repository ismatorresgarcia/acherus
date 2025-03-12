"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse in cylindrical coordinates with radial symmetry.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).
    - Absorption and defocusing due to the electron plasma.
    - Multiphotonic ionization by multiphoton absorption (MPA).
    - Nonlinear optical Kerr effect (for a third-order centrosymmetric medium).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Split-step Fourier Crank-Nicolson (FCN) scheme.
        *- Fast Fourier Transform (FFT) scheme (for GVD).
        *- Extended Crank-Nicolson (CN-AB2) scheme (for diffraction, Kerr and MPA).
    - Method (DE): 4th order Runge-Kutta (RK4) scheme.
    - Initial condition:
        *- Gaussian envelope at initial z coordinate.
        *- Constant electron density at initial t coordinate.
    - Boundary conditions: Neumann-Dirichlet (radial) and Periodic (temporal) for the envelope.

DE:          ∂N/∂t = S_K|E|^(2K)(N_n - N) + S_w N|E|^2 / U_i
UPPE:          ∂E/∂z = i/(2k) ∇²E - ik''/2 ∂²E/∂t² - i(k_0/2n_0)(N/N_c)E - iB_K|E|^(2K-2)E
                     + ik_0n_2(1-a)|E|^2 E

DISCLAIMER: UPPE uses "god-like" units, where envelope intensity and its square module are the same.
            This is equivalent to setting 0.5*c*e_0*n_0 = 1 in the UPPE when using the SI system.
            The result obtained is identical since the consistency is mantained throught the code.
            This way, the number of operations is reduced, and the code is more readable.
            However, the dictionary "MEDIA" has an entry "INT_FACTOR" where the conversion
            factor can be changed at will between the two unit systems.

E: envelope.
N: electron density (in the interacting media).
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
t: time coordinate.
k: wavenumber (in the interacting media).
k'': GVD coefficient of 2nd order.
k_0: wavenumber (in vacuum).
N_n: neutral density (of the interacting media).
N_c: critical density (of the interacting media).
n_2: nonlinear refractive index (for a third-order centrosymmetric medium).
S_K: nonlinear optical field ionization coefficient.
B_K: nonlinear multiphoton absorption coefficient.
S_w: bremsstrahlung cross-section for avalanche (cascade) ionization.
U_i: ionization energy (for the interacting media).
∇²: laplace operator (for the transverse direction).
"""

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def initial_envelope(r, t, im, amp, wnum, w, ptime, ch, f):
    """
    Set the Gaussian beam.

    Parameters:
    - r: radial coordinates array
    - t: time coordinates array
    - im: square root of -1
    - amp: initial amplitude of the beam
    - wnum: initial wavenumber of the beam
    - w: initial waist of the beam
    - ptime: initial peak time of the beam
    - ch: chirp of the beam
    - f: focal length of the beam

    Returns:
    - complex array: Initial envelope field
    """
    space_decaying_term = -((r / w) ** 2)
    time_decaying_term = -(1 + im * ch) * (t / ptime) ** 2

    if f != 0:
        space_decaying_term -= 0.5 * im * wnum * r**2 / f

    return amp * np.exp(space_decaying_term + time_decaying_term)


def crank_nicolson_matrix(n, out_diag, main_diag):
    """
    Initialize the Crank-Nicolson sparse array in CSR format using the diagonals.

    Parameters:
    - n (int): number of radial nodes

    Returns:
    - array: sparse array for the Crank-Nicolson matrix
    """
    out_diag = np.ones(n - 1, dtype=complex)
    main_diag = np.ones(n, dtype=complex)

    diags = [out_diag, main_diag, out_diag]
    offset = [-1, 0, 1]

    return diags_array(diags, offsets=offset, format="csr")


def calculate_nonlinear(e_c, w_c, n_p, m_coef, k_coef):
    """
    Compute the nonlinear terms.

    Parameters:
    - e_c: envelope at step k
    - w_c: pre-allocated array for Adam-Bashforth terms
    - n_p: multiphoton ionization exponent
    - m_coef: MPA coefficient
    - k_coef: Kerr coefficient
    """
    e_c_2 = np.abs(e_c) ** 2
    e_c_2k2 = e_c_2 ** (n_p - 1)

    w_c[:] = e_c * (m_coef[np.newaxis, :] * e_c_2k2 + k_coef[np.newaxis, :] * e_c_2)


def frequency_domain(e_c, fe_c, w_c, w_n):
    """
    Compute the FFT of the envelope and Adam-Bashforth terms.

    Parameters:
    - e_c: envelope at step k
    - fe_c: pre-allocated array for Fourier envelope at step k
    - w_c: current step nonlinear terms
    - w_n: previous step nonlinear terms
    """
    fe_c[:] = fft(e_c, axis=1, workers=-1)
    w_c[:] = fft(w_c, axis=1, workers=-1)
    w_n[:] = fft(w_n, axis=1, workers=-1)


def update_matrix(lm, rm, oc_l, mrc_l, mlc_l, up_diag, main_diag, down_diag):
    """
    Update the Crank-Nicolson matrices for the current frequency.
    """
    lm.setdiag(-oc_l * down_diag, k=-1)  # Lower diagonal
    lm.setdiag(mlc_l, k=0)  # Main diagonal
    lm.setdiag(-oc_l * up_diag, k=1)  # Upper diagonal
    rm.setdiag(oc_l * down_diag, k=-1)  # Lower diagonal
    rm.setdiag(mrc_l, k=0)  # Main diagonal
    rm.setdiag(oc_l * up_diag, k=1)  # Upper diagonal

    # Boundary condition for lower diagonal
    out_diag = lm.diagonal(-1)
    out_diag[-1] = 0  # Set last element to 0 for left matrix
    lm.setdiag(out_diag, k=-1)

    out_diag = rm.diagonal(-1)
    out_diag[-1] = 0  # Set last element to 0 for right matrix
    rm.setdiag(out_diag, k=-1)

    # Boundary condition for main diagonal
    main_diag = lm.diagonal()
    main_diag[-1] = 1  # Set last element to 1 for left matrix
    lm.setdiag(main_diag, k=0)

    main_diag = rm.diagonal()
    main_diag[-1] = 0  # Set last element to 0 for right matrix
    rm.setdiag(main_diag, k=0)

    # Boundary condition for upper diagonal
    out_diag = lm.diagonal(1)
    out_diag[0] = -2 * oc_l  # Set first element to Neumann condition
    lm.setdiag(out_diag, k=1)

    out_diag = rm.diagonal(1)
    out_diag[0] = 2 * oc_l  # Set first element to Neumann condition
    rm.setdiag(out_diag, k=1)


def solve_envelope(
    lm,
    rm,
    n,
    b,
    c,
    fe_c,
    fe_n,
    w_c,
    w_n,
    oc,
    mrc,
    mlc,
    up_diag,
    main_diag,
    down_diag,
):
    """
    Solve the envelope equation for each frequency.
    """
    for ll in range(n):
        update_matrix(
            lm,
            rm,
            oc[ll],
            mrc[ll],
            mlc[ll],
            up_diag,
            main_diag,
            down_diag,
        )

        b = rm @ fe_c[:, ll]
        c = b + 1.5 * w_c[:, ll] - 0.5 * w_n[:, ll]
        fe_n[:, ll] = spsolve(lm, c)


def time_domain(fe_c, e_c):
    """
    Compute the IFFT of the Fourier envelope at step k.

    Parameters:
    - fe_c: envelope in the frequency domain at step k
    - e_c: envelope at step k
    """
    e_c[:] = ifft(fe_c, axis=1, workers=-1)


@dataclass()
class UniversalConstants:
    "Universal constants."

    def __init__(self):
        self.light_speed = 299792458.0
        self.permittivity = 8.8541878128e-12
        self.electron_mass = 9.1093837139e-31
        self.electron_charge = 1.602176634e-19
        self.planck_bar = 1.05457182e-34
        self.pi = np.pi
        self.im_unit = 1j


@dataclass()
class MediaParameters:
    "Media parameters."

    def __init__(self):
        self.lin_ref_ind_water = 1.328
        self.nlin_ref_ind_water = 1.6e-20
        self.gvd_coef_water = 241e-28
        self.n_photons_water = 5
        self.mpa_cnt_water = 8e-64
        self.int_factor = 1
        # self.int_factor = (
        #    0.5 * const.light_speed * const.permittivity * self.lin_ref_ind_water
        # )


@dataclass
class BeamParameters:
    "Beam parameters."

    def __init__(self, const, media):
        # Basic parameters
        self.wavelength_0 = 800e-9
        self.waist_0 = 100e-6
        self.peak_time = 50e-15
        self.energy = 2.83e-6
        self.chirp = -1
        self.focal_length = 0

        # Derived parameters
        self.wavenumber_0 = 2 * const.pi / self.wavelength_0
        self.wavenumber = self.wavenumber_0 * media.lin_ref_ind_water
        self.frequency_0 = self.wavenumber_0 * const.light_speed
        self.power = self.energy / (self.peak_time * np.sqrt(0.5 * const.pi))
        self.cr_power = (
            3.77
            * self.wavelength_0**2
            / (8 * const.pi * media.lin_ref_ind_water * media.nlin_ref_ind_water)
        )
        self.intensity = 2 * self.power / (const.pi * self.waist_0**2)
        self.amplitude = np.sqrt(self.intensity / media.int_factor)


class DomainParameters:
    "Spatial and temporal domain parameters."

    def __init__(self, const, beam):
        # Radial domain
        self.ini_radi_coor = 0
        self.fin_radi_coor = 100e-5
        self.i_radi_nodes = 1500

        # Distance domain
        self.ini_dist_coor = 0
        self.fin_dist_coor = 2e-2
        self.n_steps = 1000
        self.dist_limit = 5

        # Time domain
        self.ini_time_coor = -200e-15
        self.fin_time_coor = 200e-15
        self.n_time_nodes = 4096

        # Initialize derived parameters functions
        self._setup_derived_parameters()
        self._create_arrays(const, beam)

    @property
    def n_radi_nodes(self):
        "Total number of radial nodes for boundary conditions."
        return self.i_radi_nodes + 2

    @property
    def dist_limitin(self):
        "Inner loop auxiliary parameters."
        return self.n_steps // self.dist_limit

    def _setup_derived_parameters(self):
        "Setup derived parameters."
        # Calculate steps
        self.radi_step_len = (self.fin_radi_coor - self.ini_radi_coor) / (
            self.n_radi_nodes - 1
        )
        self.dist_step_len = (self.fin_dist_coor - self.ini_dist_coor) / self.n_steps
        self.time_step_len = (self.fin_time_coor - self.ini_time_coor) / (
            self.n_time_nodes - 1
        )
        self.frq_step_len = 2 * np.pi / (self.n_time_nodes * self.time_step_len)

        # Calculate nodes
        self.axis_node = int(-self.ini_radi_coor / self.radi_step_len)
        self.peak_node = self.n_time_nodes // 2

    def _create_arrays(self, const, beam):
        "Create arrays."
        # 1D
        self.radi_array = np.linspace(
            self.ini_radi_coor, self.fin_radi_coor, self.n_radi_nodes
        )
        self.dist_array = np.linspace(
            self.ini_dist_coor, self.fin_dist_coor, self.n_steps + 1
        )
        self.time_array = np.linspace(
            self.ini_time_coor, self.fin_time_coor, self.n_time_nodes
        )
        self.frq_array = 2 * const.pi * fftfreq(self.n_time_nodes, self.time_step_len)
        self.frq_array_shift = beam.frequency_0 + self.frq_array

        # 2D
        self.radi_2d_array, self.time_2d_array = np.meshgrid(
            self.radi_array, self.time_array, indexing="ij"
        )


@dataclass
class EquationParameters:
    """Parameters for the final equation."""

    def __init__(self, const, media, beam, domain):
        # Initialize main function parameters
        self._init_coefficients(media)
        self._init_operators(const, media, beam, domain)

    def _init_coefficients(self, media):
        "Initialize equation coefficients."
        # self.mpi_exp = 2 * media.n_photons_water

    def _init_operators(self, const, media, beam, domain):
        "Initialize equation operators."
        # Pre-compute common terms
        self.u_operator = beam.wavenumber + media.gvd_coef_water * domain.frq_array
        self.delta_r = (
            0.25 * domain.dist_step_len / (self.u_operator * domain.radi_step_len**2)
        )
        self.delta_t = (
            0.25 * domain.dist_step_len * media.gvd_coef_water * domain.frq_array**2
        )

        # MPA coefficient calculation
        self.mpa_coef = (
            -0.5
            * domain.dist_step_len
            * domain.frq_array_shift
            * media.mpa_cnt_water
            * media.lin_ref_ind_water
            * media.int_factor ** (media.n_photons_water - 1)
            / (const.light_speed * self.u_operator)
        )

        # Kerr coefficient calculation
        self.kerr_coef = (
            const.im_unit
            * domain.dist_step_len
            * domain.frq_array_shift**2
            * media.lin_ref_ind_water
            * media.nlin_ref_ind_water
            * media.int_factor
            / (const.light_speed**2 * self.u_operator)
        )


class SCNSolver:
    "Solver class."

    def __init__(self, const, media, beam, domain, equation):
        self.const = const
        self.media = media
        self.beam = beam
        self.domain = domain
        self.equation = equation

        # Initialize arrays and operators
        shape = (self.domain.n_radi_nodes, self.domain.n_time_nodes)
        dist_shape = (
            self.domain.n_radi_nodes,
            self.domain.dist_limit + 1,
            self.domain.n_time_nodes,
        )
        axis_shape = (self.domain.n_steps + 1, self.domain.n_time_nodes)
        peak_shape = (self.domain.n_radi_nodes, self.domain.n_steps + 1)

        self.envelope = np.empty(shape, dtype=complex)
        self.next_envelope = np.empty_like(self.envelope)
        self.dist_envelope = np.empty(dist_shape, dtype=complex)
        self.axis_envelope = np.empty(axis_shape, dtype=complex)
        self.peak_envelope = np.empty(peak_shape, dtype=complex)
        self.fourier_envelope = np.empty_like(self.envelope)
        self.next_fourier_envelope = np.empty_like(self.envelope)
        self.w_array = np.empty_like(self.envelope)
        self.next_w_array = np.empty_like(self.envelope)
        self.b_array = np.empty(self.domain.n_radi_nodes, dtype=complex)
        self.c_array = np.empty_like(self.b_array)

        # Pre-allocate temporary arrays for matrices
        self.diag_out = np.empty(self.domain.n_radi_nodes - 1, dtype=complex)
        self.diag_main = np.empty(self.domain.n_radi_nodes, dtype=complex)

        # Setup tracking variables
        self.k_array = np.empty(self.domain.dist_limit + 1, dtype=int)

        # Setup operators and initial condition
        self.setup_operators(const, equation)
        self.set_initial_envelope()

    def setup_operators(self, const, equation):
        """Setup operators."""
        # Setup operators
        self.diff_operator = const.im_unit * equation.delta_r
        self.disp_operator = const.im_unit * equation.delta_t
        self.matrix_cnt_1 = 1 - 2 * self.diff_operator + self.disp_operator
        self.matrix_cnt_2 = 1 + 2 * self.diff_operator - self.disp_operator

        # Setup outer diagonals radial index dependence
        self.diag_up = 1 + 0.5 / np.arange(1, self.domain.n_radi_nodes - 1)
        self.diag_down = 1 - 0.5 / np.arange(1, self.domain.n_radi_nodes - 1)

        # Setup matrices
        self.left_matrix = crank_nicolson_matrix(
            self.domain.n_radi_nodes, self.diag_out, self.diag_main
        )
        self.right_matrix = crank_nicolson_matrix(
            self.domain.n_radi_nodes, self.diag_out, self.diag_main
        )

    def set_initial_envelope(self):
        "Set the initial condition for the solver."
        self.envelope = initial_envelope(
            self.domain.radi_2d_array,
            self.domain.time_2d_array,
            self.const.im_unit,
            self.beam.amplitude,
            self.beam.wavenumber,
            self.beam.waist_0,
            self.beam.peak_time,
            self.beam.chirp,
            self.beam.focal_length,
        )
        # Store initial values for diagnostics
        self.dist_envelope[:, 0, :] = self.envelope
        self.axis_envelope[0, :] = self.envelope[self.domain.axis_node, :]
        self.peak_envelope[:, 0] = self.envelope[:, self.domain.peak_node]
        self.k_array[0] = 0

    def solve_step(self, step):
        "Perform one propagation step."
        calculate_nonlinear(
            self.envelope,
            self.w_array,
            self.media.n_photons_water,
            self.equation.mpa_coef,
            self.equation.kerr_coef,
        )

        # For k = 1, initialize Adam_Bashforth second condition
        if step == 1:
            np.copyto(self.next_w_array, self.w_array)
            self.axis_envelope[1] = self.envelope[self.domain.axis_node]
            self.peak_envelope[:, 1] = self.envelope[:, self.domain.peak_node]
        frequency_domain(
            self.envelope,
            self.fourier_envelope,
            self.w_array,
            self.next_w_array,
        )
        solve_envelope(
            self.left_matrix,
            self.right_matrix,
            self.domain.n_radi_nodes,
            self.b_array,
            self.c_array,
            self.fourier_envelope,
            self.next_fourier_envelope,
            self.w_array,
            self.next_w_array,
            self.diff_operator,
            self.matrix_cnt_1,
            self.matrix_cnt_2,
            self.diag_up,
            self.diag_main,
            self.diag_down,
        )
        time_domain(self.next_fourier_envelope, self.next_envelope)

        # Update arrays
        self.envelope, self.next_envelope = self.next_envelope, self.envelope
        self.next_w_array, self.w_array = self.w_array, self.next_w_array

    def save_expensive_diagnostics(self, step):
        """Save memory expensive diagnostics data for current step."""
        self.dist_envelope[:, step, :] = self.envelope
        self.k_array[step] = self.k_array[step - 1] + self.domain.dist_limitin

    def save_cheap_diagnostics(self, step):
        """Save memory cheap diagnostics data for current step."""
        if step > 1:
            # Cache accessed arrays and parameters
            axis_node = self.domain.axis_node
            envelope = self.envelope

            # Cache axis data computations
            axis_envelope_data = envelope[axis_node]
            intensity = np.abs(axis_envelope_data)

            intensity_peak_node = np.argmax(intensity)

            self.axis_envelope[step] = axis_envelope_data
            self.peak_envelope[:, step] = envelope[:, intensity_peak_node]

    def propagate(self):
        """Propagate beam through all steps."""
        steps = self.domain.n_steps

        with tqdm(total=steps, desc="Progress") as pbar:
            for mm in range(1, self.domain.dist_limit + 1):
                for nn in range(1, self.domain.dist_limitin + 1):
                    kk = (mm - 1) * self.domain.dist_limitin + nn
                    self.solve_step(kk)
                    self.save_cheap_diagnostics(kk)
                    pbar.update(1)
                    pbar.set_postfix({"m": mm, "n": nn, "k": kk})
                self.save_expensive_diagnostics(mm)


def main():
    "Main function."
    # Initialize classes
    const = UniversalConstants()
    media = MediaParameters()
    beam = BeamParameters(const, media)
    domain = DomainParameters(const, beam)
    equation = EquationParameters(const, media, beam, domain)

    # Create and run solver
    solver = SCNSolver(const, media, beam, domain, equation)
    solver.propagate()

    # Save to file
    np.savez(
        "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/water_scn_1",
        e_dist=solver.dist_envelope,
        e_axis=solver.axis_envelope,
        e_peak=solver.peak_envelope,
        k_array=solver.k_array,
        ini_radi_coor=domain.ini_radi_coor,
        fin_radi_coor=domain.fin_radi_coor,
        ini_dist_coor=domain.ini_dist_coor,
        fin_dist_coor=domain.fin_dist_coor,
        ini_time_coor=domain.ini_time_coor,
        fin_time_coor=domain.fin_time_coor,
        axis_node=domain.axis_node,
        peak_node=domain.peak_node,
        lin_ref_ind=media.lin_ref_ind_water,
    )


if __name__ == "__main__":
    main()
