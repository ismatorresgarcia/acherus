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
        *- Extended Crank-Nicolson (CN-RK4) scheme (for diffraction, Kerr and MPA).
    - Method (DE): 4th order Runge-Kutta (RK4) scheme.
    - Initial condition:
        *- Gaussian envelope at initial z coordinate.
        *- Constant electron density at initial t coordinate.
    - Boundary conditions: Neumann-Dirichlet (radial) and Periodic (temporal) for the envelope.

DE:          ∂N/∂t = S_K|E|^(2K)(N_n - N) + S_w N|E|^2 / U_i
UPPE:          ∂E/∂z = i/(2k) ∇²E - ik''/2 ∂²E/∂t² - i(k_0/2n_0)(N/N_c)E - iB_K|E|^(2K-2)E
                     + ik_0n_2|E|^2 E

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

import sys
from dataclasses import dataclass

import numba as nb
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.sparse import diags_array
from scipy.sparse.linalg import splu
from tqdm import tqdm


def initial_envelope(r, t, im, amp, wnum, w, ptime, ch, f):
    """
    Set up the Gaussian beam.

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
    - complex 2-array: Initial envelope field
    """
    space_decaying_term = -((r / w) ** 2)
    time_decaying_term = -(1 + im * ch) * (t / ptime) ** 2

    if f != 0:
        space_decaying_term -= 0.5 * im * wnum * r**2 / f

    return amp * np.exp(space_decaying_term + time_decaying_term)


def initial_density(back_dens):
    """
    Set up the initial electron density distribution.

    Parameters:
    - back_dens: background electron density of the medium

    Returns:
    - float 1-array: Initial free electron density
    """
    return back_dens


@nb.njit
def laplacian_operator(field, radi_array, diff_array, n_r, dr_0, dr_p2, dr_m2):
    """Set up the cylindrical Laplacian operator with boundary conditions.

    Parameters:
    - field: envelope array for any time step ll
    - radi_array: radial coordinates array
    - diff_array: pre-allocated array for the Laplacian operator
    - n_r: number of radial nodes
    - dr_0: radial step length
    - dr_p2: radial step length squared
    - dr_m2: radial step length multiplied by 2

    Returns:
    - float 1-array: Laplacian operator
    """
    # Neumann homogeneous boundary condition at r=0
    diff_array[0] = 4 * (field[1] - field[0]) / dr_p2

    # Interior nodes
    for ii in range(1, n_r - 1):
        ri = radi_array[ii]
        if ri > 1e-6:
            dfield_dr = (field[ii + 1] - field[ii - 1]) / dr_m2
            dfield_drr = dfield_dr / ri
            d2field_dr2 = (field[ii + 1] - 2 * field[ii] + field[ii - 1]) / dr_p2
            diff_array[ii] = dfield_drr + d2field_dr2
        else:
            diff_array[ii] = 4 * (field[ii + 1] - field[ii]) / dr_p2

    # Dirichlet homogeneous boundary contidion at r=r_max
    ii = n_r - 1
    ri = radi_array[ii]
    dfield_dr = -field[ii - 1] / dr_0
    dfield_drr = dfield_dr / ri
    d2field_dr2 = (-2 * field[ii - 1] + field[ii - 2]) / dr_p2
    diff_array[ii] = dfield_drr + d2field_dr2

    return diff_array


@nb.njit
def nonlinear_operator(field, density, nlin_array, n_ph, p_coef, m_coef, k_coef):
    """Set up the nonlinear operator terms.

    Parameters:
    - field: envelope array for any time step ll
    - density: free electron density array for any time step ll
    - nlin_array: pre-allocated array for the nonlinear operator
    - p_coef: plasma coefficient
    - m_coef: MPA coefficient
    - k_coef: Kerr coefficient
    - n_ph: number of photons

    Returns:
    - complex 1-array: Nonlinear operator
    """
    field_2 = np.abs(field) ** 2
    field_2k2 = field_2 ** (n_ph - 1)

    nlin_array[:] = field * (p_coef * density + m_coef * field_2k2 + k_coef * field_2)

    return nlin_array


@nb.njit
def compute_field_operator(
    field,
    density,
    radi_array,
    diff_array,
    nlin_array,
    im,
    n_r,
    n_ph,
    wnum,
    dr_0,
    dr_p2,
    dr_m2,
    p_coef,
    m_coef,
    k_coef,
):
    """Set up the envelope propagation terms.

    Parameters:
    - field: envelope array for any time step ll
    - density: free electron density array for any time step ll
    - radi_array: radial coordinates array
    - diff_array: pre-allocated array for the Laplacian operator
    - nlin_array: pre-allocated array for the nonlinear operator
    - im: square root of -1
    - n_r: number of radial nodes
    - n_ph: number of photons
    - wnum: initial wavenumber of the beam
    - dr_0: radial step length
    - dr_p2: radial step length squared
    - dr_m2: radial step length multiplied by 2
    - p_coef: plasma coefficient
    - m_coef: MPA coefficient
    - k_coef: Kerr coefficient

    Returns:
    - complex 1-array: Envelope operator
    """
    # Diffraction terms
    diff_coeff = 0.5 * im / wnum
    laplacian_operator(field, radi_array, diff_array, n_r, dr_0, dr_p2, dr_m2)

    # Nonlinear terms
    nonlinear_operator(field, density, nlin_array, n_ph, p_coef, m_coef, k_coef)

    return nlin_array + diff_coeff * diff_array


@nb.njit
def compute_density_operator(density, field, n_ph, neutral_dens, ofi_coef, ava_coef):
    """Set up the electron density evolution terms.

    Parameters:
    - density: free electron density array for any time step ll
    - field: envelope array for any time step ll
    - n_ph: number of photons
    - neutral_dens: neutral density of the medium
    - ofi_coef: OFI coefficient
    - ava_coef: avalanche/cascade coefficient

    Returns:
    - float 1-array: Electron density operator
    """
    field_2 = np.abs(field) ** 2
    field_2n = field_2**n_ph

    ofi = ofi_coef * field_2n * (neutral_dens - density)
    ava = ava_coef * density * field_2

    return ofi + ava


def crank_nicolson_matrix(n_r, pos, coef):
    """
    Set the three diagonals for the Crank-Nicolson array with centered differences.

    Parameters:
    - n_r (int): number of radial nodes
    - pos (str): position of the Crank-Nicolson array (left or right)
    - coef (float): coefficient for the diagonal elements

    Returns:
    - complex 2-array: sparse 2-array for the Crank-Nicolson matrix
    """
    dc = 1 + 2 * coef
    ind = np.arange(1, n_r - 1)

    diag_m1 = -coef * (1 - 0.5 / ind)
    diag_0 = np.full(n_r, dc)
    diag_p1 = -coef * (1 + 0.5 / ind)

    diag_m1 = np.append(diag_m1, [0])
    diag_p1 = np.insert(diag_p1, 0, [0])
    if pos == "left":
        diag_0[0], diag_0[-1] = dc, 1
        diag_p1[0] = -2 * coef
    else:
        diag_0[0], diag_0[-1] = dc, 0
        diag_p1[0] = -2 * coef

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]

    return diags_array(diags, offsets=offset, format="csc")


@nb.njit
def rk4_step(
    e_curr,
    n_curr,
    e_aux,
    n_aux,
    field_op_args,
    dens_op_args,
    dz2,
    dz6,
    dt2,
    dt6,
):
    """
    Compute one step of the RK4 integration step.

    Parameters:
    - e_curr: envelope at current time slice
    - n_curr: density at current time slice
    - field_op_args: arguments for the field operator
    - dens_op_args: arguments for the density operator
    - dt2: half time step
    - dt6: time step divided by 6
    - dz2: half distance step
    - dz6: distance step divided by 6

    Returns:
    - complex 1-array: RK4 integration at current time slice
    - float 1-array: Electron density at next time slice
    """
    k1_f = compute_field_operator(e_curr, n_curr, *field_op_args)
    k1_d = compute_density_operator(n_curr, e_curr, *dens_op_args)

    e_aux = e_curr + dz2 * k1_f
    n_aux = n_curr + dt2 * k1_d

    k2_f = compute_field_operator(e_aux, n_aux, *field_op_args)
    k2_d = compute_density_operator(n_aux, e_aux, *dens_op_args)

    e_aux = e_curr + dz2 * k2_f
    n_aux = n_curr + dt2 * k2_d

    k3_f = compute_field_operator(e_aux, n_aux, *field_op_args)
    k3_d = compute_density_operator(n_aux, e_aux, *dens_op_args)

    e_aux = e_curr + dz2 * k3_f
    n_aux = n_curr + dt2 * k3_d

    k4_f = compute_field_operator(e_aux, n_aux, *field_op_args)
    k4_d = compute_density_operator(n_aux, e_aux, *dens_op_args)

    w_c = dz6 * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
    n_n = n_curr + dt6 * (k1_d + 2 * k2_d + 2 * k3_d + k4_d)

    return w_c, n_n


@nb.njit(parallel=True)
def solve_density(
    e_c,
    n_c,
    e_aux,
    n_aux,
    w_c,
    n_n,
    n_t,
    field_op_args,
    dens_op_args,
    dz2,
    dz6,
    dt2,
    dt6,
):
    """
    Solve electron density evolution for all time steps.

    Parameters:
    - e_c: envelope at current time slice
    - n_c: density at current time slice
    - e_aux: auxiliary envelope array
    - n_aux: auxiliary density array
    - w_c: pre-allocated array for the nonlinear terms
    - n_n: pre-allocated array for the density at next time slice
    - n_t: number of time nodes
    - field_op_args: arguments for the field operator
    - dens_op_args: arguments for the density operator
    - dz2: half distance step
    - dz6: distance step divided by 6
    - dt2: half time step
    - dt6: time step divided by 6
    """
    for ll in nb.prange(n_t - 1):
        e_curr = e_c[:, ll]
        n_curr = n_c[:, ll]

        w_c_ll, n_next = rk4_step(
            e_curr,
            n_curr,
            e_aux,
            n_aux,
            field_op_args,
            dens_op_args,
            dz2,
            dz6,
            dt2,
            dt6,
        )

        w_c[:, ll] = w_c_ll
        n_n[:, ll + 1] = n_next


def solve_dispersion(fc, e_c, b):
    """
    Compute one step of the FFT propagation scheme for dispersion.

    Parameters:
    - fc: precomputed Fourier coefficient
    - e_c: envelope at step k
    - b: pre-allocated array for envelope at step k + 1
    """
    b[:] = ifft(fc * fft(e_c, axis=1, workers=-1), axis=1, workers=-1)


def solve_envelope(lm, rm, n, b, w_c, e_n):
    """
    Compute one step of the Crank-Nicolson propagation scheme.

    Parameters:
    - lm: left matrix for Crank-Nicolson
    - rm: right matrix for Crank-Nicolson
    - n: number of time nodes
    - b: intermediate array from FFT step
    - w_c: current step nonlinear terms
    - e_n: pre-allocated array for envelope at step k + 1
    """
    for ll in range(n):
        c = rm @ b[:, ll]
        d = c + w_c[:, ll]
        e_n[:, ll] = lm.solve(d)


@dataclass
class UniversalConstants:
    "UniversalConstants."

    def __init__(self):
        self.light_speed = 299792458.0
        self.permittivity = 8.8541878128e-12
        self.electron_mass = 9.1093837139e-31
        self.electron_charge = 1.602176634e-19
        self.planck_bar = 1.05457182e-34
        self.pi = np.pi
        self.im_unit = 1j


@dataclass
class MediaParameters:
    "Media parameters."

    def __init__(self):
        self.lin_ref_ind_water = 1.334
        self.nlin_ref_ind_water = 4.1e-20
        self.gvd_coef_water = 248e-28
        self.n_photons_water = 5
        self.mpa_cnt_water = 1e-61
        self.mpi_cnt_water = 1.2e-72
        self.int_factor = 1
        # self.int_factor = (
        #    0.5 * const.light_speed * const.permittivity * self.lin_ref_ind_water
        # )
        self.energy_gap_water = 1.04e-18  # 6.5 eV
        self.collision_time_water = 3e-15
        self.neutral_dens_water = 6.68e28
        self.background_density_water = 1e-6


@dataclass
class BeamParameters:
    "Beam parameters."

    def __init__(self, const, media):
        # Basic parameters
        self.wavelength_0 = 800e-9
        self.waist_0 = 75e-6
        self.peak_time = 130e-15
        self.energy = 2.2e-6
        self.chirp = 0
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

    def __init__(self, const):
        # Radial domain
        self.ini_radi_coor = 0
        self.fin_radi_coor = 25e-4
        self.i_radi_nodes = 1500

        # Distance domain
        self.ini_dist_coor = 0
        self.fin_dist_coor = 3e-2
        self.n_steps = 1000
        self.dist_limit = 5

        # Time domain
        self.ini_time_coor = -250e-15
        self.fin_time_coor = 250e-15
        self.n_time_nodes = 4096

        # Initialize derived parameters functions
        self._setup_derived_parameters()
        self._create_arrays(const)

    @property
    def n_radi_nodes(self):
        "Total number of radial nodes for boundary conditions."
        return self.i_radi_nodes + 2

    @property
    def dist_limitin(self):
        "Inner loop auxiliary parameters."
        return self.n_steps // self.dist_limit

    def _setup_derived_parameters(self):
        "Setup domain parameters."
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

    def _create_arrays(self, const):
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

        # 2D
        self.radi_2d_array, self.time_2d_array = np.meshgrid(
            self.radi_array, self.time_array, indexing="ij"
        )


@dataclass
class EquationParameters:
    """Parameters for the final equation."""

    def __init__(self, const, media, beam):
        # Cache common parameters
        self.omega = beam.frequency_0
        self.omega_tau = self.omega * media.collision_time_water

        # Initialize main function parameters
        self._init_densities(const, media, beam)
        self._init_coefficients(media)
        self._init_operators(const, media, beam)

    def _init_densities(self, const, media, beam):
        "Initialize density parameters."
        self.critical_dens = (
            const.permittivity
            * const.electron_mass
            * (self.omega / const.electron_charge) ** 2
        )
        self.bremss_cs_water = (beam.wavenumber * self.omega_tau) / (
            (media.lin_ref_ind_water**2 * self.critical_dens) * (1 + self.omega_tau**2)
        )

    def _init_coefficients(self, media):
        "Initialize equation coefficients."
        self.mpi_exp = 2 * media.n_photons_water
        self.mpa_exp = self.mpi_exp - 2
        self.ofi_coef = media.mpi_cnt_water * media.int_factor**media.n_photons_water
        self.ava_coef = self.bremss_cs_water * media.int_factor / media.energy_gap_water

    def _init_operators(self, const, media, beam):
        "Initialize equation operators."
        # Plasma coefficient calculation
        self.plasma_coef = (
            -0.5
            * const.im_unit
            * beam.wavenumber_0
            / (media.lin_ref_ind_water * self.critical_dens)
        )

        # MPA coefficient calculation
        self.mpa_coef = (
            -0.5 * media.mpa_cnt_water * media.int_factor ** (media.n_photons_water - 1)
        )

        # Kerr coefficient calculation
        self.kerr_coef = (
            const.im_unit
            * beam.wavenumber_0
            * media.nlin_ref_ind_water
            * media.int_factor
        )


class FCNSolver:
    """FCN solver class for beam propagation."""

    def __init__(self, const, media, beam, domain, equation):
        self.const = const
        self.media = media
        self.beam = beam
        self.domain = domain
        self.equation = equation

        # Compute frequent constants
        self.dr = domain.radi_step_len
        self.dr2 = self.dr**2
        self.ddr = 2 * self.dr
        dz = domain.dist_step_len
        self.dz_2 = 0.5 * dz
        self.dz_6 = dz / 6
        dt = domain.time_step_len
        self.dt_2 = 0.5 * dt
        self.dt_6 = dt / 6

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
        self.density = np.empty(shape)
        self.next_envelope = np.empty_like(self.envelope)
        self.next_density = np.empty_like(self.density)
        self.dist_envelope = np.empty(dist_shape, dtype=complex)
        self.dist_density = np.empty(dist_shape)
        self.axis_envelope = np.empty(axis_shape, dtype=complex)
        self.axis_density = np.empty(axis_shape)
        self.peak_envelope = np.empty(peak_shape, dtype=complex)
        self.peak_density = np.empty(peak_shape)
        self.b_array = np.empty_like(self.envelope)
        self.w_array = np.empty_like(self.envelope)
        self.diff_array = np.empty(self.domain.n_radi_nodes, dtype=complex)
        self.nlin_array = np.empty_like(self.diff_array)

        self.field_op_args = (
            self.domain.radi_array,
            self.diff_array,
            self.nlin_array,
            self.const.im_unit,
            self.domain.n_radi_nodes,
            self.media.n_photons_water,
            self.beam.wavenumber,
            self.dr,
            self.dr2,
            self.ddr,
            self.equation.plasma_coef,
            self.equation.mpa_coef,
            self.equation.kerr_coef,
        )
        self.dens_op_args = (
            self.media.n_photons_water,
            self.media.neutral_dens_water,
            self.equation.ofi_coef,
            self.equation.ava_coef,
        )

        # Setup arrays for RK4 calculations
        self.k1_field = np.empty(self.domain.n_radi_nodes, dtype=complex)
        self.k2_field = np.empty_like(self.k1_field)
        self.k3_field = np.empty_like(self.k1_field)
        self.k4_field = np.empty_like(self.k1_field)

        self.k1_dens = np.empty(self.domain.n_radi_nodes)
        self.k2_dens = np.empty_like(self.k1_dens)
        self.k3_dens = np.empty_like(self.k1_dens)
        self.k4_dens = np.empty_like(self.k1_dens)

        self.e_temp = np.empty_like(self.k1_field)
        self.n_temp = np.empty_like(self.k1_dens)

        # Setup tracking variables
        self.k_array = np.empty(self.domain.dist_limit + 1, dtype=int)

        # Setup operators and initial condition
        self.setup_operators()
        self.set_initial_condition()

    def setup_operators(self):
        """Setup FCN operators."""
        delta_r = (
            0.25
            * self.domain.dist_step_len
            / (self.beam.wavenumber * self.domain.radi_step_len**2)
        )
        delta_t = (
            -0.25
            * self.domain.dist_step_len
            * self.media.gvd_coef_water
            / self.domain.time_step_len**2
        )

        # Setup Fourier coefficient
        self.fourier_coeff = np.exp(
            -2
            * self.const.im_unit
            * delta_t
            * (self.domain.frq_array * self.domain.time_step_len) ** 2
        )

        # Setup CN operators
        mat_cnt = self.const.im_unit * delta_r
        self.left_matrix = crank_nicolson_matrix(
            self.domain.n_radi_nodes, "left", mat_cnt
        )
        self.right_matrix = crank_nicolson_matrix(
            self.domain.n_radi_nodes, "right", -mat_cnt
        )
        self.left_matrix = splu(self.left_matrix)

    def set_initial_condition(self):
        """Set initial Gaussian beam and free electron density."""
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
        self.density[:, 0] = initial_density(self.media.background_density_water)
        # Store initial values for diagnostics
        self.dist_envelope[:, 0, :] = self.envelope
        self.dist_density[:, 0, :] = self.density
        self.axis_envelope[0, :] = self.envelope[self.domain.axis_node, :]
        self.axis_density[0, :] = self.density[self.domain.axis_node, :]
        self.peak_envelope[:, 0] = self.envelope[:, self.domain.peak_node]
        self.peak_density[:, 0] = self.density[:, self.domain.peak_node]
        self.k_array[0] = 0

    def solve_step(self):
        "Perform one propagation step."
        solve_dispersion(self.fourier_coeff, self.envelope, self.b_array)
        solve_density(
            self.b_array,
            self.density,
            self.e_temp,
            self.n_temp,
            self.w_array,
            self.next_density,
            self.domain.n_time_nodes,
            self.field_op_args,
            self.dens_op_args,
            self.dz_2,
            self.dz_6,
            self.dt_2,
            self.dt_6,
        )
        solve_envelope(
            self.left_matrix,
            self.right_matrix,
            self.domain.n_time_nodes,
            self.b_array,
            self.w_array,
            self.next_envelope,
        )

        # Update arrays
        self.envelope, self.next_envelope = self.next_envelope, self.envelope
        self.density, self.next_density = self.next_density, self.density

    def expensive_diagnostics(self, step):
        """Save memory expensive diagnostics data for current step."""
        self.dist_envelope[:, step, :] = self.envelope
        self.dist_density[:, step, :] = self.density
        self.k_array[step] = self.k_array[step - 1] + self.domain.dist_limitin

    def cheap_diagnostics(self, step):
        """Save memory cheap diagnostics data for current step."""
        # Cache accessed arrays and parameters
        axis_node = self.domain.axis_node
        envelope = self.envelope
        density = self.density

        # Cache axis data computations
        axis_envelope_data = envelope[axis_node]
        axis_density_data = density[axis_node]
        intensity = np.abs(axis_envelope_data)

        intensity_peak_node = np.argmax(intensity)
        density_peak_node = np.argmax(axis_density_data)

        self.axis_envelope[step] = axis_envelope_data
        self.axis_density[step] = axis_density_data
        self.peak_envelope[:, step] = envelope[:, intensity_peak_node]
        self.peak_density[:, step] = density[:, density_peak_node]

        # Check for non-finite values
        if np.any(~np.isfinite(self.envelope)):
            print("WARNING: Non-finite values detected in envelope")
            sys.exit(1)

        if np.any(~np.isfinite(self.density)):
            print("WARNING: Non-finite values detected in density")
            sys.exit(1)

    def propagate(self):
        """Propagate beam through all steps."""
        steps = self.domain.n_steps

        with tqdm(total=steps, desc="Progress") as pbar:
            for mm in range(1, self.domain.dist_limit + 1):
                for nn in range(1, self.domain.dist_limitin + 1):
                    kk = (mm - 1) * self.domain.dist_limitin + nn
                    self.solve_step()
                    self.cheap_diagnostics(kk)
                    pbar.update(1)
                    pbar.set_postfix({"m": mm, "n": nn, "k": kk})
                self.expensive_diagnostics(mm)


def main():
    "Main function."
    # Initialize classes
    const = UniversalConstants()
    media = MediaParameters()
    domain = DomainParameters(const)
    beam = BeamParameters(const, media)
    equation = EquationParameters(const, media, beam)

    # Create and run solver
    solver = FCNSolver(const, media, beam, domain, equation)
    solver.propagate()

    # Save to file
    np.savez(
        "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/water_fcn_1",
        e_dist=solver.dist_envelope,
        e_axis=solver.axis_envelope,
        e_peak=solver.peak_envelope,
        elec_dist=solver.dist_density,
        elec_axis=solver.axis_density,
        elec_peak=solver.peak_density,
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
