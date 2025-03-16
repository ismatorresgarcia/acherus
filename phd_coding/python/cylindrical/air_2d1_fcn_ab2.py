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
                     + ik_0n_2(1-a)|E|^2 E + ik_0n_2a (∫R(t-t')|E(t')|^2 dt') E

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
a: Raman response coefficient (for the delayed component of the Kerr effect).
R: molecular response function (for the delayed component of the Kerr effect).
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
    - complex 2D-array: Initial envelope field
    """
    space_decaying_term = -((r / w) ** 2)
    time_decaying_term = -(1 + im * ch) * (t / ptime) ** 2

    if f != 0:
        space_decaying_term -= 0.5 * im * wnum * r**2 / f

    return amp * np.exp(space_decaying_term + time_decaying_term)


def crank_nicolson_matrix(n_r, pos, coef):
    """
    Set the three diagonals for the
    Crank-Nicolson array with centered
    differences.

    Parameters:
    - n_r: number of radial nodes
    - pos: position of the Crank-Nicolson array (left or right)
    - coef: coefficient for the diagonal elements

    Returns:
    - complex 2D-array: sparse 2-array for the Crank-Nicolson matrix
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
    d_ind = [-1, 0, 1]

    return diags_array(diags, offsets=d_ind, format="csc")


@nb.njit
def set_density_operator(density, field, n_ph, neutral_dens, ofi_coef, ava_coef):
    """Set up the electron density evolution terms.

    Parameters:
    - density: density at current time slice
    - field: envelope at current time slice
    - n_ph: number of photons for MPI
    - neutral_dens: neutral density of the medium
    - ofi_coef: OFI coefficient
    - ava_coef: avalanche/cascade coefficient

    Returns:
    - float 1D-array: Electron density operator
    """
    field_2 = np.abs(field) ** 2
    field_2n = field_2**n_ph

    ofi = ofi_coef * field_2n * (neutral_dens - density)
    ava = ava_coef * density * field_2

    return ofi + ava


@nb.njit
def rk4_density_step(e_curr, n_curr, n_aux, dens_op_args, dt, dt2, dt6):
    """
    Compute one time step of the RK4 integration for electron
    density evolution.

    Parameters:
    - e_curr: envelope at current time slice
    - n_curr: density at current time slice
    - n_aux: auxiliary density array for RK4 integration
    - dens_op_args: arguments for the density operator
    - dt: time step
    - dt2: half time step
    - dt6: time step divided by 6

    Returns:
    - float 1D-array: Electron density at next time slice
    """
    k1_d = set_density_operator(n_curr, e_curr, *dens_op_args)
    n_aux = n_curr + dt2 * k1_d

    k2_d = set_density_operator(n_aux, e_curr, *dens_op_args)
    n_aux = n_curr + dt2 * k2_d

    k3_d = set_density_operator(n_aux, e_curr, *dens_op_args)
    n_aux = n_curr + dt * k3_d

    k4_d = set_density_operator(n_aux, e_curr, *dens_op_args)

    n_l = n_curr + dt6 * (k1_d + 2 * k2_d + 2 * k3_d + k4_d)

    return n_l


@nb.njit(parallel=True)
def solve_density(e_c, n_c, n_aux, n_n, n_t, dens_op_args, dt, dt2, dt6):
    """
    Solve electron density evolution for all time steps.

    Parameters:
    - e_c: envelope at current time slice
    - n_c: density at current time slice
    - n_aux: auxiliary density array for RK4 integration
    - n_n: density at next time slice
    - n_t: number of time nodes
    - dens_op_args: arguments for the density operator
    - dt: time step
    - dt2: half time step
    - dt6: time step divided by 6
    """
    # Set the initial condition
    n_n[:, 0], n_c[:, 0] = 0, 0

    # Solve the electron density evolution
    for ll in nb.prange(n_t - 1):
        e_curr = e_c[:, ll]
        n_curr = n_c[:, ll]

        n_next = rk4_density_step(e_curr, n_curr, n_aux, dens_op_args, dt, dt2, dt6)

        n_n[:, ll + 1] = n_next


@nb.njit
def set_scattering_operator(raman, draman, field, frq_coef, damp_coef):
    """Set up the Raman scattering evolution terms.

    Parameters:
    - raman: Raman response at current time slice
    - draman: Raman response time derivative at current time slice
    - field: envelope at current time slice
    - frq_coef: Raman frequency coefficient for the first ODE term
    - damp_coef: Raman frequency coefficient for the second ODE term

    Returns:
    - float 1D-array: Raman scattering operators
    """
    field_raman = field - raman

    return draman, frq_coef * field_raman + damp_coef * draman


@nb.njit
def rk4_scattering_step(
    r_curr, dr_curr, e_curr, r_aux, dr_aux, frq_coef, damp_coef, dt, dt2, dt6
):
    """
    Compute one time step of the RK4 integration for Raman
    scattering evolution.

    Parameters:
    - r_curr: raman response at current time slice
    - dr_curr: time derivative raman response at current time slice
    - e_curr: envelope at current time slice
    - r_aux: auxiliary raman response array
    - dr_aux: auxiliary raman response time derivative array
    - frq_coef: Raman frequency coefficient for the first ODE term
    - damp_coef: Raman frequency coefficient for the second ODE term
    - dt: time step
    - dt2: half time step
    - dt6: time step divided by 6

    Returns:
    - float 1D-array: Raman response at next time slice
    """
    k1_s, k1_x = set_scattering_operator(r_curr, dr_curr, e_curr, frq_coef, damp_coef)
    r_aux = r_curr + dt2 * k1_s
    dr_aux = dr_curr + dt2 * k1_x

    k2_s, k2_x = set_scattering_operator(r_aux, dr_aux, e_curr, frq_coef, damp_coef)
    r_aux = r_curr + dt2 * k2_s
    dr_aux = dr_curr + dt2 * k2_x

    k3_s, k3_x = set_scattering_operator(r_aux, dr_aux, e_curr, frq_coef, damp_coef)
    r_aux = r_curr + dt * k3_s
    dr_aux = dr_curr + dt * k3_x

    k4_s, k4_x = set_scattering_operator(r_aux, dr_aux, e_curr, frq_coef, damp_coef)

    r_l = r_curr + dt6 * (k1_s + 2 * k2_s + 2 * k3_s + k4_s)
    dr_l = dr_curr + dt6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)

    return r_l, dr_l


@nb.njit(parallel=True)
def solve_scattering(
    r_c, dr_c, e_c, r_aux, dr_aux, r_n, n_t, frq_arg, damp_arg, dt, dt2, dt6
):
    """
    Solve molecular Raman scattering delayed response for all time steps.

    Parameters:
    - r_c: raman response at current time slice
    - dr_c: raman response time derivative at current time slice
    - e_c: envelope at current time slice
    - r_aux: auxiliary raman response array
    - dr_aux: auxiliary raman response time derivative array
    - r_n: raman response at next time slice
    - n_t: number of time nodes
    - frq_coef: Raman frequency coefficient for the first ODE term
    - damp_coef: Raman frequency coefficient for the second ODE term
    - dt: time step
    - dt2: half time step
    - dt6: time step divided by 6
    """
    # Set the initial conditions
    r_n[:, 0], r_c[:, 0] = 0, 0
    dr_c[:, 0] = 0

    # Solve the raman scattering response
    for ll in nb.prange(n_t - 1):
        r_curr = r_c[:, ll]
        dr_curr = dr_c[:, ll]
        e_curr = e_c[:, ll]

        r_next, dr_next = rk4_scattering_step(
            r_curr, dr_curr, e_curr, r_aux, dr_aux, frq_arg, damp_arg, dt, dt2, dt6
        )

        r_n[:, ll + 1] = r_next
        dr_c[:, ll + 1] = dr_next


@nb.njit
def set_field_operator(field, density, raman, n_ph, p_coef, m_coef, k_coef, r_coef):
    """Set up the envelope propagation nonlinear terms.

    Parameters:
    - field: envelope at current time slice
    - density: electron density at current time slice
    - raman: Raman response at current time slice
    - n_ph: number of photons for MPI
    - p_coef: plasma coefficient
    - m_coef: MPA coefficient
    - k_coef: Kerr coefficient
    - r_coef: Raman coefficient

    Returns:
    - complex 1D-array: Nonlinear operator
    """
    field_2 = np.abs(field) ** 2
    field_2k2 = field_2 ** (n_ph - 1)

    nonlinear = field * (
        p_coef * density + m_coef * field_2k2 + k_coef * field_2 + r_coef * raman
    )

    return nonlinear


@nb.njit
def ab2_field_step(e_curr, n_curr, r_curr, field_op_args):
    """
    Compute one step of the AB2 integration for
    envelope propagation.

    Parameters:
    - e_curr: envelope at current time slice
    - n_curr: density at current time slice
    - r_curr: raman response at current time slice
    - field_op_args: arguments for the envelope operator

    Returns:
    - complex 1D-array: AB2 integration for one time slice
    """
    w_l = set_field_operator(e_curr, n_curr, r_curr, *field_op_args)

    return w_l


@nb.njit(parallel=True)
def solve_nonlinear(e_c, n_c, r_c, w_c, n_t, field_op_args):
    """
    Solve envelope propagation nonlinearities for all
    time steps.

    Parameters:
    - e_c: envelope at current time slice
    - n_c: density at current time slice
    - r_c: raman response at current time slice
    - e_aux: auxiliary envelope array for RK4 integration
    - w_c: pre-allocated array for the nonlinear terms
    - n_t: number of time nodes
    - field_op_args: arguments for the field operator
    - dz: distance step
    - dz2: half distance step
    - dz6: distance step divided by 6
    """
    for ll in nb.prange(n_t):
        e_curr = e_c[:, ll]
        n_curr = n_c[:, ll]
        r_curr = r_c[:, ll]

        w_curr = ab2_field_step(e_curr, n_curr, r_curr, field_op_args)

        w_c[:, ll] = w_curr


def solve_dispersion(fc, e_c, b):
    """
    Solve one step of the FFT
    propagation scheme for
    dispersion.

    Parameters:
    - fc: Fourier coefficient
    - e_c: envelope at current propagation step
    - b: envelope at next propagation step
    """
    b[:] = ifft(fc * fft(e_c, axis=1, workers=-1), axis=1, workers=-1)


def solve_envelope(lm, rm, n_t, b, w_p, w_c, e_n):
    """
    Solve one step of the generalized
    Crank-Nicolson scheme for envelope
    propagation.

    Parameters:
    - lm: left matrix for Crank-Nicolson
    - rm: right matrix for Crank-Nicolson
    - n_t: number of time nodes
    - b: envelope solution from FFT
    - w_p: previous propagation step nonlinear terms
    - w_c: current propagation step nonlinear terms
    - e_n: envelope at next propagation step
    """
    for ll in range(n_t):
        c = rm @ b[:, ll]
        d = c + 1.5 * w_c[:, ll] - 0.5 * w_p[:, ll]
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
        self.lin_ref_ind_air = 1.0003
        self.nlin_ref_ind_air = 5.57e-23
        self.gvd_coef_air = 2e-28
        self.n_photons_air = 7
        self.mpa_cnt_air = 6.5e-104
        self.mpi_cnt_air = 1.9e-111
        self.int_factor = 1
        # self.int_factor = (
        #    0.5 * const.light_speed * const.permittivity * self.lin_ref_ind_air
        # )
        self.energy_gap_air = 1.76e-18  # 11 eV
        self.collision_time_air = 3.5e-13
        self.neutral_dens_air = 5.4e25
        self.frq_resp_air = 16e12
        self.damp_time_air = 77e-15
        self.delay_frac_resp_air = 0.5


@dataclass
class BeamParameters:
    "Beam parameters."

    def __init__(self, const, media):
        # Basic parameters
        self.wavelength_0 = 775e-9
        self.waist_0 = 7e-4
        self.peak_time = 85e-15
        self.energy = 0.71e-3
        self.chirp = 0
        self.focal_length = 0

        # Derived parameters
        self.wavenumber_0 = 2 * const.pi / self.wavelength_0
        self.wavenumber = self.wavenumber_0 * media.lin_ref_ind_air
        self.frequency_0 = self.wavenumber_0 * const.light_speed
        self.power = self.energy / (self.peak_time * np.sqrt(0.5 * const.pi))
        self.cr_power = (
            3.77
            * self.wavelength_0**2
            / (8 * const.pi * media.lin_ref_ind_air * media.nlin_ref_ind_air)
        )
        self.intensity = 2 * self.power / (const.pi * self.waist_0**2)
        self.amplitude = np.sqrt(self.intensity / media.int_factor)


class DomainParameters:
    "Spatial and temporal domain parameters."

    def __init__(self, const):
        # Radial domain
        self.ini_radi_coor = 0
        self.fin_radi_coor = 5e-3
        self.i_radi_nodes = 5000

        # Distance domain
        self.ini_dist_coor = 0
        self.fin_dist_coor = 4
        self.n_steps = 4000
        self.dist_limit = 5

        # Time domain
        self.ini_time_coor = -250e-15
        self.fin_time_coor = 250e-15
        self.n_time_nodes = 8192

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

    def __init__(self, const, media, beam, domain):
        # Cache common parameters
        self.omega = beam.frequency_0
        self.omega_tau = self.omega * media.collision_time_air

        # Initialize main function parameters
        self._init_densities(const, media, beam)
        self._init_coefficients(media)
        self._init_operators(const, media, domain, beam)

    def _init_densities(self, const, media, beam):
        "Initialize density parameters."
        self.critical_dens = (
            const.permittivity
            * const.electron_mass
            * (self.omega / const.electron_charge) ** 2
        )
        self.bremss_cs_air = (beam.wavenumber * self.omega_tau) / (
            (media.lin_ref_ind_air**2 * self.critical_dens) * (1 + self.omega_tau**2)
        )

    def _init_coefficients(self, media):
        "Initialize equation coefficients."
        self.damp_frq = 1 / media.damp_time_air
        self.mpi_exp = 2 * media.n_photons_air
        self.mpa_exp = self.mpi_exp - 2
        self.ofi_coef = media.mpi_cnt_air * media.int_factor**media.n_photons_air
        self.ava_coef = self.bremss_cs_air * media.int_factor / media.energy_gap_air
        self.raman_coef_1 = (
            self.damp_frq**2 + media.frq_resp_air**2
        ) * media.int_factor
        self.raman_coef_2 = -2 * self.damp_frq

    def _init_operators(self, const, media, domain, beam):
        "Initialize equation operators."
        # Plasma coefficient calculation
        self.plasma_coef = (
            -0.5
            * const.im_unit
            * beam.wavenumber_0
            * domain.dist_step_len
            / (media.lin_ref_ind_air * self.critical_dens)
        )

        # MPA coefficient calculation
        self.mpa_coef = (
            -0.5
            * media.mpa_cnt_air
            * domain.dist_step_len
            * media.int_factor ** (media.n_photons_air - 1)
        )

        # Kerr coefficient calculation
        self.kerr_coef = (
            const.im_unit
            * beam.wavenumber_0
            * (1 - media.delay_frac_resp_air)
            * media.nlin_ref_ind_air
            * domain.dist_step_len
            * media.int_factor
        )

        # Raman coefficient calculation
        self.raman_coef = (
            const.im_unit
            * beam.wavenumber_0
            * media.delay_frac_resp_air
            * media.nlin_ref_ind_air
            * domain.dist_step_len
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
        self.dt = domain.time_step_len
        self.dt_2 = 0.5 * self.dt
        self.dt_6 = self.dt / 6

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
        self.raman = np.empty(shape, dtype=complex)
        self.next_envelope = np.empty_like(self.envelope)
        self.next_density = np.empty_like(self.density)
        self.next_raman = np.empty_like(self.raman)
        self.dist_envelope = np.empty(dist_shape, dtype=complex)
        self.dist_density = np.empty(dist_shape)
        self.axis_envelope = np.empty(axis_shape, dtype=complex)
        self.axis_density = np.empty(axis_shape)
        self.peak_envelope = np.empty(peak_shape, dtype=complex)
        self.peak_density = np.empty(peak_shape)
        self.draman_dt = np.empty_like(self.raman)
        self.w_array = np.empty_like(self.envelope)
        self.next_w_array = np.empty_like(self.envelope)
        self.b_array = np.empty_like(self.envelope)

        self.field_op_args = (
            self.media.n_photons_air,
            self.equation.plasma_coef,
            self.equation.mpa_coef,
            self.equation.kerr_coef,
            self.equation.raman_coef,
        )
        self.dens_op_args = (
            self.media.n_photons_air,
            self.media.neutral_dens_air,
            self.equation.ofi_coef,
            self.equation.ava_coef,
        )

        self.n_temp = np.empty(self.domain.n_radi_nodes)
        self.r_temp = np.empty(self.domain.n_radi_nodes, dtype=complex)
        self.dr_temp = np.empty_like(self.r_temp)

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
            * self.media.gvd_coef_air
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
        # Store initial values for diagnostics
        self.dist_envelope[:, 0, :] = self.envelope
        self.dist_density[:, 0, :] = 0
        self.axis_envelope[0, :] = self.envelope[self.domain.axis_node, :]
        self.axis_density[0, :] = self.density[self.domain.axis_node, :]
        self.peak_envelope[:, 0] = self.envelope[:, self.domain.peak_node]
        self.peak_density[:, 0] = self.density[:, self.domain.peak_node]
        self.k_array[0] = 0

    def solve_step(self, step):
        "Perform one propagation step."
        solve_density(
            self.envelope,
            self.density,
            self.n_temp,
            self.next_density,
            self.domain.n_time_nodes,
            self.dens_op_args,
            self.dt,
            self.dt_2,
            self.dt_6,
        )
        solve_scattering(
            self.raman,
            self.draman_dt,
            self.envelope,
            self.r_temp,
            self.dr_temp,
            self.next_raman,
            self.domain.n_time_nodes,
            self.equation.raman_coef_1,
            self.equation.raman_coef_2,
            self.dt,
            self.dt_2,
            self.dt_6,
        )
        solve_dispersion(self.fourier_coeff, self.envelope, self.b_array)
        solve_nonlinear(
            self.b_array,
            self.next_density,
            self.next_raman,
            self.next_w_array,
            self.domain.n_time_nodes,
            self.field_op_args,
        )

        # For step = 1, initialize Adam_Bashforth second condition
        if step == 1:
            np.copyto(self.w_array, self.next_w_array)
            self.axis_envelope[1] = self.envelope[self.domain.axis_node]
            self.axis_density[1] = self.density[self.domain.axis_node]
            self.peak_envelope[:, 1] = self.envelope[:, self.domain.peak_node]
            self.peak_density[:, 1] = self.density[:, self.domain.peak_node]
        solve_envelope(
            self.left_matrix,
            self.right_matrix,
            self.domain.n_time_nodes,
            self.b_array,
            self.w_array,
            self.next_w_array,
            self.next_envelope,
        )

        # Update arrays
        np.copyto(self.envelope, self.next_envelope)
        np.copyto(self.density, self.next_density)
        np.copyto(self.raman, self.next_raman)
        np.copyto(self.w_array, self.next_w_array)

    def save_cheap_diagnostics(self, step):
        """Save memory cheap diagnostics data for current step."""
        if step > 1:
            # Cache accessed arrays and parameters
            axis_node = self.domain.axis_node
            envelope = self.envelope
            density = self.density

            # Cache axis data computations
            axis_envelope_data = envelope[axis_node]
            axis_density_data = density[axis_node]
            axis_intensity_data = np.abs(axis_envelope_data)

            intensity_peak_node = np.argmax(axis_intensity_data)
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

    def save_expensive_diagnostics(self, step):
        """Save memory expensive diagnostics data for current step."""
        self.dist_envelope[:, step, :] = self.envelope
        self.dist_density[:, step, :] = self.density
        self.k_array[step] = self.k_array[step - 1] + self.domain.dist_limitin

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
    domain = DomainParameters(const)
    beam = BeamParameters(const, media)
    equation = EquationParameters(const, media, beam, domain)

    # Create and run solver
    solver = FCNSolver(const, media, beam, domain, equation)
    solver.propagate()

    # Save to file
    np.savez(
        "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/air_fcn_ab2_1",
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
        lin_ref_ind=media.lin_ref_ind_air,
    )


if __name__ == "__main__":
    main()
