"""
This program solves the Unidirectional Pulse Propagation.uppe.(UPPE) of an ultra-intense
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
            However, the dictionary "medium" has an entry "INT_FACTOR" where the conversion
            factor can be changed at will between the two unit systems.

E: envelope.
N: electron density (in the interacting medium).
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
t: time coordinate.
k: wavenumber (in the interacting medium).
k'': GVD coefficient of 2nd order.
k_0: wavenumber (in vacuum).
N_n: neutral density (of the interacting medium).
N_c: critical density (of the interacting medium).
n_2: nonlinear refractive index (for a third-order centrosymmetric medium).
S_K: nonlinear optical field ionization coefficient.
B_K: nonlinear multiphoton absorption coefficient.
S_w: bremsstrahlung cross-section for avalanche (cascade) ionization.
U_i: ionization energy (for the interacting medium).
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


def initialize_envelope(
    grid_radial,
    grid_time,
    unit_imag,
    amplitude,
    wavenumber,
    waist,
    time_peak,
    chirp,
    focal_length,
):
    """
    Set up the Gaussian beam.

    Parameters:
    - grid_radial: radial coordinates array
    - grid_time: time coordinates array
    - unit_imag: square root of -1
    - amplitude: initial amplitude of the beam
    - wavenumber: initial wavenumber of the beam
    - waist: initial waist of the beam
    - time_peak: initial peak time of the beam
    - chirp: chirp of the beam
    - focal_length: focal length of the beam

    Returns:
    - complex 2D-array: Initial envelope
    """
    space_decaying_term = -((grid_radial / waist) ** 2)
    time_decaying_term = -(1 + unit_imag * chirp) * (grid_time / time_peak) ** 2

    if focal_length != 0:
        space_decaying_term -= (
            0.5 * unit_imag * wavenumber * grid_radial**2 / focal_length
        )

    return amplitude * np.exp(space_decaying_term + time_decaying_term)


def create_crank_nicolson_matrix(nodes_radial, matrix_position, coef_delta):
    """
    Set the three diagonals for the
    Crank-Nicolson array with centered
    differences.

    Parameters:
    - nodes_radial: number of radial nodes
    - matrix_position: position of the Crank-Nicolson array (left or right)
    - coef_delta: coefficient for the diagonal elements

    Returns:
    - complex 2D-array: sparse 2-array for the Crank-Nicolson matrix
    """
    dc = 1 + 2 * coef_delta
    indices = np.arange(1, nodes_radial - 1)

    diag_lower = -coef_delta * (1 - 0.5 / indices)
    diag_main = np.full(nodes_radial, dc)
    diag_upper = -coef_delta * (1 + 0.5 / indices)

    diag_lower = np.append(diag_lower, [0])
    diag_upper = np.insert(diag_upper, 0, [0])
    if matrix_position == "left":
        diag_main[0], diag_main[-1] = dc, 1
        diag_upper[0] = -2 * coef_delta
    else:
        diag_main[0], diag_main[-1] = dc, 0
        diag_upper[0] = -2 * coef_delta

    diags = [diag_lower, diag_main, diag_upper]
    diags_ind = [-1, 0, 1]

    return diags_array(diags, offsets=diags_ind, format="csc")


@nb.njit
def _set_density_operator(
    dens_slice, env_slice, photon_number, dens_neutral, coef_ofi, coef_ava
):
    """Set up the electron density evolution terms.

    Parameters:
    - dens_slice: density at current time slice
    - env_slice: envelope at current time slice
    - photon_number: number of photons for MPI
    - dens_neutral: neutral density of the medium
    - coef_ofi: OFI coefficient
    - coef_ava: avalanche/cascade coefficient

    Returns:
    - float 1D-array: Electron density operator
    """
    env_slice_2 = np.abs(env_slice) ** 2
    env_slice_k = env_slice_2**photon_number

    term_ofi = coef_ofi * env_slice_k * (dens_neutral - dens_slice)
    term_ava = coef_ava * dens_slice * env_slice_2

    return term_ofi + term_ava


@nb.njit
def _rk4_density_step(
    env_slice, dens_slice, dens_aux, dens_op_args, dt, dt_half, dt_sixth
):
    """
    Compute one time step of the RK4 integration for electron
    density evolution.

    Parameters:
    - env_slice: envelope at current time slice
    - dens_slice: density at current time slice
    - dens_aux: auxiliary density array for RK4 integration
    - dens_op_args: arguments for the density operator
    - dt: time step
    - dt_half: half time step
    - dt_sixth: time step divided by 6

    Returns:
    - float 1D-array: Electron density at next time slice
    """
    k1_d = _set_density_operator(dens_slice, env_slice, *dens_op_args)
    dens_aux = dens_slice + dt_half * k1_d

    k2_d = _set_density_operator(dens_aux, env_slice, *dens_op_args)
    dens_aux = dens_slice + dt_half * k2_d

    k3_d = _set_density_operator(dens_aux, env_slice, *dens_op_args)
    dens_aux = dens_slice + dt * k3_d

    k4_d = _set_density_operator(dens_aux, env_slice, *dens_op_args)

    dens_next = dens_slice + dt_sixth * (k1_d + 2 * k2_d + 2 * k3_d + k4_d)

    return dens_next


@nb.njit(parallel=True)
def solve_density(env, dens, dens_aux, nodes_time, dens_op_args, dt, dt_half, dt_sixth):
    """
    Solve electron density evolution for all time steps.

    Parameters:
    - env: envelope at all time slices
    - dens: density at all time slices
    - dens_aux: auxiliary density array for RK4 integration
    - nodes_time: number of time nodes
    - dens_op_args: arguments for the density operator
    - dt: time step
    - dt_half: half time step
    - dt_sixth: time step divided by 6
    """
    # Set the initial condition
    dens[:, 0] = 0

    # Solve the electron density evolution
    for ll in nb.prange(nodes_time - 1):
        env_slice = env[:, ll]
        dens_slice = dens[:, ll]

        dens_next = _rk4_density_step(
            env_slice, dens_slice, dens_aux, dens_op_args, dt, dt_half, dt_sixth
        )

        dens[:, ll + 1] = dens_next


@nb.njit
def _set_envelope_operator(
    env_slice, dens_slice, number_photons, coef_p, coef_m, coef_k
):
    """Set up the envelope propagation nonlinear terms.

    Parameters:
    - env_slice: envelope at current time slice
    - dens_slice: electron density at current time slice
    - number_photons: number of photons for MPI
    - coef_p: plasma coefficient
    - coef_m: MPA coefficient
    - coef_k: Kerr coefficient

    Returns:
    - complex 1D-array: Nonlinear operator
    """
    env_slice_2 = np.abs(env_slice) ** 2
    env_slice_2k2 = env_slice_2 ** (number_photons - 1)

    nlin_slice = env_slice * (
        coef_p * dens_slice + coef_m * env_slice_2k2 + coef_k * env_slice_2
    )

    return nlin_slice


@nb.njit
def _rk4_envelope_step(
    env_slice, dens_slice, env_aux, env_op_args, dz, dz_half, dz_sixth
):
    """
    Compute one step of the RK4 integration for envelope propagation.

    Parameters:
    - env_slice: envelope at current time slice
    - dens_slice: density at current time slice
    - env_aux: auxiliary envelope array for RK4 integration
    - env_op_args: arguments for the envelope operator
    - dz: distance step
    - dz_half: half distance step
    - dz_sixth: distance step divided by 6

    Returns:
    - complex 1D-array: RK4 integration for one time slice
    """
    k1_f = _set_envelope_operator(env_slice, dens_slice, *env_op_args)
    env_aux = env_slice + dz_half * k1_f

    k2_f = _set_envelope_operator(env_aux, dens_slice, *env_op_args)
    env_aux = env_slice + dz_half * k2_f

    k3_f = _set_envelope_operator(env_aux, dens_slice, *env_op_args)
    env_aux = env_slice + dz * k3_f

    k4_f = _set_envelope_operator(env_aux, dens_slice, *env_op_args)

    nlin_curr_slice = dz_sixth * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)

    return nlin_curr_slice


@nb.njit(parallel=True)
def solve_nonlinear(
    env, dens, env_aux, nlin_curr, nodes_time, env_op_args, dz, dz_half, dz_sixth
):
    """
    Solve envelope propagation nonlinearities for all
    time steps.

    Parameters:
    - env: envelope at all time slices
    - dens: density at all time slices
    - env_aux: auxiliary envelope array for RK4 integration
    - nlin_curr: pre-allocated array for the nonlinear terms
    - nodes_time: number of time nodes
    - env_op_args: arguments for the envelope operator
    - dz: distance step
    - dz_half: half distance step
    - dz_sixth: distance step divided by 6
    """
    for ll in nb.prange(nodes_time):
        env_slice = env[:, ll]
        dens_slice = dens[:, ll]

        nlin_curr_slice = _rk4_envelope_step(
            env_slice, dens_slice, env_aux, env_op_args, dz, dz_half, dz_sixth
        )

        nlin_curr[:, ll] = nlin_curr_slice


def solve_dispersion(coef_fourier, env_curr, env_next):
    """
    Solve one step of the FFT
    propagation scheme for
    dispersion.

    Parameters:
    - coef_fourier: Fourier coefficient for advancing one step
    - env_curr: envelope at current propagation step
    - env_next: envelope at next propagation step
    """
    env_next[:] = ifft(
        coef_fourier * fft(env_curr, axis=1, workers=-1), axis=1, workers=-1
    )


def solve_envelope(
    matrix_left, matrix_right, nodes_time, env_curr, nlin_curr, env_next
):
    """
    Solve one step of the generalized
    Crank-Nicolson scheme for envelope
    propagation.

    Parameters:
    - matrix_left: left matrix for Crank-Nicolson
    - matrix_right: right matrix for Crank-Nicolson
    - nodes_time: number of time nodes
    - env_curr: envelope solution from FFT
    - nlin_curr: current propagation step nonlinear terms
    - env_next: envelope at next propagation step
    """
    for ll in range(nodes_time):
        c = matrix_right @ env_curr[:, ll]
        d = c + nlin_curr[:, ll]
        env_next[:, ll] = matrix_left.solve(d)


@dataclass
class Constants:
    "Physical and mathematical constants."

    def __init__(self):
        self.light_speed = 299792458.0
        self.permittivity = 8.8541878128e-12
        self.electron_mass = 9.1093837139e-31
        self.electron_charge = 1.602176634e-19
        self.planck_bar = 1.05457182e-34
        self.pi = np.pi
        self.im_unit = 1j


@dataclass
class MediumParameters:
    "Medium parameters to be chosen."

    def __init__(self):
        self.ref_ind_linear = 1.334
        self.ref_ind_nonlinear = 4.1e-20
        self.coefficient_gvd = 248e-28
        self.photons_absorbed = 5
        self.constant_mpa = 1e-61
        self.constant_mpi = 1.2e-72
        self.intensity_units = 1
        # self.intensity_units = (
        #    0.5 * const.light_speed * const.permittivity * self.lin_ref_ind
        # )
        self.energy_ionization = 1.04e-18  # 6.5 eV
        self.time_collisions = 3e-15
        self.density_neutral = 6.68e28


@dataclass
class LaserPulseParameters:
    "Laser pulse physical parameters and derived properties."

    def __init__(self, const, medium):
        # Basic parameters
        self.input_wavelength = 800e-9
        self.input_waist = 75e-6
        self.input_peak_time = 130e-15
        self.input_energy = 2.2e-6
        self.input_chirp = 0
        self.input_focal_length = 0

        # Derived parameters
        self.input_wavenumber_0 = 2 * const.pi / self.input_wavelength
        self.input_wavenumber = self.input_wavenumber_0 * medium.ref_ind_linear
        self.input_frequency = self.input_wavenumber_0 * const.light_speed
        self.input_power = self.input_energy / (
            self.input_peak_time * np.sqrt(0.5 * const.pi)
        )
        self.critical_power = (
            3.77
            * self.input_wavelength**2
            / (8 * const.pi * medium.ref_ind_linear * medium.ref_ind_nonlinear)
        )
        self.input_intensity = 2 * self.input_power / (const.pi * self.input_waist**2)
        self.input_amplitude = np.sqrt(self.input_intensity / medium.intensity_units)


class Grid:
    "Spatial and temporal grid parameters."

    def __init__(self, const):
        # Radial domain
        self.radial_coor_ini = 0
        self.radial_coor_fin = 25e-4
        self.radial_nodes_inner = 1500

        # Distance domain
        self.distance_coor_ini = 0
        self.distance_coor_fin = 3e-2
        self.distance_steps = 1000
        self.distance_limit = 5

        # Time domain
        self.time_coor_ini = -250e-15
        self.time_coor_fin = 250e-15
        self.time_nodes = 4096

        # Initialize derived parameters functions
        self._setup_derived_parameters()
        self._setup_arrays(const)

    @property
    def radial_nodes(self):
        "Total number of radial nodes for boundary conditions."
        return self.radial_nodes_inner + 2

    @property
    def distance_limitin(self):
        "Inner loop auxiliary parameters."
        return self.distance_steps // self.distance_limit

    def _setup_derived_parameters(self):
        "Setup domain parameters."
        # Calculate steps
        self.radial_step_len = (self.radial_coor_fin - self.radial_coor_ini) / (
            self.radial_nodes - 1
        )
        self.distance_step_len = (
            self.distance_coor_fin - self.distance_coor_ini
        ) / self.distance_steps
        self.time_step_len = (self.time_coor_fin - self.time_coor_ini) / (
            self.time_nodes - 1
        )
        self.frq_step_len = 2 * np.pi / (self.time_nodes * self.time_step_len)

        # Calculate nodes
        self.node_axis = int(-self.radial_coor_ini / self.radial_step_len)
        self.node_peak = self.time_nodes // 2

    def _setup_arrays(self, const):
        "Setup grid arrays."
        # 1D
        self.radial_grid = np.linspace(
            self.radial_coor_ini, self.radial_coor_fin, self.radial_nodes
        )
        self.distance_grid = np.linspace(
            self.distance_coor_ini, self.distance_coor_fin, self.distance_steps + 1
        )
        self.time_grid = np.linspace(
            self.time_coor_ini, self.time_coor_fin, self.time_nodes
        )
        self.frequency_grid = (
            2 * const.pi * fftfreq(self.time_nodes, self.time_step_len)
        )

        # 2D
        self.radial_2d_grid, self.time_2d_grid = np.meshgrid(
            self.radial_grid, self.time_grid, indexing="ij"
        )


@dataclass
class UPPEParameters:
    """Pulse propagation and electron density evolution
    parameters for the final numerical scheme."""

    def __init__(self, const, medium, laser):
        # Cache common parameters
        self.omega = laser.input_frequency
        self.omega_tau = self.omega * medium.time_collisions

        # Initialize main function parameters
        self._init_densities(const, medium, laser)
        self._init_coefficients(medium)
        self._init_operators(const, medium, laser)

    def _init_densities(self, const, medium, laser):
        "Initialize density parameters."
        self.dens_critical = (
            const.permittivity
            * const.electron_mass
            * (self.omega / const.electron_charge) ** 2
        )
        self.cross_sec_bremss = (laser.input_wavenumber * self.omega_tau) / (
            (medium.ref_ind_linear**2 * self.dens_critical) * (1 + self.omega_tau**2)
        )

    def _init_coefficients(self, medium):
        "Initialize equation coefficients."
        self.exp_mpi = 2 * medium.photons_absorbed
        self.exp_mpa = self.exp_mpi - 2
        self.coef_ofi = (
            medium.constant_mpi * medium.intensity_units**medium.photons_absorbed
        )
        self.coef_ava = (
            self.cross_sec_bremss * medium.intensity_units / medium.energy_ionization
        )

    def _init_operators(self, const, medium, laser):
        "Initialize equation operators."
        # Plasma coefficient calculation
        self.coef_plasma = (
            -0.5
            * const.im_unit
            * laser.input_wavenumber_0
            / (medium.ref_ind_linear * self.dens_critical)
        )

        # MPA coefficient calculation
        self.coef_mpa = (
            -0.5
            * medium.constant_mpa
            * medium.intensity_units ** (medium.photons_absorbed - 1)
        )

        # Kerr coefficient calculation
        self.coef_kerr = (
            const.im_unit
            * laser.input_wavenumber_0
            * medium.ref_ind_nonlinear
            * medium.intensity_units
        )


class FCNSolver:
    """FCN solver class for beam propagation."""

    def __init__(self, const, medium, laser, grid, uppe):
        # Cache common parameters
        self.const = const
        self.medium = medium
        self.laser = laser
        self.grid = grid

        # Compute frequent constants
        self.dz = grid.distance_step_len
        self.dz_2 = self.dz * 0.5
        self.dz_6 = self.dz / 6
        self.dt = grid.time_step_len
        self.dt_2 = self.dt * 0.5
        self.dt_6 = self.dt / 6

        # Initialize arrays and operators
        shape = (self.grid.radial_nodes, self.grid.time_nodes)
        shape_distance = (
            self.grid.radial_nodes,
            self.grid.distance_limit + 1,
            self.grid.time_nodes,
        )
        shape_axis = (self.grid.distance_steps + 1, self.grid.time_nodes)
        shape_peak = (self.grid.radial_nodes, self.grid.distance_steps + 1)
        self.envelope = np.empty(shape, dtype=complex)
        self.envelope_next = np.empty_like(self.envelope)
        self.envelope_split = np.empty_like(self.envelope)
        self.envelope_distance = np.empty(shape_distance, dtype=complex)
        self.envelope_axis = np.empty(shape_axis, dtype=complex)
        self.envelope_peak = np.empty(shape_peak, dtype=complex)
        self.density = np.empty(shape)
        self.density_distance = np.empty(shape_distance)
        self.density_axis = np.empty(shape_axis)
        self.density_peak = np.empty(shape_peak)
        self.nlin_current = np.empty_like(self.envelope)

        self.envelope_op_args = (
            medium.photons_absorbed,
            uppe.coef_plasma,
            uppe.coef_mpa,
            uppe.coef_kerr,
        )
        self.density_op_args = (
            medium.photons_absorbed,
            medium.density_neutral,
            uppe.coef_ofi,
            uppe.coef_ava,
        )

        self.envelope_auxiliar = np.empty(self.grid.radial_nodes, dtype=complex)
        self.density_auxiliar = np.empty(self.grid.radial_nodes)

        # Setup tracking variables
        self.indices_k = np.empty(self.grid.distance_limit + 1, dtype=int)

        # Setup operators and initial condition
        self.setup_operators()
        self.setup_initial_condition()

    def setup_operators(self):
        """Setup FCN operators."""
        delta_coef_radial = (
            0.25
            * self.grid.distance_step_len
            / (self.laser.input_wavenumber * self.grid.radial_step_len**2)
        )
        delta_coef_time = (
            -0.25
            * self.grid.distance_step_len
            * self.medium.coefficient_gvd
            / self.grid.time_step_len**2
        )

        # Setup Fourier coefficient
        self.fourier_coef = np.exp(
            -2
            * self.const.im_unit
            * delta_coef_time
            * (self.grid.frequency_grid * self.grid.time_step_len) ** 2
        )

        # Setup CN operators
        mat_cnt = self.const.im_unit * delta_coef_radial
        self.matrix_cn_left = create_crank_nicolson_matrix(
            self.grid.radial_nodes, "left", mat_cnt
        )
        self.matrix_cn_right = create_crank_nicolson_matrix(
            self.grid.radial_nodes, "right", -mat_cnt
        )
        self.matrix_cn_left = splu(self.matrix_cn_left)

    def setup_initial_condition(self):
        """Setup initial conditions."""
        # Envelope initial condition at z=0
        self.envelope = initialize_envelope(
            self.grid.radial_2d_grid,
            self.grid.time_2d_grid,
            self.const.im_unit,
            self.laser.input_amplitude,
            self.laser.input_wavenumber,
            self.laser.input_waist,
            self.laser.input_peak_time,
            self.laser.input_chirp,
            self.laser.input_focal_length,
        )
        # Store initial values for diagnostics
        self.envelope_distance[:, 0, :] = self.envelope
        self.envelope_axis[0, :] = self.envelope[self.grid.node_axis, :]
        self.envelope_peak[:, 0] = self.envelope[:, self.grid.node_peak]
        self.density_distance[:, 0, :] = 0
        self.density_axis[0, :] = self.density[self.grid.node_axis, :]
        self.density_peak[:, 0] = self.density[:, self.grid.node_peak]
        self.indices_k[0] = 0

    def solve_step(self):
        "Perform one propagation step."
        solve_density(
            self.envelope,
            self.density,
            self.density_auxiliar,
            self.grid.time_nodes,
            self.density_op_args,
            self.dt,
            self.dt_2,
            self.dt_6,
        )
        solve_dispersion(self.fourier_coef, self.envelope, self.envelope_split)
        solve_nonlinear(
            self.envelope_split,
            self.density,
            self.envelope_auxiliar,
            self.nlin_current,
            self.grid.time_nodes,
            self.envelope_op_args,
            self.dz,
            self.dz_2,
            self.dz_6,
        )
        solve_envelope(
            self.matrix_cn_left,
            self.matrix_cn_right,
            self.grid.time_nodes,
            self.envelope_split,
            self.nlin_current,
            self.envelope_next,
        )

        # Update arrays
        self.envelope, self.envelope_next = self.envelope_next, self.envelope

    def cheap_diagnostics(self, step):
        """Save memory cheap diagnostics data for current step."""
        # Cache accessed arrays and parameters
        node_axis = self.grid.node_axis
        envelope = self.envelope
        density = self.density

        # Cache axis data computations
        axis_data_envelope = envelope[node_axis]
        axis_data_density = density[node_axis]
        axis_data_intensity = np.abs(axis_data_envelope)

        peak_node_intensity = np.argmax(axis_data_intensity)
        peak_node_density = np.argmax(axis_data_density)

        self.envelope_axis[step] = axis_data_envelope
        self.envelope_peak[:, step] = envelope[:, peak_node_intensity]
        self.density_axis[step] = axis_data_density
        self.density_peak[:, step] = density[:, peak_node_density]

        # Check for non-finite values
        if np.any(~np.isfinite(self.envelope)):
            print("WARNING: Non-finite values detected in envelope")
            sys.exit(1)

        if np.any(~np.isfinite(self.density)):
            print("WARNING: Non-finite values detected in density")
            sys.exit(1)

    def expensive_diagnostics(self, step):
        """Save memory expensive diagnostics data for current step."""
        self.envelope_distance[:, step, :] = self.envelope
        self.density_distance[:, step, :] = self.density
        self.indices_k[step] = self.indices_k[step - 1] + self.grid.distance_limitin

    def propagate(self):
        """Propagate beam through all steps."""
        steps = self.grid.distance_steps

        with tqdm(total=steps, desc="Progress") as pbar:
            for mm in range(1, self.grid.distance_limit + 1):
                for nn in range(1, self.grid.distance_limitin + 1):
                    kk = (mm - 1) * self.grid.distance_limitin + nn
                    self.solve_step()
                    self.cheap_diagnostics(kk)
                    pbar.update(1)
                    pbar.set_postfix({"m": mm, "n": nn, "k": kk})
                self.expensive_diagnostics(mm)


def main():
    "Main function."
    # Initialize classes
    const = Constants()
    medium = MediumParameters()
    laser = LaserPulseParameters(const, medium)
    grid = Grid(const)
    uppe = UPPEParameters(const, medium, laser)

    # Create and run solver
    solver = FCNSolver(const, medium, laser, grid, uppe)
    solver.propagate()

    # Save to file
    np.savez(
        "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/water_fcn_rk4_1",
        e_dist=solver.envelope_distance,
        e_axis=solver.envelope_axis,
        e_peak=solver.envelope_peak,
        elec_dist=solver.density_distance,
        elec_axis=solver.density_axis,
        elec_peak=solver.density_peak,
        k_array=solver.indices_k,
        ini_radi_coor=grid.radial_coor_ini,
        fin_radi_coor=grid.radial_coor_fin,
        ini_dist_coor=grid.distance_coor_ini,
        fin_dist_coor=grid.distance_coor_fin,
        ini_time_coor=grid.time_coor_ini,
        fin_time_coor=grid.time_coor_fin,
        axis_node=grid.node_axis,
        peak_node=grid.node_peak,
        lin_ref_ind=medium.ref_ind_linear,
    )


if __name__ == "__main__":
    main()
