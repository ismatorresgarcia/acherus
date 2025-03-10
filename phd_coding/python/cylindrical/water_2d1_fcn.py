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

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.sparse import diags_array
from scipy.sparse.linalg import splu
from tqdm import tqdm


def initial_envelope(r, t, beam):
    """
    Set the post-lens chirped Gaussian beam.

    Parameters:
    - r: radial coordinates array
    - t: time coordinates array
    - beam: beam parameters object

    Returns:
    - complex array: Initial envelope field
    """
    return (
        beam.amplitude * np.exp(-((r / beam.waist_0) ** 2) - (t / beam.peak_time) ** 2)
    ).astype(complex)


def initial_density(media):
    """
    Set the initial electron density distribution.

    Parameters:
    - r: radial coordinates array
    - media: media parameters object

    Returns:
    - float array: Initial free electron density
    """
    return media.background_density_water


def crank_nicolson_matrix(n, lr, c):
    """
    Set the three diagonals for the Crank-Nicolson array with centered differences.

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

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]
    return diags_array(diags, offsets=offset, format="csc")


def density_rate(n_c, e_c, equation, media):
    """
    Compute electron density equation terms.

    Parameters:
    - n_c: electron density at step l
    - e_c: envelope at step l
    - equation: equation parameters
    - media: media parameters

    Returns:
    - ndarray: Rate of change of electron density
    """
    abs_e_c = np.abs(e_c) ** 2
    n_photons = media.n_photons_water
    neutral_dens = media.neutral_dens_water

    ofi = equation.ofi_coef * (abs_e_c**n_photons) * (neutral_dens - n_c)
    ava = equation.ava_coef * n_c * abs_e_c
    return ofi + ava


def solve_density(n_c, e_c, dt, equation, media):
    """
    Compute one step of the 4th order Runge-Kutta method for electron density.

    Parameters:
    - n_c: electron density at step l
    - e_c: envelope at step l
    - dt: time step tuple
    - equation: equation parameters
    - media: media parameters
    """
    dt_0, dt_2, dt_6 = dt
    for l in range(e_c.shape[1] - 1):
        n_c0 = n_c[:, l]
        e_c0 = e_c[:, l]
        e_c1 = e_c[:, l + 1]
        e_mid = 0.5 * (e_c0 + e_c1)

        k1 = density_rate(n_c0, e_c0, equation, media)
        k2 = density_rate(n_c0 + dt_2 * k1, e_mid, equation, media)
        k3 = density_rate(n_c0 + dt_2 * k2, e_mid, equation, media)
        k4 = density_rate(n_c0 + dt_0 * k3, e_c1, equation, media)

        n_c[:, l + 1] = n_c0 + dt_6 * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_dispersion(fc, e_c, b):
    """
    Compute one step of the FFT propagation scheme for dispersion.

    Parameters:
    - fc: precomputed Fourier coefficient
    - e_c: envelope at step k
    - b: pre-allocated array for envelope at step k + 1
    """
    b[:] = ifft(fc * fft(e_c, axis=1, workers=-1), axis=1, workers=-1)


def calculate_nonlinear(e_c, n_c, w_c, media, equation):
    """
    Compute the nonlinear terms.

    Parameters:
    - e_c: envelope at step k
    - n_c: electron density at step k
    - w_c: pre-allocated array for Adam-Bashforth terms
    - equation: equation parameters
    """
    e_c_2 = np.abs(e_c) ** 2
    e_c_2k2 = e_c_2 ** (media.n_photons_water - 1)
    w_c[:] = e_c * (
        equation.plasma_coef * n_c
        + equation.mpa_coef * e_c_2k2
        + equation.kerr_coef * e_c_2
    )


def solve_envelope(mats, b, w_c, w_n, e_n):
    """
    Compute one step of the Crank-Nicolson propagation scheme.

    Parameters:
    - mats: dict containing sparse arrays for left and right operators
    - b: intermediate array from FFT step
    - w_c: current step nonlinear terms
    - w_n: previous step nonlinear terms
    - e_n: pre-allocated array for envelope at step k + 1
    """
    lm, rm = mats
    for l in range(e_n.shape[1]):
        c = rm @ b[:, l]
        d = c + 1.5 * w_c[:, l] - 0.5 * w_n[:, l]
        e_n[:, l] = lm.solve(d)


@dataclass
class UniversalConstants:
    "UniversalConstants."

    def __init__(self):
        self.light_speed = np.float64(299792458.0)
        self.permittivity = np.float64(8.8541878128e-12)
        self.electron_mass = np.float64(9.1093837139e-31)
        self.electron_charge = np.float64(1.602176634e-19)
        self.planck_bar = np.float64(1.05457182e-34)
        self.pi = np.float64(np.pi)
        self.im_unit = 1j


@dataclass
class MediaParameters:
    "Media parameters."

    def __init__(self):
        self.lin_ref_ind_water = np.float64(1.334)
        self.nlin_ref_ind_water = np.float64(4.1e-20)
        self.gvd_coef_water = np.float64(248e-28)
        self.n_photons_water = np.int16(5)
        self.mpa_cnt_water = np.float64(1e-61)
        self.mpi_cnt_water = np.float64(1.2e-72)
        self.int_factor = np.int16(1)
        # self.int_factor = (
        #    0.5 * const.light_speed * const.permittivity * self.lin_ref_ind_water
        # )
        self.energy_gap_water = np.float64(1.04e-18)  # 6.5 eV
        self.collision_time_water = np.float64(3e-15)
        self.neutral_dens_water = np.float64(6.68e28)
        self.background_density_water = np.float64(1e-6)


@dataclass
class BeamParameters:
    "Beam parameters."

    def __init__(self, const, media):
        # Basic parameters
        self.wavelength_0 = np.float64(800e-9)
        self.waist_0 = np.float64(75e-6)
        self.peak_time = np.float64(130e-15)
        self.energy = np.float64(2.2e-6)

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

    def __init__(self, const, media, beam, domain):
        # Cache common parameters
        self.omega = beam.frequency_0
        self.omega_tau = self.omega * media.collision_time_water

        # Initialize main function parameters
        self._init_densities(const, media, beam)
        self._init_coefficients(media)
        self._init_operators(const, media, beam, domain)

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

    def _init_operators(self, const, media, beam, domain):
        "Initialize equation operators."
        # Plasma coefficient calculation
        self.plasma_coef = (
            -0.5
            * const.im_unit
            * beam.wavenumber_0
            * domain.dist_step_len
            / (media.lin_ref_ind_water * self.critical_dens)
        )

        # MPA coefficient calculation
        self.mpa_coef = (
            -0.5
            * media.mpa_cnt_water
            * domain.dist_step_len
            * media.int_factor ** (media.n_photons_water - 1)
        )

        # Kerr coefficient calculation
        self.kerr_coef = (
            const.im_unit
            * beam.wavenumber_0
            * media.nlin_ref_ind_water
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
        dt = domain.time_step_len
        dt_2 = domain.time_step_len / 2
        dt_6 = domain.time_step_len / 6
        self.dt_tuple = (dt, dt_2, dt_6)

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
        self.w_array = np.empty_like(self.envelope)
        self.next_w_array = np.empty_like(self.envelope)
        self.b_array = np.empty_like(self.envelope)

        # Setup tracking variables
        self.k_array = np.empty(self.domain.dist_limit + 1, dtype=int)

        # Setup operators and initial condition
        self.setup_operators()
        self.set_initial_condition()

    def setup_operators(self):
        """Setup FCN operators."""
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

        # Setup Fourier coefficient
        self.fourier_coeff = np.exp(
            -2
            * self.const.im_unit
            * self.delta_t
            * (self.domain.frq_array * self.domain.time_step_len) ** 2
        )

        # Setup CN operators
        mat_cnt = self.const.im_unit * self.delta_r
        self.left_matrix = crank_nicolson_matrix(
            self.domain.n_radi_nodes, "left", mat_cnt
        )
        self.right_matrix = crank_nicolson_matrix(
            self.domain.n_radi_nodes, "right", -mat_cnt
        )
        self.operators = (splu(self.left_matrix), self.right_matrix)

    def set_initial_condition(self):
        """Set initial Gaussian beam and free electron density."""
        self.envelope = initial_envelope(
            self.domain.radi_2d_array,
            self.domain.time_2d_array,
            self.beam,
        )
        self.density[:, 0] = initial_density(self.media)
        # Store initial values for diagnostics
        self.dist_envelope[:, 0, :] = self.envelope
        self.dist_density[:, 0, :] = self.density
        self.axis_envelope[0, :] = self.envelope[self.domain.axis_node, :]
        self.axis_density[0, :] = self.density[self.domain.axis_node, :]
        self.peak_envelope[:, 0] = self.envelope[:, self.domain.peak_node]
        self.peak_density[:, 0] = self.density[:, self.domain.peak_node]
        self.k_array[0] = 0

    def solve_step(self, step):
        "Perform one propagation step."
        solve_density(
            self.density,
            self.envelope,
            self.dt_tuple,
            self.equation,
            self.media,
        )
        solve_dispersion(self.fourier_coeff, self.envelope, self.b_array)
        calculate_nonlinear(
            self.b_array, self.density, self.w_array, self.media, self.equation
        )

        # For step = 1, initialize Adam_Bashforth second condition
        if step == 1:
            np.copyto(self.next_w_array, self.w_array)
            self.axis_envelope[1] = self.envelope[self.domain.axis_node]
            self.axis_density[1] = self.density[self.domain.axis_node]
            self.peak_envelope[:, 1] = self.envelope[:, self.domain.peak_node]
            self.peak_density[:, 1] = self.density[:, self.domain.peak_node]
        solve_envelope(
            self.operators,
            self.b_array,
            self.w_array,
            self.next_w_array,
            self.next_envelope,
        )

        # Update arrays
        self.envelope, self.next_envelope = self.next_envelope, self.envelope
        self.density, self.next_density = self.next_density, self.density
        self.next_w_array, self.w_array = self.w_array, self.next_w_array

    def save_expensive_diagnostics(self, step):
        """Save memory expensive diagnostics data for current step."""
        self.dist_envelope[:, step, :] = self.envelope
        self.dist_density[:, step, :] = self.density
        self.k_array[step] = self.k_array[step - 1] + self.domain.dist_limitin

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
            intensity = np.abs(axis_envelope_data)

            intensity_peak_node = np.argmax(intensity)
            density_peak_node = np.argmax(axis_density_data)

            self.axis_envelope[step] = axis_envelope_data
            self.axis_density[step] = axis_density_data
            self.peak_envelope[:, step] = envelope[:, intensity_peak_node]
            self.peak_density[:, step] = density[:, density_peak_node]

    def propagate(self):
        """Propagate beam through all steps."""
        steps = self.domain.n_steps

        with tqdm(total=steps, desc="Progress") as pbar:
            for m in range(1, self.domain.dist_limit + 1):
                for n in range(1, self.domain.dist_limitin + 1):
                    k = (m - 1) * self.domain.dist_limitin + n
                    self.solve_step(k)
                    self.save_cheap_diagnostics(k)
                    pbar.update(1)
                    pbar.set_postfix({"m": m, "n": n, "k": k})
                self.save_expensive_diagnostics(m)


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
