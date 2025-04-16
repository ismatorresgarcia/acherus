"""
This program solves the Nonlinear Envelope Equation (NEE) of an ultra-intense
and ultra-short laser pulse in cylindrical coordinates with radial symmetry.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).
    - Absorption and defocusing due to the electron plasma.
    - Multiphotonic ionization by multiphoton absorption (MPA).
    - Nonlinear optical Kerr effect (for a third-order centrosymmetric medium).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Split-step Fourier Crank-Nicolson (FSS) scheme.
        *- Fast Fourier Transform (FFT) scheme (for GVD).
        *- Extended Crank-Nicolson (CN-RK4) scheme (for diffraction, Kerr and MPA).
    - Method (DE): 4th order Runge-Kutta (RK4) scheme.
    - Initial condition:
        *- Gaussian envelope at initial z coordinate.
        *- Constant electron density at initial t coordinate.
    - Boundary conditions: Neumann-Dirichlet (radial) and Periodic (temporal) for the envelope.

DE:          ∂N/∂t = S_K|E|^(2K)(N_n - N) + S_w N|E|^2 / U_i
UPPE:          ∂E/∂z = i/(2k) ∇²E - ik''/2 ∂²E/∂t² - i(k_0/2n_0)(N/N_c)E - iB_K|E|^(2K-2)E
                     + ik_0n_2(1-a)|E|^2 E + ik_0n_2a (∫R(t-t')|E(t')|^2 dt') E

DISCLAIMER: NEE uses "natural" units, where envelope intensity and its square module are the same.
            This is equivalent to setting 0.5*c*e_0*n_0 = 1 in the NEE when using the SI system.
            The result obtained is identical since the consistency is mantained throught the code.
            This way, the number of operations is reduced, and the code is more readable.

E: envelope.
N: electron density (in the interacting media).
i: imaginary unit.
r: radial coordinate.
z: longitudinal coordinate.
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

__version__ = "0.2.0"

import argparse
import sys

import h5py
import numba as nb
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.integrate import trapezoid
from scipy.sparse import diags_array
from scipy.sparse.linalg import splu
from scipy.special import gamma
from tqdm import tqdm

DEFAULT_SAVE_PATH = "./python/storage"


def initialize_envelope(r_g, t_g, i_u, e_0, w_n, w_0, t_p, c_0, f_l, g_n):
    """
    Set up the initial envelope at z = 0.

    Parameters:
    - r_g: radial coordinates array
    - t_g: time coordinates array
    - i_u: square root of -1
    - e_0: initial amplitude of the beam
    - w_n: initial wavenumber of the beam
    - w_0: initial waist of the beam
    - t_p: initial peak time of the beam
    - c_0: chirp of the beam
    - f_l: focal length of the beam
    - g_n: super-Gaussian beam order parameter

    Returns:
    - complex 2D-array: Initial envelope
    """
    space_decaying_term = -((r_g / w_0) ** g_n)
    time_decaying_term = -(1 + i_u * c_0) * (t_g / t_p) ** 2

    if f_l != 0:  # phase curvature due to focusing lens
        space_decaying_term = space_decaying_term + i_u * 0
        space_decaying_term -= 0.5 * i_u * w_n * r_g**2 / f_l

    return e_0 * np.exp(space_decaying_term + time_decaying_term)


def create_crank_nicolson_matrix(n_r, m_p, coef_d):
    """
    Set the three diagonals for the
    Crank-Nicolson array with centered
    differences.

    Parameters:
    - n_r: number of radial nodes
    - m_p: position of the Crank-Nicolson array (left or right)
    - c_d: coefficient for the diagonal elements

    Returns:
    - complex 2D-array: sparse 2-array for the Crank-Nicolson matrix
    """
    coef_main = 1 + 2 * coef_d
    r_ind = np.arange(1, n_r - 1)

    diag_lower = -coef_d * (1 - 0.5 / r_ind)
    diag_main = np.full(n_r, coef_main)
    diag_upper = -coef_d * (1 + 0.5 / r_ind)

    diag_lower = np.append(diag_lower, [0])
    diag_upper = np.insert(diag_upper, 0, [0])
    if m_p.upper() == "LEFT":
        diag_main[0], diag_main[-1] = coef_main, 1
        diag_upper[0] = -2 * coef_d
    else:  # "RIGHT"
        diag_main[0], diag_main[-1] = coef_main, 0
        diag_upper[0] = -2 * coef_d

    diags = [diag_lower, diag_main, diag_upper]
    diags_ind = [-1, 0, 1]

    return diags_array(diags, offsets=diags_ind, format="csc")


@nb.njit
def _set_density_operator(dens_s, env_s, n_k, dens_n, coef_ofi, coef_ava):
    """Set up the electron density evolution terms.

    Parameters:
    - dens_s: density at current time slice
    - env_s: envelope at current time slice
    - n_k: number of photons for MPI
    - dens_n: neutral density of the medium
    - coef_ofi: OFI coefficient
    - coef_ava: avalanche/cascade coefficient

    Returns:
    - float 1D-array: Electron density operator
    """
    env_s_2 = np.abs(env_s) ** 2
    env_s_2k = env_s_2**n_k

    term_ofi = coef_ofi * env_s_2k * (dens_n - dens_s)
    term_ava = coef_ava * dens_s * env_s_2

    return term_ofi + term_ava


@nb.njit
def _rk4_density_step(env_s, dens_s, dens_rk4, dens_args, dt, dt_2, dt_6):
    """
    Compute one time step of the RK4 integration for electron
    density evolution.

    Parameters:
    - env_s: envelope at current time slice
    - dens_s: density at current time slice
    - dens_rk4: auxiliary density array for RK4 integration
    - dens_args: arguments for the density operator
    - dt: time step
    - dt_2: half time step
    - dt_6: time step divided by 6

    Returns:
    - float 1D-array: Electron density at next time slice
    """
    k1_dens = _set_density_operator(dens_s, env_s, *dens_args)
    dens_rk4 = dens_s + dt_2 * k1_dens

    k2_dens = _set_density_operator(dens_rk4, env_s, *dens_args)
    dens_rk4 = dens_s + dt_2 * k2_dens

    k3_dens = _set_density_operator(dens_rk4, env_s, *dens_args)
    dens_rk4 = dens_s + dt * k3_dens

    k4_dens = _set_density_operator(dens_rk4, env_s, *dens_args)

    dens_s_rk4 = dens_s + dt_6 * (k1_dens + 2 * k2_dens + 2 * k3_dens + k4_dens)

    return dens_s_rk4


@nb.njit(parallel=True)
def solve_density(env, dens, dens_rk4, n_t, dens_args, dt, dt_2, dt_6):
    """
    Solve electron density evolution for all time steps.

    Parameters:
    - env: envelope at all time slices
    - dens: density at all time slices
    - dens_rk4: auxiliary density array for RK4 integration
    - n_t: number of time nodes
    - dens_args: arguments for the density operator
    """
    # Set the initial condition
    dens[:, 0] = 0

    # Solve the electron density evolution
    for ll in nb.prange(n_t - 1):
        env_s = env[:, ll]
        dens_s = dens[:, ll]

        dens_s_rk4 = _rk4_density_step(
            env_s, dens_s, dens_rk4, dens_args, dt, dt_2, dt_6
        )

        dens[:, ll + 1] = dens_s_rk4


@nb.njit
def _set_scattering_operator(ram_s, dram_s, env_s, coef_ode1, coef_ode2):
    """Set up the Raman scattering evolution terms.

    Parameters:
    - ram_s: Raman response at current time slice
    - dram_s: Raman response time derivative at current time slice
    - env_s: envelope at current time slice
    - coef_ode1: Raman frequency coefficient for the first ODE term
    - coef_ode2: Raman frequency coefficient for the second ODE term

    Returns:
    - float 1D-array: Raman scattering operators
    """
    diff_s = env_s - ram_s

    return dram_s, coef_ode1 * diff_s + coef_ode2 * dram_s


@nb.njit
def _rk4_scattering_step(
    ram_s, dram_s, env_s, ram_rk4, dram_rk4, coef_ode1, coef_ode2, dt, dt_2, dt_6
):
    """
    Compute one time step of the RK4 integration for Raman
    scattering evolution.

    Parameters:
    - ram_s: raman response at current time slice
    - dram_s: time derivative raman response at current time slice
    - env_s: envelope at current time slice
    - ram_rk4: auxiliary raman response array for RK4 integration
    - dram_rk4: auxiliary raman response time derivative array for RK4 integration
    - coef_ode1: Raman frequency coefficient for the first ODE term
    - coef_ode2: Raman frequency coefficient for the second ODE term
    - dt: time step
    - dt_2: half time step
    - dt_6: time step divided by 6

    Returns:
    - float 1D-array: Raman response at next time slice
    """
    k1_ram, k1_dram = _set_scattering_operator(
        ram_s, dram_s, env_s, coef_ode1, coef_ode2
    )
    ram_rk4 = ram_s + dt_2 * k1_ram
    dram_rk4 = dram_s + dt_2 * k1_dram

    k2_ram, k2_dram = _set_scattering_operator(
        ram_rk4, dram_rk4, env_s, coef_ode1, coef_ode2
    )
    ram_rk4 = ram_s + dt_2 * k2_ram
    dram_rk4 = dram_s + dt_2 * k2_dram

    k3_ram, k3_dram = _set_scattering_operator(
        ram_rk4, dram_rk4, env_s, coef_ode1, coef_ode2
    )
    ram_rk4 = ram_s + dt * k3_ram
    dram_rk4 = dram_s + dt * k3_dram

    k4_ram, k4_dram = _set_scattering_operator(
        ram_rk4, dram_rk4, env_s, coef_ode1, coef_ode2
    )

    ram_s_rk4 = ram_s + dt_6 * (k1_ram + 2 * k2_ram + 2 * k3_ram + k4_ram)
    dram_s_rk4 = dram_s + dt_6 * (k1_dram + 2 * k2_dram + 2 * k3_dram + k4_dram)

    return ram_s_rk4, dram_s_rk4


@nb.njit(parallel=True)
def solve_scattering(
    ram, dram, env, ram_rk4, dram_rk4, n_t, coef_ode1, coef_ode2, dt, dt_2, dt_6
):
    """
    Solve molecular Raman scattering delayed response for all time steps.

    Parameters:
    - ram: raman response at all time slices
    - dram: raman response time derivative at all time slices
    - env: envelope at all time slices
    - ram_rk4: auxiliary raman response array
    - dram_rk4: auxiliary raman response time derivative array
    - n_t: number of time nodes
    - coef_ode1: Raman frequency coefficient for the first ODE term
    - coef_ode2: Raman frequency coefficient for the second ODE term
    - dt: time step
    - dt_2: half time step
    - dt_6: time step divided by 6
    """
    # Set the initial conditions
    ram[:, 0], dram[:, 0] = 0, 0

    # Solve the raman scattering response
    for ll in nb.prange(n_t - 1):
        ram_s = ram[:, ll]
        dram_s = dram[:, ll]
        env_s = env[:, ll]

        ram_s_rk4, dram_s_rk4 = _rk4_scattering_step(
            ram_s,
            dram_s,
            env_s,
            ram_rk4,
            dram_rk4,
            coef_ode1,
            coef_ode2,
            dt,
            dt_2,
            dt_6,
        )

        ram[:, ll + 1] = ram_s_rk4
        dram[:, ll + 1] = dram_s_rk4


@nb.njit
def _set_envelope_operator(
    env_s, dens_s, ram_s, n_k, dens_n, coef_p, coef_m, coef_k, coef_r
):
    """Set up the envelope propagation nonlinear terms.

    Parameters:
    - env_s: envelope at current time slice
    - dens_s: electron density at current time slice
    - ram_s: Raman response at current time slice
    - n_k: number of photons for MPI
    - dens_n: neutral density of the medium
    - coef_p: plasma coefficient
    - coef_m: MPA coefficient
    - coef_k: Kerr coefficient
    - coef_r: Raman coefficient

    Returns:
    - complex 1D-array: Nonlinear operator
    """
    env_s_2 = np.abs(env_s) ** 2
    env_s_2k2 = env_s_2 ** (n_k - 1)
    dens_s_sat = 1 - (dens_s / dens_n)

    nlin_s = env_s * (
        coef_p * dens_s
        + coef_m * dens_s_sat * env_s_2k2
        + coef_k * env_s_2
        + coef_r * ram_s
    )

    return nlin_s


@nb.njit
def _rk4_envelope_step(env_s, dens_s, ram_s, env_rk4, env_args, dz, dz_2, dz_6):
    """
    Compute one step of the RK4 integration for envelope propagation.

    Parameters:
    - env_s: envelope at current time slice
    - dens_s: density at current time slice
    - ram_s: raman response at current time slice
    - env_rk4: auxiliary envelope array for RK4 integration
    - env_args: arguments for the envelope operator
    - dz: z step
    - dz_2: z step divided by 2
    - dz_6: z step divided by 6

    Returns:
    - complex 1D-array: RK4 integration for one time slice
    """
    k1_env = _set_envelope_operator(env_s, dens_s, ram_s, *env_args)
    env_rk4 = env_s + dz_2 * k1_env

    k2_env = _set_envelope_operator(env_rk4, dens_s, ram_s, *env_args)
    env_rk4 = env_s + dz_2 * k2_env

    k3_env = _set_envelope_operator(env_rk4, dens_s, ram_s, *env_args)
    env_rk4 = env_s + dz * k3_env

    k4_env = _set_envelope_operator(env_rk4, dens_s, ram_s, *env_args)

    nlin_s_rk4 = dz_6 * (k1_env + 2 * k2_env + 2 * k3_env + k4_env)

    return nlin_s_rk4


@nb.njit(parallel=True)
def solve_nonlinear_rk4(env, dens, ram, env_rk4, nlin, n_t, env_args, dz, dz_2, dz_6):
    """
    Solve envelope propagation nonlinearities for all
    time steps.

    Parameters:
    - env: envelope at current propagation step
    - dens: density at current propagation step
    - ram: raman response at current propagation step
    - env_rk4: auxiliary envelope array for RK4 integration
    - nlin: pre-allocated array for the nonlinear terms
    - n_t: number of time nodes
    - env_args: arguments for the envelope operator
    - dz: z step
    - dz_2: z step divided by 2
    - dz_6: z step divided by 6
    """
    for ll in nb.prange(n_t):
        env_s = env[:, ll]
        dens_s = dens[:, ll]
        ram_s = ram[:, ll]

        nlin_s_rk4 = _rk4_envelope_step(
            env_s, dens_s, ram_s, env_rk4, env_args, dz, dz_2, dz_6
        )

        nlin[:, ll] = nlin_s_rk4


def solve_dispersion(f_prop, env_c, env_n):
    """
    Solve one step of the FFT
    propagation scheme for
    dispersion.

    Parameters:
    - f_prop: Fourier propagator for advancing one step
    - env_c: envelope at current propagation step
    - env_n: envelope at next propagation step
    """
    env_n[:] = ifft(f_prop * fft(env_c, axis=1, workers=-1), axis=1, workers=-1)


def solve_envelope(m_l, m_r, n_t, env_c, nlin, env_n):
    """
    Solve one step of the generalized
    Crank-Nicolson scheme for envelope
    propagation.

    Parameters:
    - m_l: left matrix for Crank-Nicolson
    - m_r: right matrix for Crank-Nicolson
    - n_t: number of time nodes
    - env_c: envelope solution from FFT
    - nlin: propagation step nonlinear terms
    - env_n: envelope at next propagation step
    """
    for ll in range(n_t):
        rhs_linear = m_r @ env_c[:, ll]
        lhs = rhs_linear + nlin[:, ll]
        env_n[:, ll] = m_l.solve(lhs)


def calculate_fluence(env, flu=None, dt=None):
    """
    Calculate fluence distribution for the current step.

    Parameters:
    - env: envelope at current propagation step
    - flu: fluence at current propagation step
    - dt: time step
    """
    env_2 = np.abs(env) ** 2
    fluence = trapezoid(env_2, dx=dt, axis=1)

    if flu is not None:
        flu[:] = fluence

    return fluence


def calculate_radius(flu, rad=None, r_g=None):
    """
    Calculate the beam radius (HWHM of fluence
    distribution) at the current step.

    Parameters:
    - flu: fluence at current propagation step
    - rad: beam radius at current propagation step
    - r_g: radial coordinates array

    Returns:
    - float: beam radius
    """
    maximum = np.max(flu)
    half_max = 0.5 * maximum

    half_max_idx = np.argmin(np.abs(flu - half_max))

    if half_max_idx in (0, len(flu) - 1):
        return r_g[half_max_idx]

    if flu[half_max_idx] > half_max:
        i_low, i_high = half_max_idx, half_max_idx + 1
    else:
        i_low, i_high = half_max_idx - 1, half_max_idx

    r_low, r_high = r_g[i_low], r_g[i_high]
    flu_low, flu_high = flu[i_low], flu[i_high]

    hwhm = r_low + (half_max - flu_low) * (r_high - r_low) / (flu_high - flu_low)

    if rad is not None:
        rad[0] = hwhm

    return hwhm


def create_cli_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cylindrical 2D Fourier Split-step solver "
        "for ultrashort filamentation in transparent media."
    )
    parser.add_argument(
        "-m",
        "--medium",
        choices=["oxygen800", "airDSR", "water800"],
        default="oxygen800",
        help="Propagation medium (default: oxygen at 800 nm)",
    )
    parser.add_argument(
        "-p",
        "--pulse",
        choices=["gauss", "to_be_defined"],
        default="gauss",
        help="Pulse type (default: gaussian and super-Gaussian pulses)",
    )
    parser.add_argument(
        "-g",
        "--gauss_order",
        type=int,
        default=2,
        help="Gaussian order parameter (2: regular Gaussian, > 2: super-Gaussian)",
    )
    parser.add_argument(
        "--method",
        choices=["rk4"],
        default="rk4",
        help="Integration method for nonlinear term (default: rk4)",
    )
    return parser.parse_args()


class Constants:
    "Physical and mathematical constants."

    def __init__(self):
        self.light_speed_0 = 299792458.0
        self.electric_permittivity_0 = 8.8541878128e-12
        self.electron_mass = 9.1093837139e-31
        self.electron_charge = 1.602176634e-19
        self.planck_bar = 1.05457182e-34
        self.pi = np.pi
        self.imaginary_unit = 1j


class MediumParameters:
    "Medium parameters to be chosen."

    def __init__(self, medium_opt="oxygen800"):
        if medium_opt.upper() == "OXYGEN800":
            self.medium_type = "oxygen800"
        elif medium_opt.upper() == "AIRDSR":
            self.medium_type = "airDSR"
        else:  # water at 800 nm
            self.medium_type = "water800"

        # Define parameter sets
        parameters = {
            "oxygen800": {
                "refraction_index_linear": 1.0,
                "refraction_index_nonlinear": 3.2e-23,
                "constant_gvd": 0.2e-28,
                "number_photons": 8,
                "constant_mpa": 3e-121,
                "constant_mpi": 2.81e-128,
                "ionization_energy": 1.932e-18,  # 12.06 eV
                "drude_collision_time": 3.5e-13,
                "density_neutral": 0.54e25,
                "raman_rotational_frequency": 16e12,
                "raman_response_time": 70e-15,
                "raman_partition": 0.5,
                "has_raman": True,
            },
            "airDSR": {
                "refraction_index_linear": 1.0,
                "refraction_index_nonlinear": 5.57e-23,
                "constant_gvd": 2e-28,
                "number_photons": 7,
                "constant_mpa": 6.5e-104,
                "constant_mpi": 1.3e-111,
                "ionization_energy": 1.76e-18,  # 11 eV
                "drude_collision_time": 3.5e-13,
                "density_neutral": 2.7e25,
                "raman_rotational_frequency": 16e12,
                "raman_response_time": 77e-15,
                "raman_partition": 0.5,
                "has_raman": True,
            },
            "water800": {
                "refraction_index_linear": 1.334,
                "refraction_index_nonlinear": 4.1e-20,
                "constant_gvd": 248e-28,
                "number_photons": 5,
                "constant_mpa": 1e-61,
                "constant_mpi": 1.2e-72,
                "ionization_energy": 1.04e-18,  # 6.5 eV
                "drude_collision_time": 3e-15,
                "density_neutral": 6.68e28,
                "raman_rotational_frequency": 0,
                "raman_response_time": 0,
                "raman_partition": 0,
                "has_raman": False,
            },
        }

        # Apply parameters from the dictionary
        medium_params = parameters[self.medium_type]
        for key, value in medium_params.items():
            setattr(self, key, value)


class LaserPulseParameters:
    "Laser pulse physical parameters and derived properties."

    def __init__(self, const, medium, pulse_opt="gauss", gauss_opt=2):
        self.input_wavelength = 775e-9
        self.input_waist = 7e-4  # half-width at 1/e^2
        self.input_duration = 85e-15  # half-width at 1/e^2
        self.input_energy = 0.995e-3
        self.input_chirp = 0
        self.input_focal_length = 0

        self.pulse_type = pulse_opt.upper()

        if self.pulse_type == "GAUSS":
            self.input_gauss_order = gauss_opt
        else:  # to be defined in the future
            pass

        # Derived parameters
        self.input_wavenumber_0 = 2 * const.pi / self.input_wavelength
        self.input_wavenumber = self.input_wavenumber_0 * medium.refraction_index_linear
        self.input_frequency_0 = self.input_wavenumber_0 * const.light_speed_0
        self.input_power = self.input_energy / (
            self.input_duration * np.sqrt(0.5 * const.pi)
        )
        self.critical_power = (
            3.77
            * self.input_wavelength**2
            / (
                8
                * const.pi
                * medium.refraction_index_linear
                * medium.refraction_index_nonlinear
            )
        )
        self.input_intensity = (
            self.input_gauss_order
            * self.input_power
            * 2 ** (2 / self.input_gauss_order)
            / (2 * const.pi * self.input_waist**2 * gamma(2 / self.input_gauss_order))
        )
        self.input_amplitude = np.sqrt(self.input_intensity)


class Grid:
    "Spatial and temporal grid parameters."

    def __init__(self, const):
        # Radial domain
        self.r_min = 0
        self.r_max = 5e-3
        self.nodes_r_i = 10000

        # Distance domain
        self.z_min = 0
        self.z_max = 4
        self.number_steps = 4000
        self.number_snapshots = 5

        # Time domain
        self.t_min = -250e-15
        self.t_max = 250e-15
        self.nodes_t = 8192

        # Initialize derived parameters functions
        self._setup_derived_parameters()
        self._setup_arrays(const)

    @property
    def nodes_r(self):
        "Total number of radial nodes for boundary conditions."
        return self.nodes_r_i + 2

    @property
    def steps_per_snapshot(self):
        "Number of propagation steps between saved snapshots."
        return self.number_steps // self.number_snapshots

    def _setup_derived_parameters(self):
        "Setup domain parameters."
        # Calculate steps
        self.del_r = (self.r_max - self.r_min) / (self.nodes_r - 1)
        self.del_z = (self.z_max - self.z_min) / self.number_steps
        self.del_t = (self.t_max - self.t_min) / (self.nodes_t - 1)
        self.del_w = 2 * np.pi / (self.nodes_t * self.del_t)

        # Calculate nodes for r = 0 and t = 0
        self.node_r0 = int(-self.r_min / self.del_r)
        self.node_t0 = self.nodes_t // 2

    def _setup_arrays(self, const):
        "Setup grid arrays."
        # 1D
        self.r_grid = np.linspace(self.r_min, self.r_max, self.nodes_r)
        self.z_grid = np.linspace(self.z_min, self.z_max, self.number_steps + 1)
        self.t_grid = np.linspace(self.t_min, self.t_max, self.nodes_t)
        self.w_grid = 2 * const.pi * fftfreq(self.nodes_t, self.del_t)

        # 2D
        self.r_grid_2d, self.t_grid_2d = np.meshgrid(
            self.r_grid, self.t_grid, indexing="ij"
        )


class NEEParameters:
    """Pulse propagation and electron density evolution
    parameters for the final numerical scheme."""

    def __init__(self, const, medium, laser):
        # Initialize typical parameters
        self.frequency_0 = laser.input_frequency_0
        self.frequency_tau = self.frequency_0 * medium.drude_collision_time

        # Initialize main function parameters
        self._init_densities(const, medium, laser)
        self._init_coefficients(medium)
        self._init_operators(const, medium, laser)

    def _init_densities(self, const, medium, laser):
        "Initialize density parameters."
        self.density_critical = (
            const.electric_permittivity_0
            * const.electron_mass
            * (self.frequency_0 / const.electron_charge) ** 2
        )
        self.bremsstrahlung_cross_section_0 = (
            laser.input_wavenumber_0 * self.frequency_tau
        ) / (
            (medium.refraction_index_linear * self.density_critical)
            * (1 + self.frequency_tau**2)
        )

    def _init_coefficients(self, medium):
        "Initialize equation coefficients."
        self.exponent_mpi = 2 * medium.number_photons
        self.exponent_mpa = self.exponent_mpi - 2
        self.coefficient_ofi = medium.constant_mpi
        self.coefficient_ava = (
            self.bremsstrahlung_cross_section_0 / medium.ionization_energy
        )

        if medium.has_raman:
            self.raman_response_frequency = 1 / medium.raman_response_time
            self.raman_coefficient_1 = (
                self.raman_response_frequency**2 + medium.raman_rotational_frequency**2
            )
            self.raman_coefficient_2 = -2 * self.raman_response_frequency
        else:
            self.raman_coefficient_1 = 0
            self.raman_coefficient_2 = 0

    def _init_operators(self, const, medium, laser):
        "Initialize equation operators."
        # Plasma coefficient calculation
        self.coefficient_plasma = (
            -0.5
            * self.bremsstrahlung_cross_section_0
            * (1 + const.imaginary_unit * self.frequency_tau)
        )

        # MPA coefficient calculation
        self.coefficient_mpa = -0.5 * medium.constant_mpa

        # Kerr coefficient calculation
        if medium.has_raman:
            self.coefficient_kerr = (
                const.imaginary_unit
                * laser.input_wavenumber_0
                * (1 - medium.raman_partition)
                * medium.refraction_index_nonlinear
            )

            # Raman coefficient calculation
            self.coefficient_raman = (
                const.imaginary_unit
                * laser.input_wavenumber_0
                * medium.raman_partition
                * medium.refraction_index_nonlinear
            )
        else:
            self.coefficient_kerr = (
                const.imaginary_unit
                * laser.input_wavenumber_0
                * medium.refraction_index_nonlinear
            )
            self.coefficient_raman = 0


class FSSSolver:
    """Fourier Split-Step solver class for beam propagation."""

    def __init__(self, const, medium, laser, grid, nee, method_opt="rk4"):
        self.const = const
        self.medium = medium
        self.laser = laser
        self.grid = grid
        self.nee = nee

        self.method = "rk4" if method_opt.upper() == "RK4" else "to_be_defined"

        # Compute Runge-Kutta constants
        self.del_z = grid.del_z
        self.del_z_2 = self.del_z * 0.5
        self.del_z_6 = self.del_z / 6
        self.del_t = grid.del_t
        self.del_t_2 = self.del_t * 0.5
        self.del_t_6 = self.del_t / 6

        # Initialize arrays and operators
        shape_r = (self.grid.nodes_r,)
        shape_rt = (self.grid.nodes_r, self.grid.nodes_t)
        shape_rzt = (
            self.grid.nodes_r,
            self.grid.number_snapshots + 1,
            self.grid.nodes_t,
        )
        shape_zt = (self.grid.number_steps + 1, self.grid.nodes_t)
        shape_rz = (self.grid.nodes_r, self.grid.number_steps + 1)
        self.envelope_rt = np.empty(shape_rt, dtype=complex)
        self.envelope_next_rt = np.empty_like(self.envelope_rt)
        self.envelope_split_rt = np.empty_like(self.envelope_rt)
        self.envelope_snapshot_rzt = np.empty(shape_rzt, dtype=complex)
        self.envelope_r0_zt = np.empty(shape_zt, dtype=complex)
        self.envelope_tp_rz = np.empty(shape_rz, dtype=complex)
        self.density_rt = np.empty(shape_rt)
        self.density_snapshot_rzt = np.empty(shape_rzt)
        self.density_r0_zt = np.empty(shape_zt)
        self.density_tp_rz = np.empty(shape_rz)
        self.fluence_r = np.empty(shape_r)
        self.fluence_rz = np.empty(shape_rz)
        self.radius = np.empty(1)
        self.radius_z = np.empty(self.grid.number_steps + 1)
        self.raman_rt = np.empty(shape_rt, dtype=complex)
        self.draman_rt = np.empty_like(self.raman_rt)
        self.nonlinear_rt = np.empty_like(self.envelope_rt)

        self.envelope_arguments = (
            self.medium.number_photons,
            self.medium.density_neutral,
            nee.coefficient_plasma,
            nee.coefficient_mpa,
            nee.coefficient_kerr,
            nee.coefficient_raman,
        )
        self.density_arguments = (
            self.medium.number_photons,
            self.medium.density_neutral,
            nee.coefficient_ofi,
            nee.coefficient_ava,
        )

        self.envelope_rk4_stage = np.empty(self.grid.nodes_r, dtype=complex)
        self.density_rk4_stage = np.empty(self.grid.nodes_r)
        self.raman_rk4_stage = np.empty(self.grid.nodes_r, dtype=complex)
        self.draman_rk4_stage = np.empty_like(self.raman_rk4_stage)

        # Setup flags
        self.use_raman = medium.has_raman

        # Setup tracking variables
        self.snapshot_z_index = np.empty(self.grid.number_snapshots + 1, dtype=int)

        # Setup operators and initial condition
        self.setup_operators()
        self.setup_initial_condition()

    def setup_operators(self):
        """Setup FSS operators."""
        coefficient_diffraction = (
            0.25 * self.grid.del_z / (self.laser.input_wavenumber * self.grid.del_r**2)
        )
        coefficient_dispersion = (
            -0.25 * self.grid.del_z * self.medium.constant_gvd / self.grid.del_t**2
        )

        # Setup Fourier propagator
        self.propagator_fft = np.exp(
            -2
            * self.const.imaginary_unit
            * coefficient_dispersion
            * (self.grid.w_grid * self.grid.del_t) ** 2
        )

        # Setup CN operators
        matrix_constant = self.const.imaginary_unit * coefficient_diffraction
        self.matrix_cn_left = create_crank_nicolson_matrix(
            self.grid.nodes_r, "left", matrix_constant
        )
        self.matrix_cn_right = create_crank_nicolson_matrix(
            self.grid.nodes_r, "right", -matrix_constant
        )
        self.matrix_cn_left = splu(self.matrix_cn_left)

    def setup_initial_condition(self):
        """Setup initial conditions."""
        # Initial conditions
        self.envelope_rt = initialize_envelope(
            self.grid.r_grid_2d,
            self.grid.t_grid_2d,
            self.const.imaginary_unit,
            self.laser.input_amplitude,
            self.laser.input_wavenumber,
            self.laser.input_waist,
            self.laser.input_duration,
            self.laser.input_chirp,
            self.laser.input_focal_length,
            self.laser.input_gauss_order,
        )
        self.density_rt[:, 0] = 0
        self.fluence_rz[:, 0] = calculate_fluence(self.envelope_rt, dt=self.grid.del_t)
        self.radius_z[0] = calculate_radius(self.fluence_rz[:, 0], r_g=self.grid.r_grid)

        # Store initial values for diagnostics
        self.envelope_snapshot_rzt[:, 0, :] = self.envelope_rt
        self.envelope_r0_zt[0, :] = self.envelope_rt[self.grid.node_r0, :]
        self.envelope_tp_rz[:, 0] = self.envelope_rt[:, self.grid.node_t0]
        self.density_snapshot_rzt[:, 0, :].fill(0)
        self.density_r0_zt[0, :] = self.density_rt[self.grid.node_r0, :]
        self.density_tp_rz[:, 0] = self.density_rt[:, self.grid.node_t0]
        self.snapshot_z_index[0] = 0

    def solve_step(self):
        "Perform one propagation step."
        solve_density(
            self.envelope_rt,
            self.density_rt,
            self.density_rk4_stage,
            self.grid.nodes_t,
            self.density_arguments,
            self.del_t,
            self.del_t_2,
            self.del_t_6,
        )
        if self.use_raman:
            solve_scattering(
                self.raman_rt,
                self.draman_rt,
                self.envelope_rt,
                self.raman_rk4_stage,
                self.draman_rk4_stage,
                self.grid.nodes_t,
                self.nee.raman_coefficient_1,
                self.nee.raman_coefficient_2,
                self.del_t,
                self.del_t_2,
                self.del_t_6,
            )
        else:
            self.raman_rt.fill(0)
        solve_dispersion(self.propagator_fft, self.envelope_rt, self.envelope_split_rt)
        if self.method.upper() == "RK4":
            solve_nonlinear_rk4(
                self.envelope_split_rt,
                self.density_rt,
                self.raman_rt,
                self.envelope_rk4_stage,
                self.nonlinear_rt,
                self.grid.nodes_t,
                self.envelope_arguments,
                self.del_z,
                self.del_z_2,
                self.del_z_6,
            )
        else:  # to be defined in the future
            pass
        solve_envelope(
            self.matrix_cn_left,
            self.matrix_cn_right,
            self.grid.nodes_t,
            self.envelope_split_rt,
            self.nonlinear_rt,
            self.envelope_next_rt,
        )
        calculate_fluence(self.envelope_next_rt, self.fluence_r, self.grid.del_t)
        calculate_radius(self.fluence_r, self.radius, self.grid.r_grid)

        # Update arrays
        self.envelope_rt, self.envelope_next_rt = (
            self.envelope_next_rt,
            self.envelope_rt,
        )

    def cheap_diagnostics(self, step):
        """Save memory cheap diagnostics data for current step."""
        node_r0 = self.grid.node_r0
        envelope_rt = self.envelope_rt
        density_rt = self.density_rt

        axis_data_envelope = envelope_rt[node_r0]
        axis_data_density = density_rt[node_r0]
        axis_data_intensity = np.abs(axis_data_envelope)

        peak_node_intensity = np.argmax(axis_data_intensity)
        peak_node_density = np.argmax(axis_data_density)

        self.envelope_r0_zt[step] = axis_data_envelope
        self.envelope_tp_rz[:, step] = envelope_rt[:, peak_node_intensity]
        self.density_r0_zt[step] = axis_data_density
        self.density_tp_rz[:, step] = density_rt[:, peak_node_density]
        self.fluence_rz[:, step] = self.fluence_r
        self.radius_z[step] = self.radius[0]

        # Check for non-finite values
        if np.any(~np.isfinite(self.envelope_rt)):
            print("WARNING: Non-finite values detected in envelope")
            sys.exit(1)

        if np.any(~np.isfinite(self.density_rt)):
            print("WARNING: Non-finite values detected in density")
            sys.exit(1)

    def expensive_diagnostics(self, step):
        """Save memory expensive diagnostics data for current step."""
        self.envelope_snapshot_rzt[:, step, :] = self.envelope_rt
        self.density_snapshot_rzt[:, step, :] = self.density_rt
        self.snapshot_z_index[step] = (
            self.snapshot_z_index[step - 1] + self.grid.steps_per_snapshot
        )

    def propagate(self):
        """Propagate beam through all steps."""
        steps = self.grid.number_steps
        steps_snap = self.grid.steps_per_snapshot
        n_snaps = self.grid.number_snapshots

        with tqdm(total=steps, desc="Progress") as pbar:
            for snap_idx in range(1, n_snaps + 1):
                for steps_snap_idx in range(1, steps_snap + 1):
                    step_idx = (snap_idx - 1) * steps_snap + steps_snap_idx
                    self.solve_step()
                    self.cheap_diagnostics(step_idx)
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "snap": snap_idx,
                            "step_per_snap": steps_snap_idx,
                            "step": step_idx,
                        }
                    )
                self.expensive_diagnostics(snap_idx)


def main():
    "Main function."
    # Initialize CLI arguments
    args = create_cli_arguments()

    # Initialize classes
    const = Constants()
    medium = MediumParameters(medium_opt=args.medium)
    grid = Grid(const)
    laser = LaserPulseParameters(
        const, medium, pulse_opt=args.pulse, gauss_opt=args.gauss_order
    )
    nee = NEEParameters(const, medium, laser)

    # Initialize and run solver class
    solver = FSSSolver(const, medium, laser, grid, nee, method_opt=args.method)
    solver.propagate()

    # Store snapshot data
    with h5py.File(f"{DEFAULT_SAVE_PATH}/snapshots.h5", "w") as f:
        f.create_dataset(
            "envelope_snapshot_rzt",
            data=solver.envelope_snapshot_rzt,
            compression="gzip",
            chunks=True,
        )
        f.create_dataset(
            "density_snapshot_rzt",
            data=solver.density_snapshot_rzt,
            compression="gzip",
            chunks=True,
        )
        f.create_dataset("snap_z_idx", data=solver.snapshot_z_index, compression="gzip")

        # Store smaller datasets together
        with h5py.File(f"{DEFAULT_SAVE_PATH}/final_diagnostic.h5", "w") as f:
            envelope_grp = f.create_group("envelope")
            envelope_grp.create_dataset(
                "axis_zt", data=solver.envelope_r0_zt, compression="gzip"
            )
            envelope_grp.create_dataset(
                "peak_rz", data=solver.envelope_tp_rz, compression="gzip"
            )

            density_grp = f.create_group("density")
            density_grp.create_dataset(
                "axis_zt", data=solver.density_r0_zt, compression="gzip"
            )
            density_grp.create_dataset(
                "peak_rz", data=solver.density_tp_rz, compression="gzip"
            )

            pulse_grp = f.create_group("pulse")
            pulse_grp.create_dataset(
                "fluence_rz", data=solver.fluence_rz, compression="gzip"
            )
            pulse_grp.create_dataset(
                "radius_z", data=solver.radius_z, compression="gzip"
            )

            coords_grp = f.create_group("coordinates")
            coords_grp.create_dataset("r_min", data=grid.r_min)
            coords_grp.create_dataset("r_max", data=grid.r_max)
            coords_grp.create_dataset("z_min", data=grid.z_min)
            coords_grp.create_dataset("z_max", data=grid.z_max)
            coords_grp.create_dataset("t_min", data=grid.t_min)
            coords_grp.create_dataset("t_max", data=grid.t_max)


if __name__ == "__main__":
    main()
