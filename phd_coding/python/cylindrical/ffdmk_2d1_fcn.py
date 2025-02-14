"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse in cylindrical coordinates with radial symmetry.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).
    - Multiphotonic ionization by multiphoton absorption (MPA).
    - Nonlinear optical Kerr effect (for a third-order centrosymmetric medium).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Split-step Fourier Crank-Nicolson (FCN) scheme.
        *- Fast Fourier Transform (FFT) scheme (for GVD).
        *- Extended Crank-Nicolson (CN) scheme (for diffraction, Kerr and MPA).
    - Initial condition: Gaussian.
    - Boundary conditions: Neumann-Dirichlet (radial) and Periodic (temporal).

UPPE:          ∂E/∂z = i/(2k) ∇²E - ik''/2 ∂²E/∂t² - iB_K|E|^(2K-2)E + ik_0n_2|E|^2 E 

DISCLAIMER: UPPE uses "god-like" units, where envelope intensity and its square module are the same.
            This is equivalent to setting 0.5*c*e_0*n_0 = 1 in the UPPE when using the SI system.
            The result obtained is identical since the consistency is mantained throught the code.
            This way, the number of operations is reduced, and the code is more readable.
            However, the dictionary "MEDIA" has an entry "INT_FACTOR" where the conversion 
            factor can be changed at will between the two unit systems.

E: envelope.
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
t: time coordinate.
k: wavenumber (in the interacting media).
k'': GVD coefficient of 2nd order.
k_0: wavenumber (in vacuum).
n_2: nonlinear refractive index (for a third-order centrosymmetric medium).
B_K: nonlinear multiphoton absorption coefficient.
∇²: laplace operator (for the transverse direction).
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def initial_condition(radius, time, im_unit, beam_parameters):
    """
    Set the post-lens chirped Gaussian beam.

    Parameters:
    - radius (array): radial array
    - time (array): time array
    - im_unit (complex): square root of -1
    - beam_parameters (dict): dictionary containing the beam parameters
        - amplitude (float): amplitude of the Gaussian beam
        - waist (float): waist of the Gaussian beam
        - wave_number (float): wavenumber of the Gaussian beam
        - focal_length (float): focal length of the initial lens
        - peak_time (float): time at which the Gaussian beam reaches its peak intensity
        - chirp (float): initial chirping introduced by some optical system
    """
    amplitude = beam_parameters["AMPLITUDE"]
    waist = beam_parameters["WAIST_0"]
    wave_number = beam_parameters["WAVENUMBER"]
    focal_length = beam_parameters["FOCAL_LENGTH"]
    peak_time = beam_parameters["PEAK_TIME"]
    chirp = beam_parameters["CHIRP"]
    gaussian_envelope = amplitude * np.exp(
        -((radius / waist) ** 2)
        - 0.5 * im_unit * wave_number * radius**2 / focal_length
        - (1 + im_unit * chirp) * (time / peak_time) ** 2
    )

    return gaussian_envelope


def crank_nicolson_diags(nodes, position, coefficient):
    """
    Set the three diagonals for the Crank-Nicolson array with centered differences.

    Parameters:
    - nodes (int): number of radial nodes
    - position (str): position of the Crank-Nicolson array (left or right)
    - coefficient (float): coefficient for the diagonal elements

    Returns:
    - tuple: upper, main, and lower diagonals
    """
    main_coefficient = 1 + 2 * coefficient
    indices = np.arange(1, nodes - 1)

    diag_m1 = -coefficient * (1 - 0.5 / indices)
    diag_0 = np.full(nodes, main_coefficient)
    diag_p1 = -coefficient * (1 + 0.5 / indices)

    diag_m1 = np.append(diag_m1, [0])
    diag_p1 = np.insert(diag_p1, 0, [0])
    if position == "LEFT":
        diag_0[0], diag_0[-1] = main_coefficient, 1
        diag_p1[0] = -2 * coefficient
    else:
        diag_0[0], diag_0[-1] = main_coefficient, 0
        diag_p1[0] = -2 * coefficient

    return diag_m1, diag_0, diag_p1


def crank_nicolson_array(nodes, position, coefficient):
    """
    Set the Crank-Nicolson sparse array in CSR format using the diagonals.

    Parameters:
    - nodes (int): number of radial nodes
    - position (str): position of the Crank-Nicolson array (left or right)
    - coefficient (float): coefficient for the diagonal elements

    Returns:
    - array: Crank-Nicolson sparse array in CSR format
    """
    diag_m1, diag_0, diag_p1 = crank_nicolson_diags(nodes, position, coefficient)

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]
    crank_nicolson_output = diags_array(diags, offsets=offset, format="csr")

    return crank_nicolson_output


def fft_step(fourier_coefficient, current_envelope, array_inter):
    """
    Compute one step of the FFT propagation scheme.

    Parameters:
    - fourier_coefficient: precomputed Fourier coefficient
    - current_envelope: envelope at step k
    - array_inter: pre-allocated array for envelope at step k + 1
    """
    array_inter[:] = ifft(fourier_coefficient * fft(current_envelope, axis=1), axis=1)


def nonlinear_terms(current_envelope, array_inter, media_params):
    """
    Set the terms for nonlinear contributions.

    Parameters:
    - current_envelope: pre-allocated array for envelope at step k
    - inter_array: pre-allocated array for intermediate results
    - media_params: dictionary with media parameters
    """
    abs_envelope = np.abs(current_envelope)
    array_inter[:, :, 0] = current_envelope
    array_inter[:, :, 1] = abs_envelope**2
    array_inter[:, :, 2] = abs_envelope ** media_params["MPA_EXP"]


def adam_bashforth_step(array_inter, current_w_array, media_params):
    """
    Compute one step of the Adam-Bashforth scheme for the nonlinear terms.

    Parameters:
    - array_inter: pre-allocated array for intermediate results
    - current_w_array: pre-allocated array for Adam-Bashforth terms
    - media_params: dictionary with media parameters
    Compute one step of the Adam-Bashforth scheme for the nonlinear terms.
    """
    current_w_array[:] = (
        media_params["KERR_COEF"] * array_inter[:, :, 1]
        + media_params["MPA_COEF"] * array_inter[:, :, 2]
    ) * array_inter[:, :, 0]


def crank_nicolson_step(
    operator, array_inter, current_w_array, next_w_array, next_envelope
):
    """
    Compute one step of the Crank-Nicolson propagation scheme.

    Parameters:
    - operator: dict containing sparse arrays for left and right operators
    - array_inter: intermediate array from FFT step
    - current_w_array: current step nonlinear terms
    - next_w_array: previous step nonlinear terms
    - next_envelope: pre-allocated array for envelope at step k + 1
    """
    for l in range(array_inter.shape[1]):
        d_array = operator["right"] @ array_inter[:, l]
        f_array = d_array + 1.5 * current_w_array[:, l] - 0.5 * next_w_array[:, l]
        next_envelope[:, l] = spsolve(operator["left"], f_array)


IM_UNIT = 1j
PI = np.pi

## Set parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
INI_RADI_COOR, FIN_RADI_COOR, I_RADI_NODES = 0, 25e-4, 1500
N_RADI_NODES = I_RADI_NODES + 2
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)  # On-axis node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0, 2e-2, 1000
DIST_STEP_LEN = FIN_DIST_COOR / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES = -200e-15, 200e-15, 4096
TIME_STEP_LEN = (FIN_TIME_COOR - INI_TIME_COOR) / (N_TIME_NODES - 1)
PEAK_NODE = N_TIME_NODES // 2  # Peak intensity node
# Angular frequency (ω) grid
FRQ_STEP_LEN = 2 * PI / (N_TIME_NODES * TIME_STEP_LEN)
INI_FRQ_COOR_W1 = 0
FIN_FRQ_COOR_W1 = PI / TIME_STEP_LEN - FRQ_STEP_LEN
INI_FRQ_COOR_W2 = -PI / TIME_STEP_LEN
FIN_FRQ_COOR_W2 = -FRQ_STEP_LEN
w1 = np.linspace(INI_FRQ_COOR_W1, FIN_FRQ_COOR_W1, N_TIME_NODES // 2)
w2 = np.linspace(INI_FRQ_COOR_W2, FIN_FRQ_COOR_W2, N_TIME_NODES // 2)
radi_array = np.linspace(INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES)
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_STEPS + 1)
time_array = np.linspace(INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES)
frq_array = np.append(w1, w2)
radi_2d_array, time_2d_array = np.meshgrid(radi_array, time_array, indexing="ij")

## Set beam and media parameters
LIGHT_SPEED = 299792458
PERMITTIVITY = 8.8541878128e-12
LIN_REF_IND_WATER = 1.328
NLIN_REF_IND_WATER = 1.6e-20
GVD_COEF_WATER = 241e-28
N_PHOTONS_WATER = 5
BETA_COEF_WATER = 8e-64

WAVELENGTH_0 = 800e-9
WAIST_0 = 100e-6
PEAK_TIME = 50e-15
ENERGY = 2.83e-6
FOCAL_LENGTH = 20
CHIRP = -1

# INT_FACTOR = 0.5 * LIGHT_SPEED * PERMITTIVITY * LIN_REF_IND_WATER
INT_FACTOR = 1
WAVENUMBER_0 = 2 * PI / WAVELENGTH_0
WAVENUMBER = 2 * PI * LIN_REF_IND_WATER / WAVELENGTH_0
POWER = ENERGY / (PEAK_TIME * np.sqrt(0.5 * PI))
CR_POWER = 3.77 * WAVELENGTH_0**2 / (8 * PI * LIN_REF_IND_WATER * NLIN_REF_IND_WATER)
INTENSITY = 2 * POWER / (PI * WAIST_0**2)
AMPLITUDE = np.sqrt(INTENSITY / INT_FACTOR)

MPA_EXP = 2 * N_PHOTONS_WATER - 2
KERR_COEF = IM_UNIT * WAVENUMBER_0 * NLIN_REF_IND_WATER * DIST_STEP_LEN * INT_FACTOR
MPA_COEF = -0.5 * BETA_COEF_WATER * DIST_STEP_LEN * INT_FACTOR ** (N_PHOTONS_WATER - 1)

## Set dictionaries for better organization
MEDIA = {
    "WATER": {
        "LIN_REF_IND": LIN_REF_IND_WATER,
        "NLIN_REF_IND": NLIN_REF_IND_WATER,
        "GVD_COEF": GVD_COEF_WATER,
        "N_PHOTONS": N_PHOTONS_WATER,  # Number of photons absorbed [-]
        "BETA_COEF": BETA_COEF_WATER,  # MPA coefficient [m(2K-3) / W-(K-1)]
        "MPA_EXP": MPA_EXP,  # MPA exponent [-]
        "KERR_COEF": KERR_COEF,  # Kerr coefficient [m^2 / W]
        "MPA_COEF": MPA_COEF,  # MPA coefficient [m^2 / W]
        "INT_FACTOR": INT_FACTOR,
    },
    "VACUUM": {
        "LIGHT_SPEED": LIGHT_SPEED,
        "PERMITTIVITY": PERMITTIVITY,
    },
}

## Set dictionaries for better organization
BEAM = {
    "WAVELENGTH_0": WAVELENGTH_0,
    "WAIST_0": WAIST_0,
    "PEAK_TIME": PEAK_TIME,
    "ENERGY": ENERGY,
    "FOCAL_LENGTH": FOCAL_LENGTH,
    "CHIRP": CHIRP,
    "WAVENUMBER_0": WAVENUMBER_0,
    "WAVENUMBER": WAVENUMBER,
    "POWER": POWER,
    "CR_POWER": CR_POWER,
    "INTENSITY": INTENSITY,
    "AMPLITUDE": AMPLITUDE,
}

## Set loop variables
DELTA_R = 0.25 * DIST_STEP_LEN / (BEAM["WAVENUMBER"] * RADI_STEP_LEN**2)
DELTA_T = -0.25 * DIST_STEP_LEN * MEDIA["WATER"]["GVD_COEF"] / TIME_STEP_LEN**2
fourier_coeff = np.exp(-2 * IM_UNIT * DELTA_T * (frq_array * TIME_STEP_LEN) ** 2)
envelope_current = np.empty([N_RADI_NODES, N_TIME_NODES], dtype=complex)
envelope_next = np.empty_like(envelope_current)
envelope_axis = np.empty([N_STEPS + 1, N_TIME_NODES], dtype=complex)
b_array = np.empty_like(envelope_current)
c_array = np.empty([N_RADI_NODES, N_TIME_NODES, 3], dtype=complex)
w_array_current = np.empty_like(envelope_current)
w_array_next = np.empty_like(envelope_current)

## Set tridiagonal Crank-Nicolson matrices in csr_array format
MATRIX_CNT_1 = IM_UNIT * DELTA_R
left_operator = crank_nicolson_array(N_RADI_NODES, "LEFT", MATRIX_CNT_1)
right_operator = crank_nicolson_array(N_RADI_NODES, "RIGHT", -MATRIX_CNT_1)

## Set initial electric field wave packet
envelope_current = initial_condition(radi_2d_array, time_2d_array, IM_UNIT, BEAM)
envelope_axis[0, :] = envelope_current[AXIS_NODE, :]

## Set dictionaries for better organization
operators = {"left": left_operator, "right": right_operator}

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS - 1)):
    fft_step(fourier_coeff, envelope_current, b_array)
    nonlinear_terms(b_array, c_array, MEDIA["WATER"])
    adam_bashforth_step(c_array, w_array_current, MEDIA["WATER"])

    # For k = 0, initialize Adam_Bashforth second condition
    if k == 0:
        w_array_next = w_array_current.copy()
        envelope_axis[1, :] = envelope_current[AXIS_NODE, :]

    crank_nicolson_step(
        operators, b_array, w_array_current, w_array_next, envelope_next
    )

    # Update arrays for the next step
    envelope_current, envelope_next = envelope_next, envelope_current
    w_array_next = w_array_current

    # Store axis data
    if k > 0:
        envelope_axis[k + 1, :] = envelope_current[AXIS_NODE, :]

np.savez(
    "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/ffdmk_fcn_1",
    INI_RADI_COOR=INI_RADI_COOR,
    FIN_RADI_COOR=FIN_RADI_COOR,
    INI_DIST_COOR=INI_DIST_COOR,
    FIN_DIST_COOR=FIN_DIST_COOR,
    INI_TIME_COOR=INI_TIME_COOR,
    FIN_TIME_COOR=FIN_TIME_COOR,
    AXIS_NODE=AXIS_NODE,
    PEAK_NODE=PEAK_NODE,
    LIN_REF_IND=MEDIA["WATER"]["LIN_REF_IND"],
    e=envelope_current,
    e_axis=envelope_axis,
)
