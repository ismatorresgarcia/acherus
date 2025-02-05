"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).
    - Nonlinear optical Kerr effect (for a third-order centrosymmetric medium).
    - Multiphotonic ionization by multiphoton absorption (MPA).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Split-step Fourier Crank-Nicolson (FCN) scheme.
        *- Fast Fourier Transform (FFT) scheme (for diffraction).
        *- Extended Crank-Nicolson (CN) scheme (for diffraction, Kerr and MPA).
    - Initial condition: Gaussian.
    - Boundary conditions: Neumann-Dirichlet (radial) and Periodic (temporal).

UPPE:           ∂E/∂z = i/(2k) ∇²E - ik''/2 ∂²E/∂t² + ik_0n_2|E|^2 E - iB|E|^(2K-2)E

DISCLAIMER: UPPE uses "god-like" units, where envelope intensity and its square module are the same.
            This is equivalent to setting 0.5*c*e_0*n_0 = 1 in the UPPE when using the SI system.
            The result obtained is identical since the consistency is mantained throught the code.
            This way, the number of operations is reduced, and the code is more readable.

E: envelope.
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
t: time coordinate.
k: wavenumber (in the interacting media).
k_0: wavenumber (in vacuum).
n_2: nonlinear refractive index (for a third-order centrosymmetric medium).
B: nonlinear multiphoton absorption coefficient.
∇: nabla operator (for the tranverse direction).
∇²: laplace operator (for the transverse direction).
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def gaussian_beam(r, t, amplitude, waist, wavenumber, focal, peak_time, chirp):
    """
    Set the post-lens chirped Gaussian beam.

    Parameters:
    - r (array): Radial array
    - t (array): Time array
    - amplitude (float): Amplitude of the Gaussian beam
    - waist (float): Waist of the Gaussian beam
    - focal (float): Focal length of the initial lens
    - peak_time (float): Time at which the Gaussian beam reaches its peaks
    - chirp (float): Initial chirping introduced by some optical system
    """
    gaussian = amplitude * np.exp(
        -((r / waist) ** 2)
        - IMAG_UNIT * 0.5 * wavenumber * r**2 / focal
        - (1 + IMAG_UNIT * chirp) * (t / peak_time) ** 2
    )

    return gaussian


def crank_nicolson_diagonals(nodes, off_coeff, main_coeff, coor_system):
    """
    Generate the three diagonals for a Crank-Nicolson array with centered differences.

    Parameters:
    - nodes (int): Number of radial nodes
    - off_coeff (float): Coefficient for the off-diagonal elements
    - main_coeff (float): Coefficient for the main diagonal elements
    - coor_system (int): Parameter for planar (0) or cylindrical (1) geometry

    Returns:
    - tuple: Containing the upper, main, and lower diagonals
    """
    indices = np.arange(1, nodes - 1)

    lower_diag = off_coeff * (1 - 0.5 * coor_system / indices)
    main_diag = np.full(nodes, main_coeff)
    upper_diag = off_coeff * (1 + 0.5 * coor_system / indices)
    lower_diag = np.append(lower_diag, [0])
    upper_diag = np.insert(upper_diag, 0, [0])

    return lower_diag, main_diag, upper_diag


def crank_nicolson_array(nodes, off_coeff, main_coeff, coor_system):
    """
    Generate a Crank-Nicolson sparse array in CSR format using the diagonals.

    Parameters:
    - nodes (int): Number of radial nodes
    - off_coeff (float): Coefficient for the off-diagonal elements
    - main_coeff (float): Coefficient for the main diagonal elements
    - coor_system (int): Parameter for planar (0) or cylindrical (1) geometry

    Returns:
    - array: Containing the Crank-Nicolson sparse array in CSR format
    """
    lower_diag, main_diag, upper_diag = crank_nicolson_diagonals(
        nodes, off_coeff, main_coeff, coor_system
    )

    diagonals = [lower_diag, main_diag, upper_diag]
    offset = [-1, 0, 1]
    array = diags_array(diagonals, offsets=offset, format="csr")

    return array


## Set physical and mathematical constants
IMAG_UNIT = 1j
PI_NUMBER = np.pi
ELEC_PERMITTIVITY_0 = 8.8541878128e-12
LIGHT_SPEED_0 = 299792458.0

## Set physical variables (for water at 800 nm)
BEAM_WLEN_0 = 800e-9
LINEAR_REFF = 1.328
NON_LINEAR_REFF = 1.6e-20
# GVD_COEFF = 0
GVD_COEFF = 241e-28  # 2nd order GVD coefficient [s2 / m]
PHOTON_NUMBER = 5  # Number of photons absorbed [-]
BETA_K = 8e-64  # MPA coefficient [m(2K-3) / W-(K-1)]
BEAM_WNUMBER_0 = 2 * PI_NUMBER / BEAM_WLEN_0
BEAM_WNUMBER = BEAM_WNUMBER_0 * LINEAR_REFF
# KERR_COEFF = 0
KERR_COEFF = IMAG_UNIT * BEAM_WNUMBER_0 * NON_LINEAR_REFF  # [m/W]
# MPA_COEFF = 0
MPA_COEFF = -0.5 * BETA_K  # [m(2K-3) / W-(K-1)]
MPA_EXPONENT = 2 * PHOTON_NUMBER - 2

## Set parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
INI_RADI_COOR, FIN_RADI_COOR, I_RADI_NODES = 0.0, 25e-4, 100
N_RADI_NODES = I_RADI_NODES + 2
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)  # On-axis node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0.0, 2e-2, 100
DIST_STEP_LEN = FIN_DIST_COOR / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES = -200e-15, 200e-15, 410
TIME_STEP_LEN = (FIN_TIME_COOR - INI_TIME_COOR) / (N_TIME_NODES - 1)
PEAK_NODE = N_TIME_NODES // 2  # Peak intensity node
# Angular frequency (ω) grid
FRQ_STEP_LEN = 2 * PI_NUMBER / (N_TIME_NODES * TIME_STEP_LEN)
INI_FRQ_COOR_W1 = 0.0
FIN_FRQ_COOR_W1 = PI_NUMBER / TIME_STEP_LEN - FRQ_STEP_LEN
INI_FRQ_COOR_W2 = -PI_NUMBER / TIME_STEP_LEN
FIN_FRQ_COOR_W2 = -FRQ_STEP_LEN
w1 = np.linspace(INI_FRQ_COOR_W1, FIN_FRQ_COOR_W1, N_TIME_NODES // 2)
w2 = np.linspace(INI_FRQ_COOR_W2, FIN_FRQ_COOR_W2, N_TIME_NODES // 2)
radi_array = np.linspace(INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES)
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_STEPS + 1)
time_array = np.linspace(INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES)
frq_array = np.append(w1, w2)
radi_2d_array, dist_2d_array = np.meshgrid(radi_array, dist_array, indexing="ij")
radi_2d_array_2, time_2d_array_2 = np.meshgrid(radi_array, time_array, indexing="ij")
dist_2d_array_3, time_2d_array_3 = np.meshgrid(dist_array, time_array, indexing="ij")

## Set loop variables
EU_CYL = 1  # Parameter for planar (0) or cylindrical (1) geometry
DELTA_R = 0.25 * DIST_STEP_LEN / (BEAM_WNUMBER * RADI_STEP_LEN**2)
DELTA_T = -0.25 * DIST_STEP_LEN * GVD_COEFF / TIME_STEP_LEN**2
envelope = np.empty_like(radi_2d_array_2, dtype=complex)
envelope_axis = np.empty_like(dist_2d_array_3, dtype=complex)
envelope_fourier = np.empty_like(time_array, dtype=complex)
envelope_store = np.empty_like(envelope)
fourier_coeff = np.exp(-2 * IMAG_UNIT * DELTA_T * (frq_array * TIME_STEP_LEN) ** 2)
b_array = np.empty_like(envelope)
c_array = np.empty([N_RADI_NODES, N_TIME_NODES, 3], dtype=complex)
d_array = np.empty_like(radi_array)
f_array = np.empty_like(radi_array)
w_array = np.empty([N_RADI_NODES, N_TIME_NODES, 2], dtype=complex)

## Set tridiagonal Crank-Nicolson matrices in csr_array format
MATRIX_CNT_1 = IMAG_UNIT * DELTA_R
left_cn_matrix = crank_nicolson_array(
    N_RADI_NODES, -MATRIX_CNT_1, 1 + 2 * MATRIX_CNT_1, EU_CYL
)
right_cn_matrix = crank_nicolson_array(
    N_RADI_NODES, MATRIX_CNT_1, 1 - 2 * MATRIX_CNT_1, EU_CYL
)

# Convert to lil_array (dia_array class does not support slicing) class to manipulate BCs easier
left_cn_matrix = left_cn_matrix.tolil()
right_cn_matrix = right_cn_matrix.tolil()

# Set boundary conditions
if EU_CYL == 0:  # (Dirichlet type)
    left_cn_matrix[0, 0], right_cn_matrix[0, 0] = 1, 0
    left_cn_matrix[0, 1], right_cn_matrix[0, 1] = 0, 0
    left_cn_matrix[-1, -1], right_cn_matrix[-1, -1] = 1, 0
else:  # (Neumann-Dirichlet type)
    right_cn_matrix[0, 0] = 1 - 2 * MATRIX_CNT_1
    left_cn_matrix[0, 0] = 1 + 2 * MATRIX_CNT_1
    right_cn_matrix[0, 1] = 2 * MATRIX_CNT_1
    left_cn_matrix[0, 1] = -2 * MATRIX_CNT_1
    right_cn_matrix[-1, -1] = 0
    left_cn_matrix[-1, -1] = 1

# Convert to csr_array class (better for conversion from lil_array class) to perform operations
left_cn_matrix = left_cn_matrix.tocsr()
right_cn_matrix = right_cn_matrix.tocsr()

## Set electric field wave packet
BEAM_WAIST_0 = 100e-6
BEAM_PEAK_TIME = 50e-15
BEAM_ENERGY = 2.2e-6
BEAM_CHIRP = -1
FOCAL_LEN = 20
BEAM_CR_POWER = 3.77 * BEAM_WLEN_0**2 / (8 * PI_NUMBER * NON_LINEAR_REFF * LINEAR_REFF)
BEAM_POWER = BEAM_ENERGY / (BEAM_PEAK_TIME * np.sqrt(0.5 * PI_NUMBER))
BEAM_INTENSITY = 2 * BEAM_POWER / (PI_NUMBER * BEAM_WAIST_0**2)
BEAM_AMPLITUDE = np.sqrt(BEAM_INTENSITY)
# Wave packet's initial condition
envelope = gaussian_beam(
    radi_2d_array_2,
    time_2d_array_2,
    BEAM_AMPLITUDE,
    BEAM_WAIST_0,
    BEAM_WNUMBER,
    FOCAL_LEN,
    BEAM_PEAK_TIME,
    BEAM_CHIRP,
)
# Save on-axis envelope initial state
envelope_axis[0, :] = envelope[AXIS_NODE, :]

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS - 1)):
    # Compute first half-step (Spectral domain)
    for i in range(N_RADI_NODES):
        envelope_fourier = fourier_coeff * fft(envelope[i, :])
        # Compute first half-step solution
        b_array[i, :] = ifft(envelope_fourier)

    # Compute second half-step (Time domain)
    for l in range(N_TIME_NODES):
        c_array[:, l, 0] = b_array[:, l]
        c_array[:, l, 1] = np.abs(c_array[:, l, 0]) ** 2
        c_array[:, l, 2] = np.abs(c_array[:, l, 0]) ** MPA_EXPONENT
        if k == 0:  # I'm guessing a value for starting the AB2 method
            w_array[:, l, 0] = (
                DIST_STEP_LEN
                * (KERR_COEFF * c_array[:, l, 1] + MPA_COEFF * c_array[:, l, 2])
                * c_array[:, l, 0]
            )
            G = 1.0
            c_array[:, l, 0] = G * c_array[:, l, 0]
            c_array[:, l, 1] = np.abs(c_array[:, l, 0]) ** 2
            c_array[:, l, 2] = np.abs(c_array[:, l, 0]) ** MPA_EXPONENT
            w_array[:, l, 1] = (
                DIST_STEP_LEN
                * (KERR_COEFF * c_array[:, l, 1] + MPA_COEFF * c_array[:, l, 2])
                * c_array[:, l, 0]
            )
            envelope_axis[k + 1, l] = c_array[
                AXIS_NODE, l, 0
            ]  # Save on-axis envelope 1-step
        else:
            w_array[:, l, 1] = (
                DIST_STEP_LEN
                * (KERR_COEFF * c_array[:, l, 1] + MPA_COEFF * c_array[:, l, 2])
                * c_array[:, l, 0]
            )

        # Compute intermediate arrays
        d_array = right_cn_matrix @ c_array[:, l, 0]
        f_array = d_array + 0.5 * (3 * w_array[:, l, 1] - w_array[:, l, 0])

        # Compute second half-step solution
        envelope_store[:, l] = spsolve(left_cn_matrix, f_array)

    # Update arrays for the next step
    w_array[:, :, 0] = w_array[:, :, 1]
    envelope = envelope_store
    envelope_axis[k + 2, :] = envelope_store[
        AXIS_NODE, :
    ]  # Save on-axis envelope k-step

np.savez(
    "/Users/ytoga/projects/phd_thesis/phd_coding/python/practitioners_guide/storage/ffdmk_fcn_1",
    INI_RADI_COOR=INI_RADI_COOR,
    FIN_RADI_COOR=FIN_RADI_COOR,
    INI_DIST_COOR=INI_DIST_COOR,
    FIN_DIST_COOR=FIN_DIST_COOR,
    INI_TIME_COOR=INI_TIME_COOR,
    FIN_TIME_COOR=FIN_TIME_COOR,
    AXIS_NODE=AXIS_NODE,
    PEAK_NODE=PEAK_NODE,
    LINEAR_REFF=LINEAR_REFF,
    e=envelope,
    e_axis=envelope_axis,
)
