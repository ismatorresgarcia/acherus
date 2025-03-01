"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse in cylindrical coordinates with radial symmetry.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).
    - Absorption and defocusing due to the electron plasma. 
    - Multiphotonic ionization by multiphoton absorption (MPA).
    - Nonlinear optical Kerr effect (for a third-order centrosymmetric medium).
        *- Instantaneous component (1-a) due to the electronic response in the polarization.
        *- Delayed component (a) due to stimulated molecular Raman scattering.

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
R: molecular response function (for the delayed component of the Kerr effect).
S_K: nonlinear optical field ionization coefficient.
B_K: nonlinear multiphoton absorption coefficient.
S_w: bremsstrahlung cross-section for avalanche (cascade) ionization.
U_i: ionization energy (for the interacting media).
∇²: laplace operator (for the transverse direction).
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def initial_condition(r, t, im, beam):
    """
    Set the post-lens chirped Gaussian beam.

    Parameters:
    - r (array): radial array
    - t (array): time array
    - im (complex): square root of -1
    - beam (dict): dictionary containing the beam parameters
        - a (float): amplitude of the Gaussian beam
        - w (float): waist of the Gaussian beam
        - wn (float): wavenumber of the Gaussian beam
        - f (float): focal length of the initial lens
        - pt (float): time at which the Gaussian beam reaches its peak intensity
        - ch (float): initial chirping introduced by some optical system
    """
    a = beam["AMPLITUDE"]
    w = beam["WAIST_0"]
    # wn = beam["WAVENUMBER"]
    # f = beam["FOCAL_LENGTH"]
    pt = beam["PEAK_TIME"]
    ch = beam["CHIRP"]
    gaussian = a * np.exp(
        -((r / w) ** 2)
        # - 0.5 * im_unit * wave_number * radius**2 / focal_length
        - (1 + im * ch) * (t / pt) ** 2
    )

    return gaussian


def crank_nicolson_diags(n, lr, c):
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
    if lr == "LEFT":
        diag_0[0], diag_0[-1] = dc, 1
        diag_p1[0] = -2 * c
    else:
        diag_0[0], diag_0[-1] = dc, 0
        diag_p1[0] = -2 * c

    return diag_m1, diag_0, diag_p1


def crank_nicolson_array(n, lr, c):
    """
    Set the Crank-Nicolson sparse array in CSR format using the diagonals.

    Parameters:
    - n (int): number of radial nodes
    - lr (str): position of the Crank-Nicolson array (left or right)
    - c (float): coefficient for the diagonal elements

    Returns:
    - array: Crank-Nicolson sparse array in CSR format
    """
    diag_m1, diag_0, diag_p1 = crank_nicolson_diags(n, lr, c)

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]
    matrix = diags_array(diags, offsets=offset, format="csr")

    return matrix


def density_rate(n_c, e_c, media):
    """
    Calculate the electron density rate equation.

    Parameters:
    - n_c: Current electron density
    - e_c: Electric field intensity
    - media (dict): Media parameters

    Returns:
    - ndarray: Rate of change of electron density
    """
    abs_e_c = np.abs(e_c) ** 2

    ofi = (
        media["OFI_COEF"]
        * (abs_e_c ** media["N_PHOTONS"])
        * (media["NEUTRAL_DENS"] - n_c)
    )
    ava = media["AVA_COEF"] * n_c * abs_e_c
    return ofi + ava


def runge_kutta_4(n_c, e_c, e_n, dt, media):
    """
    Solve the electron density equation using 4th order Runge-Kutta method.

    Parameters:
    - n_c: electron density at position l
    - e_c: envelope at step l
    - e_n: envelope at step l + 1
    - dt: time step length
    - media: dictionary with media parameters

    Returns:
    - ndarray: Updated electron density
    """
    e_mid = 0.5 * (e_c + e_n)

    k1 = density_rate(n_c, e_c, media)
    k2 = density_rate(n_c + 0.5 * dt * k1, e_mid, media)
    k3 = density_rate(n_c + 0.5 * dt * k2, e_mid, media)
    k4 = density_rate(n_c + dt * k3, e_n, media)

    return n_c + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


IM_UNIT = 1j
PI = np.pi

## Set parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
INI_RADI_COOR, FIN_RADI_COOR, I_RADI_NODES = 0, 50e-3, 2000
N_RADI_NODES = I_RADI_NODES + 2
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)  # On-axis node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0, 10, 5000
DIST_STEP_LEN = (FIN_DIST_COOR - INI_DIST_COOR) / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES = -500e-15, 500e-15, 8192
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
ELECTRON_MASS = 9.1093837139e-31
ELECTRON_CHARGE = 1.602176634e-19
PERMITTIVITY = 8.8541878128e-12
PLANCK_CNT = 1.05457182e-34
LIN_REF_IND_AIR = 1
NLIN_REF_IND_AIR = 5.57e-23
GVD_COEF_AIR = 2e-28
N_PHOTONS_AIR = 7
CS_MPA_AIR = 6.5e-104
CS_MPI_AIR = 1.87e-111
IONIZATION_ENERGY_AIR = 1.76e-18
COLLISION_TIME_AIR = 3.5e-13
NEUTRAL_DENS = 2e25
RAMAN_FRQ = 16e12
RAMAN_TIME = 77e-15
RAMAN_FRAC = 0.5

WAVELENGTH_0 = 775e-9
WAIST_0 = 7e-4
PEAK_TIME = 85e-15
ENERGY = 0.95e-3
FOCAL_LENGTH = 0
CHIRP = 0

# INT_FACTOR = 0.5 * LIGHT_SPEED * PERMITTIVITY * LIN_REF_IND_AIR
INT_FACTOR = 1
WAVENUMBER_0 = 2 * PI / WAVELENGTH_0
WAVENUMBER = 2 * PI * LIN_REF_IND_AIR / WAVELENGTH_0
ANGULAR_FRQ = WAVENUMBER_0 * LIGHT_SPEED
POWER = ENERGY / (PEAK_TIME * np.sqrt(0.5 * PI))
CR_POWER = 3.77 * WAVELENGTH_0**2 / (8 * PI * LIN_REF_IND_AIR * NLIN_REF_IND_AIR)
INTENSITY = 2 * POWER / (PI * WAIST_0**2)
AMPLITUDE = np.sqrt(INTENSITY / INT_FACTOR)

CRITICAL_DENS = PERMITTIVITY * ELECTRON_MASS * (ANGULAR_FRQ / ELECTRON_CHARGE) ** 2
CS_BREMSSTRAHLUNG_AIR = (
    WAVENUMBER
    * ANGULAR_FRQ
    * COLLISION_TIME_AIR
    / (
        (LIN_REF_IND_AIR**2 * CRITICAL_DENS)
        * (1 + (ANGULAR_FRQ * COLLISION_TIME_AIR) ** 2)
    )
)
RAMAN_CTE = (1 + (RAMAN_FRQ * RAMAN_TIME) ** 2) / (RAMAN_FRQ * RAMAN_TIME**2)
MPI_EXP = 2 * N_PHOTONS_AIR
MPA_EXP = MPI_EXP - 2
OFI_COEF = CS_MPI_AIR * INT_FACTOR**N_PHOTONS_AIR
AVA_COEF = CS_BREMSSTRAHLUNG_AIR * INT_FACTOR / IONIZATION_ENERGY_AIR
MPA_COEF = -0.5 * CS_MPA_AIR * DIST_STEP_LEN * INT_FACTOR ** (N_PHOTONS_AIR - 1)
REF_COEF = (
    -0.5 * IM_UNIT * WAVENUMBER_0 * DIST_STEP_LEN / (LIN_REF_IND_AIR * CRITICAL_DENS)
)
KERR_INS_COEF = (
    IM_UNIT
    * WAVENUMBER_0
    * NLIN_REF_IND_AIR
    * (1 - RAMAN_FRAC)
    * DIST_STEP_LEN
    * INT_FACTOR
)
KERR_DEL_COEF = (
    IM_UNIT * WAVENUMBER_0 * NLIN_REF_IND_AIR * RAMAN_FRAC * DIST_STEP_LEN * INT_FACTOR
)

## Set dictionaries for better organization
MEDIA = {
    "AIR": {
        "LIN_REF_IND": LIN_REF_IND_AIR,
        "NLIN_REF_IND": NLIN_REF_IND_AIR,
        "GVD_COEF": GVD_COEF_AIR,
        "NEUTRAL_DENS": NEUTRAL_DENS,
        "CRITICAL_DENS": CRITICAL_DENS,
        "N_PHOTONS": N_PHOTONS_AIR,  # Number of photons absorbed [-]
        "CS_MPA": CS_MPA_AIR,  # K-photon MPA coefficient [m(2K-3) - W-(K-1)]
        "CS_MPI": CS_MPI_AIR,  # K-photon MPI coefficient [s-1 - m(2K) - W-K]
        "MPA_EXP": MPA_EXP,  # MPA exponent [-]
        "MPI_EXP": MPI_EXP,  # MPI exponent [-]
        "REF_COEF": REF_COEF,  # Refraction coefficient
        "MPA_COEF": MPA_COEF,  # MPA coefficient
        "KERR_INS_COEF": KERR_INS_COEF,  # Instantaneous coefficient
        "KERR_DEL_COEF": KERR_DEL_COEF,  # Raman-Kerr delayed coefficient
        "OFI_COEF": OFI_COEF,  # OFI coefficient
        "AVA_COEF": AVA_COEF,  # Avalanche ionization coefficient
        "INT_FACTOR": INT_FACTOR,
    },
    "VACUUM": {
        "LIGHT_SPEED": LIGHT_SPEED,
        "PERMITTIVITY": PERMITTIVITY,
        "ELECTRON_MASS": ELECTRON_MASS,
        "ELECTRON_CHARGE": ELECTRON_CHARGE,
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
DIST_INDEX = 0
DIST_LIMIT = 5
DELTA_R = 0.25 * DIST_STEP_LEN / (BEAM["WAVENUMBER"] * RADI_STEP_LEN**2)
DELTA_T = -0.25 * DIST_STEP_LEN * MEDIA["AIR"]["GVD_COEF"] / TIME_STEP_LEN**2
DENS_CNT_1 = -0.5 * TIME_STEP_LEN * MEDIA["AIR"]["OFI_COEF"]
DENS_CNT_2 = 0.5 * TIME_STEP_LEN * MEDIA["AIR"]["AVA_COEF"]
DENS_CNT_3 = (
    0.5 * TIME_STEP_LEN * MEDIA["AIR"]["NEUTRAL_DENS"] * MEDIA["AIR"]["OFI_COEF"]
)
RAMAN_CNT_1 = np.exp((-(1 / RAMAN_TIME) + IM_UNIT * RAMAN_FRQ) * TIME_STEP_LEN)
RAMAN_CNT_2 = 0.5 * RAMAN_CTE * TIME_STEP_LEN
RAMAN_CNT_3 = RAMAN_CNT_1 * RAMAN_CNT_2
fourier_coeff = np.exp(-2 * IM_UNIT * DELTA_T * (frq_array * TIME_STEP_LEN) ** 2)

current_envelope = np.empty([N_RADI_NODES, N_TIME_NODES], dtype=complex)
next_envelope = np.empty_like(current_envelope)
current_electron_dens = np.empty([N_RADI_NODES, N_TIME_NODES])
next_electron_dens = np.empty_like(current_electron_dens)
current_kerr_raman = np.empty_like(current_envelope)
next_kerr_raman = np.empty_like(current_envelope)

dist_envelope = np.empty([N_RADI_NODES, DIST_LIMIT + 1, N_TIME_NODES], dtype=complex)
axis_envelope = np.empty([N_STEPS + 1, N_TIME_NODES], dtype=complex)
peak_envelope = np.empty([N_RADI_NODES, N_STEPS + 1], dtype=complex)
dist_electron_dens = np.empty([N_RADI_NODES, DIST_LIMIT + 1, N_TIME_NODES])
axis_electron_dens = np.empty([N_STEPS + 1, N_TIME_NODES])
peak_electron_dens = np.empty([N_RADI_NODES, N_STEPS + 1])

b_array = np.empty_like(current_envelope)
c_array = np.empty([N_RADI_NODES, N_TIME_NODES, 5], dtype=complex)
current_w_array = np.empty_like(current_envelope)
next_w_array = np.empty_like(current_envelope)

k_indices = np.empty(DIST_LIMIT + 1, dtype=int)

## Set tridiagonal Crank-Nicolson matrices in csr_array format
MATRIX_CNT_1 = IM_UNIT * DELTA_R
left_operator = crank_nicolson_array(N_RADI_NODES, "LEFT", MATRIX_CNT_1)
right_operator = crank_nicolson_array(N_RADI_NODES, "RIGHT", -MATRIX_CNT_1)

## Set initial electric field wave packet and electron density
current_envelope = initial_condition(radi_2d_array, time_2d_array, IM_UNIT, BEAM)
current_electron_dens[:, 0] = 0
current_kerr_raman[:, 0] = 0
axis_envelope[0, :] = current_envelope[AXIS_NODE, :]
peak_envelope[:, 0] = current_envelope[:, PEAK_NODE]
axis_electron_dens[0, :] = current_electron_dens[AXIS_NODE, :]
peak_electron_dens[:, 0] = current_electron_dens[:, PEAK_NODE]

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    # Electron density evolution update
    for l in range(N_TIME_NODES - 1):
        # Update density
        current_electron_dens[:, l + 1] = runge_kutta_4(
            current_electron_dens[:, l],
            current_envelope[:, l],
            current_envelope[:, l + 1],
            TIME_STEP_LEN,
            MEDIA["AIR"],
        )

        current_kerr_raman[:, l + 1] = (
            RAMAN_CNT_1 * current_kerr_raman[:, l]
            + RAMAN_CNT_2 * np.abs(current_envelope[:, l + 1]) ** 2
            + RAMAN_CNT_3 * np.abs(current_envelope[:, l]) ** 2
        )
    current_kerr_raman[:, 0] = current_kerr_raman[:, 2]

    # FFT step in vectorized form
    b_array = ifft(fourier_coeff * fft(current_envelope, axis=1), axis=1)

    # Nonlinear terms calculation
    c_array[:, :, 0] = b_array
    c_array[:, :, 1] = np.abs(c_array[:, :, 0]) ** 2
    c_array[:, :, 2] = np.abs(c_array[:, :, 0]) ** MEDIA["AIR"]["MPA_EXP"]
    c_array[:, :, 3] = current_electron_dens
    c_array[:, :, 4] = np.imag(current_kerr_raman)

    # Calculate Adam-Bashforth term for current step
    current_w_array = c_array[:, :, 0] * (
        MEDIA["AIR"]["KERR_INS_COEF"] * c_array[:, :, 1]
        + MEDIA["AIR"]["MPA_COEF"] * c_array[:, :, 2]
        + MEDIA["AIR"]["REF_COEF"] * c_array[:, :, 3]
        + MEDIA["AIR"]["KERR_DEL_COEF"] * c_array[:, :, 4]
    )

    # For k = 0, initialize Adam_Bashforth second condition
    if k == 0:
        next_w_array = current_w_array.copy()
        axis_envelope[1, :] = current_envelope[AXIS_NODE, :]
        axis_electron_dens[1, :] = current_electron_dens[AXIS_NODE, :]

    # Solve propagation equation for all time slices
    for l in range(N_TIME_NODES):
        d_array = right_operator @ b_array[:, l]
        f_array = d_array + 1.5 * current_w_array[:, l] - 0.5 * next_w_array[:, l]
        next_envelope[:, l] = spsolve(left_operator, f_array)

    # Update arrays for the next step
    current_envelope, next_envelope = next_envelope, current_envelope
    current_electron_dens, electron_dens_next = (
        electron_dens_next,
        current_electron_dens,
    )
    current_kerr_raman, next_kerr_raman = next_kerr_raman, current_kerr_raman
    next_w_array = current_w_array

    # Store data
    if (
        (k % (N_STEPS // DIST_LIMIT) == 0) or (k == N_STEPS - 1)
    ) and DIST_INDEX <= DIST_LIMIT:

        dist_envelope[:, DIST_INDEX, :] = current_envelope
        dist_electron_dens[:, DIST_INDEX, :] = current_electron_dens
        k_indices[DIST_INDEX] = k
        DIST_INDEX += 1

    # Store axis data
    if k > 0:
        axis_envelope[k + 1, :] = current_envelope[AXIS_NODE, :]
        peak_envelope[:, k + 1] = current_envelope[:, PEAK_NODE]
        axis_electron_dens[k + 1, :] = current_electron_dens[AXIS_NODE, :]
        peak_electron_dens[:, k + 1] = current_electron_dens[:, PEAK_NODE]

# Save to file
np.savez(
    "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/hastur_fcn_1",
    e_dist=dist_envelope,
    e_axis=axis_envelope,
    e_peak=peak_envelope,
    elec_dist=dist_electron_dens,
    elec_axis=axis_electron_dens,
    elec_peak=peak_electron_dens,
    k_indices=k_indices,
    INI_RADI_COOR=INI_RADI_COOR,
    FIN_RADI_COOR=FIN_RADI_COOR,
    INI_DIST_COOR=INI_DIST_COOR,
    FIN_DIST_COOR=FIN_DIST_COOR,
    INI_TIME_COOR=INI_TIME_COOR,
    FIN_TIME_COOR=FIN_TIME_COOR,
    AXIS_NODE=AXIS_NODE,
    PEAK_NODE=PEAK_NODE,
    LIN_REF_IND=MEDIA["AIR"]["LIN_REF_IND"],
)
