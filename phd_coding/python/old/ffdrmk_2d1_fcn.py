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
    - Method (UPPE): Split-step Fourier Crank-Nicolson (FCN) scheme.
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

import numpy as np
from numpy.fft import fft, ifft
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def init_gaussian(r, t, im, beam):
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
    wn = beam["WAVENUMBER"]
    f = beam["FOCAL_LENGTH"]
    pt = beam["PEAK_TIME"]
    ch = beam["CHIRP"]
    gaussian = a * np.exp(
        -((r / w) ** 2) - 0.5 * im * wn * r**2 / f - (1 + im * ch) * (t / pt) ** 2
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


def solve_dispersion(fc, e_c, b):
    """
    Compute one step of the FFT propagation scheme.

    Parameters:
    - fc: precomputed Fourier coefficient
    - e_c: envelope at step k
    - b: pre-allocated array for envelope at step k + 1
    """
    b[:] = ifft(fc * fft(e_c, axis=1), axis=1)


def calc_nonlinear(e_c, n_c, w_c, media):
    """
    Compute one step of the Adam-Bashforth scheme for the nonlinear terms.

    Parameters:
    - e_c: envelope at step k
    - n_c: electron density at step k
    - w_c: pre-allocated array for Adam-Bashforth terms
    - media: dictionary with media parameters
    Compute one step of the Adam-Bashforth scheme for the nonlinear terms.
    """
    e_c_2 = np.abs(e_c) ** 2
    e_c_2k2 = np.abs(e_c) ** media["MPA_EXP"]
    w_c[:] = e_c * (
        media["KERR_COEF"] * e_c_2
        + media["MPA_COEF"] * e_c_2k2
        + media["REF_COEF"] * n_c
    )


def solve_propagation(mats, b, w_c, w_n, e_n):
    """
    Compute one step of the Crank-Nicolson propagation scheme.

    Parameters:
    - mats: dict containing sparse arrays for left and right operators
    - b: intermediate array from FFT step
    - w_c: current step nonlinear terms
    - w_n: previous step nonlinear terms
    - e_n: pre-allocated array for envelope at step k + 1
    """
    for m in range(e_n.shape[1]):
        c = mats["rm"] @ b[:, m]
        d = c + 1.5 * w_c[:, m] - 0.5 * w_n[:, m]
        e_n[:, m] = spsolve(mats["lm"], d)


def calc_density_rate(n_c, e_c, media):
    """
    Calculate the electron density rate equation.

    Parameters:
    - n_c: electron density at step l
    - e_c: envelope at step l
    - media: dictionary with media parameters

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


def solve_density(n_c, e_c, dt, media):
    """
    Solve the electron density equation using 4th order Runge-Kutta method.

    Parameters:
    - n_c: electron density at step l
    - e_c: envelope at step l
    - dt: time step length
    - media: dictionary with media parameters
    """
    for l in range(e_c.shape[1] - 1):
        n_c0 = n_c[:, l]
        e_c0 = e_c[:, l]
        e_c1 = e_c[:, l + 1]
        e_mid = 0.5 * (e_c0 + e_c1)

        k1 = calc_density_rate(n_c0, e_c0, media)
        k2 = calc_density_rate(n_c0 + 0.5 * dt * k1, e_mid, media)
        k3 = calc_density_rate(n_c0 + 0.5 * dt * k2, e_mid, media)
        k4 = calc_density_rate(n_c0 + dt * k3, e_c1, media)

        n_c[:, l + 1] = n_c0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


IM_UNIT = 1j
PI = np.pi

## Set parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
INI_RADI_COOR, FIN_RADI_COOR, I_RADI_NODES = 0, 25e-4, 1500
N_RADI_NODES = I_RADI_NODES + 2
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)  # On-axis node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0, 3e-2, 1000
DIST_STEP_LEN = (FIN_DIST_COOR - INI_DIST_COOR) / N_STEPS
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
ELECTRON_MASS = 9.1093837139e-31
ELECTRON_CHARGE = 1.602176634e-19
PERMITTIVITY = 8.8541878128e-12
PLANCK_CNT = 1.05457182e-34
LIN_REF_IND_WATER = 1.334
NLIN_REF_IND_WATER = 4.1e-20
GVD_COEF_WATER = 248e-28
N_PHOTONS_WATER = 5
CS_MPA_WATER = 1e-61
CS_MPI_WATER = 1.2e-72
IONIZATION_ENERGY_WATER = 6.5 * ELECTRON_CHARGE
COLLISION_TIME_WATER = 3e-15
NEUTRAL_DENS = 6.68e28

WAVELENGTH_0 = 800e-9
WAIST_0 = 75e-6
PEAK_TIME = 130e-15
ENERGY = 2.2e-6
FOCAL_LENGTH = 100
CHIRP = 0

# INT_FACTOR = 0.5 * LIGHT_SPEED * PERMITTIVITY * LIN_REF_IND_WATER
INT_FACTOR = 1
WAVENUMBER_0 = 2 * PI / WAVELENGTH_0
WAVENUMBER = 2 * PI * LIN_REF_IND_WATER / WAVELENGTH_0
ANGULAR_FRQ = WAVENUMBER_0 * LIGHT_SPEED
POWER = ENERGY / (PEAK_TIME * np.sqrt(0.5 * PI))
CR_POWER = 3.77 * WAVELENGTH_0**2 / (8 * PI * LIN_REF_IND_WATER * NLIN_REF_IND_WATER)
INTENSITY = 2 * POWER / (PI * WAIST_0**2)
AMPLITUDE = np.sqrt(INTENSITY / INT_FACTOR)

CRITICAL_DENS = PERMITTIVITY * ELECTRON_MASS * (ANGULAR_FRQ / ELECTRON_CHARGE) ** 2
CS_BREMSSTRAHLUNG_WATER = (
    WAVENUMBER
    * ANGULAR_FRQ
    * COLLISION_TIME_WATER
    / (
        (LIN_REF_IND_WATER**2 * CRITICAL_DENS)
        * (1 + (ANGULAR_FRQ * COLLISION_TIME_WATER) ** 2)
    )
)
MPI_EXP = 2 * N_PHOTONS_WATER
MPA_EXP = MPI_EXP - 2
OFI_COEF = CS_MPI_WATER * INT_FACTOR**N_PHOTONS_WATER
AVA_COEF = CS_BREMSSTRAHLUNG_WATER * INT_FACTOR / IONIZATION_ENERGY_WATER
MPA_COEF = -0.5 * CS_MPA_WATER * DIST_STEP_LEN * INT_FACTOR ** (N_PHOTONS_WATER - 1)
REF_COEF = (
    -0.5 * IM_UNIT * WAVENUMBER_0 * DIST_STEP_LEN / (LIN_REF_IND_WATER * CRITICAL_DENS)
)
KERR_COEF = IM_UNIT * WAVENUMBER_0 * NLIN_REF_IND_WATER * DIST_STEP_LEN * INT_FACTOR

## Set dictionaries for better organization
MEDIA = {
    "WATER": {
        "LIN_REF_IND": LIN_REF_IND_WATER,
        "NLIN_REF_IND": NLIN_REF_IND_WATER,
        "GVD_COEF": GVD_COEF_WATER,
        "NEUTRAL_DENS": NEUTRAL_DENS,
        "CRITICAL_DENS": CRITICAL_DENS,
        "N_PHOTONS": N_PHOTONS_WATER,  # Number of photons absorbed [-]
        "CS_MPA": CS_MPA_WATER,  # K-photon MPA coefficient [m(2K-3) - W-(K-1)]
        "CS_MPI": CS_MPI_WATER,  # K-photon MPI coefficient [s-1 - m(2K) - W-K]
        "MPA_EXP": MPA_EXP,  # MPA exponent [-]
        "MPI_EXP": MPI_EXP,  # MPI exponent [-]
        "REF_COEF": REF_COEF,  # Refraction coefficient
        "MPA_COEF": MPA_COEF,  # MPA coefficient
        "KERR_COEF": KERR_COEF,  # Kerr coefficient
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
DELTA_T = -0.25 * DIST_STEP_LEN * MEDIA["WATER"]["GVD_COEF"] / TIME_STEP_LEN**2
DENS_CNT_1 = -0.5 * TIME_STEP_LEN * MEDIA["WATER"]["OFI_COEF"]
DENS_CNT_2 = 0.5 * TIME_STEP_LEN * MEDIA["WATER"]["AVA_COEF"]
DENS_CNT_3 = (
    0.5 * TIME_STEP_LEN * MEDIA["WATER"]["NEUTRAL_DENS"] * MEDIA["WATER"]["OFI_COEF"]
)
fourier_coeff = np.exp(-2 * IM_UNIT * DELTA_T * (frq_array * TIME_STEP_LEN) ** 2)

current_envelope = np.empty([N_RADI_NODES, N_TIME_NODES], dtype=complex)
next_envelope = np.empty_like(current_envelope)
current_electron_dens = np.empty([N_RADI_NODES, N_TIME_NODES])
next_electron_dens = np.empty_like(current_electron_dens)

dist_envelope = np.empty([N_RADI_NODES, DIST_LIMIT + 1, N_TIME_NODES], dtype=complex)
axis_envelope = np.empty([N_STEPS + 1, N_TIME_NODES], dtype=complex)
peak_envelope = np.empty([N_RADI_NODES, N_STEPS + 1], dtype=complex)
dist_electron_dens = np.empty([N_RADI_NODES, DIST_LIMIT + 1, N_TIME_NODES])
axis_electron_dens = np.empty([N_STEPS + 1, N_TIME_NODES])
peak_electron_dens = np.empty([N_RADI_NODES, N_STEPS + 1])
current_w_array = np.empty_like(current_envelope)
next_w_array = np.empty_like(current_envelope)

b_array = np.empty_like(current_envelope)
k_indices = np.empty(DIST_LIMIT + 1, dtype=int)

## Set tridiagonal Crank-Nicolson matrices in csr_array format
MATRIX_CNT_1 = IM_UNIT * DELTA_R
left_operator = crank_nicolson_array(N_RADI_NODES, "LEFT", MATRIX_CNT_1)
right_operator = crank_nicolson_array(N_RADI_NODES, "RIGHT", -MATRIX_CNT_1)

## Set initial electric field wave packet and electron density
current_envelope = init_gaussian(radi_2d_array, time_2d_array, IM_UNIT, BEAM)
current_electron_dens[:, 0] = 0
axis_envelope[0, :] = current_envelope[AXIS_NODE, :]
peak_envelope[:, 0] = current_envelope[:, PEAK_NODE]
axis_electron_dens[0, :] = current_electron_dens[AXIS_NODE, :]
peak_electron_dens[:, 0] = current_electron_dens[:, PEAK_NODE]

## Set dictionaries for better organization
operators = {"lm": left_operator, "rm": right_operator}

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    solve_density(
        current_electron_dens, current_envelope, TIME_STEP_LEN, MEDIA["WATER"]
    )

    solve_dispersion(fourier_coeff, current_envelope, b_array)
    calc_nonlinear(b_array, current_electron_dens, current_w_array, MEDIA["WATER"])

    # For k = 0, initialize Adam_Bashforth second condition
    if k == 0:
        next_w_array = current_w_array.copy()
        axis_envelope[1, :] = current_envelope[AXIS_NODE, :]
        axis_electron_dens[1, :] = current_electron_dens[AXIS_NODE, :]

    solve_propagation(operators, b_array, current_w_array, next_w_array, next_envelope)

    # Update arrays for the next step
    current_envelope, next_envelope = next_envelope, current_envelope
    current_electron_dens, next_electron_dens = (
        next_electron_dens,
        current_electron_dens,
    )
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
    "/Users/ytoga/projects/phd_thesis/phd_coding/python/storage/ffdrmk_fcn_1",
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
    LIN_REF_IND=MEDIA["WATER"]["LIN_REF_IND"],
)
