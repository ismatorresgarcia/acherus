"""
Solves the unidirectional Envelope Propagation Equation (EPE) of an ultra-intense and
ultra-short laser pulse using Finite Differences and the split-step Fourier's spectral
method.

UEPE:           ∂ℰ/∂z = 𝑖/(2k) ∂²ℰ/∂x² - 𝑖k₀⁽²⁾/2 ∂²ℰ/∂t²
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from numpy.fft import fft, ifft
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

## Initialize physical and mathematical constants
IMAG_UNIT = 1j
PI_NUM = 3.141592653589793
# BOLTZMANN_CN = 1.380649e-23
ELEC_PERMITTIVITY_0 = 8.8541878128e-12
ELEC_MASS = 9.1093837139e-31
ELEC_CHARGE = 1.602176634e-19
# PLANK_CN = 6.62607015e-34
# PLANK_R = PLANK_CN / (2 * PI_NUM)
LIGHT_SPEED_0 = 299792458.0

## Initialize physical magnitudes (for water at 800 nm)
BEAM_WLENGTH_0 = 800e-9
LINEAR_REFF = 1.334
NON_LINEAR_REFF = 4.1e-20
# GVD_COEFF = 0
GVD_COEFF = 2.41e-26  # 2nd order GVD coefficient [s2 / m]
#
PHOTON_NUMBER = 5  # Number of photons absorbed [-]
BETA_K = 1e-61  # MPA coefficient [m(2K-3) / W-(K-1)]
SIGMA_K = 1.2e-72  # MPI coefficient [s-1 - m2K / W-K]
#
BEAM_WNUMBER_0 = 2 * PI_NUM / BEAM_WLENGTH_0
# CONV_FACTOR = 1
CONV_FACTOR = 0.5 * LIGHT_SPEED_0 * ELEC_PERMITTIVITY_0 * LINEAR_REFF
# KERR_COEFF = 0
KERR_COEFF = IMAG_UNIT * BEAM_WNUMBER_0 * NON_LINEAR_REFF * CONV_FACTOR  # [m/W]
# MPA_COEFF = 0
MPA_COEFF = -0.5 * BETA_K * CONV_FACTOR ** (PHOTON_NUMBER - 1)  # [m(2K-3)) / W-(K-1)]
MPA_EXPONENT = 2 * PHOTON_NUMBER - 2
# MPI_COEFF = 0
MPI_COEFF = SIGMA_K * CONV_FACTOR**PHOTON_NUMBER  # [s-1 - m2K / W-K]
MPI_EXPONENT = 2 * PHOTON_NUMBER

## Initialize parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
STARTING_RADIAL_COOR, ENDING_RADIAL_COOR = 0.0, 75e-4
N_RADIAL_NODES = 1000
RADIAL_STEP_LENGTH = (ENDING_RADIAL_COOR - STARTING_RADIAL_COOR) / (N_RADIAL_NODES + 1)
AXIS_NODE = int(-STARTING_RADIAL_COOR / RADIAL_STEP_LENGTH)  # On-axis node
# Propagation (z) grid
STARTING_PROPAGATION_COOR, ENDING_PROPAGATION_COOR = 0.0, 2e-2
N_PROPAGATION_STEPS = 1000
PROPAGATION_STEP_LENGTH = ENDING_PROPAGATION_COOR / N_PROPAGATION_STEPS
# Time (t) grid
STARTING_TIME_COOR, ENDING_TIME_COOR = -300e-15, 300e-15
N_TIME_NODES = 4096
TIME_STEP_LENGTH = (ENDING_TIME_COOR - STARTING_TIME_COOR) / (N_TIME_NODES - 1)
PEAK_TIME_NODE = N_TIME_NODES // 2  # Peak intensity node
# Angular frequency grid
SPECTRAL_STEP_LENGTH_W = 2 * PI_NUM / (N_TIME_NODES * TIME_STEP_LENGTH)
STARTING_SPECTRAL_COOR_W1, ENDING_SPECTRAL_COOR_W1 = (
    0.0,
    PI_NUM / TIME_STEP_LENGTH - SPECTRAL_STEP_LENGTH_W,
)
STARTING_SPECTRAL_COOR_W2, ENDING_SPECTRAL_COOR_W2 = (
    -PI_NUM / TIME_STEP_LENGTH,
    -SPECTRAL_STEP_LENGTH_W,
)
r = np.linspace(STARTING_RADIAL_COOR, ENDING_RADIAL_COOR, N_RADIAL_NODES + 2)
z = np.linspace(
    STARTING_PROPAGATION_COOR, ENDING_PROPAGATION_COOR, N_PROPAGATION_STEPS + 1
)
t = np.linspace(STARTING_TIME_COOR, ENDING_TIME_COOR, N_TIME_NODES)
w1 = np.linspace(STARTING_SPECTRAL_COOR_W1, ENDING_SPECTRAL_COOR_W1, N_TIME_NODES // 2)
w2 = np.linspace(STARTING_SPECTRAL_COOR_W2, ENDING_SPECTRAL_COOR_W2, N_TIME_NODES // 2)
w = np.append(w1, w2)
t1, z1 = np.meshgrid(z, t)
t2, r2 = np.meshgrid(r, t)

## Initialize electric field wave packet
BEAM_WAIST_0 = 75e-6
BEAM_PEAK_TIME = 130e-15
BEAM_ENERGY = 2.2e-6
BEAM_CHIRP = -10
FOCAL_LENGTH = 20
BEAM_WNUMBER = BEAM_WNUMBER_0 * LINEAR_REFF
BEAM_FRECUENCY_0 = BEAM_WNUMBER_0 / LIGHT_SPEED_0
BEAM_CR_POWER = 3.77 * BEAM_WLENGTH_0**2 / (8 * PI_NUM * NON_LINEAR_REFF * LINEAR_REFF)
BEAM_POWER = BEAM_ENERGY / (BEAM_PEAK_TIME * np.sqrt(0.5 * PI_NUM))
BEAM_INTENSITY = 2 * BEAM_POWER / (PI_NUM * BEAM_WAIST_0**2)
BEAM_AMPLITUDE = np.sqrt(BEAM_INTENSITY / CONV_FACTOR)
e = np.empty([N_RADIAL_NODES + 2, N_TIME_NODES], dtype=complex)
# Wave packet's initial condition
for n in range(N_TIME_NODES):
    for i in range(N_RADIAL_NODES + 2):
        e[i, n] = BEAM_AMPLITUDE * np.exp(
            -((r[i] / BEAM_WAIST_0) ** 2)
            - IMAG_UNIT * 0.5 * BEAM_WNUMBER * r[i] ** 2 / FOCAL_LENGTH
            - (1 + IMAG_UNIT * BEAM_CHIRP) * (t[n] / BEAM_PEAK_TIME) ** 2
        )

## Initialize electron density
RHO_NEUTRAL = 6.68e-28  # [m-3]
RHO_CRITICAL = (
    ELEC_PERMITTIVITY_0 * ELEC_MASS * BEAM_FRECUENCY_0**2 / ELEC_CHARGE**2
)  # [m-3]
RHO_CONSTANT_2 = 0.5 * TIME_STEP_LENGTH * MPI_COEFF
RHO_CONSTANT_3 = RHO_NEUTRAL * RHO_CONSTANT_2
REFF_COEFF = -0.5 * IMAG_UNIT * BEAM_WNUMBER_0 / (LINEAR_REFF * RHO_CRITICAL)
# Electron density initial condition
rho = np.zeros([N_RADIAL_NODES + 2, N_PROPAGATION_STEPS], dtype=complex)

## Initialize loop storage arrays
b_arr = np.empty([N_RADIAL_NODES + 2, N_TIME_NODES], dtype=complex)
c_arr = np.empty([N_RADIAL_NODES + 2, N_TIME_NODES, 4], dtype=complex)
d_arr = np.empty([N_RADIAL_NODES + 2], dtype=complex)
f_arr = np.empty([N_RADIAL_NODES + 2], dtype=complex)
w_arr = np.empty([N_RADIAL_NODES + 2, N_TIME_NODES, 2], dtype=complex)
e_store = np.empty([N_RADIAL_NODES + 2, N_TIME_NODES], dtype=complex)
e_axis = np.empty([N_PROPAGATION_STEPS + 1, N_TIME_NODES], dtype=complex)
e_axis[0, :] = e[AXIS_NODE, :]  # Save on-axis envelope initial state
# rho_axis = np.empty([N_PROPAGATION_STEPS + 1, N_TIME_NODES], dtype=complex)
# rho_axis[0, :] = rho[AXIS_NODE, :]  # Save on-axis electron density initial state

## Initialize Crank-Nicolson arrays (for ADI procedure)
EU_CYL = 1  # Parameter for planar (0) or cylindrical (1) geometry
DELTA_R = 0.25 * PROPAGATION_STEP_LENGTH / (BEAM_WNUMBER * RADIAL_STEP_LENGTH**2)
DELTA_T = -0.25 * PROPAGATION_STEP_LENGTH * GVD_COEFF / TIME_STEP_LENGTH**2
e_fourier = np.empty([N_TIME_NODES], dtype=complex)
fourier_coeff = np.exp(-2 * IMAG_UNIT * DELTA_T * (w * TIME_STEP_LENGTH) ** 2)
# Set lower, main, and upper diagonals
MAIN_DIAG_ENTRY_P = 1 + 2 * IMAG_UNIT * DELTA_R
MAIN_DIAG_ENTRY_M = 1 - 2 * IMAG_UNIT * DELTA_R
OUTER_DIAG_ENTRY = IMAG_UNIT * DELTA_R
lower_diag = np.empty([N_RADIAL_NODES + 2], dtype=complex)
main_diag_p = np.empty([N_RADIAL_NODES + 2], dtype=complex)
main_diag_m = np.empty([N_RADIAL_NODES + 2], dtype=complex)
upper_diag = np.empty([N_RADIAL_NODES + 2], dtype=complex)
for i in range(1, N_RADIAL_NODES + 1):
    lower_diag[i - 1] = OUTER_DIAG_ENTRY * (1 - 0.5 * EU_CYL / i)
    main_diag_p[i] = MAIN_DIAG_ENTRY_P
    main_diag_m[i] = MAIN_DIAG_ENTRY_M
    upper_diag[i + 1] = OUTER_DIAG_ENTRY * (1 + 0.5 * EU_CYL / i)
data_m = [-lower_diag, main_diag_p, -upper_diag]
data_p = [lower_diag, main_diag_m, upper_diag]
offsets = [-1, 0, 1]

## Store tridiagonal matrices in sparse form
lm = sp.dia_array((data_m, offsets), shape=(N_RADIAL_NODES + 2, N_RADIAL_NODES + 2))
lp = sp.dia_array((data_p, offsets), shape=(N_RADIAL_NODES + 2, N_RADIAL_NODES + 2))
# Convert to lil_array (dia_array class does not support slicing) class to manipulate BCs easier
lm, lp = lm.tolil(), lp.tolil()

## Modify first and last rows for BCs
# (Neumann-Dirichlet BCs)
if EU_CYL == 0:  # (Dirichlet BCs)
    lm[0, 0], lp[0, 0] = 1, 0
    lm[0, 1], lp[0, 1] = 0, 0
    lm[-1, -1], lp[-1, -1] = 1, 0
else:  # (Neumann-Dirichlet BCs)
    lm[0, 0], lp[0, 0] = MAIN_DIAG_ENTRY_P, MAIN_DIAG_ENTRY_M
    lm[0, 1], lp[0, 1] = -2 * OUTER_DIAG_ENTRY, 2 * OUTER_DIAG_ENTRY
    lm[-1, -1], lp[-1, -1] = 1, 0
# Convert to csr_array class (better for conversion from lil_array class) to perform operations
lm, lp = lm.tocsr(), lp.tocsr()

## Propagation loop over desired number of steps
for k in tqdm(range(N_PROPAGATION_STEPS - 1)):
    # Compute electron density (Time domain)
    for n in range(N_TIME_NODES - 1):
        for i in range(N_RADIAL_NODES + 2):
            RHO_CONSTANT_1 = np.exp(
                -RHO_CONSTANT_2
                * (
                    np.abs(e[i, n + 1]) ** MPI_EXPONENT
                    + np.abs(e[i, n]) ** MPI_EXPONENT
                )
            )
            rho[i, n + 1] = (
                RHO_CONSTANT_1
                * (rho[i, n] + RHO_CONSTANT_3 * np.abs(e[i, n]) ** MPI_EXPONENT)
                + RHO_CONSTANT_3 * np.abs(e[i, n + 1]) ** MPI_EXPONENT
            )

    # Compute first half-step (Spectral domain)
    for i in range(N_RADIAL_NODES + 2):
        e_fourier = fourier_coeff * fft(e[i, :])
        # Compute first half-step solution
        b_arr[i, :] = ifft(e_fourier)

    # Compute second half-step (Time domain)
    for n in range(N_TIME_NODES):
        c_arr[:, n, 0] = b_arr[:, n]
        c_arr[:, n, 1] = rho[:, n]
        c_arr[:, n, 2] = np.abs(c_arr[:, n, 0]) ** 2
        c_arr[:, n, 3] = np.abs(c_arr[:, n, 0]) ** MPA_EXPONENT
        if k == 0:  # I'm guessing a value for starting the AB2 method
            w_arr[:, n, 0] = (
                PROPAGATION_STEP_LENGTH
                * (
                    REFF_COEFF * c_arr[:, n, 1]
                    + KERR_COEFF * c_arr[:, n, 2]
                    + MPA_COEFF * c_arr[:, n, 3]
                )
                * c_arr[:, n, 0]
            )
            G = 1.0
            c_arr[:, n, 0] = G * c_arr[:, n, 0]
            c_arr[:, n, 1] = G * rho[:, n]
            c_arr[:, n, 2] = np.abs(c_arr[:, n, 0]) ** 2
            c_arr[:, n, 3] = np.abs(c_arr[:, n, 0]) ** MPA_EXPONENT
            w_arr[:, n, 1] = (
                PROPAGATION_STEP_LENGTH
                * (
                    REFF_COEFF * c_arr[:, n, 1]
                    + KERR_COEFF * c_arr[:, n, 2]
                    + MPA_COEFF * c_arr[:, n, 3]
                )
                * c_arr[:, n, 0]
            )
            e_axis[k + 1, n] = c_arr[AXIS_NODE, n, 0]  # Save on-axis envelope 1-step
            # rho_axis[k + 2, n] = c_arr[
            # AXIS_NODE, n, 1
            # ]  # Save on-axis electron density 2-step
        else:
            w_arr[:, n, 1] = (
                PROPAGATION_STEP_LENGTH
                * (
                    REFF_COEFF * c_arr[:, n, 1]
                    + KERR_COEFF * c_arr[:, n, 2]
                    + MPA_COEFF * c_arr[:, n, 3]
                )
                * c_arr[:, n, 0]
            )

        # Compute intermediate arrays
        d_arr = lp @ c_arr[:, n, 0]
        f_arr = d_arr + 0.5 * (3 * w_arr[:, n, 1] - w_arr[:, n, 0])

        # Compute second half-step solution
        e_store[:, n] = spsolve(lm, f_arr)

    # Update arrays for the next step
    w_arr[:, :, 0] = w_arr[:, :, 1]
    e = e_store
    e_axis[k + 2, :] = e_store[AXIS_NODE, :]  # Save on-axis envelope k-step
    # rho_axis[k + 2, :] = rho[AXIS_NODE, :]  # Save on-axis electron density k-step

## Plots
plt.style.use("dark_background")
cmap = mpl.colormaps["magma"]

p_final = CONV_FACTOR * np.abs(e) ** 2
p_axis = CONV_FACTOR * np.abs(e_axis) ** 2
p_axis_sta = p_axis[0, :]
p_axis_end = p_axis[-1, :]
p_axis_tmax = p_axis[:, PEAK_TIME_NODE]

# Set up figure
f1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4.8))
# First subplot
ax1.plot(t, p_axis_sta, "b--", label=r"On-axis solution at beginning $z$ step")
ax1.plot(t, p_axis_end, "r--", label=r"On-axis solution at final $z$ step")
ax1.set(xlabel=r"$t$ ($\mathrm{s}$)", ylabel=r"$I(t)$ ($\mathrm{W/m^2}$)")
ax1.legend()

# Second subplot
ax2.plot(z, p_axis_tmax, "r", label="On-axis peak time solution")
ax2.set(xlabel=r"$z$ ($\mathrm{m}$)", ylabel=r"$I(z)$ ($\mathrm{W/m^2}$)")
ax2.legend()

f1.tight_layout()
plt.show()

# Set up figure
f2 = plt.figure(figsize=(12, 4.8))
## First subplot
ax3 = f2.add_subplot(1, 2, 1)
fs1 = ax3.pcolormesh(t1, z1, p_axis, cmap=cmap)
f2.colorbar(fs1, ax=ax3)
ax3.set(xlabel=r"$t$ ($\mathrm{s}$)", ylabel=r"$z$ ($\mathrm{m}$)")
ax3.set_title("On-axis solution in 2D")
## Second subplot
ax4 = f2.add_subplot(1, 2, 2, projection="3d")
ax4.plot_surface(t1, z1, p_axis, cmap=cmap, linewidth=0, antialiased=False)
ax4.set(
    xlabel=r"$t$ ($\mathrm{m}$)",
    ylabel=r"$z$ ($\mathrm{s}$)",
    zlabel=r"$I(z,t)$ ($\mathrm{W/m^2}$)",
)
ax4.set_title("On-axis solution in 3D")

f2.tight_layout()
plt.show()

# Set up figure
f3 = plt.figure(figsize=(12, 4.8))
## First subplot
ax5 = f3.add_subplot(1, 2, 1)
fs2 = ax5.pcolormesh(t2, r2, p_final, cmap=cmap)
f3.colorbar(fs2, ax=ax5)
ax5.set(xlabel=r"$t$ ($\mathrm{s}$)", ylabel=r"$r$ ($\mathrm{m}$)")
ax5.set_title(r"Final step solution in 2D")
## Second subplot
ax6 = f3.add_subplot(1, 2, 2, projection="3d")
ax6.plot_surface(t2, r2, p_final, cmap=cmap, linewidth=0, antialiased=False)
ax6.set(
    xlabel=r"$t$ ($\mathrm{s}$)",
    ylabel=r"$r$ ($\mathrm{m}$)",
    zlabel=r"$I(r,t)$ ($\mathrm{W/m^2}$)",
)
ax6.set_title(r"Final step solution in 3D")

f3.tight_layout()
plt.show()
