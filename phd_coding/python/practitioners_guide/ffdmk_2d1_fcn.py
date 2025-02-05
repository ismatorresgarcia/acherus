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
MPA_COEFF = -0.5 * BETA_K ** (PHOTON_NUMBER - 1)  # [m(2K-3) / W-(K-1)]
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

# Set lower, main, and upper diagonals
MATRIX_CNT_1 = IMAG_UNIT * DELTA_R
MATRIX_CNT_2 = 1 - 2 * MATRIX_CNT_1
MATRIX_CNT_3 = 1 + 2 * MATRIX_CNT_1
left_m1_diag = np.empty_like(radi_array, dtype=complex)
right_m1_diag = np.empty_like(left_m1_diag)
left_main_diag = np.empty_like(left_m1_diag)
right_main_diag = np.empty_like(left_m1_diag)
left_p1_diag = np.empty_like(left_m1_diag)
right_p1_diag = np.empty_like(left_m1_diag)
for i in range(1, N_RADI_NODES - 1):
    right_m1_diag[i - 1] = MATRIX_CNT_1 * (1 - 0.5 * EU_CYL / i)
    left_m1_diag[i - 1] = -right_m1_diag[i - 1]
    right_main_diag[i] = MATRIX_CNT_2
    left_main_diag[i] = MATRIX_CNT_3
    right_p1_diag[i + 1] = MATRIX_CNT_1 * (1 + 0.5 * EU_CYL / i)
    left_p1_diag[i + 1] = -right_p1_diag[i + 1]

# Store diagonals in a list of arrays
left_diagonals = [left_m1_diag, left_main_diag, left_p1_diag]
right_diagonals = [right_m1_diag, right_main_diag, right_p1_diag]
offsets = [-1, 0, 1]

# Store tridiagonal matrices in sparse form
left_cn_matrix = sp.dia_array(
    (left_diagonals, offsets), shape=(N_RADI_NODES, N_RADI_NODES)
)
right_cn_matrix = sp.dia_array(
    (right_diagonals, offsets), shape=(N_RADI_NODES, N_RADI_NODES)
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
    right_cn_matrix[0, 0] = MATRIX_CNT_2
    left_cn_matrix[0, 0] = MATRIX_CNT_3
    right_cn_matrix[0, 1] = 2 * MATRIX_CNT_1
    left_cn_matrix[0, 1] = -right_cn_matrix[0, 1]
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
envelope = BEAM_AMPLITUDE * np.exp(
    -((radi_2d_array_2 / BEAM_WAIST_0) ** 2)
    - IMAG_UNIT * 0.5 * BEAM_WNUMBER * radi_2d_array_2**2 / FOCAL_LEN
    - (1 + IMAG_UNIT * BEAM_CHIRP) * (time_2d_array_2 / BEAM_PEAK_TIME) ** 2
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

### Plots
plt.style.use("dark_background")
cmap_option = mpl.colormaps["plasma"]
figsize_option = (13, 7)

# Set up conversion factors
# INTENSITY_FACTOR = 0.5 * LIGHT_SPEED_0 * ELEC_PERMITTIVITY_0 * LINEAR_REFF
INTENSITY_FACTOR = 1
RADI_FACTOR = 1.0e6
DIST_FACTOR = 100.0
TIME_FACTOR = 1.0e15
AREA_FACTOR = 1.0e-4
# Set up plotting grid (µm, cm and fs)
new_radi_2d_array_2 = RADI_FACTOR * radi_2d_array_2
new_dist_2d_array_3 = DIST_FACTOR * dist_2d_array_3
new_time_2d_array_2 = TIME_FACTOR * time_2d_array_2
new_time_2d_array_3 = TIME_FACTOR * time_2d_array_3
new_dist_array = new_dist_2d_array_3[:, 0]
new_time_array = new_time_2d_array_3[0, :]

# Set up intensities (W/cm^2)
plot_intensity_axis = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope_axis) ** 2
plot_intensity_end = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope) ** 2

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# First subplot
ax1.plot(
    new_time_array,
    plot_intensity_axis[0, :],
    color="#32CD32",  # Lime green
    linestyle="--",
    label=r"On-axis numerical solution at beginning $z$ step",
)
ax1.plot(
    new_time_array,
    plot_intensity_axis[-1, :],
    color="#1E90FF",  # Electric Blue
    linestyle="--",
    label=r"On-axis numerical solution at final $z$ step",
)
ax1.set(xlabel=r"$t$ ($\mathrm{fs}$)", ylabel=r"$I(t)$ ($\mathrm{W/{cm}^2}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Second subplot
ax2.plot(
    new_dist_array,
    plot_intensity_axis[:, PEAK_NODE],
    color="#FFFF00",  # Pure yellow
    linestyle="-",
    label="On-axis peak time numerical solution",
)
ax2.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
ax2.legend(facecolor="black", edgecolor="white")

# fig1.tight_layout()
plt.show()

## Set up figure 2
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize_option)
# First subplot
fig2_1 = ax3.pcolormesh(
    new_dist_2d_array_3, new_time_2d_array_3, plot_intensity_axis, cmap=cmap_option
)
fig2.colorbar(fig2_1, ax=ax3)
ax3.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
ax3.set_title("On-axis solution in 2D")
# Second subplot
fig2_2 = ax4.pcolormesh(
    new_radi_2d_array_2, new_time_2d_array_2, plot_intensity_end, cmap=cmap_option
)
fig2.colorbar(fig2_2, ax=ax4)
ax4.set(xlabel=r"$r$ ($\mathrm{\mu m}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
ax4.set_title(r"Final step solution in 2D")

# fig2.tight_layout()
plt.show()

## Set up figure 3
fig3, (ax5, ax6) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# First subplot
ax5.plot_surface(
    new_dist_2d_array_3,
    new_time_2d_array_3,
    plot_intensity_axis,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax5.set(
    xlabel=r"$z$ ($\mathrm{cm}$)",
    ylabel=r"$t$ ($\mathrm{fs}$)",
    zlabel=r"$I(z,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax5.set_title("On-axis solution in 3D")

# Second subplot
ax6.plot_surface(
    new_radi_2d_array_2,
    new_time_2d_array_2,
    plot_intensity_end,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax6.set(
    xlabel=r"$r$ ($\mathrm{\mu m}$)",
    ylabel=r"$t$ ($\mathrm{fs}$)",
    zlabel=r"$I(r,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax6.set_title(r"Final step solution in 3D")

# fig3.tight_layout()
plt.show()
