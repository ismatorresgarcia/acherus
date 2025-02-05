"""
Solves the unidirectional Envelope Propagation Equation (EPE) of an ultra-intense and
ultra-short laser pulse using Finite Differences and Alternate Direction Implicit method.

UEPE:           ∂ℰ/∂z = 𝑖/(2k) ∂²ℰ/∂x² - 𝑖k₀⁽²⁾/2 ∂²ℰ/∂t²


ℰ:     Envelope (3d complex vector)
𝑖:     Imaginary unit
k₀⁽²⁾: GVD coefficient of 2nd order 
x:     X-coordinate
z:     Z-coordinate
t:     Time-coordinate
k:     Wavenumber (in the interacting media)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

## Set physical and mathematical constants
IMAG_UNIT = 1j
PI_NUMBER = np.pi
ELEC_PERMITTIVITY_0 = 8.8541878128e-12
LIGHT_SPEED_0 = 299792458.0

## Set physical variables (for water at 800 nm)
BEAM_WLEN_0 = 800e-9
LINEAR_REFF = 1.334
NON_LINEAR_REFF = 4.1e-20
# GVD_COEFF = 0
GVD_COEFF = 241e-28  # 2nd order GVD coefficient [s2 / m]
PHOTON_NUMBER = 5  # Number of photons absorbed [-]
BETA_K = 1e-61  # MPA coefficient [m(2K-3) / W-(K-1)]
BEAM_WNUMBER_0 = 2 * PI_NUMBER / BEAM_WLEN_0
BEAM_WNUMBER = BEAM_WNUMBER_0 * LINEAR_REFF
# KERR_COEFF = 0
KERR_COEFF = IMAG_UNIT * BEAM_WNUMBER_0 * NON_LINEAR_REFF  # [m/W]
# MPA_COEFF = 0
MPA_COEFF = -0.5 * BETA_K ** (PHOTON_NUMBER - 1)  # [m(2K-3) / W-(K-1)]
MPA_EXPONENT = 2 * PHOTON_NUMBER - 2

## Set parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
INI_RADI_COOR, FIN_RADI_COOR, I_RADI_NODES = 0.0, 75e-4, 200
N_RADI_NODES = I_RADI_NODES + 2
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)  # On-axis node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0.0, 6e-2, 300
DIST_STEP_LEN = FIN_DIST_COOR / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, I_TIME_NODES = -300e-15, 300e-15, 1024
N_TIME_NODES = I_TIME_NODES + 2
TIME_STEP_LEN = (FIN_TIME_COOR - INI_TIME_COOR) / (N_TIME_NODES - 1)
PEAK_NODE = N_TIME_NODES // 2  # Peak intensity node
radi_array = np.linspace(INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES)
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_STEPS + 1)
time_array = np.linspace(INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES)
radi_2d_array, dist_2d_array = np.meshgrid(radi_array, dist_array, indexing="ij")
radi_2d_array_2, time_2d_array_2 = np.meshgrid(radi_array, time_array, indexing="ij")
dist_2d_array_3, time_2d_array_3 = np.meshgrid(dist_array, time_array, indexing="ij")

## Set loop variables
EU_CYL = 1  # Parameter for planar (0) or cylindrical (1) geometry
DELTA_R = 0.25 * DIST_STEP_LEN / (BEAM_WNUMBER * RADI_STEP_LEN**2)
DELTA_T = -0.25 * DIST_STEP_LEN * GVD_COEFF / TIME_STEP_LEN**2
envelope = np.empty_like(radi_2d_array_2, dtype=complex)
envelope_axis = np.empty_like(dist_2d_array_3, dtype=complex)
envelope_store = np.empty_like(envelope)
b_array = np.empty_like(envelope)
c_array = np.empty_like(envelope)
d_array = np.empty([N_RADI_NODES, N_TIME_NODES, 3], dtype=complex)
w_array = np.empty([N_RADI_NODES, N_TIME_NODES, 2], dtype=complex)

# Set lower, main, and upper diagonals
MATRIX_CNT_1R = IMAG_UNIT * DELTA_R
MATRIX_CNT_2R = 1 - 2 * MATRIX_CNT_1R
MATRIX_CNT_3R = 1 + 2 * MATRIX_CNT_1R
MATRIX_CNT_1T = IMAG_UNIT * DELTA_T
MATRIX_CNT_2T = 1 - 2 * MATRIX_CNT_1T
MATRIX_CNT_3T = 1 + 2 * MATRIX_CNT_1T
left_m1_diag_r = np.empty_like(radi_array, dtype=complex)
right_m1_diag_r = np.empty_like(left_m1_diag_r)
left_main_diag_r = np.empty_like(left_m1_diag_r)
right_main_diag_r = np.empty_like(left_m1_diag_r)
left_p1_diag_r = np.empty_like(left_m1_diag_r)
right_p1_diag_r = np.empty_like(left_m1_diag_r)
left_m1_diag_t = np.empty_like(time_array, dtype=complex)
right_m1_diag_t = np.empty_like(left_m1_diag_t)
left_main_diag_t = np.empty_like(left_m1_diag_t)
right_main_diag_t = np.empty_like(left_m1_diag_t)
left_p1_diag_t = np.empty_like(left_m1_diag_t)
right_p1_diag_t = np.empty_like(left_m1_diag_t)
for i in range(1, N_RADI_NODES - 1):
    right_m1_diag_r[i - 1] = MATRIX_CNT_1R * (1 - 0.5 * EU_CYL / i)
    left_m1_diag_r[i - 1] = -right_m1_diag_r[i - 1]
    right_main_diag_r[i] = MATRIX_CNT_2R
    left_main_diag_r[i] = MATRIX_CNT_3R
    right_p1_diag_r[i + 1] = MATRIX_CNT_1R * (1 + 0.5 * EU_CYL / i)
    left_p1_diag_r[i + 1] = -right_p1_diag_r[i + 1]
for l in range(1, N_TIME_NODES - 1):
    right_m1_diag_t[l - 1] = MATRIX_CNT_1T
    left_m1_diag_t[l - 1] = -right_m1_diag_t[l - 1]
    right_main_diag_t[l] = MATRIX_CNT_2T
    left_main_diag_t[l] = MATRIX_CNT_3T
    right_p1_diag_t[l + 1] = MATRIX_CNT_1T
    left_p1_diag_t[l + 1] = -right_p1_diag_t[l + 1]

# Store diagonals in a list of arrays
left_diagonals_r = [left_m1_diag_r, left_main_diag_r, left_p1_diag_r]
right_diagonals_r = [right_m1_diag_r, right_main_diag_r, right_p1_diag_r]
left_diagonals_t = [left_m1_diag_t, left_main_diag_t, left_p1_diag_t]
right_diagonals_t = [right_m1_diag_t, right_main_diag_t, right_p1_diag_t]
offsets = [-1, 0, 1]

# Store tridiagonal matrices in sparse form
left_cn_matrix_r = sp.dia_array(
    (left_diagonals_r, offsets), shape=(N_RADI_NODES, N_RADI_NODES)
)
right_cn_matrix_r = sp.dia_array(
    (right_diagonals_r, offsets), shape=(N_RADI_NODES, N_RADI_NODES)
)
left_cn_matrix_t = sp.dia_array(
    (left_diagonals_t, offsets), shape=(N_TIME_NODES, N_TIME_NODES)
)
right_cn_matrix_t = sp.dia_array(
    (right_diagonals_t, offsets), shape=(N_TIME_NODES, N_TIME_NODES)
)

# Convert to lil_array (dia_array does not support slicing) class to manipulate BCs easier
left_cn_matrix_r = left_cn_matrix_r.tolil()
right_cn_matrix_r = right_cn_matrix_r.tolil()
left_cn_matrix_t = left_cn_matrix_t.tolil()
right_cn_matrix_t = right_cn_matrix_t.tolil()

# Set boundary conditions
# Radial array
if EU_CYL == 0:  # (Dirichlet type)
    left_cn_matrix_r[0, 0], right_cn_matrix_r[0, 0] = 1, 0
    left_cn_matrix_r[0, 1], right_cn_matrix_r[0, 1] = 0, 0
    left_cn_matrix_r[-1, -1], right_cn_matrix_r[-1, -1] = 1, 0
else:  # (Neumann-Dirichlet type)
    right_cn_matrix_r[0, 0] = MATRIX_CNT_2R
    left_cn_matrix_r[0, 0] = MATRIX_CNT_3R
    right_cn_matrix_r[0, 1] = 2 * MATRIX_CNT_1R
    left_cn_matrix_r[0, 1] = -2 * MATRIX_CNT_1R
    right_cn_matrix_r[-1, -1] = 0
    left_cn_matrix_r[-1, -1] = 1
# Temporal array
left_cn_matrix_t[0, 0], right_cn_matrix_t[0, 0] = 1, 0
left_cn_matrix_t[0, 1], right_cn_matrix_t[0, 1] = 0, 0
left_cn_matrix_t[-1, -1], right_cn_matrix_t[-1, -1] = 1, 0

## Convert to csr_array class (better for conversion from lil_array class) to perform operations
left_cn_matrix_r = left_cn_matrix_r.tocsr()
right_cn_matrix_r = right_cn_matrix_r.tocsr()
left_cn_matrix_t = left_cn_matrix_t.tocsr()
right_cn_matrix_t = right_cn_matrix_t.tocsr()

## Set electric field wave packet
BEAM_WAIST_0 = 75e-6
BEAM_PEAK_TIME = 130e-15
BEAM_ENERGY = 2.2e-6
BEAM_CHIRP = -10
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
    ## Compute first half-step (ADI transverse direction)
    # Compute right-hand side matrix product row by row
    for i in range(N_RADI_NODES):
        b_array[i, :] = right_cn_matrix_t @ envelope[i, :]

    # Compute first half-step solution
    for l in range(N_TIME_NODES):
        c_array[:, l] = spsolve(left_cn_matrix_r, b_array[:, l])

    ## Compute second half-step (ADI time direction)
    # Compute right-hand side matrix column by column
    for l in range(N_TIME_NODES):
        b_array[:, l] = right_cn_matrix_r @ c_array[:, l]

    # Compute second half-step solution
    for i in range(N_RADI_NODES):
        c_array[i, :] = spsolve(left_cn_matrix_t, b_array[i, :])

    ## Compute second step (Nonlinear terms)
    for l in range(N_TIME_NODES):
        d_array[:, l, 0] = c_array[:, l]
        d_array[:, l, 1] = np.abs(d_array[:, l, 0]) ** 2
        d_array[:, l, 2] = np.abs(d_array[:, l, 0]) ** MPA_EXPONENT
        if k == 0:  # I'm guessing a value for starting the AB2 method
            w_array[:, l, 0] = (
                DIST_STEP_LEN
                * (KERR_COEFF * d_array[:, l, 1] + MPA_COEFF * d_array[:, l, 2])
                * d_array[:, l, 0]
            )
            G = 1.0
            d_array[:, l, 0] = G * d_array[:, l, 0]
            d_array[:, l, 1] = np.abs(d_array[:, l, 0]) ** 2
            d_array[:, l, 2] = np.abs(d_array[:, l, 0]) ** MPA_EXPONENT
            w_array[:, l, 1] = (
                DIST_STEP_LEN
                * (KERR_COEFF * d_array[:, l, 1] + MPA_COEFF * d_array[:, l, 2])
                * d_array[:, l, 0]
            )
            envelope_axis[k + 1, l] = d_array[
                AXIS_NODE, l, 0
            ]  # Save on-axis envelope 1-step
        else:
            w_array[:, l, 1] = (
                DIST_STEP_LEN
                * (KERR_COEFF * d_array[:, l, 1] + MPA_COEFF * d_array[:, l, 2])
                * d_array[:, l, 0]
            )

        # Compute second step solution
        envelope_store[:, l] = d_array[:, l, 0] + 0.5 * (
            3 * w_array[:, l, 1] - w_array[:, l, 0]
        )

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
