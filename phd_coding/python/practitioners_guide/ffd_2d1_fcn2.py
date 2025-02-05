"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Spectral (in frequency) Crank-Nicolson (CN) scheme.
    - Initial condition: Gaussian.
    - Boundary conditions: Neumann-Dirichlet (radial) and Periodic (temporal).

UPPE:           ∂E/∂z = i/(2k) ∇²E - ik''/2 ∂²E/∂t²


E: envelope.
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
t: time coordinate.
k: wavenumber (in the interacting media).
∇: nabla operator (for the tranverse direction).
∇²: laplace operator (for the transverse direction).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
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
LINEAR_REFF = 1.334
# GVD_COEFF = 0
GVD_COEFF = 241e-28  # 2nd order GVD coefficient [s2 / m]
BEAM_WNUMBER_0 = 2 * PI_NUMBER / BEAM_WLEN_0
BEAM_WNUMBER = BEAM_WNUMBER_0 * LINEAR_REFF
INTENSITY_FACTOR = 0.5 * LIGHT_SPEED_0 * ELEC_PERMITTIVITY_0 * LINEAR_REFF

## Set parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
INI_RADI_COOR, FIN_RADI_COOR, I_RADI_NODES = 0.0, 75e-4, 1000
N_RADI_NODES = I_RADI_NODES + 2
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)  # On-axis node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0.0, 6e-2, 500
DIST_STEP_LEN = FIN_DIST_COOR / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES = -300e-15, 300e-15, 2048
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
envelope_fourier = np.empty_like(envelope)
envelope_store = np.empty_like(envelope)
fourier_coeff = IMAG_UNIT * DELTA_T * frq_array
b_array = np.empty_like(radi_array, dtype=complex)
c_array = np.empty_like(envelope)

## Set tridiagonal Crank-Nicolson matrices in csr_array format
MATRIX_CNT_1 = IMAG_UNIT * DELTA_R
matrix_cnt_2 = 1 - 2 * MATRIX_CNT_1 + fourier_coeff
matrix_cnt_3 = 1 + 2 * MATRIX_CNT_1 - fourier_coeff
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
BEAM_WAIST_0 = 75e-5
BEAM_PEAK_TIME = 130e-15
BEAM_ENERGY = 2.2e-6
BEAM_CHIRP = -10
FOCAL_LEN = 20
BEAM_POWER = BEAM_ENERGY / (BEAM_PEAK_TIME * np.sqrt(0.5 * PI_NUMBER))
BEAM_INTENSITY = 2 * BEAM_POWER / (PI_NUMBER * BEAM_WAIST_0**2)
BEAM_AMPLITUDE = np.sqrt(BEAM_INTENSITY / INTENSITY_FACTOR)
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

## Propagation loop over desired number of steps (Spectral domain)
for k in tqdm(range(N_STEPS)):
    ## Compute Direct Fast Fourier Transform (DFFT)
    for i in range(N_RADI_NODES):
        envelope_fourier[i, :] = fft(envelope[i, :])

    ## Compute terms in the Spectral domain
    for l in range(N_TIME_NODES):
        ## Compute Crank-Nicolson array new diagonal terms
        # Update only the diagonal elements for each matrix (in-place)
        right_cn_matrix.setdiag(matrix_cnt_2[l])
        left_cn_matrix.setdiag(matrix_cnt_3[l])
        ## Compute Crank-Nicolson array operations
        b_array = right_cn_matrix @ envelope_fourier[:, l]
        c_array[:, l] = spsolve(left_cn_matrix, b_array)

    ## Compute Inverse Fast Fourier Transform (IFFT)
    for i in range(N_RADI_NODES):
        envelope_store[i, :] = ifft(c_array[i, :])

    ## Update arrays for the next step
    envelope = envelope_store
    envelope_axis[k + 1, :] = envelope_store[
        AXIS_NODE, :
    ]  # Save on-axis envelope k-step

## Analytical solution for a Gaussian beam
# Set arrays
envelope_radial_s = np.empty_like(radi_2d_array, dtype=complex)
envelope_time_s = np.empty_like(envelope_axis)
envelope_axis_s = np.empty_like(envelope_axis)
envelope_end_s = np.empty_like(envelope)

# Set variables
RAYLEIGH_LEN = 0.5 * BEAM_WNUMBER * BEAM_WAIST_0**2
DISPERSION_LEN = 0.5 * BEAM_PEAK_TIME**2 / GVD_COEFF
LENS_DIST = FOCAL_LEN / (1 + (FOCAL_LEN / RAYLEIGH_LEN) ** 2)
beam_waist = BEAM_WAIST_0 * np.sqrt(
    (1 - dist_array / FOCAL_LEN) ** 2 + (dist_array / RAYLEIGH_LEN) ** 2
)
beam_duration = BEAM_PEAK_TIME * np.sqrt(
    (1 + BEAM_CHIRP * dist_array / DISPERSION_LEN) ** 2
    + (dist_array / DISPERSION_LEN) ** 2
)
beam_radius = (
    dist_array
    - LENS_DIST
    + (LENS_DIST * (FOCAL_LEN - LENS_DIST)) / (dist_array - LENS_DIST)
)
gouy_radial_phase = np.atan(
    (dist_array - LENS_DIST) / np.sqrt(FOCAL_LEN * LENS_DIST - LENS_DIST**2)
)
gouy_time_phase = 0.5 * np.atan(
    -dist_array / (DISPERSION_LEN + BEAM_CHIRP * dist_array)
)
#
ratio_term = BEAM_WAIST_0 / beam_waist[np.newaxis, :]
sqrt_term = np.sqrt(BEAM_PEAK_TIME / beam_duration[:, np.newaxis])
decay_radial_exp_term = (radi_array[:, np.newaxis] / beam_waist) ** 2
decay_time_exp_term = (time_array / beam_duration[:, np.newaxis]) ** 2
prop_radial_exp_term = (
    0.5 * IMAG_UNIT * BEAM_WNUMBER * radi_array[:, np.newaxis] ** 2 / beam_radius
)
prop_time_exp_term = 1 + IMAG_UNIT * (
    BEAM_CHIRP + (1 + BEAM_CHIRP**2) * (dist_array[:, np.newaxis] / DISPERSION_LEN)
)
gouy_radial_exp_term = IMAG_UNIT * gouy_radial_phase[np.newaxis, :]
gouy_time_exp_term = IMAG_UNIT * gouy_time_phase[:, np.newaxis]

# Compute solution
envelope_radial_s = ratio_term * np.exp(
    -decay_radial_exp_term + prop_radial_exp_term - gouy_radial_exp_term
)
envelope_time_s = sqrt_term * np.exp(
    -decay_time_exp_term * prop_time_exp_term - gouy_time_exp_term
)
envelope_end_s = BEAM_AMPLITUDE * (
    envelope_radial_s[:, -1, np.newaxis] * envelope_time_s[-1, :]
)
envelope_axis_s = BEAM_AMPLITUDE * (
    envelope_radial_s[AXIS_NODE, :, np.newaxis] * envelope_time_s
)

### Plots
plt.style.use("dark_background")
cmap_option = mpl.colormaps["plasma"]
figsize_option = (13, 7)

# Set up conversion factors
RADI_FACTOR = 1000.0
DIST_FACTOR = 100.0
TIME_FACTOR = 1.0e15
AREA_FACTOR = 1.0e-4
# Set up plotting grid (mm, cm and s)
new_radi_2d_array_2 = RADI_FACTOR * radi_2d_array_2
new_dist_2d_array_3 = DIST_FACTOR * dist_2d_array_3
new_time_2d_array_2 = TIME_FACTOR * time_2d_array_2
new_time_2d_array_3 = TIME_FACTOR * time_2d_array_3
new_dist_array = new_dist_2d_array_3[:, 0]
new_time_array = new_time_2d_array_3[0, :]

# Set up intensities (W/cm^2)
plot_intensity_axis = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope_axis) ** 2
plot_intensity_end = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope) ** 2
plot_intensity_axis_s = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope_axis_s) ** 2
plot_intensity_end_s = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope_end_s) ** 2

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
intensity_list = [
    (
        plot_intensity_axis_s[0, :],
        "#FF00FF",  # Magenta
        "-",
        r"On-axis analytical solution at beginning $z$ step",
    ),
    (
        plot_intensity_axis_s[-1, :],
        "#FFFF00",  # Pure yellow
        "-",
        r"On-axis analytical solution at final $z$ step",
    ),
    (
        plot_intensity_axis[0, :],
        "#32CD32",  # Lime green
        "--",
        r"On-axis numerical solution at beginning $z$ step",
    ),
    (
        plot_intensity_axis[-1, :],
        "#1E90FF",  # Electric Blue
        "--",
        r"On-axis numerical solution at final $z$ step",
    ),
]
for data, color, style, label in intensity_list:
    ax1.plot(new_time_array, data, color, linestyle=style, linewidth=2, label=label)
ax1.set(xlabel=r"$t$ ($\mathrm{s}$)", ylabel=r"$I(t)$ ($\mathrm{W/{cm}^2}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax2.plot(
    new_dist_array,
    plot_intensity_axis_s[:, PEAK_NODE],
    "#FF00FF",  # Magenta
    linestyle="-",
    linewidth=2,
    label="On-axis peak time analytical solution",
)
ax2.plot(
    new_dist_array,
    plot_intensity_axis[:, PEAK_NODE],
    "#32CD32",  # Lime green
    linestyle="--",
    linewidth=2,
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
    new_dist_2d_array_3,
    new_time_2d_array_3,
    plot_intensity_axis,
    cmap=cmap_option,
)
fig2.colorbar(fig2_1, ax=ax3)
ax3.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{s}$)")
ax3.set_title("On-axis numerical solution in 2D")
# Second subplot
fig2_2 = ax4.pcolormesh(
    new_dist_2d_array_3,
    new_time_2d_array_3,
    plot_intensity_axis_s,
    cmap=cmap_option,
)
fig2.colorbar(fig2_2, ax=ax4)
ax4.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{s}$)")
ax4.set_title("On-axis analytical solution in 2D")

# fig2.tight_layout()
plt.show()

## Set up figure 3
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=figsize_option)
# First subplot
fig3_1 = ax5.pcolormesh(
    new_radi_2d_array_2,
    new_time_2d_array_2,
    plot_intensity_end,
    cmap=cmap_option,
)
fig3.colorbar(fig3_1, ax=ax5)
ax5.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$t$ ($\mathrm{s}$)")
ax5.set_title("Final step numerical solution in 2D")
# Second subplot
fig3_2 = ax6.pcolormesh(
    new_radi_2d_array_2,
    new_time_2d_array_2,
    plot_intensity_end_s,
    cmap=cmap_option,
)
fig3.colorbar(fig3_2, ax=ax6)
ax6.set(xlabel=r"$t$ ($\mathrm{mm}$)", ylabel=r"$r$ ($\mathrm{s}$)")
ax6.set_title("Final step analytical solution in 2D")

# fig3.tight_layout()
plt.show()

## Set up figure 4
fig4, (ax7, ax8) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# First subplot
ax7.plot_surface(
    new_dist_2d_array_3,
    new_time_2d_array_3,
    plot_intensity_axis,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax7.set(
    xlabel=r"$z$ ($\mathrm{cm}$)",
    ylabel=r"$t$ ($\mathrm{s}$)",
    zlabel=r"$I(z,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax7.set_title("On-axis numerical solution in 3D")
# Second subplot
ax8.plot_surface(
    new_dist_2d_array_3,
    new_time_2d_array_3,
    plot_intensity_axis_s,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax8.set(
    xlabel=r"$z$ ($\mathrm{cm}$)",
    ylabel=r"$t$ ($\mathrm{s}$)",
    zlabel=r"$I(z,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax8.set_title("On-axis analytical solution in 3D")

# fig4.tight_layout()
plt.show()

## Set up figure 5
fig5, (ax9, ax10) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# First subplot
ax9.plot_surface(
    new_radi_2d_array_2,
    new_time_2d_array_2,
    plot_intensity_end,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax9.set(
    xlabel=r"$r$ ($\mathrm{mm}$)",
    ylabel=r"$t$ ($\mathrm{s}$)",
    zlabel=r"$I(r,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax9.set_title("Final step numerical solution in 3D")
## Second subplot
ax10.plot_surface(
    new_radi_2d_array_2,
    new_time_2d_array_2,
    plot_intensity_end_s,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax10.set(
    xlabel=r"$r$ ($\mathrm{mm}$)",
    ylabel=r"$t$ ($\mathrm{s}$)",
    zlabel=r"$I(r,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax10.set_title("Final step analytical solution in 3D")

# fig5.tight_layout()
plt.show()
