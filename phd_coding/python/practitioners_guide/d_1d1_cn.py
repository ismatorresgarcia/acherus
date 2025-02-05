"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse.
This program only includes second order group velocity dispersion (GVD).

Numerical discretization: Finite Differences Method (FDM)
- Method: Crank-Nicolson (CN) scheme
- Initial condition: Gaussian
- Boundary conditions: homogeneous Dirichlet

UPPE:           ∂ℰ/∂z = -ik''/2 ∂²E/∂t²


E: envelope (2d complex array)
i: imaginary unit
k'': GVD coefficient of 2nd order
z: distance coordinate
t: time coordinate
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def crank_nicolson_diagonals(nodes, off_coeff, main_coeff):
    """
    Generate the three diagonals for a Crank-Nicolson array with centered differences.

    Parameters:
    - nodes (int): Number of time nodes
    - off_coeff (float): Coefficient for the off-diagonal elements
    - main_coeff (float): Coefficient for the main diagonal elements

    Returns:
    - tuple: Containing the upper, main, and lower diagonals
    """
    lower_diag = np.full(nodes - 1, off_coeff)
    main_diag = np.full(nodes, main_coeff)
    upper_diag = np.full(nodes - 1, off_coeff)

    return lower_diag, main_diag, upper_diag


def crank_nicolson_array(nodes, off_coeff, main_coeff):
    """
    Generate a Crank-Nicolson sparse array in CSR format using the diagonals.

    Parameters:
    - nodes (int): Number of time nodes
    - off_coeff (float): Coefficient for the off-diagonal elements
    - main_coeff (float): Coefficient for the main diagonal elements

    Returns:
    - array: Containing the Crank-Nicolson sparse array in CSR format
    """
    lower_diag, main_diag, upper_diag = crank_nicolson_diagonals(
        nodes, off_coeff, main_coeff
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
GVD_COEFF = 241e-28  # 2nd order GVD coefficient [s2 / m]
BEAM_WNUMBER_0 = 2 * PI_NUMBER / BEAM_WLEN_0
BEAM_WNUMBER = BEAM_WNUMBER_0 * LINEAR_REFF
INTENSITY_FACTOR = 0.5 * LIGHT_SPEED_0 * ELEC_PERMITTIVITY_0 * LINEAR_REFF

## Set parameters (grid spacing, propagation step, etc.)
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0.0, 5e-2, 1000
DIST_STEP_LEN = (FIN_DIST_COOR - INI_DIST_COOR) / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, I_TIME_NODES = -300e-15, 300e-15, 2048
N_TIME_NODES = I_TIME_NODES + 2
TIME_STEP_LEN = (FIN_TIME_COOR - INI_TIME_COOR) / (N_TIME_NODES - 1)
PEAK_NODE = N_TIME_NODES // 2
# Angular frequency (ω) grid
FRQ_STEP_LEN = 2 * PI_NUMBER / (N_TIME_NODES * TIME_STEP_LEN)
INI_FRQ_COOR_W1 = 0.0
FIN_FRQ_COOR_W1 = PI_NUMBER / TIME_STEP_LEN - FRQ_STEP_LEN
INI_FRQ_COOR_W2 = -PI_NUMBER / TIME_STEP_LEN
FIN_FRQ_COOR_W2 = -FRQ_STEP_LEN
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_STEPS + 1)
time_array = np.linspace(INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES)
dist_2d_array, time_2d_array = np.meshgrid(dist_array, time_array, indexing="ij")

## Set loop variables
DELTA_T = -0.25 * DIST_STEP_LEN * GVD_COEFF / TIME_STEP_LEN**2
envelope = np.empty_like(dist_2d_array, dtype=complex)
b_array = np.empty_like(time_array, dtype=complex)

## Set tridiagonal Crank-Nicolson matrices in csr_array format
MATRIX_CNT_1 = IMAG_UNIT * DELTA_T
left_cn_matrix = crank_nicolson_array(N_TIME_NODES, -MATRIX_CNT_1, 1 + 2 * MATRIX_CNT_1)
right_cn_matrix = crank_nicolson_array(N_TIME_NODES, MATRIX_CNT_1, 1 - 2 * MATRIX_CNT_1)

# Convert to lil_array (dia_array class does not support slicing) class to manipulate BCs easier
left_cn_matrix = left_cn_matrix.tolil()
right_cn_matrix = right_cn_matrix.tolil()

# Set boundary conditions (Dirichlet type)
left_cn_matrix[0, 0], right_cn_matrix[0, 0] = 1, 0
left_cn_matrix[0, 1], right_cn_matrix[0, 1] = 0, 0
left_cn_matrix[-1, -1], right_cn_matrix[-1, -1] = 1, 0

# Convert to csr_array class (better for conversion from lil_array class) to perform operations
left_cn_matrix = left_cn_matrix.tocsr()
right_cn_matrix = right_cn_matrix.tocsr()

## Set electric field wave packet
BEAM_WAIST_0 = 75e-6
BEAM_PEAK_TIME = 130e-15
BEAM_ENERGY = 2.2e-6
BEAM_CHIRP = -10
FOCAL_LEN = 20
BEAM_POWER = BEAM_ENERGY / (BEAM_PEAK_TIME * np.sqrt(0.5 * PI_NUMBER))
BEAM_INTENSITY = 2 * BEAM_POWER / (PI_NUMBER * BEAM_WAIST_0**2)
BEAM_AMPLITUDE = np.sqrt(BEAM_INTENSITY / INTENSITY_FACTOR)
# Wave packet's initial condition
envelope[0, :] = BEAM_AMPLITUDE * np.exp(
    -(1 + IMAG_UNIT * BEAM_CHIRP) * (time_array / BEAM_PEAK_TIME) ** 2
)

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    # Compute solution
    b_array = right_cn_matrix @ envelope[k, :]
    envelope[k + 1, :] = spsolve(left_cn_matrix, b_array)

## Analytical solution for a Gaussian beam
# Set arrays
envelope_s = np.empty_like(envelope)

# Set variables
DISPERSION_LEN = 0.5 * BEAM_PEAK_TIME**2 / GVD_COEFF
beam_duration = BEAM_PEAK_TIME * np.sqrt(
    (1 + BEAM_CHIRP * dist_array / DISPERSION_LEN) ** 2
    + (dist_array / DISPERSION_LEN) ** 2
)
gouy_phase = 0.5 * np.atan(-dist_array / (DISPERSION_LEN + BEAM_CHIRP * dist_array))
#
sqrt_term = np.sqrt(BEAM_PEAK_TIME / beam_duration[:, np.newaxis])
decay_exp_term = (time_array / beam_duration[:, np.newaxis]) ** 2
prop_exp_term = 1 + IMAG_UNIT * (
    BEAM_CHIRP + (1 + BEAM_CHIRP**2) * (dist_array[:, np.newaxis] / DISPERSION_LEN)
)
gouy_exp_term = IMAG_UNIT * gouy_phase[:, np.newaxis]

# Compute solution
envelope_s = (
    BEAM_AMPLITUDE * sqrt_term * np.exp(-decay_exp_term * prop_exp_term - gouy_exp_term)
)

### Plots
plt.style.use("dark_background")
cmap_option = mpl.colormaps["plasma"]
figsize_option = (13, 7)

# Set up conversion factors
DIST_FACTOR = 100.0
AREA_FACTOR = 1.0e-4
# Set up plotting grid (cm, s)
new_dist_2d_array = DIST_FACTOR * dist_2d_array
new_time_2d_array = time_2d_array
new_dist_array = new_dist_2d_array[:, 0]
new_time_array = new_time_2d_array[0, :]

# Set up intensities (W/cm^2)
plot_intensity = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope) ** 2
plot_intensity_s = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope_s) ** 2

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
intensity_list = [
    (
        plot_intensity_s[0, :],
        "#FF00FF",  # Magenta
        "-",
        r"Analytical solution at beginning $z$ step",
    ),
    (
        plot_intensity_s[-1, :],
        "#FFFF00",  # Pure yellow
        "-",
        r"Analytical solution at final $z$ step",
    ),
    (
        plot_intensity[0, :],
        "#32CD32",  # Lime green
        "--",
        r"Numerical solution at beginning $z$ step",
    ),
    (
        plot_intensity[-1, :],
        "#1E90FF",  # Electric blue
        "--",
        r"Numerical solution at final $z$ step",
    ),
]
for data, color, style, label in intensity_list:
    ax1.plot(new_time_array, data, color, linestyle=style, linewidth=2, label=label)
ax1.set(xlabel=r"$t$ ($\mathrm{s}$)", ylabel=r"$I(t)$ ($\mathrm{W/{cm}^2}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax2.plot(
    new_dist_array,
    plot_intensity_s[:, PEAK_NODE],
    "#FF00FF",  # Magenta
    linestyle="-",
    linewidth=2,
    label="Peak time analytical solution",
)
ax2.plot(
    new_dist_array,
    plot_intensity[:, PEAK_NODE],
    "#32CD32",  # Lime green
    linestyle="--",
    linewidth=2,
    label="Peak time numerical solution",
)
ax2.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
ax2.legend(facecolor="black", edgecolor="white")

# fig1.tight_layout()
plt.show()

## Set up figure 2
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize_option)
# Subplot 1
fig2_1 = ax3.pcolormesh(
    new_dist_2d_array, new_time_2d_array, plot_intensity, cmap=cmap_option
)
fig2.colorbar(fig2_1, ax=ax3)
ax3.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{s}$)")
ax3.set_title("Numerical solution in 2D")
# Subplot 2
fig2_2 = ax4.pcolormesh(
    new_dist_2d_array, new_time_2d_array, plot_intensity_s, cmap=cmap_option
)
fig2.colorbar(fig2_2, ax=ax4)
ax4.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{s}$)")
ax4.set_title("Analytical solution in 2D")

# fig2.tight_layout()
plt.show()

## Set up figure 3
fig3, (ax5, ax6) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# Subplot 1
ax5.plot_surface(
    new_dist_2d_array,
    new_time_2d_array,
    plot_intensity,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax5.set(
    xlabel=r"$z$ ($\mathrm{cm}$)",
    ylabel=r"$t$ ($\mathrm{s}$)",
    zlabel=r"$I(z,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax5.set_title("Numerical solution")
# Subplot 2
ax6.plot_surface(
    new_dist_2d_array,
    new_time_2d_array,
    plot_intensity_s,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax6.set(
    xlabel=r"$z$ ($\mathrm{cm}$)",
    ylabel=r"$t$ ($\mathrm{s}$)",
    zlabel=r"$I(z,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax6.set_title("Analytical solution")

# fig3.tight_layout()
plt.show()
