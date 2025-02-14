"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse in cylindrical coordinates with radial symmetry.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Alternating Direction Implicit (ADI) scheme.
    - Initial condition: Gaussian.
    - Boundary conditions: Neumann-Dirichlet (radial) and homogeneous Dirichlet (temporal).

UPPE:          ∂E/∂z = i/(2k) ∇²E - ik''/2 ∂²E/∂t²


E: envelope.
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
t: time coordinate.
k: wavenumber (in the interacting media).
k'': GVD coefficient of 2nd order.
∇²: laplace operator (for the transverse direction).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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


def crank_nicolson_radial_diags(nodes, position, coefficient):
    """
    Set the three diagonals for the Crank-Nicolson radial array with centered differences.

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


def crank_nicolson_time_diags(nodes, position, coefficient):
    """
    Set the three diagonals for a Crank-Nicolson time array with centered differences.

    Parameters:
    - nodes (int): number of time nodes
    - position (str): position of the Crank-Nicolson array (left or right)
    - coefficient (float): coefficient for the diagonal elements

    Returns:
    - tuple: upper, main, and lower diagonals
    """
    main_coefficient = 1 + 2 * coefficient

    diag_m1 = np.full(nodes - 1, -coefficient)
    diag_0 = np.full(nodes, main_coefficient)
    diag_p1 = np.full(nodes - 1, -coefficient)

    diag_p1[0], diag_m1[-1] = 0, 0
    if position == "LEFT":
        diag_0[0], diag_0[-1] = 1, 1
    else:
        diag_0[0], diag_0[-1] = 0, 0

    return diag_m1, diag_0, diag_p1


def crank_nicolson_radial_array(nodes, position, coefficient):
    """
    Set the Crank-Nicolson radial sparse array in CSR format using the diagonals.

    Parameters:
    - nodes (int): number of radial nodes
    - position (str): position of the Crank-Nicolson array (left or right)
    - coefficient (float): coefficient for the diagonal elements

    Returns:
    - array: Crank-Nicolson sparse array in CSR format
    """
    diag_m1, diag_0, diag_p1 = crank_nicolson_radial_diags(nodes, position, coefficient)

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]
    radial_output = diags_array(diags, offsets=offset, format="csr")

    return radial_output


def crank_nicolson_time_array(nodes, position, coefficient):
    """
    Set the Crank-Nicolson sparse time array in CSR format using the diagonals.

    Parameters:
    - nodes (int): number of time nodes
    - position (str): position of the Crank-Nicolson array (left or right)
    - coefficient (float): coefficient for the diagonal elements

    Returns:
    - array: Crank-Nicolson sparse array in CSR format
    """
    diag_m1, diag_0, diag_p1 = crank_nicolson_time_diags(nodes, position, coefficient)

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]
    time_output = diags_array(diags, offsets=offset, format="csr")

    return time_output


def adi_radial_step(
    left_array_r, right_array_t, current_envelope, inter_array, semi_array
):
    """
    Compute first half-step (ADI radial direction).

    Parameters:
    - left_array_r: sparse array for left-hand side (radial)
    - right_array_t: sparse array for right-hand side (time)
    - current_envelope: envelope at step k
    - inter_array: pre-allocated array for intermediate results
    - semi_array: pre-allocated array for envelope at step k + 1
    """
    # Compute right-hand side matrix product row by row
    for i in range(current_envelope.shape[0]):
        inter_array[i, :] = right_array_t @ current_envelope[i, :]

    # Compute first half-step solution
    for l in range(current_envelope.shape[1]):
        semi_array[:, l] = spsolve(left_array_r, inter_array[:, l])


def adi_time_step(left_array_t, right_array_r, inter_array, semi_array, next_envelope):
    """
    Compute second half-step (ADI time direction).

    Parameters:
    - left_array_t: sparse array for left-hand side (time)
    - right_array_r: sparse array for right-hand side (radial)
    - inter_array: envelope at step k
    - semi_array: preallocated array for intermediate results
    - next_envelope: pre-allocated array for envelope at step k + 1
    """
    # Compute right-hand side matrix product column by column
    for l in range(inter_array.shape[1]):
        semi_array[:, l] = right_array_r @ inter_array[:, l]

    # Compute second half-step solution
    for i in range(inter_array.shape[0]):
        next_envelope[i, :] = spsolve(left_array_t, semi_array[i, :])


IM_UNIT = 1j
PI = np.pi

## Set parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
INI_RADI_COOR, FIN_RADI_COOR, I_RADI_NODES = 0, 75e-4, 1000
N_RADI_NODES = I_RADI_NODES + 2
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)  # On-axis node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0, 6e-2, 100
DIST_STEP_LEN = FIN_DIST_COOR / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, I_TIME_NODES = -300e-15, 300e-15, 1024
N_TIME_NODES = I_TIME_NODES + 2
TIME_STEP_LEN = (FIN_TIME_COOR - INI_TIME_COOR) / (N_TIME_NODES - 1)
PEAK_NODE = N_TIME_NODES // 2  # Peak intensity node
radi_array = np.linspace(INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES)
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_STEPS + 1)
time_array = np.linspace(INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES)
radi_2d_array, time_2d_array = np.meshgrid(radi_array, time_array, indexing="ij")
dist_2d_array_2, time_2d_array_2 = np.meshgrid(dist_array, time_array, indexing="ij")

## Set beam and media parameters
LIGHT_SPEED = 299792458
PERMITTIVITY = 8.8541878128e-12
LIN_REF_IND_WATER = 1.334
GVD_COEF_WATER = 241e-28

WAVELENGTH_0 = 800e-9
WAIST_0 = 75e-5
PEAK_TIME = 130e-15
ENERGY = 2.2e-6
FOCAL_LENGTH = 20
CHIRP = -10

## Set dictionaries for better organization
MEDIA = {
    "WATER": {
        "LIN_REF_IND": LIN_REF_IND_WATER,
        "GVD_COEF": GVD_COEF_WATER,
        "INT_FACTOR": 0.5 * LIGHT_SPEED * PERMITTIVITY * LIN_REF_IND_WATER,
    },
    "VACUUM": {
        "LIGHT_SPEED": LIGHT_SPEED,
        "PERMITTIVITY": PERMITTIVITY,
    },
}

WAVENUMBER_0 = 2 * PI / WAVELENGTH_0
WAVENUMBER = 2 * PI * LIN_REF_IND_WATER / WAVELENGTH_0
POWER = ENERGY / (PEAK_TIME * np.sqrt(0.5 * PI))
INTENSITY = 2 * POWER / (PI * WAIST_0**2)
AMPLITUDE = np.sqrt(INTENSITY / MEDIA["WATER"]["INT_FACTOR"])

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
    "INTENSITY": INTENSITY,
    "AMPLITUDE": AMPLITUDE,
}

## Set loop variables
DELTA_R = 0.25 * DIST_STEP_LEN / (BEAM["WAVENUMBER"] * RADI_STEP_LEN**2)
DELTA_T = -0.25 * DIST_STEP_LEN * MEDIA["WATER"]["GVD_COEF"] / TIME_STEP_LEN**2
envelope_current = np.empty([N_RADI_NODES, N_TIME_NODES], dtype=complex)
envelope_next = np.empty_like(envelope_current)
envelope_axis = np.empty([N_STEPS + 1, N_TIME_NODES], dtype=complex)
b_array = np.empty_like(envelope_current)
c_array = np.empty_like(envelope_current)

## Set tridiagonal Crank-Nicolson matrices in csr_array format
MAT_CNT_1R = IM_UNIT * DELTA_R
MAT_CNT_1T = IM_UNIT * DELTA_T
left_operator_r = crank_nicolson_radial_array(N_RADI_NODES, "LEFT", MAT_CNT_1R)
right_operator_r = crank_nicolson_radial_array(N_RADI_NODES, "RIGHT", -MAT_CNT_1R)
left_operator_t = crank_nicolson_time_array(N_TIME_NODES, "LEFT", MAT_CNT_1T)
right_operator_t = crank_nicolson_time_array(N_TIME_NODES, "RIGHT", -MAT_CNT_1T)

## Set initial electric field wave packet
envelope_current = initial_condition(radi_2d_array, time_2d_array, IM_UNIT, BEAM)
envelope_axis[0, :] = envelope_current[AXIS_NODE, :]

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    adi_radial_step(
        left_operator_r, right_operator_t, envelope_current, b_array, c_array
    )
    adi_time_step(left_operator_t, right_operator_r, c_array, b_array, envelope_next)

    # Update arrays for the next step
    envelope_current, envelope_next = envelope_next, envelope_current

    # Store axis data
    envelope_axis[k + 1, :] = envelope_current[AXIS_NODE, :]

## Analytical solution for a Gaussian beam
# Set arrays
envelope_radial_s = np.empty([N_RADI_NODES, N_STEPS + 1], dtype=complex)
envelope_time_s = np.empty([N_TIME_NODES, N_STEPS + 1], dtype=complex)
envelope_fin_s = np.empty([N_RADI_NODES, N_TIME_NODES], dtype=complex)
envelope_axis_s = np.empty_like(envelope_time_s)

# Set variables
RAYLEIGH_LEN = 0.5 * BEAM["WAVENUMBER"] * BEAM["WAIST_0"] ** 2
DISPERSION_LEN = 0.5 * BEAM["PEAK_TIME"] ** 2 / MEDIA["WATER"]["GVD_COEF"]
LENS_DIST = BEAM["FOCAL_LENGTH"] / (1 + (BEAM["FOCAL_LENGTH"] / RAYLEIGH_LEN) ** 2)
beam_waist = BEAM["WAIST_0"] * np.sqrt(
    (1 - dist_array / BEAM["FOCAL_LENGTH"]) ** 2 + (dist_array / RAYLEIGH_LEN) ** 2
)
beam_duration = BEAM["PEAK_TIME"] * np.sqrt(
    (1 + BEAM["CHIRP"] * dist_array / DISPERSION_LEN) ** 2
    + (dist_array / DISPERSION_LEN) ** 2
)
beam_radius = (
    dist_array
    - LENS_DIST
    + (LENS_DIST * (BEAM["FOCAL_LENGTH"] - LENS_DIST)) / (dist_array - LENS_DIST)
)
gouy_radial_phase = np.atan(
    (dist_array - LENS_DIST) / np.sqrt(BEAM["FOCAL_LENGTH"] * LENS_DIST - LENS_DIST**2)
)
gouy_time_phase = 0.5 * np.atan(
    -dist_array / (DISPERSION_LEN + BEAM["CHIRP"] * dist_array)
)
#
ratio_term = BEAM["WAIST_0"] / beam_waist[np.newaxis, :]
sqrt_term = np.sqrt(BEAM["PEAK_TIME"] / beam_duration[:, np.newaxis])
decay_radial_exp_term = (radi_array[:, np.newaxis] / beam_waist) ** 2
decay_time_exp_term = (time_array / beam_duration[:, np.newaxis]) ** 2
prop_radial_exp_term = (
    0.5 * IM_UNIT * BEAM["WAVENUMBER"] * radi_array[:, np.newaxis] ** 2 / beam_radius
)
prop_time_exp_term = 1 + IM_UNIT * (
    BEAM["CHIRP"]
    + (1 + BEAM["CHIRP"] ** 2) * (dist_array[:, np.newaxis] / DISPERSION_LEN)
)
gouy_radial_exp_term = IM_UNIT * gouy_radial_phase[np.newaxis, :]
gouy_time_exp_term = IM_UNIT * gouy_time_phase[:, np.newaxis]

# Compute solution
envelope_radial_s = ratio_term * np.exp(
    -decay_radial_exp_term + prop_radial_exp_term - gouy_radial_exp_term
)
envelope_time_s = sqrt_term * np.exp(
    -decay_time_exp_term * prop_time_exp_term - gouy_time_exp_term
)
envelope_fin_s = BEAM["AMPLITUDE"] * (
    envelope_radial_s[:, -1, np.newaxis] * envelope_time_s[-1, :]
)
envelope_axis_s = BEAM["AMPLITUDE"] * (
    envelope_radial_s[AXIS_NODE, :, np.newaxis] * envelope_time_s
)

### Plots
plt.style.use("dark_background")
cmap_option = mpl.colormaps["plasma"]
figsize_option = (13, 7)

# Set up conversion factors
RADI_FACTOR = 1000
DIST_FACTOR = 100
TIME_FACTOR = 1e15
AREA_FACTOR = 1e-4
# Set up plotting grid (mm, cm and s)
radi_2d_array = RADI_FACTOR * radi_2d_array
dist_2d_array_2 = DIST_FACTOR * dist_2d_array_2
time_2d_array = TIME_FACTOR * time_2d_array
time_2d_array_2 = TIME_FACTOR * time_2d_array_2
dist_array = dist_2d_array_2[:, 0]
time_array = time_2d_array_2[0, :]

# Set up intensities (W/cm^2)
plot_int_axis = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_axis) ** 2
plot_int_fin = (
    AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_current) ** 2
)
plot_int_axis_s = (
    AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_axis_s) ** 2
)
plot_int_fin_s = (
    AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_fin_s) ** 2
)

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
intensity_list = [
    (
        plot_int_axis_s[0, :],
        "#FF00FF",  # Magenta
        "-",
        r"On-axis analytical solution at beginning $z$ step",
    ),
    (
        plot_int_axis_s[-1, :],
        "#FFFF00",  # Pure yellow
        "-",
        r"On-axis analytical solution at final $z$ step",
    ),
    (
        plot_int_axis[0, :],
        "#32CD32",  # Lime green
        "--",
        r"On-axis numerical solution at beginning $z$ step",
    ),
    (
        plot_int_axis[-1, :],
        "#1E90FF",  # Electric Blue
        "--",
        r"On-axis numerical solution at final $z$ step",
    ),
]
for data, color, style, label in intensity_list:
    ax1.plot(time_array, data, color, linestyle=style, linewidth=2, label=label)
ax1.set(xlabel=r"$t$ ($\mathrm{s}$)", ylabel=r"$I(t)$ ($\mathrm{W/{cm}^2}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax2.plot(
    dist_array,
    plot_int_axis_s[:, PEAK_NODE],
    "#FF00FF",  # Magenta
    linestyle="-",
    linewidth=2,
    label="On-axis peak time analytical solution",
)
ax2.plot(
    dist_array,
    plot_int_axis[:, PEAK_NODE],
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
    dist_2d_array_2,
    time_2d_array_2,
    plot_int_axis,
    cmap=cmap_option,
)
fig2.colorbar(fig2_1, ax=ax3)
ax3.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{s}$)")
ax3.set_title("On-axis numerical solution in 2D")
# Second subplot
fig2_2 = ax4.pcolormesh(
    dist_2d_array_2,
    time_2d_array_2,
    plot_int_axis_s,
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
    radi_2d_array,
    time_2d_array,
    plot_int_fin,
    cmap=cmap_option,
)
fig3.colorbar(fig3_1, ax=ax5)
ax5.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$t$ ($\mathrm{s}$)")
ax5.set_title("Final step numerical solution in 2D")
# Second subplot
fig3_2 = ax6.pcolormesh(
    radi_2d_array,
    time_2d_array,
    plot_int_fin_s,
    cmap=cmap_option,
)
fig3.colorbar(fig3_2, ax=ax6)
ax6.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$t$ ($\mathrm{s}$)")
ax6.set_title("Final step analytical solution in 2D")

# fig3.tight_layout()
plt.show()

## Set up figure 4
fig4, (ax7, ax8) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# First subplot
ax7.plot_surface(
    dist_2d_array_2,
    time_2d_array_2,
    plot_int_axis,
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
    dist_2d_array_2,
    time_2d_array_2,
    plot_int_axis_s,
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
    radi_2d_array,
    time_2d_array,
    plot_int_fin,
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
    radi_2d_array,
    time_2d_array,
    plot_int_fin_s,
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
