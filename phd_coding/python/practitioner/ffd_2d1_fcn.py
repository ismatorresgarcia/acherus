"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse.
This program includes:
    - Diffraction (for the transverse direction).
    - Second order group velocity dispersion (GVD).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Split-step Fourier Crank-Nicolson (FCN) scheme.
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


def initial_condition(r, t, imu, bpm):
    """
    Set the post-lens chirped Gaussian beam.

    Parameters:
    - r (array): Radial array
    - t (array): Time array
    - imu (complex): Square root of -1
    - bpm (dict): Dictionary containing the beam parameters
        - ampli (float): Amplitude of the Gaussian beam
        - waist (float): Waist of the Gaussian beam
        - wnum (float): Wavenumber of the Gaussian beam
        - f (float): Focal length of the initial lens
        - pkt (float): Time at which the Gaussian beam reaches its peak intensity
        - ch (float): Initial chirping introduced by some optical system
    """
    ampli = bpm["AMPLITUDE"]
    waist = bpm["WAIST_0"]
    wnum = bpm["WAVENUMBER"]
    f = bpm["FOCAL_LENGTH"]
    pkt = bpm["PEAK_TIME"]
    ch = bpm["CHIRP"]
    gauss = ampli * np.exp(
        -((r / waist) ** 2)
        - 0.5 * imu * wnum * r**2 / f
        - (1 + imu * ch) * (t / pkt) ** 2
    )

    return gauss


def crank_nicolson_diags(n, pos, coor, coef):
    """
    Generate the three diagonals for a Crank-Nicolson array with centered differences.

    Parameters:
    - n (int): Number of radial nodes
    - pos (str): Position of the Crank-Nicolson array (left or right)
    - coor (int): Parameter for planar (0) or cylindrical (1) geometry
    - coef (float): Coefficient for the diagonal elements

    Returns:
    - tuple: Containing the upper, main, and lower diagonals
    """
    mcf = 1 + 2 * coef
    ind = np.arange(1, n - 1)

    diag_m1 = -coef * (1 - 0.5 * coor / ind)
    diag_0 = np.full(n, mcf)
    diag_p1 = -coef * (1 + 0.5 * coor / ind)

    diag_m1 = np.append(diag_m1, [0])
    diag_p1 = np.insert(diag_p1, 0, [0])
    if coor == 0 and pos == "LEFT":
        diag_0[0], diag_0[-1] = 1, 1
    elif coor == 0 and pos == "RIGHT":
        diag_0[0], diag_0[-1] = 0, 0
    elif coor == 1 and pos == "LEFT":
        diag_0[0], diag_0[-1] = mcf, 1
        diag_p1[0] = -2 * coef
    elif coor == 1 and pos == "RIGHT":
        diag_0[0], diag_0[-1] = mcf, 0
        diag_p1[0] = -2 * coef

    return diag_m1, diag_0, diag_p1


def crank_nicolson_array(n, pos, coor, coef):
    """
    Generate a Crank-Nicolson sparse array in CSR format using the diagonals.

    Parameters:
    - n (int): Number of radial nodes
    - pos (str): Position of the Crank-Nicolson array (left or right)
    - coor (int): Parameter for planar (0) or cylindrical (1) geometry
    - coef (float): Coefficient for the diagonal elements

    Returns:
    - array: Containing the Crank-Nicolson sparse array in CSR format
    """
    diag_m1, diag_0, diag_p1 = crank_nicolson_diags(n, pos, coor, coef)

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]
    cn_array = diags_array(diags, offsets=offset, format="csr")

    return cn_array


IM_UNIT = 1j
PI = np.pi

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

MEDIA = {
    "WATER": {
        "LIN_REF_IND": LIN_REF_IND_WATER,
        "GVD_COEF": GVD_COEF_WATER,
        "INT_FACTOR": 0.5 * LIGHT_SPEED * PERMITTIVITY * LIN_REF_IND_WATER,
    },
    "VACUUM": {
        "LIGHT_SPEED": 299792458,
        "PERMITTIVITY": 8.8541878128e-12,
    },
}

WAVENUMBER_0 = 2 * PI / WAVELENGTH_0
WAVENUMBER = 2 * PI * LIN_REF_IND_WATER / WAVELENGTH_0
POWER = ENERGY / (PEAK_TIME * np.sqrt(0.5 * PI))
INTENSITY = 2 * POWER / (PI * WAIST_0**2)
AMPLITUDE = np.sqrt(INTENSITY / MEDIA["WATER"]["INT_FACTOR"])

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
INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES = -300e-15, 300e-15, 1024
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
radi_2d_array, dist_2d_array = np.meshgrid(radi_array, dist_array, indexing="ij")
radi_2d_array_2, time_2d_array_2 = np.meshgrid(radi_array, time_array, indexing="ij")
dist_2d_array_3, time_2d_array_3 = np.meshgrid(dist_array, time_array, indexing="ij")

## Set loop variables
EU_CYL = 1  # Parameter for planar (0) or cylindrical (1) geometry
DELTA_R = 0.25 * DIST_STEP_LEN / (BEAM["WAVENUMBER"] * RADI_STEP_LEN**2)
DELTA_T = -0.25 * DIST_STEP_LEN * MEDIA["WATER"]["GVD_COEF"] / TIME_STEP_LEN**2
envelope = np.empty_like(radi_2d_array_2, dtype=complex)
envelope_axis = np.empty_like(dist_2d_array_3, dtype=complex)
envelope_fourier = np.empty_like(time_array, dtype=complex)
envelope_store = np.empty_like(envelope)
fourier_coeff = np.exp(-2 * IM_UNIT * DELTA_T * (frq_array * TIME_STEP_LEN) ** 2)
b_array = np.empty_like(envelope)
c_array = np.empty_like(radi_array, dtype=complex)

## Set tridiagonal Crank-Nicolson matrices in csr_array format
MATRIX_CNT_1 = IM_UNIT * DELTA_R
left_cn_matrix = crank_nicolson_array(N_RADI_NODES, "LEFT", EU_CYL, MATRIX_CNT_1)
right_cn_matrix = crank_nicolson_array(N_RADI_NODES, "RIGHT", EU_CYL, -MATRIX_CNT_1)

## Set initial electric field wave packet
envelope = initial_condition(radi_2d_array_2, time_2d_array_2, IM_UNIT, BEAM)
# Save on-axis envelope initial state
envelope_axis[0, :] = envelope[AXIS_NODE, :]

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    # Compute first half-step (Spectral domain)
    for i in range(N_RADI_NODES):
        envelope_fourier = fourier_coeff * fft(envelope[i, :])
        b_array[i, :] = ifft(envelope_fourier)

    # Compute second half-step (Time domain)
    for l in range(N_TIME_NODES):
        c_array = right_cn_matrix @ b_array[:, l]
        envelope_store[:, l] = spsolve(left_cn_matrix, c_array)

    # Update arrays for the next step
    envelope = envelope_store
    envelope_axis[k + 1, :] = envelope_store[AXIS_NODE, :]

## Analytical solution for a Gaussian beam
# Set arrays
envelope_radial_s = np.empty_like(radi_2d_array, dtype=complex)
envelope_time_s = np.empty_like(envelope_axis)
envelope_axis_s = np.empty_like(envelope_axis)
envelope_end_s = np.empty_like(envelope)

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
envelope_end_s = BEAM["AMPLITUDE"] * (
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
new_radi_2d_array_2 = RADI_FACTOR * radi_2d_array_2
new_dist_2d_array_3 = DIST_FACTOR * dist_2d_array_3
new_time_2d_array_2 = TIME_FACTOR * time_2d_array_2
new_time_2d_array_3 = TIME_FACTOR * time_2d_array_3
new_dist_array = new_dist_2d_array_3[:, 0]
new_time_array = new_time_2d_array_3[0, :]

# Set up intensities (W/cm^2)
plot_intensity_axis = (
    AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_axis) ** 2
)
plot_intensity_end = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope) ** 2
plot_intensity_axis_s = (
    AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_axis_s) ** 2
)
plot_intensity_end_s = (
    AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_end_s) ** 2
)

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
