"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse in cartesian coordinates.
This program includes:
    - Diffraction (for two transverse directions).

Numerical discretization: Finite Differences Method (FDM)
    - Method: Fast Fourier Transform (FFT).
    - Initial condition: Gaussian.
    - Boundary conditions: Periodic.

UPPE:          ∂E/∂z = i/(2k) (∂²E/∂x² + ∂²E/∂y²)


E: envelope.
i: imaginary unit.
x: independent transverse coordinate.
y: independent transverse coordinate.
z: distance coordinate.
k: wavenumber (in the interacting media).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2
from tqdm import tqdm


def initial_condition(x, y, im_unit, beam_parameters):
    """
    Set the post-lens chirped Gaussian beam.

    Parameters:
    - x (array): x array
    - y (array): y array
    - im_unit (complex): square root of -1
    - beam_parameters (dict): dictionary containing the beam parameters
        - amplitude (float): amplitude of the Gaussian beam
        - waist (float): waist of the Gaussian beam
        - wave_number (float): wavenumber of the Gaussian beam
        - focal_length (float): focal length of the initial lens

    Returns:
    - array: Gaussian beam envelope's initial condition
    """
    amplitude = beam_parameters["AMPLITUDE"]
    waist = beam_parameters["WAIST_0"]
    wave_number = beam_parameters["WAVENUMBER"]
    focal_length = beam_parameters["FOCAL_LENGTH"]
    gaussian_envelope = amplitude * np.exp(
        -(x**2 + y**2) / waist**2
        - 0.5 * im_unit * wave_number * (x**2 + y**2) / focal_length
    )

    return gaussian_envelope


def fft_step(fourier_coefficient, current_envelope, next_envelope):
    """
    Compute one step of the FFT propagation scheme.

    Parameters:
    - fourier_coefficient: precomputed Fourier coefficient
    - current_envelope: envelope at step k
    - next_envelope: pre-allocated array for envelope at step k + 1
    """
    next_envelope[:, :] = ifft2(fourier_coefficient * fft2(current_envelope))


IM_UNIT = 1j
PI = np.pi

## Set parameters (grid spacing, propagation step, etc.)
# Transverse (x) grid
INI_X_COOR, FIN_X_COOR, N_X_NODES = -1e-2, 1e-2, 128
X_STEP_LEN = (FIN_X_COOR - INI_X_COOR) / (N_X_NODES - 1)
AXIS_X_NODE = int(-INI_X_COOR / X_STEP_LEN)  # On-axis x node
# Transverse (y) grid
INI_Y_COOR, FIN_Y_COOR, N_Y_NODES = -2e-2, 2e-2, 512
Y_STEP_LEN = (FIN_Y_COOR - INI_Y_COOR) / (N_Y_NODES - 1)
AXIS_Y_NODE = int(-INI_Y_COOR / Y_STEP_LEN)  # On-axis y node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0, 3, 1000
DIST_STEP_LEN = (FIN_DIST_COOR - INI_DIST_COOR) / N_STEPS
# Wavenumber's (kx) grid
XFRQ_STEP_LEN = 2 * PI / (N_X_NODES * X_STEP_LEN)
INI_FRQ_COOR_XX1 = 0
FIN_FRQ_COOR_XX1 = PI / X_STEP_LEN - XFRQ_STEP_LEN
INI_FRQ_COOR_XX2 = -PI / X_STEP_LEN
FIN_FRQ_COOR_XX2 = -XFRQ_STEP_LEN
# Wavenumber's (ky) grid
YFRQ_STEP_LEN = 2 * PI / (N_Y_NODES * Y_STEP_LEN)
INI_FRQ_COOR_YX1 = 0
FIN_FRQ_COOR_YX1 = PI / Y_STEP_LEN - YFRQ_STEP_LEN
INI_FRQ_COOR_YX2 = -PI / Y_STEP_LEN
FIN_FRQ_COOR_YX2 = -YFRQ_STEP_LEN
kx1 = np.linspace(INI_FRQ_COOR_XX1, FIN_FRQ_COOR_XX1, N_X_NODES // 2)
kx2 = np.linspace(INI_FRQ_COOR_XX2, FIN_FRQ_COOR_XX2, N_X_NODES // 2)
ky1 = np.linspace(INI_FRQ_COOR_YX1, FIN_FRQ_COOR_YX1, N_Y_NODES // 2)
ky2 = np.linspace(INI_FRQ_COOR_YX2, FIN_FRQ_COOR_YX2, N_Y_NODES // 2)
x_array = np.linspace(INI_X_COOR, FIN_X_COOR, N_X_NODES)
y_array = np.linspace(INI_Y_COOR, FIN_Y_COOR, N_Y_NODES)
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_STEPS + 1)
kx_array = np.append(kx1, kx2)
ky_array = np.append(ky1, ky2)
x_2d_array, y_2d_array = np.meshgrid(x_array, y_array, indexing="ij")
x_3d_array, y_3d_array, dist_3d_array = np.meshgrid(
    x_array, y_array, dist_array, indexing="ij"
)

## Set beam and media parameters
LIGHT_SPEED = 299792458
PERMITTIVITY = 8.8541878128e-12
LIN_REF_IND_WATER = 1.334

WAVELENGTH_0 = 800e-9
WAIST_0 = 9e-3
PEAK_TIME = 130e-15
ENERGY = 4e-3
FOCAL_LENGTH = 10

## Set dictionaries for better organization
MEDIA = {
    "WATER": {
        "LIN_REF_IND": LIN_REF_IND_WATER,
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
    "WAVENUMBER_0": WAVENUMBER_0,
    "WAVENUMBER": WAVENUMBER,
    "POWER": POWER,
    "INTENSITY": INTENSITY,
    "AMPLITUDE": AMPLITUDE,
}

## Set loop variables
DELTA_X = 0.25 * DIST_STEP_LEN / (BEAM["WAVENUMBER"] * X_STEP_LEN**2)
DELTA_Y = 0.25 * DIST_STEP_LEN / (BEAM["WAVENUMBER"] * Y_STEP_LEN**2)
fourier_coeff = np.exp(
    -2
    * IM_UNIT
    * (
        DELTA_X * (kx_array[:, np.newaxis] * X_STEP_LEN) ** 2
        + DELTA_Y * (ky_array * Y_STEP_LEN) ** 2
    )
)
envelope_current = np.empty([N_X_NODES, N_Y_NODES], dtype=complex)
envelope_next = np.empty_like(envelope_current)
envelope_ini = np.empty_like(envelope_current)
envelope_axis = np.empty(N_STEPS + 1, dtype=complex)

## Set initial electric field wave packet
envelope_current = initial_condition(x_2d_array, y_2d_array, IM_UNIT, BEAM)
envelope_ini = envelope_current
# Save on-axis envelope initial state
envelope_axis[0] = envelope_current[AXIS_X_NODE, AXIS_Y_NODE]

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    fft_step(fourier_coeff, envelope_current, envelope_next)
    envelope_current = envelope_next
    envelope_axis[k + 1] = envelope_next[AXIS_X_NODE, AXIS_Y_NODE]

## Analytical solution for a Gaussian beam
# Set arrays
envelope_s = np.empty_like(x_3d_array, dtype=complex)

# Set variables
RAYLEIGH_LEN = 0.5 * BEAM["WAVENUMBER"] * BEAM["WAIST_0"] ** 2
LENS_DIST = BEAM["FOCAL_LENGTH"] / (1 + (BEAM["FOCAL_LENGTH"] / RAYLEIGH_LEN) ** 2)
beam_waist = BEAM["WAIST_0"] * np.sqrt(
    (1 - dist_array / BEAM["FOCAL_LENGTH"]) ** 2 + (dist_array / RAYLEIGH_LEN) ** 2
)
beam_radius = (
    dist_array
    - LENS_DIST
    + (LENS_DIST * (BEAM["FOCAL_LENGTH"] - LENS_DIST)) / (dist_array - LENS_DIST)
)
gouy_phase = np.atan(
    (dist_array - LENS_DIST) / np.sqrt(BEAM["FOCAL_LENGTH"] * LENS_DIST - LENS_DIST**2)
)
#
ratio_term = BEAM["WAIST_0"] / beam_waist[np.newaxis, np.newaxis, :]
decay_exp_term = (
    x_array[:, np.newaxis, np.newaxis] ** 2 + y_array[np.newaxis, :, np.newaxis] ** 2
) / beam_waist[np.newaxis, np.newaxis, :] ** 2
prop_exp_term = (
    0.5
    * IM_UNIT
    * BEAM["WAVENUMBER"]
    * (
        x_array[:, np.newaxis, np.newaxis] ** 2
        + y_array[np.newaxis, :, np.newaxis] ** 2
    )
    / beam_radius[np.newaxis, np.newaxis, :]
)
gouy_exp_term = IM_UNIT * gouy_phase[np.newaxis, np.newaxis, :]

# Compute solution
envelope_s = (
    BEAM["AMPLITUDE"]
    * ratio_term
    * np.exp(-decay_exp_term + prop_exp_term - gouy_exp_term)
)

## Plots
plt.style.use("dark_background")
cmap_option = mpl.colormaps["plasma"]
figsize_option = (13, 7)

# Set up conversion factors
X_FACTOR = 1000
Y_FACTOR = 1000
DIST_FACTOR = 100
AREA_FACTOR = 1e-4
# Set up plotting grid (mm, mm, cm)
x_2d_array = X_FACTOR * x_2d_array
y_2d_array = Y_FACTOR * y_2d_array
dist_array = DIST_FACTOR * dist_array
x_array = x_2d_array[:, 0]
y_array = y_2d_array[0, :]

# Set up intensities (W/cm^2)
plot_int_s = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_s) ** 2
plot_int_ini = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_ini) ** 2
plot_int_fin = (
    AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_current) ** 2
)
plot_int_axis = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_axis) ** 2

## Set up figure 1
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize_option)
# Subplot 1
fig1_2 = ax1.pcolormesh(x_2d_array, y_2d_array, plot_int_ini, cmap=cmap_option)
fig1.colorbar(fig1_2, ax=ax1)
ax1.set(xlabel=r"$x$ ($\mathrm{mm}$)", ylabel=r"$y$ ($\mathrm{mm}$)")
ax1.set_title("Initial numerical solution in 2D")
# Subplot 2
fig1_2 = ax2.pcolormesh(x_2d_array, y_2d_array, plot_int_s[:, :, 0], cmap=cmap_option)
fig1.colorbar(fig1_2, ax=ax2)
ax2.set(xlabel=r"$x$ ($\mathrm{mm}$)", ylabel=r"$y$ ($\mathrm{mm}$)")
ax2.set_title("Initial analytical solution in 2D")
# Subplot 3
fig1_3 = ax3.pcolormesh(x_2d_array, y_2d_array, plot_int_fin, cmap=cmap_option)
fig1.colorbar(fig1_3, ax=ax3)
ax3.set(xlabel=r"$x$ ($\mathrm{mm}$)", ylabel=r"$y$ ($\mathrm{mm}$)")
ax3.set_title("Final numerical solution in 2D")
# Subplot 4
fig1_4 = ax4.pcolormesh(x_2d_array, y_2d_array, plot_int_s[:, :, -1], cmap=cmap_option)
fig1.colorbar(fig1_4, ax=ax4)
ax4.set(xlabel=r"$x$ ($\mathrm{mm}$)", ylabel=r"$y$ ($\mathrm{mm}$)")
ax4.set_title("Final analytical solution in 2D")

fig1.tight_layout()
plt.show()

## Set up figure 2
fig2, (ax5, ax6) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# Subplot 1
ax5.plot_surface(
    x_2d_array,
    y_2d_array,
    plot_int_ini,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax5.set(
    xlabel=r"$x$ ($\mathrm{mm}$)",
    ylabel=r"$y$ ($\mathrm{mm}$)",
    zlabel=r"$I(x,y)$ ($\mathrm{W/{cm}^2}$)",
)
ax5.set_title("Initial numerical solution in 3D")
# Subplot 2
ax6.plot_surface(
    x_2d_array,
    y_2d_array,
    plot_int_s[:, :, 0],
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax6.set(
    xlabel=r"$x$ ($\mathrm{mm}$)",
    ylabel=r"$y$ ($\mathrm{mm}$)",
    zlabel=r"$I(x,y)$ ($\mathrm{W/{cm}^2}$)",
)
ax6.set_title("Initial analytical solution in 3D")

# fig2.tight_layout()
plt.show()

## Set up figure 2
fig3, (ax7, ax8) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# Subplot 1
ax7.plot_surface(
    x_2d_array,
    y_2d_array,
    plot_int_fin,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax7.set(
    xlabel=r"$x$ ($\mathrm{mm}$)",
    ylabel=r"$y$ ($\mathrm{mm}$)",
    zlabel=r"$I(x,y)$ ($\mathrm{W/{cm}^2}$)",
)
ax7.set_title("Final numerical solution in 3D")
# Subplot 2
ax8.plot_surface(
    x_2d_array,
    y_2d_array,
    plot_int_s[:, :, -1],
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax8.set(
    xlabel=r"$x$ ($\mathrm{mm}$)",
    ylabel=r"$y$ ($\mathrm{mm}$)",
    zlabel=r"$I(x,y)$ ($\mathrm{W/{cm}^2}$)",
)
ax8.set_title("Final analytical solution in 3D")

# fig3.tight_layout()
plt.show()

# Set up figure 4
fig4, (ax9, ax10) = plt.subplots(1, 2, figsize=figsize_option)
# Subplot 1
ax9.plot(
    x_array,
    plot_int_s[:, AXIS_Y_NODE, 0],
    color="#FFFF00",
    linestyle="-",
    label=r"Initial analytical solution at $y=0$",
)
ax9.plot(
    x_array,
    plot_int_s[:, AXIS_Y_NODE, -1],
    color="#1E90FF",
    linestyle="-",
    label=r"Final analytical solution at $y=0$",
)
ax9.plot(
    x_array,
    plot_int_ini[:, AXIS_Y_NODE],
    color="#32CD32",
    linestyle="--",
    label=r"Initial numerical solution at $y=0$",
)
ax9.plot(
    x_array,
    plot_int_fin[:, AXIS_Y_NODE],
    color="#FF00FF",
    linestyle="--",
    label=r"Final numerical solution at $y=0$",
)
ax9.set(xlabel=r"$x$ ($\mathrm{mm}$)", ylabel=r"$I(x)$ ($\mathrm{W/{cm}^2}$)")
ax9.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax10.plot(
    y_array,
    plot_int_s[AXIS_X_NODE, :, 0],
    color="#FFFF00",
    linestyle="-",
    label=r"Initial analytical solution at $x=0$",
)
ax10.plot(
    y_array,
    plot_int_s[AXIS_X_NODE, :, -1],
    color="#1E90FF",
    linestyle="-",
    label=r"Final analytical solution at $x=0$",
)
ax10.plot(
    y_array,
    plot_int_ini[AXIS_X_NODE, :],
    color="#32CD32",
    linestyle="--",
    label=r"Initial numerical solution at $x=0$",
)
ax10.plot(
    y_array,
    plot_int_fin[AXIS_X_NODE, :],
    color="#FF00FF",
    linestyle="--",
    label=r"Final numerical solution at $x=0$",
)
ax10.set(xlabel=r"$y$ ($\mathrm{mm}$)", ylabel=r"$I(y)$ ($\mathrm{W/{cm}^2}$)")
ax10.legend(facecolor="black", edgecolor="white")

fig4.tight_layout()
plt.show()

# Set up figure 5
fig5, ax11 = plt.subplots(figsize=figsize_option)
ax11.plot(
    dist_array,
    plot_int_s[AXIS_X_NODE, AXIS_Y_NODE, :],
    color="#32CD32",
    linestyle="-",
    label=r"On-axis analytical solution",
)
ax11.plot(
    dist_array,
    plot_int_axis,
    color="#FF00FF",
    linestyle="--",
    label=r"On-axis numerical solution",
)
ax11.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
ax11.legend(facecolor="black", edgecolor="white")

fig5.tight_layout()
plt.show()
