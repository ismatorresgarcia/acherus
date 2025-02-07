"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse.
This program includes:
    - Diffraction (for the transverse direction).

Numerical discretization: Finite Differences Method (FDM)
    - Method: Fast Fourier Transform (FFT).
    - Initial condition: Gaussian.
    - Boundary conditions: Periodic.

UPPE:           ∂E/∂z = i/(2k) ∂²E/∂x²


E: envelope.
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
k: wavenumber (in the interacting media).
∇: nabla operator (for the tranverse direction).
∇²: laplace operator (for the transverse direction).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from tqdm import tqdm


def initial_condition(r, imu, bpm):
    """
    Set the post-lens chirped Gaussian beam.

    Parameters:
    - r (array): Radial array
    - imu (complex): Square root of -1
    - bpm (dict): Dictionary containing the beam parameters
        - ampli (float): Amplitude of the Gaussian beam
        - waist (float): Waist of the Gaussian beam
        - wnum (float): Wavenumber of the Gaussian beam
        - f (float): Focal length of the initial lens

    Returns:
    - array: Gaussian beam envelope's initial condition
    """
    ampli = bpm["AMPLITUDE"]
    waist = bpm["WAIST_0"]
    wnum = bpm["WAVENUMBER"]
    f = bpm["FOCAL_LENGTH"]
    gauss = ampli * np.exp(-((r / waist) ** 2) - 0.5 * imu * wnum * r**2 / f)

    return gauss


IM_UNIT = 1j
PI = np.pi

MEDIA = {
    "WATER": {
        "LIN_REF_IND": 1.334,
    },
    "VACUUM": {
        "LIN_REF_IND": 1,
        "LIGHT_SPEED": 299792458,
        "PERMITTIVITY": 8.8541878128e-12,
    },
}
MEDIA["WATER"].update(
    {
        "INT_FACTOR": 0.5
        * MEDIA["VACUUM"]["LIGHT_SPEED"]
        * MEDIA["VACUUM"]["PERMITTIVITY"]
        * MEDIA["WATER"]["LIN_REF_IND"],
    }
)
BEAM = {
    "WAVELENGTH_0": 800e-9,
    "WAIST_0": 9e-3,
    "PEAK_TIME": 130e-15,
    "ENERGY": 4e-3,
    "FOCAL_LENGTH": 10,
}
BEAM.update(
    {
        "WAVENUMBER_0": 2 * PI / BEAM["WAVELENGTH_0"],
        "WAVENUMBER": 2 * PI * MEDIA["WATER"]["LIN_REF_IND"] / BEAM["WAVELENGTH_0"],
        "POWER": BEAM["ENERGY"] / (BEAM["PEAK_TIME"] * np.sqrt(0.5 * PI)),
    }
)
BEAM.update({"INTENSITY": 2 * BEAM["POWER"] / (PI * BEAM["WAIST_0"] ** 2)})
BEAM.update({"AMPLITUDE": np.sqrt(BEAM["INTENSITY"] / MEDIA["WATER"]["INT_FACTOR"])})

## Set parameters (grid spacing, propagation step, etc.)
# Radial (x) grid
INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES = -2e-2, 2e-2, 1000
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)  # On-axis node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0, 3, 1000
DIST_STEP_LEN = (FIN_DIST_COOR - INI_DIST_COOR) / N_STEPS
# Wavenumber's (kx) grid
FRQ_STEP_LEN = 2 * PI / (N_RADI_NODES * RADI_STEP_LEN)
INI_FRQ_COOR_X1 = 0
FIN_FRQ_COOR_X1 = PI / RADI_STEP_LEN - FRQ_STEP_LEN
INI_FRQ_COOR_X2 = -PI / RADI_STEP_LEN
FIN_FRQ_COOR_X2 = -FRQ_STEP_LEN
kx1 = np.linspace(INI_FRQ_COOR_X1, FIN_FRQ_COOR_X1, N_RADI_NODES // 2)
kx2 = np.linspace(INI_FRQ_COOR_X2, FIN_FRQ_COOR_X2, N_RADI_NODES // 2)
radi_array = np.linspace(INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES)
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_STEPS + 1)
kx_array = np.append(kx1, kx2)
radi_2d_array, dist_2d_array = np.meshgrid(radi_array, dist_array, indexing="ij")

## Set loop variables
DELTA_X = 0.25 * DIST_STEP_LEN / (BEAM["WAVENUMBER"] * RADI_STEP_LEN**2)
envelope = np.empty_like(radi_2d_array, dtype=complex)
envelope_fourier = np.empty_like(radi_array, dtype=complex)
fourier_coeff = np.exp(-2 * IM_UNIT * DELTA_X * (kx_array * RADI_STEP_LEN) ** 2)

## Set initial electric field wave packet
envelope[:, 0] = initial_condition(radi_array, IM_UNIT, BEAM)

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    # Compute solution
    envelope_fourier = fourier_coeff * fft(envelope[:, k])
    envelope[:, k + 1] = ifft(envelope_fourier)

## Analytical solution for a Gaussian beam
# Set arrays
envelope_s = np.empty_like(envelope)

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
ratio_term = BEAM["WAIST_0"] / beam_waist[np.newaxis, :]
decay_exp_term = (radi_array[:, np.newaxis] / beam_waist) ** 2
prop_exp_term = (
    0.5 * IM_UNIT * BEAM["WAVENUMBER"] * radi_array[:, np.newaxis] ** 2 / beam_radius
)  # (1002, 1001)
gouy_exp_term = IM_UNIT * gouy_phase[np.newaxis, :]

# Compute solution
envelope_s = (
    BEAM["AMPLITUDE"]
    * np.sqrt(ratio_term)
    * np.exp(-decay_exp_term + prop_exp_term - gouy_exp_term)
)

### Plots
plt.style.use("dark_background")
cmap_option = mpl.colormaps["plasma"]
figsize_option = (13, 7)

# Set up conversion factors
RADI_FACTOR = 1000
DIST_FACTOR = 100
AREA_FACTOR = 1e-4
# Set up plotting grid (mm, cm)
new_radi_2d_array = RADI_FACTOR * radi_2d_array
new_dist_2d_array = DIST_FACTOR * dist_2d_array
new_radi_array = new_radi_2d_array[:, 0]
new_dist_array = new_dist_2d_array[0, :]

# Set up intensities (W/cm^2)
plot_intensity = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope) ** 2
plot_intensity_s = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_s) ** 2

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
intensity_list = [
    (
        plot_intensity_s[:, 0],
        "#FF00FF",  # Magenta
        "-",
        r"Analytical solution at beginning $z$ step",
    ),
    (
        plot_intensity_s[:, -1],
        "#FFFF00",  # Pure yellow
        "-",
        r"Analytical solution at final $z$ step",
    ),
    (
        plot_intensity[:, 0],
        "#32CD32",  # Lime green
        "--",
        r"Numerical solution at beginning $z$ step",
    ),
    (
        plot_intensity[:, -1],
        "#1E90FF",  # Electric Blue
        "--",
        r"Numerical solution at final $z$ step",
    ),
]
for data, color, style, label in intensity_list:
    ax1.plot(new_radi_array, data, color, linestyle=style, linewidth=2, label=label)
ax1.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$I(r)$ ($\mathrm{W/{cm}^2}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax2.plot(
    new_dist_array,
    plot_intensity_s[AXIS_NODE, :],
    "#FF00FF",  # Magenta
    linestyle="-",
    linewidth=2,
    label="On-axis analytical solution",
)
ax2.plot(
    new_dist_array,
    plot_intensity[AXIS_NODE, :],
    "#32CD32",  # Lime green
    linestyle="--",
    linewidth=2,
    label="On-axis numerical solution",
)
ax2.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
ax2.legend(facecolor="black", edgecolor="white")

# fig1.tight_layout()
plt.show()

## Set up figure 2
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize_option)
# Subplot 1
fig2_1 = ax3.pcolormesh(
    new_radi_2d_array, new_dist_2d_array, plot_intensity, cmap=cmap_option
)
fig2.colorbar(fig2_1, ax=ax3)
ax3.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$z$ ($\mathrm{cm}$)")
ax3.set_title("Numerical solution in 2D")
# Subplot 2
fig2_2 = ax4.pcolormesh(
    new_radi_2d_array, new_dist_2d_array, plot_intensity_s, cmap=cmap_option
)
fig2.colorbar(fig2_2, ax=ax4)
ax4.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$z$ ($\mathrm{cm}$)")
ax4.set_title("Analytical solution in 2D")

# fig2.tight_layout()
plt.show()

## Set up figure 3
fig3, (ax5, ax6) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# Subplot 1
ax5.plot_surface(
    new_radi_2d_array,
    new_dist_2d_array,
    plot_intensity,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax5.set(
    xlabel=r"$r$ ($\mathrm{mm}$)",
    ylabel=r"$z$ ($\mathrm{cm}$)",
    zlabel=r"$I(r,z)$ ($\mathrm{W/{cm}^2}$)",
)
ax5.set_title("Numerical solution")
# Subplot 2
ax6.plot_surface(
    new_radi_2d_array,
    new_dist_2d_array,
    plot_intensity_s,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax6.set(
    xlabel=r"$r$ ($\mathrm{mm}$)",
    ylabel=r"$z$ ($\mathrm{cm}$)",
    zlabel=r"$I(r,z)$ ($\mathrm{W/{cm}^2}$)",
)
ax6.set_title("Analytical solution")

# fig3.tight_layout()
plt.show()
