"""
This program represents graphically the 2D propagation equation of a laser pulse
due to diffraction and second order group velocity dispersion (GVD) with cylindrical
coordinates and radial symmetry.

UPPE:           ∂E/∂z = i/(2k) ∇²E - ik''/2 ∂²E/∂t²

E: envelope.
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
t: time coordinate.
k: wavenumber (in the interacting media).
∇²: laplace operator (for the transverse direction).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

IM_UNIT = 1j
PI = np.pi

## Set parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES = 0, 75e-5, 500
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)
# Propagation (z) grid
INI_DISTANCE_COOR, FIN_DISTANCE_COOR, N_STEPS = 0, 6e-2, 500
DISTANCE_STEP_LEN = FIN_DISTANCE_COOR / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES = -150e-15, 150e-15, 2048
TIME_STEP_LEN = (FIN_TIME_COOR - INI_TIME_COOR) / (N_TIME_NODES - 1)
PEAK_NODE = N_TIME_NODES // 2
radi_array = np.linspace(INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES)
dist_array = np.linspace(INI_DISTANCE_COOR, FIN_DISTANCE_COOR, N_STEPS + 1)
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

## Analytical solution for a Gaussian beam
# Set arrays
envelope_radial_sol = np.empty([N_RADI_NODES, N_STEPS + 1], dtype=complex)
envelope_time_sol = np.empty([N_STEPS + 1, N_TIME_NODES], dtype=complex)
envelope_fin_sol = np.empty([N_RADI_NODES, N_TIME_NODES], dtype=complex)
envelope_axis_sol = np.empty_like(envelope_time_sol)

# Set variables
RAYLEIGH_LEN = 0.5 * BEAM["WAVENUMBER"] * BEAM["WAIST_0"] ** 2
DISPERSION_LEN = 0.5 * BEAM["PEAK_TIME"] ** 2 / MEDIA["WATER"]["GVD_COEF"]
LENS_DISTANCE = BEAM["FOCAL_LENGTH"] / (1 + (BEAM["FOCAL_LENGTH"] / RAYLEIGH_LEN) ** 2)
beam_waist = BEAM["WAIST_0"] * np.sqrt(
    (1 - dist_array / BEAM["FOCAL_LENGTH"]) ** 2 + (dist_array / RAYLEIGH_LEN) ** 2
)
beam_duration = BEAM["PEAK_TIME"] * np.sqrt(
    (1 + BEAM["CHIRP"] * dist_array / DISPERSION_LEN) ** 2
    + (dist_array / DISPERSION_LEN) ** 2
)
beam_radius = (
    dist_array
    - LENS_DISTANCE
    + (LENS_DISTANCE * (BEAM["FOCAL_LENGTH"] - LENS_DISTANCE))
    / (dist_array - LENS_DISTANCE)
)
gouy_radial_phase = np.atan(
    (dist_array - LENS_DISTANCE)
    / np.sqrt(BEAM["FOCAL_LENGTH"] * LENS_DISTANCE - LENS_DISTANCE**2)
)
gouy_time_phase = 0.5 * np.atan(
    -dist_array / (DISPERSION_LEN + BEAM["CHIRP"] * dist_array)
)
#
ratio_term = BEAM["WAIST_0"] / beam_waist[np.newaxis, :]
sqrt_term = np.sqrt(BEAM["PEAK_TIME"] / beam_duration[:, np.newaxis])
decay_radial_term = (radi_array[:, np.newaxis] / beam_waist) ** 2
decay_time_term = (time_array / beam_duration[:, np.newaxis]) ** 2
prop_radial_term = (
    0.5 * IM_UNIT * BEAM["WAVENUMBER"] * radi_array[:, np.newaxis] ** 2 / beam_radius
)
prop_time_term = 1 + IM_UNIT * (
    BEAM["CHIRP"]
    + (1 + BEAM["CHIRP"] ** 2) * (dist_array[:, np.newaxis] / DISPERSION_LEN)
)
gouy_radial_term = IM_UNIT * gouy_radial_phase[np.newaxis, :]
gouy_time_term = IM_UNIT * gouy_time_phase[:, np.newaxis]

# Compute solution
envelope_radial_sol = ratio_term * np.exp(
    -decay_radial_term + prop_radial_term - gouy_radial_term
)
envelope_time_sol = sqrt_term * np.exp(
    -decay_time_term * prop_time_term - gouy_time_term
)
envelope_fin_sol = BEAM["AMPLITUDE"] * (
    envelope_radial_sol[:, -1, np.newaxis] * envelope_time_sol[-1, :]
)
envelope_axis_sol = BEAM["AMPLITUDE"] * (
    envelope_radial_sol[AXIS_NODE, :, np.newaxis] * envelope_time_sol
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
plot_beam_waist = RADI_FACTOR * beam_waist
plot_beam_duration = TIME_FACTOR * beam_duration
plot_int_axis_sol = (
    AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_axis_sol) ** 2
)
plot_int_fin_sol = (
    AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_fin_sol) ** 2
)

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
ax1.plot(
    dist_array, plot_beam_waist, color="#FF00FF", linestyle="-", label="Beam waist"
)
ax1.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$w(z)$ ($\mathrm{\mu m}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax2.plot(
    dist_array,
    plot_beam_duration,
    color="#FFFF00",
    linestyle="-",
    label="Beam duration",
)
ax2.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$T(z)$ ($\mathrm{fs}$)")
ax2.legend(facecolor="black", edgecolor="white")

fig1.tight_layout()
plt.show()

## Set up figure 2
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
ax3.plot(
    time_array,
    plot_int_axis_sol[0, :],
    color="#32CD32",  # Lime green
    linestyle="-",
    label=r"On-axis analytical solution at beginning $z$ step",
)
ax3.plot(
    time_array,
    plot_int_axis_sol[-1, :],
    color="#1E90FF",  # Electric Blue
    linestyle="-",
    label=r"On-axis analytical solution at final $z$ step",
)
ax3.set(xlabel=r"$t$ ($\mathrm{fs}$)", ylabel=r"$I(t)$ ($\mathrm{W/{cm}^2}$)")
ax3.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax4.plot(
    dist_array,
    plot_int_axis_sol[:, PEAK_NODE],
    color="#FFFF00",  # Pure yellow
    linestyle="-",
    label="On-axis at peak-time",
)
ax4.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
ax4.legend(facecolor="black", edgecolor="white")

fig2.tight_layout()
plt.show()

## Set up figure 3
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=figsize_option)
# Subplot 1
fig3_1 = ax5.pcolormesh(
    dist_2d_array_2,
    time_2d_array_2,
    plot_int_axis_sol,
    cmap=cmap_option,
)
fig3.colorbar(fig3_1, ax=ax5)
ax5.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
ax5.set_title("On-axis analytical solution in 2D")
# Subplot 2
fig3_1 = ax6.pcolormesh(
    radi_2d_array, time_2d_array, plot_int_fin_sol, cmap=cmap_option
)
fig3.colorbar(fig3_1, ax=ax6)
ax6.set(xlabel=r"$r$ ($\mathrm{\mu m}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
ax6.set_title(r"Ending $z$ step analytical solution in 2D")

fig3.tight_layout()
plt.show()

## Set up figure 4
fig4, (ax7, ax8) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# Subplot 1
ax7.plot_surface(
    dist_2d_array_2,
    time_2d_array_2,
    plot_int_axis_sol,
    cmap=cmap_option,
    linewidth=0,
    antialiased=True,
)
ax7.set(
    xlabel=r"$z$ ($\mathrm{cm}$)",
    ylabel=r"$t$ ($\mathrm{fs}$)",
    zlabel=r"$I(z,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax7.set_title("On-axis analytical solution in 3D")
# Subplot 2
ax8.plot_surface(
    radi_2d_array,
    time_2d_array,
    plot_int_fin_sol,
    cmap=cmap_option,
    linewidth=0,
    antialiased=True,
)
ax8.set(
    xlabel=r"$r$ ($\mathrm{\mu m}$)",
    ylabel=r"$t$ ($\mathrm{fs}$)",
    zlabel=r"$I(r,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax8.set_title(r"Ending $z$ step analytical solution in 3D")

# fig4.tight_layout()  # Add more padding between subplots
plt.show()
