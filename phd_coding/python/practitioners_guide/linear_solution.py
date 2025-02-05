"""
Analytical solution to the linear propagation equation
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES = 0.0, 75e-5, 500
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)
# Propagation (z) grid
INI_DISTANCE_COOR, FIN_DISTANCE_COOR, N_STEPS = 0.0, 6e-2, 500
DISTANCE_STEP_LEN = FIN_DISTANCE_COOR / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES = -150e-15, 150e-15, 2048
TIME_STEP_LEN = (FIN_TIME_COOR - INI_TIME_COOR) / (N_TIME_NODES - 1)
PEAK_NODE = N_TIME_NODES // 2
radi_array = np.linspace(INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES)
dist_array = np.linspace(INI_DISTANCE_COOR, FIN_DISTANCE_COOR, N_STEPS + 1)
time_array = np.linspace(INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES)
radi_2d_array, dist_2d_array = np.meshgrid(radi_array, dist_array, indexing="ij")
radi_2d_array_2, time_2d_array_2 = np.meshgrid(radi_array, time_array, indexing="ij")
dist_2d_array_3, time_2d_array_3 = np.meshgrid(dist_array, time_array, indexing="ij")

# Set wave packet
BEAM_WAIST_0 = 75e-6
BEAM_PEAK_TIME = 130e-15
BEAM_ENERGY = 2.2e-6
FOCAL_LEN = 20
BEAM_CHIRP = -10
BEAM_POWER = BEAM_ENERGY / (BEAM_PEAK_TIME * np.sqrt(0.5 * PI_NUMBER))  # W
BEAM_INTENSITY = 2 * BEAM_POWER / (PI_NUMBER * BEAM_WAIST_0**2)  # W/m^2
BEAM_AMPLITUDE = np.sqrt(BEAM_INTENSITY / INTENSITY_FACTOR)  # V/m

## Analytical solution for a Gaussian beam
# Set arrays
envelope_radial_s = np.empty_like(radi_2d_array, dtype=complex)
envelope_time_s = np.empty_like(dist_2d_array_3, dtype=complex)
envelope_axis_s = np.empty_like(envelope_time_s)
envelope_end_s = np.empty_like(radi_2d_array_2, dtype=complex)

# Set variables
RAYLEIGH_LEN = 0.5 * BEAM_WNUMBER * BEAM_WAIST_0**2
DISPERSION_LEN = 0.5 * BEAM_PEAK_TIME**2 / GVD_COEFF
LENS_DISTANCE = FOCAL_LEN / (1 + (FOCAL_LEN / RAYLEIGH_LEN) ** 2)
beam_waist = BEAM_WAIST_0 * np.sqrt(
    (1 - dist_array / FOCAL_LEN) ** 2 + (dist_array / RAYLEIGH_LEN) ** 2
)
beam_duration = BEAM_PEAK_TIME * np.sqrt(
    (1 + BEAM_CHIRP * dist_array / DISPERSION_LEN) ** 2
    + (dist_array / DISPERSION_LEN) ** 2
)
beam_radius = (
    dist_array
    - LENS_DISTANCE
    + (LENS_DISTANCE * (FOCAL_LEN - LENS_DISTANCE)) / (dist_array - LENS_DISTANCE)
)
gouy_radial_phase = np.atan(
    (dist_array - LENS_DISTANCE) / np.sqrt(FOCAL_LEN * LENS_DISTANCE - LENS_DISTANCE**2)
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
plot_beam_waist = RADI_FACTOR * beam_waist
plot_beam_duration = TIME_FACTOR * beam_duration
plot_intensity_axis_s = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope_axis_s) ** 2
plot_intensity_end_s = AREA_FACTOR * INTENSITY_FACTOR * np.abs(envelope_end_s) ** 2

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
ax1.plot(
    new_dist_array, plot_beam_waist, color="#FF00FF", linestyle="-", label="Beam waist"
)
ax1.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$w(z)$ ($\mathrm{\mu m}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax2.plot(
    new_dist_array,
    plot_beam_duration,
    color="#FFFF00",
    linestyle="-",
    label="Beam duration",
)
ax2.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$T(z)$ ($\mathrm{fs}$)")
ax2.legend(facecolor="black", edgecolor="white")

# fig1.tight_layout()
plt.show()

## Set up figure 2
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
ax3.plot(
    new_time_array,
    plot_intensity_axis_s[0, :],
    color="#32CD32",  # Lime green
    linestyle="-",
    label=r"On-axis analytical solution at beginning $z$ step",
)
ax3.plot(
    new_time_array,
    plot_intensity_axis_s[-1, :],
    color="#1E90FF",  # Electric Blue
    linestyle="-",
    label=r"On-axis analytical solution at final $z$ step",
)
ax3.set(xlabel=r"$t$ ($\mathrm{fs}$)", ylabel=r"$I(t)$ ($\mathrm{W/{cm}^2}$)")
ax3.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax4.plot(
    new_dist_array,
    plot_intensity_axis_s[:, PEAK_NODE],
    color="#FFFF00",  # Pure yellow
    linestyle="-",
    label="On-axis at peak-time",
)
ax4.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
ax4.legend(facecolor="black", edgecolor="white")

# fig2.tight_layout()
plt.show()

## Set up figure 3
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=figsize_option)
# Subplot 1
fig3_1 = ax5.pcolormesh(
    new_dist_2d_array_3,
    new_time_2d_array_3,
    plot_intensity_axis_s,
    cmap=cmap_option,
)
fig3.colorbar(fig3_1, ax=ax5)
ax5.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
ax5.set_title("On-axis analytical solution in 2D")
# Subplot 2
fig3_1 = ax6.pcolormesh(
    new_radi_2d_array_2, new_time_2d_array_2, plot_intensity_end_s, cmap=cmap_option
)
fig3.colorbar(fig3_1, ax=ax6)
ax6.set(xlabel=r"$r$ ($\mathrm{\mu m}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
ax6.set_title(r"Ending $z$ step analytical solution in 2D")

# fig3.tight_layout()
plt.show()

## Set up figure 4
fig4, (ax7, ax8) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# Subplot 1
ax7.plot_surface(
    new_dist_2d_array_3,
    new_time_2d_array_3,
    plot_intensity_axis_s,
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
    new_radi_2d_array_2,
    new_time_2d_array_2,
    plot_intensity_end_s,
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
