"""
Python script for plotting NumPy arrays saved during the simulations.
This script uses the matplotlib library to plot the results.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

## Load arrays
data = np.load("~/projects/phd_thesis/phd_coding/python/storage/ffdmk_fcn_1.npz")
INI_RADI_COOR = data["INI_RADI_COOR"]
FIN_RADI_COOR = data["FIN_RADI_COOR"]
INI_DIST_COOR = data["INI_DIST_COOR"]
FIN_DIST_COOR = data["FIN_DIST_COOR"]
INI_TIME_COOR = data["INI_TIME_COOR"]
FIN_TIME_COOR = data["FIN_TIME_COOR"]
AXIS_NODE = data["AXIS_NODE"]
PEAK_NODE = data["PEAK_NODE"]
LIN_REF_IND = data["LIN_REF_IND"]
envelope = data["e"]
envelope_axis = data["e_axis"]

## Initialize physical and mathematical constants
ELEC_PERMITTIVITY_0 = 8.8541878128e-12
LIGHT_SPEED_0 = 299792458

# Set up parameters
N_RADI_NODES = envelope.shape[0]
N_DIST_NODES = envelope_axis.shape[0]
N_TIME_NODES = envelope_axis.shape[1]

## Choose the computational domain which you would like to graph
RADI_COOR_A, RADI_COOR_B = INI_RADI_COOR, FIN_RADI_COOR / 20
DIST_COOR_A, DIST_COOR_B = INI_DIST_COOR, FIN_DIST_COOR
TIME_COOR_A, TIME_COOR_B = -100e-15, 100e-15
# Compute nodes and arrays
RADI_FLOAT_A, RADI_FLOAT_B = (
    (RADI_COOR_A - INI_RADI_COOR)
    * (N_RADI_NODES - 1)
    / (FIN_RADI_COOR - INI_RADI_COOR),
    (RADI_COOR_B - INI_RADI_COOR)
    * (N_RADI_NODES - 1)
    / (FIN_RADI_COOR - INI_RADI_COOR),
)
DIST_FLOAT_A, DIST_FLOAT_B = (
    (DIST_COOR_A - INI_DIST_COOR)
    * (N_DIST_NODES - 1)
    / (FIN_DIST_COOR - INI_DIST_COOR),
    (DIST_COOR_B - INI_DIST_COOR)
    * (N_DIST_NODES - 1)
    / (FIN_DIST_COOR - INI_DIST_COOR),
)
TIME_FLOAT_A, TIME_FLOAT_B = (
    (TIME_COOR_A - INI_TIME_COOR)
    * (N_TIME_NODES - 1)
    / (FIN_TIME_COOR - INI_TIME_COOR),
    (TIME_COOR_B - INI_TIME_COOR)
    * (N_TIME_NODES - 1)
    / (FIN_TIME_COOR - INI_TIME_COOR),
)
RADI_NODE_A, RADI_NODE_B = int(RADI_FLOAT_A), int(RADI_FLOAT_B)
DIST_NODE_A, DIST_NODE_B = int(DIST_FLOAT_A), int(DIST_FLOAT_B)
TIME_NODE_A, TIME_NODE_B = int(TIME_FLOAT_A), int(TIME_FLOAT_B)
PEAK_NODE = -int(TIME_COOR_A * (N_TIME_NODES - 1) // (FIN_TIME_COOR - INI_TIME_COOR))
slice_nodes_r = slice(RADI_NODE_A, RADI_NODE_B + 1)
slice_nodes_z = slice(DIST_NODE_A, DIST_NODE_B + 1)
slice_nodes_t = slice(TIME_NODE_A, TIME_NODE_B + 1)
radi_array = np.linspace(INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES)
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_DIST_NODES)
time_array = np.linspace(INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES)
radi_array = radi_array[slice_nodes_r]
dist_array = dist_array[slice_nodes_z]
time_array = time_array[slice_nodes_t]
envelope = envelope[slice_nodes_r, slice_nodes_t]
envelope_axis = envelope_axis[slice_nodes_z, slice_nodes_t]
radi_2d_array_2, time_2d_array_2 = np.meshgrid(radi_array, time_array, indexing="ij")
dist_2d_array_3, time_2d_array_3 = np.meshgrid(dist_array, time_array, indexing="ij")

### Plots
plt.style.use("dark_background")
cmap_option = mpl.colormaps["plasma"]
figsize_option = (13, 7)

# Set up conversion factors
INT_FACTOR = 1
# INT_FACTOR = 0.5 * LIGHT_SPEED_0 * ELEC_PERMITTIVITY_0 * LIN_REF_IND
RADI_FACTOR = 1e6
DIST_FACTOR = 100
TIME_FACTOR = 1e15
AREA_FACTOR = 1e-4
# Set up plotting grid (µm, cm and fs)
new_radi_2d_array_2 = RADI_FACTOR * radi_2d_array_2
new_dist_2d_array_3 = DIST_FACTOR * dist_2d_array_3
new_time_2d_array_2 = TIME_FACTOR * time_2d_array_2
new_time_2d_array_3 = TIME_FACTOR * time_2d_array_3
new_dist_array = new_dist_2d_array_3[:, 0]
new_time_array = new_time_2d_array_3[0, :]

# Set up intensities (W/cm^2)
plot_intensity_end = AREA_FACTOR * INT_FACTOR * np.abs(envelope) ** 2
plot_intensity_axis = AREA_FACTOR * INT_FACTOR * np.abs(envelope_axis) ** 2

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
    linestyle="-",
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
