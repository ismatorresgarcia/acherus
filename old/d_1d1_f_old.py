"""
This program solves the 1D propagation equation of a laser pulse
due to second order group velocity dispersion (GVD).

Numerical discretization: Finite Differences Method (FDM).
    - Method: Fast Fourier Transform (FFT).
    - Initial condition: Gaussian.
    - Boundary conditions: Periodic.

UPPE:           ∂ℰ/∂z = -ik''/2 ∂²E/∂t²


E: envelope.
i: imaginary unit.
z: distance coordinate.
t: time coordinate.
k'': GVD coefficient of 2nd order.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from tqdm import tqdm


def initial_condition(t, im, beam):
    """
    Set the chirped Gaussian beam.

    Parameters:
    - t (array): time array
    - im (complex): square root of -1
    - beam (dict): dictionary containing the beam parameters
        - a (float): amplitude of the Gaussian beam
        - pt (float): time when the Gaussian beam reaches its peak intensity
        - ch (float): chirp of the Gaussian beam introduced by some optical system

    Returns:
    - array: Gaussian beam envelope's initial condition
    """
    a = beam["AMPLITUDE"]
    pt = beam["PEAK_TIME"]
    ch = beam["CHIRP"]
    gaussian = a * np.exp(-(1 + im * ch) * (t / pt) ** 2)

    return gaussian


def fft_step(fc, e_c, e_n):
    """
    Compute one step of the FFT propagation scheme.

    Parameters:
    - fc: precomputed Fourier coefficient
    - e_c: envelope at step k
    - e_n: pre-allocated array for envelope at step k + 1
    """
    e_n[:] = ifft(fc * fft(e_c))


IM_UNIT = 1j
PI = np.pi

## Set parameters (grid spacing, propagation step, etc.)
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0, 5e-2, 1000
DIST_STEP_LEN = (FIN_DIST_COOR - INI_DIST_COOR) / N_STEPS
# Time (t) grid
INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES = -300e-15, 300e-15, 2048
TIME_STEP_LEN = (FIN_TIME_COOR - INI_TIME_COOR) / (N_TIME_NODES - 1)
PEAK_NODE = N_TIME_NODES // 2
# Angular frequency (ω) grid
FRQ_STEP_LEN = 2 * PI / (N_TIME_NODES * TIME_STEP_LEN)
INI_FRQ_COOR_W1 = 0
FIN_FRQ_COOR_W1 = PI / TIME_STEP_LEN - FRQ_STEP_LEN
INI_FRQ_COOR_W2 = -PI / TIME_STEP_LEN
FIN_FRQ_COOR_W2 = -FRQ_STEP_LEN
w1 = np.linspace(INI_FRQ_COOR_W1, FIN_FRQ_COOR_W1, N_TIME_NODES // 2)
w2 = np.linspace(INI_FRQ_COOR_W2, FIN_FRQ_COOR_W2, N_TIME_NODES // 2)
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_STEPS + 1)
time_array = np.linspace(INI_TIME_COOR, FIN_TIME_COOR, N_TIME_NODES)
frq_array = np.append(w1, w2)
dist_2d_array, time_2d_array = np.meshgrid(dist_array, time_array, indexing="ij")

## Set beam and media parameters
LIGHT_SPEED = 299792458
PERMITTIVITY = 8.8541878128e-12
LIN_REF_IND_WATER = 1.334
GVD_COEF_WATER = 241e-28

WAVELENGTH_0 = 800e-9
WAIST_0 = 75e-6
PEAK_TIME = 130e-15
ENERGY = 2.2e-6
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
    "CHIRP": CHIRP,
    "WAVENUMBER_0": WAVENUMBER_0,
    "WAVENUMBER": WAVENUMBER,
    "POWER": POWER,
    "INTENSITY": INTENSITY,
    "AMPLITUDE": AMPLITUDE,
}

## Set loop variables
DELTA_T = -0.25 * DIST_STEP_LEN * MEDIA["WATER"]["GVD_COEF"] / TIME_STEP_LEN**2
current_envelope = np.empty([N_STEPS + 1, N_TIME_NODES], dtype=complex)
next_envelope = np.empty(N_TIME_NODES, dtype=complex)
fourier_coeff = np.exp(-2 * IM_UNIT * DELTA_T * (frq_array * TIME_STEP_LEN) ** 2)

## Set initial electric field wave packet
current_envelope[0, :] = initial_condition(time_array, IM_UNIT, BEAM)

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    fft_step(fourier_coeff, current_envelope[k, :], next_envelope)
    current_envelope[k + 1, :] = next_envelope

## Analytical solution for a Gaussian beam
# Set arrays
envelope_sol = np.empty_like(current_envelope)

# Set variables
DISPERSION_LEN = 0.5 * BEAM["PEAK_TIME"] ** 2 / MEDIA["WATER"]["GVD_COEF"]
beam_duration = BEAM["PEAK_TIME"] * np.sqrt(
    (1 + BEAM["CHIRP"] * dist_array / DISPERSION_LEN) ** 2
    + (dist_array / DISPERSION_LEN) ** 2
)
gouy_phase = 0.5 * np.atan(-dist_array / (DISPERSION_LEN + BEAM["CHIRP"] * dist_array))
#
sqrt_term = np.sqrt(BEAM["PEAK_TIME"] / beam_duration[:, np.newaxis])
decay_term = (time_array / beam_duration[:, np.newaxis]) ** 2
prop_term = 1 + IM_UNIT * (
    BEAM["CHIRP"]
    + (1 + BEAM["CHIRP"] ** 2) * (dist_array[:, np.newaxis] / DISPERSION_LEN)
)
gouy_term = IM_UNIT * gouy_phase[:, np.newaxis]

# Compute solution
envelope_sol = (
    BEAM["AMPLITUDE"] * sqrt_term * np.exp(-decay_term * prop_term - gouy_term)
)

### Plots
plt.style.use("dark_background")
cmap_option = mpl.colormaps["plasma"]
figsize_option = (13, 7)

# Set up conversion factors
DIST_FACTOR = 100
TIME_FACTOR = 1e15
AREA_FACTOR = 1e-4
# Set up plotting grid (cm, s)
dist_2d_array = DIST_FACTOR * dist_2d_array
time_2d_array = TIME_FACTOR * time_2d_array
dist_array = dist_2d_array[:, 0]
time_array = time_2d_array[0, :]

## Set up intensities (W/cm^2)
plot_int = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(current_envelope) ** 2
plot_int_sol = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_sol) ** 2

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
intensity_list = [
    (
        plot_int_sol[0, :],
        "#FF00FF",  # Magenta
        "-",
        r"Analytical solution at beginning $z$ step",
    ),
    (
        plot_int_sol[-1, :],
        "#FFFF00",  # Pure yellow
        "-",
        r"Analytical solution at final $z$ step",
    ),
    (
        plot_int[0, :],
        "#32CD32",  # Lime green
        "--",
        r"Numerical solution at beginning $z$ step",
    ),
    (
        plot_int[-1, :],
        "#1E90FF",  # Electric blue
        "--",
        r"Numerical solution at final $z$ step",
    ),
]
for data, color, style, label in intensity_list:
    ax1.plot(time_array, data, color, linestyle=style, linewidth=2, label=label)
ax1.set(xlabel=r"$t$ ($\mathrm{fs}$)", ylabel=r"$I(t)$ ($\mathrm{W/{cm}^2}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax2.plot(
    dist_array,
    plot_int_sol[:, PEAK_NODE],
    "#FF00FF",  # Magenta
    linestyle="-",
    linewidth=2,
    label="Peak time analytical solution",
)
ax2.plot(
    dist_array,
    plot_int[:, PEAK_NODE],
    "#32CD32",  # Lime green
    linestyle="--",
    linewidth=2,
    label="Peak time numerical solution",
)
ax2.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
ax2.legend(facecolor="black", edgecolor="white")

fig1.tight_layout()
plt.show()

## Set up figure 2
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize_option)
# Subplot 1
fig2_1 = ax3.pcolormesh(dist_2d_array, time_2d_array, plot_int, cmap=cmap_option)
fig2.colorbar(fig2_1, ax=ax3)
ax3.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
ax3.set_title("Numerical solution in 2D")
# Subplot 2
fig2_2 = ax4.pcolormesh(dist_2d_array, time_2d_array, plot_int_sol, cmap=cmap_option)
fig2.colorbar(fig2_2, ax=ax4)
ax4.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
ax4.set_title("Analytical solution in 2D")

fig2.tight_layout()
plt.show()

## Set up figure 3
fig3, (ax5, ax6) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# Subplot 1
ax5.plot_surface(
    dist_2d_array,
    time_2d_array,
    plot_int,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax5.set(
    xlabel=r"$z$ ($\mathrm{cm}$)",
    ylabel=r"$t$ ($\mathrm{fs}$)",
    zlabel=r"$I(z,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax5.set_title("Numerical solution")
# Subplot 2
ax6.plot_surface(
    dist_2d_array,
    time_2d_array,
    plot_int_sol,
    cmap=cmap_option,
    linewidth=0,
    antialiased=False,
)
ax6.set(
    xlabel=r"$z$ ($\mathrm{cm}$)",
    ylabel=r"$t$ ($\mathrm{fs}$)",
    zlabel=r"$I(z,t)$ ($\mathrm{W/{cm}^2}$)",
)
ax6.set_title("Analytical solution")

# fig3.tight_layout()
plt.show()
