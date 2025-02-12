"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse.
This program includes:
    - Second order group velocity dispersion (GVD).

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


def initial_condition(time, im_unit, beam_parameters):
    """
    Set the chirped Gaussian beam.

    Parameters:
    - time (array): time array
    - im_unit (complex): square root of -1
    - beam_parameters (dict): dictionary containing the beam parameters
        - amplitude (float): amplitude of the Gaussian beam
        - peak_time (float): time when the Gaussian beam reaches its peak intensity
        - chirp (float): chirp of the Gaussian beam introduced by some optical system

    Returns:
    - array: Gaussian beam envelope's initial condition
    """
    amplitude = beam_parameters["AMPLITUDE"]
    peak_time = beam_parameters["PEAK_TIME"]
    chirp = beam_parameters["CHIRP"]
    gaussian_envelope = amplitude * np.exp(
        -(1 + im_unit * chirp) * (time / peak_time) ** 2
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
    next_envelope[:] = ifft(fourier_coefficient * fft(current_envelope))


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
envelope = np.empty_like(dist_2d_array, dtype=complex)
envelope_store = np.empty_like(time_array, dtype=complex)
fourier_coeff = np.exp(-2 * IM_UNIT * DELTA_T * (frq_array * TIME_STEP_LEN) ** 2)

## Set initial electric field wave packet
envelope[0, :] = initial_condition(time_array, IM_UNIT, BEAM)

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    fft_step(fourier_coeff, envelope[k, :], envelope_store)
    envelope[k + 1, :] = envelope_store

## Analytical solution for a Gaussian beam
# Set arrays
envelope_s = np.empty_like(envelope)

# Set variables
DISPERSION_LEN = 0.5 * BEAM["PEAK_TIME"] ** 2 / MEDIA["WATER"]["GVD_COEF"]
beam_duration = BEAM["PEAK_TIME"] * np.sqrt(
    (1 + BEAM["CHIRP"] * dist_array / DISPERSION_LEN) ** 2
    + (dist_array / DISPERSION_LEN) ** 2
)
gouy_phase = 0.5 * np.atan(-dist_array / (DISPERSION_LEN + BEAM["CHIRP"] * dist_array))
#
sqrt_term = np.sqrt(BEAM["PEAK_TIME"] / beam_duration[:, np.newaxis])
decay_exp_term = (time_array / beam_duration[:, np.newaxis]) ** 2
prop_exp_term = 1 + IM_UNIT * (
    BEAM["CHIRP"]
    + (1 + BEAM["CHIRP"] ** 2) * (dist_array[:, np.newaxis] / DISPERSION_LEN)
)
gouy_exp_term = IM_UNIT * gouy_phase[:, np.newaxis]

# Compute solution
envelope_s = (
    BEAM["AMPLITUDE"]
    * sqrt_term
    * np.exp(-decay_exp_term * prop_exp_term - gouy_exp_term)
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
new_dist_2d_array = DIST_FACTOR * dist_2d_array
new_time_2d_array = TIME_FACTOR * time_2d_array
new_dist_array = new_dist_2d_array[:, 0]
new_time_array = new_time_2d_array[0, :]

## Set up intensities (W/cm^2)
plot_int = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope) ** 2
plot_int_s = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_s) ** 2

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
intensity_list = [
    (
        plot_int_s[0, :],
        "#FF00FF",  # Magenta
        "-",
        r"Analytical solution at beginning $z$ step",
    ),
    (
        plot_int_s[-1, :],
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
        "#1E90FF",  # Electric Blue
        "--",
        r"Numerical solution at final $z$ step",
    ),
]
for data, color, style, label in intensity_list:
    ax1.plot(new_time_array, data, color, linestyle=style, linewidth=2, label=label)
ax1.set(xlabel=r"$t$ ($\mathrm{fs}$)", ylabel=r"$I(t)$ ($\mathrm{W/{cm}^2}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax2.plot(
    new_dist_array,
    plot_int_s[:, PEAK_NODE],
    "#FF00FF",  # Magenta
    linestyle="-",
    linewidth=2,
    label="Peak time analytical solution",
)
ax2.plot(
    new_dist_array,
    plot_int[:, PEAK_NODE],
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
    new_dist_2d_array, new_time_2d_array, plot_int, cmap=cmap_option
)
fig2.colorbar(fig2_1, ax=ax3)
ax3.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
ax3.set_title("Numerical solution in 2D")
# Subplot 2
fig2_2 = ax4.pcolormesh(
    new_dist_2d_array, new_time_2d_array, plot_int_s, cmap=cmap_option
)
fig2.colorbar(fig2_2, ax=ax4)
ax4.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$t$ ($\mathrm{fs}$)")
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
    new_dist_2d_array,
    new_time_2d_array,
    plot_int_s,
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
