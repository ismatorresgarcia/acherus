"""
This program solves the Unidirectional Pulse Propagation Equation (UPPE) of an ultra-intense
and ultra-short laser pulse in cylindrical coordinates with radial symmetry.
This program includes:
    - Diffraction (for the radial direction).

Numerical discretization: Finite Differences Method (FDM)
    - Method: Crank-Nicolson (CN) scheme.
    - Initial condition: Gaussian.
    - Boundary conditions: Neumann-Dirichlet.

UPPE:          ∂E/∂z = i/(2k) ∇²E


E: envelope.
i: imaginary unit.
r: radial coordinate.
z: distance coordinate.
k: wavenumber (in the interacting media).
∇²: laplace operator (for the radial direction).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags_array
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def init_gaussian(r, im, beam):
    """
    Set the post-lens chirped Gaussian beam.

    Parameters:
    - r (array): radial array
    - im (complex): square root of -1
    - beam (dict): dictionary containing the beam parameters
        - a (float): amplitude of the Gaussian beam
        - w (float): waist of the Gaussian beam
        - k (float): wavenumber of the Gaussian beam
        - f (float): focal length of the initial lens

    Returns:
    - array: Gaussian beam envelope's initial condition
    """
    a = beam["AMPLITUDE"]
    w = beam["WAIST_0"]
    wn = beam["WAVENUMBER"]
    f = beam["FOCAL_LENGTH"]
    gaussian = a * np.exp(-((r / w) ** 2) - 0.5 * im * wn * r**2 / f)

    return gaussian


def crank_nicolson_diags(n, lr, c):
    """
    Set the three diagonals for the Crank-Nicolson array with centered differences.

    Parameters:
    - n (int): number of radial nodes
    - lr (str): position of the Crank-Nicolson array (left or right)
    - c (float): coefficient for the diagonal elements

    Returns:
    - tuple: upper, main, and lower diagonals
    """
    dc = 1 + 2 * c
    ind = np.arange(1, n - 1)

    diag_m1 = -c * (1 - 0.5 / ind)
    diag_0 = np.full(n, dc)
    diag_p1 = -c * (1 + 0.5 / ind)

    diag_m1 = np.append(diag_m1, [0])
    diag_p1 = np.insert(diag_p1, 0, [0])
    if lr == "LEFT":
        diag_0[0], diag_0[-1] = dc, 1
        diag_p1[0] = -2 * c
    else:
        diag_0[0], diag_0[-1] = dc, 0
        diag_p1[0] = -2 * c

    return diag_m1, diag_0, diag_p1


def crank_nicolson_array(n, lr, c):
    """
    Set the Crank-Nicolson sparse array in CSR format using the diagonals.

    Parameters:
    - n (int): number of radial nodes
    - lr (str): position of the Crank-Nicolson array (left or right)
    - c (float): coefficient for the diagonal elements

    Returns:
    - array: Crank-Nicolson sparse array in CSR format
    """
    diag_m1, diag_0, diag_p1 = crank_nicolson_diags(n, lr, c)

    diags = [diag_m1, diag_0, diag_p1]
    offset = [-1, 0, 1]
    matrix = diags_array(diags, offsets=offset, format="csr")

    return matrix


def solve_diffraction(lm, rm, e_c, step):
    """
    Compute one step of the Crank-Nicolson propagation scheme.

    Parameters:
    - lm: sparse array for left-hand side
    - rm: sparse array for right-hand side
    - e_c: envelope at step k
    - step: current propagation step
    """
    b = rm @ e_c[:, step]
    e_c[:, step + 1] = spsolve(lm, b)


IM_UNIT = 1j
PI = np.pi

## Set parameters (grid spacing, propagation step, etc.)
# Radial (r) grid
INI_RADI_COOR, FIN_RADI_COOR, I_RADI_NODES = 0, 2e-2, 1000
N_RADI_NODES = I_RADI_NODES + 2
RADI_STEP_LEN = (FIN_RADI_COOR - INI_RADI_COOR) / (N_RADI_NODES - 1)
AXIS_NODE = int(-INI_RADI_COOR / RADI_STEP_LEN)  # On-axis node
# Propagation (z) grid
INI_DIST_COOR, FIN_DIST_COOR, N_STEPS = 0, 3, 1000
DIST_STEP_LEN = (FIN_DIST_COOR - INI_DIST_COOR) / N_STEPS
radi_array = np.linspace(INI_RADI_COOR, FIN_RADI_COOR, N_RADI_NODES)
dist_array = np.linspace(INI_DIST_COOR, FIN_DIST_COOR, N_STEPS + 1)
radi_2d_array, dist_2d_array = np.meshgrid(radi_array, dist_array, indexing="ij")

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
DELTA_R = 0.25 * DIST_STEP_LEN / (BEAM["WAVENUMBER"] * RADI_STEP_LEN**2)
current_envelope = np.empty([N_RADI_NODES, N_STEPS + 1], dtype=complex)

## Set tridiagonal Crank-Nicolson matrices in csr_array format
MATRIX_CNT_1 = IM_UNIT * DELTA_R
left_operator = crank_nicolson_array(N_RADI_NODES, "LEFT", MATRIX_CNT_1)
right_operator = crank_nicolson_array(N_RADI_NODES, "RIGHT", -MATRIX_CNT_1)

## Set initial electric field wave packet
current_envelope[:, 0] = init_gaussian(radi_array, IM_UNIT, BEAM)

## Propagation loop over desired number of steps
for k in tqdm(range(N_STEPS)):
    solve_diffraction(left_operator, right_operator, current_envelope, k)

## Analytical solution for a Gaussian beam
# Set arrays
envelope_sol = np.empty_like(current_envelope)

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
decay_term = (radi_array[:, np.newaxis] / beam_waist) ** 2
prop_term = (
    0.5 * IM_UNIT * BEAM["WAVENUMBER"] * radi_array[:, np.newaxis] ** 2 / beam_radius
)
gouy_term = IM_UNIT * gouy_phase[np.newaxis, :]

# Compute solution
envelope_sol = (
    BEAM["AMPLITUDE"] * ratio_term * np.exp(-decay_term + prop_term - gouy_term)
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
radi_2d_array = RADI_FACTOR * radi_2d_array
dist_2d_array = DIST_FACTOR * dist_2d_array
radi_array = radi_2d_array[:, 0]
dist_array = dist_2d_array[0, :]

# Set up intensities (W/cm^2)
plot_int = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(current_envelope) ** 2
plot_int_sol = AREA_FACTOR * MEDIA["WATER"]["INT_FACTOR"] * np.abs(envelope_sol) ** 2

## Set up figure 1
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_option)
# Subplot 1
intensity_list = [
    (
        plot_int_sol[:, 0],
        "#FF00FF",  # Magenta
        "-",
        r"Analytical solution at beginning $z$ step",
    ),
    (
        plot_int_sol[:, -1],
        "#FFFF00",  # Pure yellow
        "-",
        r"Analytical solution at final $z$ step",
    ),
    (
        plot_int[:, 0],
        "#32CD32",  # Lime green
        "--",
        r"Numerical solution at beginning $z$ step",
    ),
    (
        plot_int[:, -1],
        "#1E90FF",  # Electric Blue
        "--",
        r"Numerical solution at final $z$ step",
    ),
]
for data, color, style, label in intensity_list:
    ax1.plot(radi_array, data, color, linestyle=style, linewidth=2, label=label)
ax1.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$I(r)$ ($\mathrm{W/{cm}^2}$)")
ax1.legend(facecolor="black", edgecolor="white")
# Subplot 2
ax2.plot(
    dist_array,
    plot_int_sol[AXIS_NODE, :],
    "#FF00FF",  # Magenta
    linestyle="-",
    linewidth=2,
    label="On-axis analytical solution",
)
ax2.plot(
    dist_array,
    plot_int[AXIS_NODE, :],
    "#32CD32",  # Lime green
    linestyle="--",
    linewidth=2,
    label="On-axis numerical solution",
)
ax2.set(xlabel=r"$z$ ($\mathrm{cm}$)", ylabel=r"$I(z)$ ($\mathrm{W/{cm}^2}$)")
ax2.legend(facecolor="black", edgecolor="white")

fig1.tight_layout()
plt.show()

## Set up figure 2
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize_option)
# Subplot 1
fig2_1 = ax3.pcolormesh(radi_2d_array, dist_2d_array, plot_int, cmap=cmap_option)
fig2.colorbar(fig2_1, ax=ax3)
ax3.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$z$ ($\mathrm{cm}$)")
ax3.set_title("Numerical solution in 2D")
# Subplot 2
fig2_2 = ax4.pcolormesh(radi_2d_array, dist_2d_array, plot_int_sol, cmap=cmap_option)
fig2.colorbar(fig2_2, ax=ax4)
ax4.set(xlabel=r"$r$ ($\mathrm{mm}$)", ylabel=r"$z$ ($\mathrm{cm}$)")
ax4.set_title("Analytical solution in 2D")

fig2.tight_layout()
plt.show()

## Set up figure 3
fig3, (ax5, ax6) = plt.subplots(
    1, 2, figsize=figsize_option, subplot_kw={"projection": "3d"}
)
# Subplot 1
ax5.plot_surface(
    radi_2d_array,
    dist_2d_array,
    plot_int,
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
    radi_2d_array,
    dist_2d_array,
    plot_int_sol,
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
