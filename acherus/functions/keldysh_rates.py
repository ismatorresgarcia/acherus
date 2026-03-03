"""Helper module injected into "keldysh.py" for computing ionization rates."""

import numpy as np
from scipy.special import ellipe, ellipk, gamma  # pylint: disable=no-name-in-module

from ..constants import E_CHARGE as e_charge
from ..constants import HBAR as hbar
from ..constants import M_E as m_e
from ..constants import PHYSICAL_CONSTANTS as phys
from ..constants import PI as pi
from .keldysh_sum import series_sum


def mpi_rate(intensity, photons, cross_section):
    """Compute the MPI rate."""
    k = np.ceil(photons)
    return cross_section * intensity**k


def keldysh_gas_rate(
    intensity,
    field_factor,
    omega,
    energy_gap,
    photons,
    tolerance,
    max_iterations,
):
    """Compute the Keldysh ionization rate for gaseous media."""
    u_hy = 0.5 * phys["Hartree energy"][0]
    w_au = 1 / phys["atomic unit of time"][0]
    f_au = phys["atomic unit of electric field"][0]

    field = np.sqrt(intensity / field_factor)

    f_a = f_au * np.sqrt(energy_gap / u_hy) ** 3
    n_q = 1 / np.sqrt(energy_gap / u_hy)

    g = (omega / (e_charge * field)) * np.sqrt(2 * m_e * energy_gap)
    asinh = np.arcsinh(g)
    idx = 1 + 0.5 / g**2
    nu = photons * idx
    beta = 2 * g / np.sqrt(1 + g**2)
    alpha = 2 * asinh - beta
    exp = 1.5 * (idx * asinh - 1 / beta) / g
    b_momentum = 1 + 0.5 * (g**4 + 4 * g**2 + 1) / (photons * g * (1 + g**2) ** 1.5)
    c_momentum = 0.5 * g**2 * (g**2 - 1) / (photons * (1 + g**2) ** 2)
    alpha += 2 * c_momentum
    beta *= b_momentum

    ion_sum = series_sum(alpha, beta, nu, "gas", tolerance, max_iterations)

    return (
        4
        * (16 / 3)
        * ((2 * g**2 + 3) / (1 + g**2))
        * w_au
        * (2 ** (2 * n_q) / (gamma(n_q + 1)) ** 2)
        * (0.5 * energy_gap / u_hy)
        * (4 * np.sqrt(2) / pi)
        * (g**2 / (1 + g**2))
        / np.sqrt(b_momentum)
        * ion_sum
        * (2 * f_a / (field * np.sqrt(1 + g**2))) ** (2 * n_q - 1.5)
        * np.exp(-2 * f_a * exp / (3 * field))
    )


def keldysh_condensed_rate(
    intensity,
    field_factor,
    omega,
    energy_gap,
    photons,
    reduced_mass,
    neutral_density,
    tolerance,
    max_iterations,
):
    """Compute the Keldysh ionization rate for condensed media."""
    field = np.sqrt(intensity / field_factor)

    g = (omega / (e_charge * field)) * np.sqrt(reduced_mass * m_e * energy_gap)
    g_2 = 1 / (1 + g**2)
    g_1 = g_2 * g**2
    ek_1 = ellipk(g_1)
    ek_2 = ellipk(g_2)
    ee_1 = ellipe(g_1)
    ee_2 = ellipe(g_2)

    alpha = pi * (ek_1 - ee_1) / ee_2
    beta = 0.25 * pi**2 / (ek_2 * ee_2)
    x = (2 / pi) * photons * ee_2 / np.sqrt(g_1)

    ion_sum = series_sum(alpha, beta, x, "condensed", tolerance, max_iterations)

    return (
        2
        * omega
        / (9 * pi)
        * (omega * reduced_mass * m_e / (hbar * np.sqrt(g_1))) ** 1.5
        * np.sqrt(0.5 * pi / ek_2)
        * ion_sum
        * np.exp(-alpha * np.floor(x + 1))
    ) / neutral_density
