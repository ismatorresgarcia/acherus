"""Module for laser pulse initial condition for a Gaussian or
super-Gaussian beams in cylindrical coordinates."""

import numpy as np


def initialize_envelope(grid, laser):
    """
    Set up the initial envelope at z = 0.

    Parameters:
    - grid: GridParameters object containing simulation parameters
    - laser: LaserPulseParameters object containing laser parameters

    Returns:
    - complex 2D-array: Initial envelope
    """
    space_decaying_term = -(
        (grid.r_grid_2d / laser.params.waist) ** laser.gaussian_order
    )
    time_decaying_term = (
        -(1 + 1j * laser.params.chirp) * (grid.t_grid_2d / laser.params.duration) ** 2
    )

    if laser.params.focal_length != 0:  # phase curvature due to focusing lens
        space_decaying_term = space_decaying_term + 0j
        space_decaying_term -= (
            0.5j * laser.wavenumber * grid.r_grid_2d**2 / laser.params.focal_length
        )

    return laser.amplitude * np.exp(space_decaying_term + time_decaying_term)
