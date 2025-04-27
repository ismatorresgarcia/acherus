"""Module for laser pulse initial condition for a Gaussian or
super-Gaussian beams in cylindrical coordinates."""

import numpy as np


def initialize_envelope(const, grid, laser):
    """
    Set up the initial envelope at z = 0.

    Parameters:
    - const: Constants object containing physical and mathmatical constants
    - grid: GridParameters object containing simulation parameters

    Returns:
    - complex 2D-array: Initial envelope
    """
    space_decaying_term = -(
        (grid.r_grid_2d / laser.input_waist) ** laser.input_gauss_order
    )
    time_decaying_term = (
        -(1 + const.imaginary_unit * laser.input_chirp)
        * (grid.t_grid_2d / laser.input_duration) ** 2
    )

    if laser.input_focal_length != 0:  # phase curvature due to focusing lens
        space_decaying_term = space_decaying_term + const.imaginary_unit * 0
        space_decaying_term -= (
            0.5
            * const.imaginary_unit
            * laser.input_wavenumber
            * grid.r_grid_2d**2
            / laser.input_focal_length
        )

    return laser.input_amplitude * np.exp(space_decaying_term + time_decaying_term)
