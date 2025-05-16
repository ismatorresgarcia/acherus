"""Module for laser pulse initial condition for a Gaussian or
super-Gaussian beams in cylindrical coordinates."""

import numpy as np


def initialize_envelope(grid, laser):
    """
    Compute the initial condition for an envelope at z = 0.

    Parameters
    ----------
    grid : object
        Contains the grid input parameters.
    laser : object
        Contains the laser input parameters.

    Returns
    -------
    out : (M, N) ndarray
        The initial complex envelope. M is the number of radial
        nodes and N the number of time nodes.
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
