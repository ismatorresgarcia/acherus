"""Module for laser pulse initial condition for Gaussian or
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
    r_grid_2d, t_grid_2d = np.meshgrid(grid.r_grid, grid.t_grid, indexing="ij")
    exp_r2 = -((r_grid_2d / laser.waist) ** laser.gauss_opt).astype(np.complex128)
    exp_t2 = -((t_grid_2d / laser.duration) ** 2).astype(np.complex128)

    if laser.focal_length != 0:  # phase curvature due to the focusing lens
        exp_r2 -= 0.5j * laser.wavenumber * r_grid_2d**2 / laser.focal_length
    if laser.chirp != 0:  # temporal phase due to the chirping system
        exp_t2 -= 1j * laser.chirp * (t_grid_2d / laser.duration) ** 2

    return np.sqrt(laser.ini_intensity) * np.exp(exp_r2 + exp_t2)
