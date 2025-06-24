"""Intensity module."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def compute_intensity(env_a, inten_a, r_a, t_a):
    """
    Compute the intensity interpolated object
    for the current step.

    Parameters
    ----------
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    r_a : (M,) array_like
        Radial coordinates grid.
    t_a : (N,) array_like
        Time coordinates grid.

    Returns
    -------
    intensity : function
        Interpolation function for intensity at current propagation step.

    """
    inten_a[:] = np.abs(env_a) ** 2
    intensity_function = RegularGridInterpolator(
        (r_a, t_a),
        inten_a,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    return intensity_function
