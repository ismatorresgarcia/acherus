"""Intensity module."""

import numpy as np


def compute_intensity(env_a, inten_a):
    """
    Compute the intensity for the current step.

    Parameters
    ----------
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    inten_a : (M, N) array_like
        Intensity at current propagation step (output).

    Returns
    -------
    intensity : (M, N) ndarray
        Intensity at current propagation step

    """
    inten_a[:] = np.abs(env_a) ** 2
