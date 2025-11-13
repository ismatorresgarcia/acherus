"""Helper module for computing fluence distribution."""

import numpy as np
from scipy.integrate import simpson


def compute_fluence(env_a, t_g_a, flu_a=None):
    """
    Compute fluence distribution for the current step.

    Parameters
    ----------
    env_a : (M, N) array_like
        Complex envelope at current propagation step.
    t_g_a : float
        Time coordinates grid.
    flu_a : (M,) array_like
        Fluence at current propagation step (output).

    Returns
    -------
    fluence : (M, N) ndarray
        Fluence distribution at current propagation step

    """
    fluence = simpson(np.abs(env_a) ** 2, x=t_g_a, axis=1)

    if flu_a is None:
        return fluence

    if flu_a is not None:
        flu_a[:] = fluence
