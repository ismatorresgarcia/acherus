"""Fluence distribution module for cylindrical NEE simulations."""

import numpy as np
from scipy.integrate import trapezoid


def compute_fluence(env, flu=None, dt=None):
    """
    Compute fluence distribution for the current step.

    Parameters
    ----------
    env : (M, N) array_like
        Complex envelope at current propagation step.
    flu : (M,) array_like
        Fluence at current propagation step (output).
    dt : float
        Time step.

    Returns
    -------
    fluence : (M, N) ndarray
        Fluence distribution at current propagation step

    """
    env_2 = np.abs(env) ** 2
    fluence = trapezoid(env_2, dx=dt, axis=1)

    if flu is not None:
        flu[:] = fluence

    return fluence
