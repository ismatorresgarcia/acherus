"""Fluence distribution module for cylindrical NEE simulations."""

import numpy as np
from scipy.integrate import trapezoid


def compute_fluence(int, flu=None, dt=None):
    """
    Compute fluence distribution for the current step.

    Parameters
    ----------
    int : (M, N) array_like
        Laser field intensity at current propagation step.
    flu : (M,) array_like
        Fluence at current propagation step (output).
    dt : float
        Time step.

    Returns
    -------
    fluence : (M, N) ndarray
        Fluence distribution at current propagation step

    """
    fluence = trapezoid(int, dx=dt, axis=1)

    if flu is not None:
        flu[:] = fluence

    return fluence
