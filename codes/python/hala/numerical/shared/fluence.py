"""Fluence distribution module for cylindrical NEE simulations."""

import numpy as np
from scipy.integrate import trapezoid


def calculate_fluence(env, flu=None, dt=None):
    """
    Calculate fluence distribution for the current step.

    Parameters:
    - env: envelope at current propagation step
    - flu: fluence at current propagation step (output)
    - dt: time step

    Returns:
    - Fluence array
    """
    env_2 = np.abs(env) ** 2
    fluence = trapezoid(env_2, dx=dt, axis=1)

    if flu is not None:
        flu[:] = fluence

    return fluence
