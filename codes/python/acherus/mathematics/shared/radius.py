"""Beam radius module."""

import numpy as np


def compute_radius(flu_a, rad_a=None, r_g_a=None):
    """
    Compute the beam radius (HWHM of fluence distribution)
    at the current step.

    Parameters
    ----------
    flu_a : (M,) array_like
        Fluence distribution at current propagation step.
    rad_a : float
        Beam radius at current propagation step (output).
    r_g_a : (M,) array_like
        Radial coordinates grid.

    Returns
    -------
    hwhm : float
        Beam radius (HWHM) at current propagation step.

    """
    half_max = 0.5 * np.max(flu_a)
    indices = np.where(flu_a >= half_max)[0]

    if len(indices) == 0:
        print(f"Warning: HWHM not found!")
        return np.nan

    hwhm_idx = indices[-1]
    hwhm = r_g_a[hwhm_idx]

    if rad_a is not None:
        rad_a[0] = hwhm

    return hwhm
