"""Helper module for computing beam radius."""

import numpy as np


def compute_radius(flu_a, r_g_a, rad_a=None):
    """
    Compute the beam radius (HWHM of fluence distribution)
    at the current step.

    Parameters
    ----------
    flu_a : (M,) array_like
        Fluence distribution at current propagation step.
    r_g_a : (M,) array_like
        Radial coordinates grid.
    rad_a : float (optional)
        Beam radius at current propagation step (output).

    Returns
    -------
    hwhm : float
        Beam radius (HWHM) at current propagation step.

    """
    half_max = 0.5 * np.max(flu_a)
    indices = np.where(flu_a >= half_max)[0]

    if len(indices) == 0:
        print("Warning: HWHM not found!")
        return np.nan

    hwhm_idx = indices[-1]
    hwhm = r_g_a[hwhm_idx]

    if rad_a is None:
        return hwhm

    if rad_a is not None:
        rad_a[0] = hwhm
