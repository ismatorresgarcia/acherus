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
    peak_idx = np.argmax(flu_a)
    half_peak = 0.5 * np.max(flu_a)

    delta = flu_a - half_peak

    idx_range = delta[peak_idx + 1 : -1] * delta[peak_idx + 2 :]
    idx_change = np.where(idx_range <= 0)[0]

    if len(idx_change) == 0:
        print(f"Warning: HWHM found for peak at index {peak_idx}.")
        return np.nan

    i_1 = peak_idx + 1 + idx_change[0]
    i_2 = i_1 + 1

    r_1, r_2 = r_g_a[i_1], r_g_a[i_2]
    f_1, f_2 = flu_a[i_1], flu_a[i_2]

    if f_2 != f_1:
        hwhm = r_1 + (half_peak - f_1) * (r_2 - r_1) / (f_2 - f_1)
    else:
        hwhm = r_1

    if rad_a is not None:
        rad_a[0] = hwhm

    return hwhm
