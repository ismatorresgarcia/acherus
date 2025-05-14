"""Beam radius module."""

import numpy as np


def calculate_radius(flu, rad=None, r_g=None):
    """
    Calculate the beam radius (HWHM of fluence
    distribution) at the current step.

    Parameters:
    -> flu: fluence at current propagation step
    -> rad: beam radius at current propagation step (output)
    -> r_g: radial coordinates array

    Returns:
    -> float: Beam radius at current propagation step
    """
    peak_idx = np.argmax(flu)
    half_peak = 0.5 * np.max(flu)

    diff = flu - half_peak

    idx_range = diff[peak_idx + 1 : -1] * diff[peak_idx + 2 :]
    idx_change = np.where(idx_range <= 0)[0]

    if len(idx_change) == 0:
        return r_g[-1]

    i_1 = peak_idx + 1 + idx_change[0]
    i_2 = i_1 + 1

    r_1, r_2 = r_g[i_1], r_g[i_2]
    f_1, f_2 = flu[i_1], flu[i_2]

    hwhm = r_1 + (half_peak - f_1) * (r_2 - r_1) / (f_2 - f_1)

    if rad is not None:
        rad[0] = hwhm

    return hwhm
