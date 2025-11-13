"""Helper module for interpolating from intensity to ionization rates."""


def compute_ion_rate(inten_a, ion_a, ion_inten_a):
    """
    Compute the ionization rate array for the current step.

    Parameters
    ----------
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    ion_a : (M, N) array_like
        Ionization rate at current propagation step.
    ion_inten_a : object
        Interpolation function from intensity to ionization arrays.

    """
    ion_a[:] = ion_inten_a(inten_a)
