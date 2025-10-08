"""
Photo-ionization rate interpolation module.

How this works
--------------

1. The module first checks what ionization model
has been requested, either "MPI" for multiphoton
ionization or "PPT" for the generalized PPT model.

2. If the "MPI" option was selected, it updates
in-place the ionization rate as a power of the intensity
multiplied by the multiphoton ionization coefficient,
according to the PPT model in the low intensity limit.

3. If the "PPT" option was selected, it takes the
two 1D arrays `rates_ppt` and `peak_inten` which
contain the peak intensity values and their corresponding
ionization rates computed in the `ppt_rate` module.

4. Then, it interpolates the two datasets to generate an
interpolating function, which is used to compute
the ionization rate for the given 2D intensity values. 
The interpolated ionization rates for this given 2D 
intensity array are updated in-place.

"""

from scipy.interpolate import interp1d


def compute_ionization(
    inten_a,
    ionz_rate_a,
    n_k_a,
    mpi_a,
    ion_model="MPI",
    peak_inten=None,
    ppt_rate=None,
):
    """
    Compute the ionization rates from the "MPI" or "PPT" models.

    Parameters
    ----------
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    ionz_rate_a : (M, N) array_like
        Pre-allocated ionization rate array.
    n_k_a : integer
        Number of photons required for MPI.
    mpi_a : float
        MPI "cross-section" coefficient.
    ion_model : str, default: "MPI"
        Ionization model to use, "MPI" or "PPT".
    peak_intensity : (K,) array_like, optional
        Peak intensity values for PPT model.
    ppt_rate : (K,) array_like, optional
        Ionization rates for PPT model.

    """
    if ion_model == "MPI":
        ionz_rate_a[:] = mpi_a * inten_a**n_k_a

    elif ion_model == "PPT":
        if peak_inten is None or ppt_rate is None:
            raise ValueError(
                "Both `peak_intensity` and `ppt_rate` must be"
                "given for computing PPT rates."
            )

        rates_interpolator = interp1d(
            peak_inten,
            ppt_rate,
            kind="linear",
            fill_value="extrapolate",
        )
        ionz_rate_a[:] = rates_interpolator(inten_a)
