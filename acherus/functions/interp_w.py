"""
Photo-ionization rate interpolation module.

How this works
--------------

1. The module first checks what ionization model
has been requested, either "MPI" for multiphoton
ionization or "PPT" for the generalized Keldysh-PPT
model.

2. If the "MPI" option was selected, it updates
in-place the ionization rate as a power of the intensity
multiplied by the multiphoton ionization coefficient,
according to the PPT model in the low intensity limit.

3. If the "PPT" option was selected, it takes the
interpolating function `inten_ion_a` returned by 
the `keldysh` ionization module.

4. Then, it interpolates element-wise each entry in 
the 2D `inten_a` input array and provided as an output
the corresponding ionization rate `ion_a` 2D array. The 
output is updated in-place.

"""

def compute_ionization(
    inten_a,
    ion_a,
    n_k_a,
    mpi_a,
    ion_model,
    inten_ion_a,
):
    """
    Compute the ionization rates from the "MPI" or "PPT" models.

    Parameters
    ----------
    inten_a : (M, N) array_like
        Intensity at current propagation step.
    ion_a : (M, N) array_like
        Pre-allocated ionization rate array.
    n_k_a : integer
        Number of photons required for MPI.
    mpi_a : float
        MPI "cross-section" coefficient.
    ion_model : str, default: "MPI"
        Ionization model to use, "MPI" or "PPT".
    inten_ion_a : object
        Interpolating function from intensity to
        ionization rates.

    """
    if ion_model == "MPI":
        ion_a[:] = mpi_a * inten_a**n_k_a

    elif ion_model == "PPT":
        ion_a[:] = inten_ion_a(inten_a)
