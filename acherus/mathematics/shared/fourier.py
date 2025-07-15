"""
Fast Fourier Transform algorithm module.

How this works
--------------

1. The module first tries to import the CuPy library
and the `cufft` routine for GPU-accelerated FFT
computations, and sets it as the global backend.
SciPy's API can be used to perform the FFT operation
with the target backend.

2. If the `CuPy` library is imported successfully,
it sets the `CUPY` variable to `True`, indicating that
the CuPy backend is available for FFT computations, which
has been set to use the `cufft` backend. If CuPy is not
available, it sets `CUPY` to `False`.

3. `compute_fft` and `compute_ifft` functions are the user's
API functions which perform the required FFT or IFFT. By default,
they use the second dimension (axis=1) for the transform, but
the axis can be specified as an argument if needed.

4. If CuPy is available, which means variable `CUPY` is
`True` it uses SciPy's API to return the FFT/IFFT of the
given data, where the target backend was set to use CuPy's
`cufft`. When `CUPY` is `False` CuPy's import wasn't successful,
and returns the FFT/IFFT using SciPy's `fft` and `ifft`
functions with the `workers=-1` argument, which allows
parallelization across all available CPU cores.

"""

import scipy.fft

try:
    import cupy as xp
    import cupyx.scipy.fft as cufft

    scipy.fft.set_global_backend(cufft)
    CUPY = True
except ImportError:
    CUPY = False


def compute_fft(data, axis=1):
    """
    Transform data from time domain to frequency domain
    using FFT (cufft if available, scipy.fft otherwise).
    """
    if CUPY:
        return scipy.fft.fft(data, axis=axis)
    return scipy.fft.fft(data, axis=axis, workers=-1)


def compute_ifft(data, axis=1):
    """
    Transform data from to frequency domain time domain
    using FFT (cufft if available, scipy.fft otherwise).
    """
    if CUPY:
        return scipy.fft.ifft(data, axis=axis)
    return scipy.fft.ifft(data, axis=axis, workers=-1)
