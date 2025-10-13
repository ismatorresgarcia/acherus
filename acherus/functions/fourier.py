"""
Fast Fourier Transform algorithm module.

How this works
--------------

1. The module first tries to import the CuPy library
and the `cufft` routine for GPU-accelerated FFT
computations, and sets it as the global backend.
SciPy's API can be used to perform the FFT operation
with the target backend. Notice that the `GPU` variable
determines initially whether the user wants to use
GPU-acceleration or not.

2. If the `CuPy` library is imported successfully and
the `GPU` engine was chosen, this will indicate that
the `CuPy` backend is available for FFT computations, which
has been set to use the `cufft` backend. If `CuPy` is not
available, it sets `GPU` to `CPU`.

3. `compute_fft` and `compute_ifft` functions are the user's
API functions which perform the required FFT or IFFT. By default,
they use the second dimension (axis=1) for the transform, but
the axis can be specified as an argument if needed.

4. In the end, if the engine is `GPU` this means that the
`CuPy` library is available, and the user wants to use
GPU-acceleration, so the FFT/IFFT is computed using the
`cufft` routines. Otherwise, `CPU` is used, and the
FFT/IFFT is computed using SciPy's `fft` and `ifft`
functions with the `workers=-1` argument, which allows
parallelization across all available CPU cores.

"""

import scipy.fft

from ..config import ConfigOptions

engine = ConfigOptions.computing_engine

try:
    if engine == "GPU":
        import cupyx.scipy.fft as cufft

        scipy.fft.set_global_backend(cufft)
except ImportError:
    engine = "CPU"


def compute_fft(data, axis=1):
    """
    Transform data from time domain to frequency domain
    using FFT (cufft if available, scipy.fft otherwise).
    """
    if engine == "GPU":
        return scipy.fft.fft(data, axis=axis)
    return scipy.fft.fft(data, axis=axis, workers=-1)


def compute_ifft(data, axis=1):
    """
    Transform data from to frequency domain time domain
    using FFT (cufft if available, scipy.fft otherwise).
    """
    if engine == "GPU":
        return scipy.fft.ifft(data, axis=axis)
    return scipy.fft.ifft(data, axis=axis, workers=-1)
