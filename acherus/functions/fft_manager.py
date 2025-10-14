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
the `GPU` backend was chosen, this will indicate that
the `CuPy` backend is available for FFT computations, which
has been set to use the `cufft` backend. If `CuPy` is not
available, it sets `GPU` to `CPU`.

3. `compute_fft` and `compute_ifft` functions are the user's
API functions which perform the required FFT or IFFT. By default,
they use the second dimension (axis=1) for the transform, but
the axis can be specified as an argument if needed.

4. In the end, if the backend is `GPU` this means that the
`CuPy` library is available, and the user wants to use
GPU-acceleration, so the FFT/IFFT is computed using the
`cufft` routines. Otherwise, `CPU` is used, and the
FFT/IFFT is computed using pyFFTW's `fft` and `ifft`
functions with the `workers=-1` argument, which allows
parallelization across all available CPU cores.

"""

import scipy.fft


class FFTManager:
    """Fast Fourier Transform options configuration."""

    def __init__(self):
        self.compute_fft = None
        self.compute_ifft = None
        self._fft_plan_cache = {}

    def set_fft_backend(self, backend: str):
        """FFT backend configuration"""
        backend_opt = backend
        if backend_opt == "GPU":
            try:
                import cupyx.scipy.fft as cufft

                scipy.fft.set_global_backend(cufft)
                self.compute_fft = lambda data, axis=1: scipy.fft.fft(data, axis=axis)
                self.compute_ifft = lambda data, axis=1: scipy.fft.ifft(data, axis=axis)
                print("Using GPU FFT with CuPy")
            except ImportError:
                print("CuPy not available. Falling back to CPU with pyFFTW")
                self._setup_fft_cpu()
        else:
            self._setup_fft_cpu()

    def _setup_fft_cpu(self):
        try:
            import pyfftw.interfaces.scipy_fft as fftw_backend

            scipy.fft.set_global_backend(fftw_backend)
            print("Using pyFFTW with SciPy backend")

            self.compute_fft = lambda data, axis=1: scipy.fft.fft(
                data, axis=axis, workers=-1
            )
            self.compute_ifft = lambda data, axis=1: scipy.fft.ifft(
                data, axis=axis, workers=-1
            )

        except ImportError:
            print("pyFFTW not available. Falling back to SciPy")
            self.compute_fft = lambda data, axis=1: scipy.fft.fft(
                data, axis=axis, workers=-1
            )
            self.compute_ifft = lambda data, axis=1: scipy.fft.ifft(
                data, axis=axis, workers=-1
            )

    def fft(self, data, axis=1):
        """
        Transform data from time domain to frequency domain
        using FFT (cufft if available, scipy.fft otherwise).
        """
        if not self.compute_fft:
            raise RuntimeError("FFT backend not set up yet.")
        return self.compute_fft(data, axis=axis)

    def ifft(self, data, axis=1):
        """
        Transform data from frequency domain to time domain
        using FFT (cufft if available, scipy.fft otherwise).
        """
        if not self.compute_ifft:
            raise RuntimeError("FFT backend not set up yet.")
        return self.compute_ifft(data, axis=axis)
