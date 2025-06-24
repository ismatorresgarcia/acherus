"""
Fast Fourier Transform algorithm module.

How this works
--------------

1. The module first tries to import the `pyfftw`
library utilities needed and available CPU cores.
If `pyfftw` is not available, it falls back to
importing the `fft` and `ifft` functions from
scipy's `fft` module.

2. If the `pyfftw` import is successful, it defines
the number of threads to use and the planning strategy
for the FFTW.

3. Then it defines two wrapper functions `fft` and `ifft`
that use the `_set_fftw_plan` function to create the
optimized FFTW plans for the given array shape, data type,
direction (inverse of direct transform), and axis with
respect to which the transform is applied. The input data
is then copied into the plan's aligned input array and the
plan executed.

4. What does `_set_fftw_plan` do? The idea is keeping a
dictionary (the cache) of FFTW plans for the given array's
arguments (shape, data type, direction, and axis). When the
plan for the requested arguments already exists in the cache,
it will reuse it instead of creating a new one. If the plan
does not exist, it creates a new aligned array and builds a
new FFTW plan with the number of threads and planning strategy
chosen, then caches and returns it.

5. `compute_fft` and `compute_ifft` functions are the user's
API functions which perform the required FFT or IFFT. By default,
they use the second dimension (axis=1) for the transform, but
the axis can be specified as an argument if needed.

"""

try:
    from os import cpu_count

    from pyfftw import builders, empty_aligned

    _FFTW_THREADS = cpu_count() or 1
    _FFTW_PLANNER = "FFTW_MEASURE"

    def fft(data, axis):
        """FFTW plan wrapper function for FFT."""
        plan = _set_fftw_plan(data.shape, data.dtype, "fft", axis)
        plan.input_array[:] = data
        return plan()

    def ifft(data, axis):
        """FFTW plan wrapper function for IFFT."""
        plan = _set_fftw_plan(data.shape, data.dtype, "ifft", axis)
        plan.input_array[:] = data
        return plan()

except ImportError:
    from scipy.fft import fft, ifft


# FFTW plans and aligned arrays are reused for improving
# algorithm performance.
def _set_fftw_plan(shape, dtype, direction, axis):
    args = (shape, dtype, direction, axis)
    if not hasattr(_set_fftw_plan, "cache"):
        _set_fftw_plan.cache = {}
    cache = _set_fftw_plan.cache
    if args in cache:
        return cache[args]
    a = empty_aligned(shape, dtype=dtype)
    if direction == "fft":
        plan = builders.fft(
            a, axis=axis, threads=_FFTW_THREADS, planner_effort=_FFTW_PLANNER
        )
    else:
        plan = builders.ifft(
            a, axis=axis, threads=_FFTW_THREADS, planner_effort=_FFTW_PLANNER
        )
    cache[args] = plan
    return plan


def compute_fft(data, axis=1):
    """
    Transform data from time domain to frequency domain using FFT
    (FFTW if available, with planning/alignment).
    """
    return fft(data, axis=axis)


def compute_ifft(data, axis=1):
    """
    Transform data from frequency domain to time domain using IFFT
    (FFTW if available, with planning/alignment).
    """
    return ifft(data, axis=axis)
