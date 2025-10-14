"""Fast Fourier Transform shared instance and method exposure."""
from .fft_manager import FFTManager

fft_manager = FFTManager()

def fft(data, axis=1):
    """Exposed method for using the FFT."""
    return fft_manager.fft(data, axis=axis)

def ifft(data, axis=1):
    """Exposed method for using the IFFT."""
    return fft_manager.ifft(data, axis=axis)
