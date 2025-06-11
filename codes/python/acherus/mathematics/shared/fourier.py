"""Fourier FFT algorithms module."""

from scipy.fft import fft, ifft


def compute_fft(data):
    """Transform data from time domain to frequency domain using FFT."""
    return fft(data, axis=1, workers=-1)


def compute_ifft(data):
    """Transform data from frequency domain to time domain using IFFT."""
    return ifft(data, axis=1, workers=-1)
