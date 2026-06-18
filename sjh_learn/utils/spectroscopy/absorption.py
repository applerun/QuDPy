"""Absorption-only spectroscopy helpers."""

from .spectra import diagnose_uniform_time_axis, lab_frame_absorption_response, lab_frame_fft_response, safe_complex_ratio

__all__ = [
    "diagnose_uniform_time_axis",
    "lab_frame_absorption_response",
    "lab_frame_fft_response",
    "safe_complex_ratio",
]
