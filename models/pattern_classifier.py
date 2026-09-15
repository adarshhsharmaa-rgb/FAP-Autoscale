"""
Workload Pattern Classifier (Person A skeleton & implementation).

Computes 4 time-series features:
1. Autocorrelation peak count
2. Wavelet energy ratio
3. Coefficient of Variation (CV)
4. Hurst exponent

Classifies workload into 'periodic', 'bursty', or 'hybrid'.
"""

from typing import Union
import numpy as np
import pandas as pd


def compute_autocorr_peaks(series: np.ndarray) -> int:
    """Compute number of prominent autocorrelation peaks."""
    if len(series) < 10:
        return 0
    # Demean series
    norm_series = series - np.mean(series)
    autocorr = np.correlate(norm_series, norm_series, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]

    # Simple peak detection
    peaks = 0
    for i in range(1, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1] and autocorr[i] > 0.3:
            peaks += 1
    return peaks


def compute_cv(series: np.ndarray) -> float:
    """Compute Coefficient of Variation (std / mean)."""
    mean = np.mean(series)
    if abs(mean) < 1e-6:
        return 0.0
    return float(np.std(series) / abs(mean))


def compute_hurst_exponent(series: np.ndarray) -> float:
    """Compute simplified Hurst exponent estimation."""
    if len(series) < 20:
        return 0.5
    lags = range(2, min(20, len(series) // 2))
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    if any(t <= 0 for t in tau):
        return 0.5
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(np.clip(poly[0] * 2.0, 0.0, 1.0))


def compute_wavelet_energy(series: np.ndarray) -> float:
    """Compute energy ratio across low/high frequency components."""
    if len(series) < 8:
        return 0.5
    fft_vals = np.abs(np.fft.fft(series))
    mid = len(fft_vals) // 2
    low_energy = np.sum(fft_vals[:mid] ** 2)
    total_energy = np.sum(fft_vals ** 2)
    if total_energy == 0:
        return 0.5
    return float(low_energy / total_energy)


def classify_workload_pattern(window_data: Union[np.ndarray, pd.Series, list]) -> str:
    """
    Person A deliverable function:
    Classify input workload window data into pattern: 'periodic', 'bursty', or 'hybrid'.

    Args:
        window_data: Time series array or Series of request rates / cpu demand.

    Returns:
        pattern_label: 'periodic', 'bursty', or 'hybrid'
    """
    series = np.array(window_data, dtype=float)
    if len(series) == 0:
        return "periodic"

    cv = compute_cv(series)
    peaks = compute_autocorr_peaks(series)
    hurst = compute_hurst_exponent(series)
    wavelet_ratio = compute_wavelet_energy(series)

    # Classification logic using all 4 features
    # High CV + few autocorr peaks + low hurst = bursty (random spikes, no long-range structure)
    if cv > 0.4 and peaks < 2 and hurst < 0.55:
        return "bursty"
    # Multiple autocorr peaks + low CV + high wavelet low-freq energy = periodic
    elif peaks >= 2 and cv < 0.35 and wavelet_ratio > 0.6:
        return "periodic"
    # Strong long-range correlation alone also indicates periodic
    elif hurst > 0.65 and peaks >= 1:
        return "periodic"
    else:
        return "hybrid"
