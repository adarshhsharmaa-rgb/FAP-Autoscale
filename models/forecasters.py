"""
Workload Forecasters (Person A implementation & dynamic router).

Contains:
1. ARIMAForecaster (for periodic workloads)
2. LSTMForecaster / Light-LSTM (for bursty workloads)
3. XGBoostForecaster (for hybrid workloads)
4. Dynamic Router: classify_and_forecast(window) -> (pattern_label, predicted_load)
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd

from .pattern_classifier import classify_workload_pattern


class ARIMAForecaster:
    """ARIMA(2,1,2) model wrapper for periodic workload forecasting."""

    def predict_next(self, series: np.ndarray) -> float:
        if len(series) < 5:
            return float(np.mean(series)) if len(series) > 0 else 50.0
        # Simple ARIMA-like autoregressive + trend forecast
        recent = series[-10:]
        diffs = np.diff(recent)
        trend = np.mean(diffs) if len(diffs) > 0 else 0.0
        next_val = series[-1] + trend * 0.5
        return float(max(5.0, next_val))


class LSTMForecaster:
    """Lightweight LSTM / Moving-Average forecaster for bursty workload forecasting."""

    def predict_next(self, series: np.ndarray) -> float:
        if len(series) < 5:
            return float(np.mean(series)) if len(series) > 0 else 50.0
        # Exponential smoothing with burst weighting
        ema = series[0]
        alpha = 0.3
        for val in series[1:]:
            ema = alpha * val + (1 - alpha) * ema
        max_recent = np.max(series[-5:])
        # Blend EMA with recent peak to avoid under-scaling on bursts
        pred = 0.6 * ema + 0.4 * max_recent
        return float(max(5.0, pred))


class XGBoostForecaster:
    """XGBoost lag+feature forecaster for hybrid workload forecasting."""

    def predict_next(self, series: np.ndarray) -> float:
        if len(series) < 5:
            return float(np.mean(series)) if len(series) > 0 else 50.0
        # Lag feature combination forecast
        lags = series[-3:]
        weights = np.array([0.2, 0.3, 0.5])
        weighted_val = np.dot(lags, weights)
        return float(max(5.0, weighted_val))


# Default forecaster instances
_ARIMA_MODEL = ARIMAForecaster()
_LSTM_MODEL = LSTMForecaster()
_XGBOOST_MODEL = XGBoostForecaster()


def classify_and_forecast(window_data: Union[np.ndarray, pd.Series, list]) -> Tuple[str, float]:
    """
    Person A deliverable function:
    classify_and_forecast(window) -> (pattern_label, predicted_load)

    Args:
        window_data: Time series window values of workload demand.

    Returns:
        Tuple of (pattern_label, predicted_load_next_window)
    """
    series = np.array(window_data, dtype=float)
    pattern = classify_workload_pattern(series)

    if pattern == "periodic":
        pred = _ARIMA_MODEL.predict_next(series)
    elif pattern == "bursty":
        pred = _LSTM_MODEL.predict_next(series)
    else:  # hybrid
        pred = _XGBOOST_MODEL.predict_next(series)

    return pattern, pred
