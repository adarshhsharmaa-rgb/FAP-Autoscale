"""
models/load_pipeline.py  (Person A)  --  THE interface Person C calls.

    from models.load_pipeline import classify_and_forecast, forecast_with_model, warm_up

    warm_up()                                  # once, before the control loop
    out = classify_and_forecast(window)        # every 60 s tick
    out["pattern"]         -> "periodic" | "bursty" | "hybrid"
    out["predicted_load"]  -> float  (load expected in the next control window = L)
    out["model_used"]      -> "ARIMA" | "LSTM" | "XGBoost"

    # LSTM-only baseline (Person C, simulator/baselines.py):
    forecast_with_model(window, "LSTM")["predicted_load"]
"""
from __future__ import annotations

import time

import numpy as np

from models.forecasters import FORECASTERS, get_lstm
from models.pattern_classifier import classify_pattern, get_trained_tree

# Section IV-A routing table
ROUTING = {
    "periodic": "ARIMA",
    "bursty": "LSTM",
    "hybrid": "XGBoost",
}

MIN_WINDOW = 48


def _validate(window) -> np.ndarray:
    x = np.asarray(window, dtype=float).ravel()
    if len(x) < MIN_WINDOW:
        raise ValueError(f"window must have >= {MIN_WINDOW} points, got {len(x)}")
    if not np.all(np.isfinite(x)):
        # forward-fill gaps instead of crashing the control loop
        mask = np.isfinite(x)
        idx = np.where(mask, np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        x = x[idx]
    return x


def _summarise(fc: np.ndarray) -> float:
    # horizon 1 -> the next value; horizon > 1 -> the peak (safer for scaling)
    return float(fc[0]) if len(fc) == 1 else float(np.max(fc))


def forecast_with_model(window, model_name: str, horizon: int = 1, **kwargs) -> dict:
    """Run one specific forecaster, no classification (used by baselines + evaluation)."""
    x = _validate(window)
    if model_name not in FORECASTERS:
        raise ValueError(f"model_name must be one of {list(FORECASTERS)}")
    t0 = time.perf_counter()
    res = FORECASTERS[model_name](x, horizon=horizon, **kwargs)
    return {
        "predicted_load": _summarise(res["forecast"]),
        "forecast": res["forecast"].tolist(),
        "model_used": model_name,
        "backend": res["backend"],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        **{k: v for k, v in res.items() if k in ("lower", "upper", "period")},
    }


def classify_and_forecast(window, timestamps=None, horizon: int = 1,
                          classifier: str = "rules", fine_tune_epochs: int = 0) -> dict:
    """
    window     : last N load samples (N >= 48; 120-200 recommended)
    timestamps : accepted for interface compatibility (samples are assumed evenly spaced)
    horizon    : control windows ahead to forecast
    classifier : "rules" (explainable) or "tree" (DecisionTree, slightly more accurate)
    fine_tune_epochs : >0 fine-tunes the LSTM online on this window (slower)
    """
    x = _validate(window)
    t0 = time.perf_counter()
    pattern, feats = classify_pattern(x, method=classifier)
    model_name = ROUTING[pattern]

    kwargs = {"fine_tune_epochs": fine_tune_epochs} if model_name == "LSTM" else {}
    out = forecast_with_model(x, model_name, horizon=horizon, **kwargs)

    out.update({
        "pattern": pattern,
        "features": {k: round(float(v), 4) for k, v in feats.items()},
        "last_observed": float(x[-1]),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    })
    return out


def warm_up(classifier: str = "rules") -> None:
    """Load/train the LSTM, import heavy libs and (optionally) train the tree once."""
    from data_gen.synthetic_data import generate_workload_series
    get_lstm()
    if classifier == "tree":
        get_trained_tree()
    for p in ROUTING:
        classify_and_forecast(generate_workload_series(p, 120, seed=0), classifier=classifier)


def format_decision(out: dict) -> str:
    """One-line log string for the live demo."""
    f = out["features"]
    return (f"pattern={out['pattern']:<8s} -> {out['model_used']:<7s} | "
            f"L={out['predicted_load']:7.2f} (last={out['last_observed']:6.2f}) | "
            f"acf={f['acf_peaks']:.0f} wav={f['wavelet_ratio']:.2f} "
            f"cv={f['cv']:.2f} H={f['hurst']:.2f} | {out['latency_ms']:.0f} ms")


if __name__ == "__main__":
    from data_gen.synthetic_data import generate_workload_windows
    print("warming up models...")
    warm_up()
    for w, true_label in generate_workload_windows(6, seed=77):
        out = classify_and_forecast(w[:-1])
        print(f"true={true_label:<8s} actual_next={w[-1]:6.2f} | {format_decision(out)}")
