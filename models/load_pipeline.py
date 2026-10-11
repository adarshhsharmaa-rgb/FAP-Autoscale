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

import os
import time
from collections import deque

import numpy as np

from models.forecasters import FORECASTERS, LSTM_REAL_PATH, get_lstm, use_lstm
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
    if np.ptp(x) == 0:
        # an idle service reports a constant (usually zero) load: nothing to model
        return {
            "predicted_load": float(x[-1]), "forecast": [float(x[-1])] * horizon,
            "model_used": ROUTING["periodic"], "backend": "last value (flat window)",
            "pattern": "periodic",
            "features": {"acf_peaks": 0.0, "wavelet_ratio": 0.0, "cv": 0.0, "hurst": 0.5},
            "last_observed": float(x[-1]),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
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


def warm_up(classifier: str = "rules", real: bool = False) -> str:
    """
    Load/train the LSTM, import heavy libs and (optionally) train the tree once.

    real=True activates the LSTM trained on the GenTD26 request log
    (models/train_real.py) instead of the one trained on synthetic bursty
    series. Returns the name of the LSTM backend in use.
    """
    from data_gen.synthetic_data import generate_workload_series
    if real and os.path.exists(LSTM_REAL_PATH):
        use_lstm(LSTM_REAL_PATH, "trained on GenTD26 request log")
    backend = get_lstm()[1]
    if classifier == "tree":
        get_trained_tree()
    for p in ROUTING:
        classify_and_forecast(generate_workload_series(p, 120, seed=0), classifier=classifier)
    return backend


class ForecastMargin:
    """
    Safety buffer for a load forecast, learned from the forecaster's own misses.

    Sizing replicas for exactly the predicted load under-provisions about half
    the time. The buffer is the q-quantile of the recent shortfalls
    (actual - predicted) over a sliding window, so a forecaster that has been
    accurate lately earns a small buffer and one that keeps missing gets a
    large one. Scaling for `predicted + margin` then leaves demand above
    capacity roughly (1 - q) of the time.
    """

    def __init__(self, quantile: float = 0.9, window: int = 120, min_samples: int = 20):
        self.quantile = quantile
        self.window = window
        self.min_samples = min_samples
        self.shortfalls: deque = deque(maxlen=window)

    def update(self, predicted: float, actual: float) -> None:
        self.shortfalls.append(float(actual) - float(predicted))

    def value(self, history=None) -> float:
        """Current buffer, in load units (never negative)."""
        if len(self.shortfalls) >= self.min_samples:
            return max(0.0, float(np.quantile(self.shortfalls, self.quantile)))
        if history is not None and len(history) > 2:
            # no track record yet: fall back to how much the load itself jumps
            jumps = np.diff(np.asarray(history, dtype=float)[-self.window:])
            return max(0.0, float(np.quantile(jumps, self.quantile)))
        return 0.0

    def reset(self) -> None:
        self.shortfalls.clear()


# --------------------------------------------------------------------------- #
# Dynamic model selection that learns from its own track record
# --------------------------------------------------------------------------- #
class DynamicSelector:
    """
    Pattern-aware model selection with feedback.

    The routing table of Section IV-A (periodic -> ARIMA, bursty -> LSTM,
    hybrid -> XGBoost) was tuned on synthetic workloads. On a real trace the
    best forecaster for a pattern is not known in advance, so the selector
    treats that table as a starting point and corrects it online: after each
    control window every forecaster is scored on the load that actually
    arrived, and an exponentially weighted error is kept per (pattern, model).

    The forecast for a pattern is then the error-weighted blend of the models,
    with weight proportional to (1 / tracked error) ** sharpness. A model that
    is clearly best takes almost all the weight, which is ordinary selection.
    Models that are neck and neck share it, which is steadier than flipping
    between them on noise.

    decay       weight of the past in the tracked error (0.97 ~ last 30 windows)
    sharpness   how strongly the weights favour the best model
    min_scores  windows of evidence a pattern needs before the table may change
    """

    def __init__(self, decay: float = 0.97, sharpness: float = 4.0, min_scores: int = 10):
        self.decay = decay
        self.sharpness = sharpness
        self.min_scores = min_scores
        self.error = {p: {m: 0.0 for m in FORECASTERS} for p in ROUTING}
        self.scores = {p: 0 for p in ROUTING}

    def weights(self, pattern: str) -> dict:
        """{model: weight}, summing to 1. One-hot on the prior table until there is evidence."""
        if self.scores[pattern] < self.min_scores:
            return {m: float(m == ROUTING[pattern]) for m in FORECASTERS}
        inv = {m: (1.0 / max(e, 1e-9)) ** self.sharpness for m, e in self.error[pattern].items()}
        total = sum(inv.values())
        return {m: v / total for m, v in inv.items()}

    def choose(self, pattern: str) -> str:
        """The model currently trusted most for this pattern."""
        w = self.weights(pattern)
        return max(w, key=w.get)

    def blend(self, pattern: str, forecasts: dict) -> float:
        """Weighted forecast from {model: forecast}."""
        w = self.weights(pattern)
        return float(sum(w[m] * forecasts[m] for m in w))

    def update(self, pattern: str, abs_errors: dict) -> None:
        """abs_errors: {model_name: |forecast - actual|} for the window just ended."""
        first = self.scores[pattern] == 0
        for m, e in abs_errors.items():
            prev = self.error[pattern][m]
            self.error[pattern][m] = float(e) if first else self.decay * prev + (1 - self.decay) * float(e)
        self.scores[pattern] += 1

    def table(self) -> dict:
        """Current routing table {pattern: most trusted model}."""
        return {p: self.choose(p) for p in ROUTING}

    def reset(self) -> None:
        self.__init__(self.decay, self.sharpness, self.min_scores)


def rolling_forecasts(series, start: int, stop: int = None, window: int = 150, horizon: int = 1,
                      classifier: str = "rules", progress=None):
    """
    Replay the control loop over a load series and record what every
    forecaster would have predicted at each step.

    At step t the models see series[t - window : t + 1] (everything up to and
    including the window that just ended) and predict the next `horizon`
    windows. One row per step:

        step, pattern, <features>, ARIMA, LSTM, XGBoost   (peak of the horizon)
        actual                                            (peak that then occurred)
        ARIMA_next, LSTM_next, XGBoost_next, actual_next  (one step ahead)

    Rows near the end of the series, where the horizon runs past the data,
    are left out.
    """
    import pandas as pd
    x = np.asarray(series, dtype=float)
    stop = len(x) - horizon if stop is None else min(stop, len(x) - horizon)
    rows = []
    for t in range(start, stop):
        w = x[max(0, t + 1 - window): t + 1]
        row = {"step": t, "actual_next": x[t + 1], "actual": float(np.max(x[t + 1: t + 1 + horizon]))}
        if np.ptp(w) == 0:
            row.update({"pattern": "periodic", "acf_peaks": 0.0, "wavelet_ratio": 0.0,
                        "cv": 0.0, "hurst": 0.5})
            for m in FORECASTERS:
                row[m] = row[m + "_next"] = float(w[-1])
        else:
            pattern, feats = classify_pattern(w, method=classifier)
            row["pattern"] = pattern
            row.update({k: float(v) for k, v in feats.items()})
            for m, fn in FORECASTERS.items():
                fc = fn(w, horizon=horizon)["forecast"]
                row[m], row[m + "_next"] = _summarise(fc), float(fc[0])
        rows.append(row)
        if progress and (t - start) % 100 == 0:
            progress(t - start, stop - start)
    return pd.DataFrame(rows)


def add_dynamic_selection(table, selector: "DynamicSelector" = None):
    """
    Add what dynamic model selection would have forecast to a
    rolling_forecasts() table: columns `Dynamic`, `Dynamic_next`, `routed_to`
    (adaptive, error-weighted selection; `routed_to` is the most trusted
    model) and `Static`, `Static_next` (the fixed Section IV-A table).
    Selection at a step only uses errors from steps already over.
    """
    selector = selector or DynamicSelector()
    models = list(FORECASTERS)
    horizon_lag = 1
    out = table.copy()
    routed, dyn, dyn_next = [], [], []
    pending = []          # rows whose outcome is not known yet at selection time
    for row in out.itertuples(index=False):
        # outcomes become visible one window after the forecast was made
        while pending and pending[0].step <= row.step - horizon_lag:
            done = pending.pop(0)
            selector.update(done.pattern, {m: abs(getattr(done, m + "_next") - done.actual_next)
                                           for m in models})
        routed.append(selector.choose(row.pattern))
        dyn.append(selector.blend(row.pattern, {m: getattr(row, m) for m in models}))
        dyn_next.append(selector.blend(row.pattern, {m: getattr(row, m + "_next") for m in models}))
        pending.append(row)
    out["routed_to"] = routed
    out["Dynamic"], out["Dynamic_next"] = dyn, dyn_next
    out["Static"] = [getattr(r, ROUTING[r.pattern]) for r in out.itertuples(index=False)]
    out["Static_next"] = [getattr(r, ROUTING[r.pattern] + "_next") for r in out.itertuples(index=False)]
    return out


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
