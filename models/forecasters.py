"""
models/forecasters.py  (Person A)

The three load forecasters from FAP-Scale Section IV-A. Each takes one window
of past load and returns a dict:

    {"forecast": np.ndarray (horizon,), "backend": str, ...extras}

    forecast_arima    ARIMA(2,1,2) + Fourier regressors -> periodic workloads
    forecast_lstm     2-layer LSTM (64 units, drop .2) -> bursty workloads
    forecast_xgboost  XGBoost, robust lags + Fourier   -> hybrid workloads

The LSTM runs on whichever deep-learning library is installed, Keras
(TensorFlow) or PyTorch, with the same architecture either way. If a library
is missing the function degrades to a lighter model and says so in "backend",
so the demo never crashes on a teammate's laptop.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # silences oneDNN banner

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
LSTM_PATH = os.path.join(ARTIFACT_DIR, "lstm_bursty.keras")      # Keras, synthetic bursty series
LSTM_TORCH_PATH = os.path.join(ARTIFACT_DIR, "lstm_bursty.pt")   # PyTorch, same training data
LSTM_REAL_PATH = os.path.join(ARTIFACT_DIR, "lstm_gentd26.pt")   # PyTorch, GenTD26 request log
LOOKBACK = 24          # LSTM / XGBoost input length (24 control steps)
ARIMA_ORDER = (2, 1, 2)
ARIMA_MAX_HISTORY = 120  # fitting on the last 120 points keeps ARIMA fast


def _clip(arr):
    return np.clip(np.asarray(arr, dtype=float), 0.0, None)


# =========================================================================== #
# 1) ARIMA(2,1,2)  -- periodic
# =========================================================================== #
def _holt_fallback(x, horizon, alpha=0.5, beta=0.1):
    level, trend = x[0], x[1] - x[0]
    for v in x[1:]:
        prev = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return np.array([level + (h + 1) * trend for h in range(horizon)])


def fourier_terms(t, period, k=2):
    """Seasonal regressors sin/cos(2*pi*j*t/period), j = 1..k."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([f(2 * np.pi * j * t / period)
                            for j in range(1, k + 1) for f in (np.sin, np.cos)])


def forecast_arima(window, horizon: int = 1, seasonal: bool = True) -> dict:
    """
    ARIMA(2,1,2). When the window has a detectable cycle, Fourier terms for that
    cycle are added as exogenous regressors (dynamic harmonic regression): plain
    ARIMA(2,1,2) cannot represent a 12-30 step season on its own, and this cut
    periodic-workload error by ~40% in our tests.
    """
    x = np.asarray(window, dtype=float)[-ARIMA_MAX_HISTORY:]
    period = estimate_period(x, default=None) if seasonal else None
    try:
        from statsmodels.tsa.arima.model import ARIMA
        t = np.arange(len(x))
        t_future = np.arange(len(x), len(x) + horizon)
        exog = fourier_terms(t, period) if period else None
        exog_f = fourier_terms(t_future, period) if period else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(x, exog=exog, order=ARIMA_ORDER).fit()
            fc = res.get_forecast(horizon, exog=exog_f)
            ci = np.asarray(fc.conf_int(alpha=0.2))  # 80% interval
        return {
            "forecast": _clip(fc.predicted_mean),
            "lower": _clip(ci[:, 0]),
            "upper": _clip(ci[:, 1]),
            "period": period,
            "backend": "statsmodels ARIMA(2,1,2)" + (f" + Fourier(P={period})" if period else ""),
        }
    except ImportError:
        return {"forecast": _clip(_holt_fallback(x, horizon)), "backend": "Holt smoothing (fallback)"}
    except Exception:  # non-convergence etc. -> safe fallback, never crash the loop
        return {"forecast": _clip(_holt_fallback(x, horizon)), "backend": "Holt smoothing (ARIMA failed)"}


# =========================================================================== #
# 2) LSTM  -- bursty
# Trained once on z-scored series, cached to disk, then optionally fine-tuned
# on the live window ("fine-tuned online"). Two training sets exist:
#   lstm_bursty.*    synthetic bursty series (the original Review-2 artifact)
#   lstm_gentd26.pt  windows of the real GenTD26 request log (models/train_real.py)
# =========================================================================== #
_LSTM = None
_LSTM_BACKEND = None
KERAS_NAME = "Keras LSTM 2x64 (dropout 0.2)"
TORCH_NAME = "PyTorch LSTM 2x64 (dropout 0.2)"


def _make_supervised(z, lookback=LOOKBACK):
    X = np.lib.stride_tricks.sliding_window_view(z[:-1], lookback)
    y = z[lookback:]
    return X, y


def _zscore(x):
    mu, sd = float(np.mean(x)), float(np.std(x)) or 1.0
    return (x - mu) / sd, mu, sd


def _build_keras_lstm():
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input(shape=(LOOKBACK, 1)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber")
    return model


def _build_torch_lstm():
    """Same network as the Keras one: LSTM(64) -> drop -> LSTM(64) -> drop -> Dense(1)."""
    from torch import nn

    class LoadLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, num_layers=2, dropout=0.2, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(64, 1)

        def forward(self, x):                      # x: [batch, LOOKBACK, 1]
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    return LoadLSTM()


def _fit_torch(model, X, y, epochs, batch_size=128, lr=1e-3, seed=11, verbose=0):
    import torch
    Xt = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32).unsqueeze(-1)
    yt = torch.tensor(np.ascontiguousarray(y), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.HuberLoss()
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs}  huber={total / len(Xt):.4f}")
    model.eval()
    return model


def _is_torch(model) -> bool:
    return hasattr(model, "lstm") and hasattr(model, "state_dict")


def _synthetic_bursty(n_series, seed):
    from data_gen.synthetic_data import generate_workload_series
    rng = np.random.default_rng(seed)
    return [generate_workload_series("bursty", 200, seed=int(rng.integers(1e9)))
            for _ in range(n_series)]


def train_lstm(n_series: int = 60, epochs: int = 8, seed: int = 11, verbose: int = 0,
               series=None, save_path=None):
    """
    Train the load LSTM and save it.

    series : list of 1-D load arrays to learn from. Each is z-scored on its
             own, so only the shape of a series matters, not its scale.
             Defaults to synthetic bursty series (the original artifact).
    """
    if series is None:
        series = _synthetic_bursty(n_series, seed)
    Xs, ys = [], []
    for one in series:
        z, _, _ = _zscore(np.asarray(one, dtype=float))
        if len(z) > LOOKBACK + 1:
            X, y = _make_supervised(z)
            Xs.append(X)
            ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if save_path is None or save_path.endswith(".keras"):
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            model = _build_keras_lstm()
            model.fit(X[..., None], y, epochs=epochs, batch_size=128, verbose=verbose,
                      validation_split=0.1)
            model.save(save_path or LSTM_PATH)
            return model, KERAS_NAME
        except ImportError:
            pass
    try:
        import torch
        torch.manual_seed(seed)
        model = _fit_torch(_build_torch_lstm(), X, y, epochs, seed=seed, verbose=verbose)
        path = save_path if save_path and save_path.endswith(".pt") else LSTM_TORCH_PATH
        torch.save(model.state_dict(), path)
        return model, TORCH_NAME
    except ImportError:
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=seed)
        model.fit(X, y)
        return model, "MLP 'LSTM-lite' (neither TensorFlow nor PyTorch installed)"


def _load_torch(path):
    import torch
    model = _build_torch_lstm()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def use_lstm(path: str, label: str = "") -> str:
    """Make the PyTorch LSTM saved at `path` the active one (e.g. LSTM_REAL_PATH)."""
    global _LSTM, _LSTM_BACKEND
    _LSTM = _load_torch(path)
    _LSTM_BACKEND = TORCH_NAME + (f", {label}" if label else "")
    return _LSTM_BACKEND


def reset_lstm() -> None:
    """Forget the active LSTM so the next call reloads the default artifact."""
    global _LSTM, _LSTM_BACKEND
    _LSTM = _LSTM_BACKEND = None


def get_lstm():
    """Return the active LSTM: load a cached artifact, or train on first use."""
    global _LSTM, _LSTM_BACKEND
    if _LSTM is None:
        try:
            if os.path.exists(LSTM_PATH):
                from tensorflow import keras
                _LSTM = keras.models.load_model(LSTM_PATH)
                _LSTM_BACKEND = KERAS_NAME
        except ImportError:
            pass
        if _LSTM is None and os.path.exists(LSTM_TORCH_PATH):
            try:
                _LSTM, _LSTM_BACKEND = _load_torch(LSTM_TORCH_PATH), TORCH_NAME
            except ImportError:
                pass
        if _LSTM is None:
            _LSTM, _LSTM_BACKEND = train_lstm()
    return _LSTM, _LSTM_BACKEND


def _lstm_predict_one(model, ctx):
    if _is_torch(model):
        import torch
        with torch.no_grad():
            x = torch.tensor(ctx, dtype=torch.float32).reshape(1, LOOKBACK, 1)
            return float(model(x)[0])
    if hasattr(model, "predict") and hasattr(model, "layers"):  # keras
        return float(model(ctx.reshape(1, LOOKBACK, 1), training=False).numpy()[0, 0])
    return float(model.predict(ctx.reshape(1, -1))[0])


def forecast_lstm(window, horizon: int = 1, fine_tune_epochs: int = 0) -> dict:
    x = np.asarray(window, dtype=float)
    model, backend = get_lstm()
    z, mu, sd = _zscore(x)

    if fine_tune_epochs > 0 and len(z) > LOOKBACK + 8:
        X, y = _make_supervised(z)
        if _is_torch(model):
            _fit_torch(model, X, y, fine_tune_epochs, batch_size=32)
            backend += " + online fine-tune"
        elif "Keras" in backend:
            model.fit(X[..., None], y, epochs=fine_tune_epochs, batch_size=32, verbose=0)
            backend += " + online fine-tune"

    ctx = z[-LOOKBACK:].copy()
    preds = []
    for _ in range(horizon):  # recursive multi-step
        p = _lstm_predict_one(model, ctx)
        preds.append(p)
        ctx = np.append(ctx[1:], p)
    return {"forecast": _clip(np.array(preds) * sd + mu), "backend": backend}


# =========================================================================== #
# 3) XGBoost on lag + Fourier features  -- hybrid
# Hybrid = cycle + spikes. Raw lags make the model chase the last spike, so the
# features are spike-robust (medians), include a seasonal lag, and the loss is
# pseudo-Huber so bursts in the target don't dominate the fit.
# =========================================================================== #
def estimate_period(x, min_p=4, max_p=72, default=24):
    """Dominant cycle length = lag of the highest ACF peak (> 0.2), else `default`."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    max_p = int(min(max_p, len(x) // 2))
    denom = np.dot(x, x) or 1.0
    r = np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, max_p + 1)])
    best, best_val = None, 0.2
    for k in range(min_p, max_p - 1):
        if r[k - 1] > best_val and r[k - 1] >= r[k - 2] and r[k - 1] >= r[k]:
            best, best_val = k, r[k - 1]
    return best if best is not None else default


XGB_FEATURES = ["lag1", "median_lag1_3", "median_lag1_6", "median_lag1_12",
                "seasonal_lag_median", "sin1", "cos1", "sin2", "cos2"]


def _xgb_features(series, t, period):
    """Feature row for predicting series[t] using only values before t."""
    s = series
    seas_lo = t - period - 2
    seasonal = float(np.median(s[seas_lo: t - period + 3])) if seas_lo >= 0 else float(s[t - 1])
    w = 2 * np.pi * t / period
    return [
        float(s[t - 1]),
        float(np.median(s[t - 3:t])),
        float(np.median(s[t - 6:t])),
        float(np.median(s[t - 12:t])),
        seasonal,
        np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w),
    ]


def _xgb_regressor(y):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.9,
                            objective="reg:pseudohubererror",
                            huber_slope=max(float(np.std(y)) * 0.5, 1e-3),
                            base_score=float(np.median(y)),
                            n_jobs=1, verbosity=0), "XGBoost (robust lags + Fourier)"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=150, max_depth=3, loss="huber"), \
            "sklearn GradientBoosting (xgboost not installed)"


def forecast_xgboost(window, horizon: int = 1) -> dict:
    x = np.asarray(window, dtype=float)
    period = estimate_period(x)
    start = max(12, period + 3)
    X = np.array([_xgb_features(x, t, period) for t in range(start, len(x))])
    y = x[start:]
    model, backend = _xgb_regressor(y)
    model.fit(X, y)

    hist = list(x)
    preds = []
    for _ in range(horizon):  # recursive multi-step
        p = float(model.predict(np.array([_xgb_features(hist, len(hist), period)]))[0])
        preds.append(p)
        hist.append(p)
    return {"forecast": _clip(preds), "backend": backend, "period": period}


# =========================================================================== #
FORECASTERS = {
    "ARIMA": forecast_arima,
    "LSTM": forecast_lstm,
    "XGBoost": forecast_xgboost,
}

# ─── Backward-compat shim ───────────────────────────────────────────────────
# test_person_b.py imports classify_and_forecast from models.forecasters
# with the old tuple signature (pattern, predicted_load).
# New code should use models.load_pipeline.classify_and_forecast() instead.
from typing import Tuple, Union  # noqa: E402


def classify_and_forecast(
    window_data: Union[np.ndarray, list],
) -> Tuple[str, float]:
    """
    Legacy shim: classify_and_forecast(window) -> (pattern_label, predicted_load).
    Prefer models.load_pipeline.classify_and_forecast() for new code.
    Handles short windows (< 32 points) by using a simple EMA fallback.
    """
    from models.pattern_classifier import classify_pattern
    series = np.asarray(window_data, dtype=float)
    if len(series) < 32:
        # Too short to classify — use hybrid default + EMA forecast
        pred = float(series[-1]) if len(series) > 0 else 50.0
        return "hybrid", pred
    pattern, _ = classify_pattern(series)
    model_name = {"periodic": "ARIMA", "bursty": "LSTM", "hybrid": "XGBoost"}[pattern]
    out = FORECASTERS[model_name](series)
    return pattern, float(out["forecast"][0])

if __name__ == "__main__":
    import time
    from data_gen.synthetic_data import generate_workload_series
    for pattern, name in (("periodic", "ARIMA"), ("bursty", "LSTM"), ("hybrid", "XGBoost")):
        s = generate_workload_series(pattern, 200, seed=5)
        t0 = time.time()
        out = FORECASTERS[name](s[:-1])
        print(f"{pattern:9s} -> {name:7s} pred={out['forecast'][0]:7.2f} actual={s[-1]:7.2f} "
              f"({time.time() - t0:.2f}s, {out['backend']})")
