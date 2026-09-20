"""Run with:  python -m pytest tests -q"""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from data_gen.synthetic_data import (PATTERNS, generate_service_traces,
                                     generate_workload_series, generate_workload_windows)
from models.forecasters import FORECASTERS, estimate_period
from models.load_pipeline import ROUTING, classify_and_forecast, forecast_with_model
from models.pattern_classifier import FEATURE_NAMES, classify_pattern, extract_features


def test_generator_shapes_and_labels():
    data = generate_workload_windows(9, length=150, seed=1)
    assert len(data) == 9
    assert {l for _, l in data} == set(PATTERNS)
    assert all(len(w) == 150 and np.all(w > 0) for w, _ in data)


def test_generator_is_reproducible():
    a = generate_workload_series("hybrid", 100, seed=3)
    b = generate_workload_series("hybrid", 100, seed=3)
    assert np.array_equal(a, b)


def test_service_traces():
    tr = generate_service_traces(3, length=200)
    assert len(tr) == 3 and all(len(v["load"]) == 200 for v in tr.values())


def test_features_present_and_finite():
    f = extract_features(generate_workload_series("periodic", 150, seed=2))
    assert list(f) == FEATURE_NAMES
    assert all(np.isfinite(v) for v in f.values())


@pytest.mark.parametrize("method", ["rules", "tree"])
def test_classifier_accuracy_above_85pct(method):
    data = generate_workload_windows(30, seed=31337)
    acc = np.mean([classify_pattern(w, method)[0] == l for w, l in data])
    assert acc >= 0.85


def test_period_detection():
    t = np.arange(200)
    assert abs(estimate_period(10 + 5 * np.sin(2 * np.pi * t / 20)) - 20) <= 1


@pytest.mark.parametrize("name", list(FORECASTERS))
def test_each_forecaster_is_sane(name):
    w = generate_workload_series("periodic", 150, seed=4)
    out = FORECASTERS[name](w, horizon=3)
    assert out["forecast"].shape == (3,)
    assert np.all(out["forecast"] >= 0)
    assert out["forecast"].max() < 3 * w.max()


def test_classify_and_forecast_contract():
    out = classify_and_forecast(generate_workload_series("bursty", 150, seed=5))
    assert out["pattern"] in PATTERNS
    assert out["model_used"] == ROUTING[out["pattern"]]
    assert isinstance(out["predicted_load"], float) and out["predicted_load"] >= 0


def test_multi_step_uses_peak():
    out = classify_and_forecast(generate_workload_series("periodic", 150, seed=6), horizon=5)
    assert len(out["forecast"]) == 5
    assert out["predicted_load"] == pytest.approx(max(out["forecast"]))


def test_forecast_with_model_for_baseline():
    out = forecast_with_model(generate_workload_series("hybrid", 150, seed=7), "LSTM")
    assert out["model_used"] == "LSTM" and out["predicted_load"] >= 0


def test_short_window_rejected():
    with pytest.raises(ValueError):
        classify_and_forecast(np.ones(10))


def test_nan_gaps_are_filled():
    w = generate_workload_series("periodic", 150, seed=8)
    w[50] = np.nan
    assert np.isfinite(classify_and_forecast(w)["predicted_load"])
