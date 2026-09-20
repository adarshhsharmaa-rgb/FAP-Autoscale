# Load-side interface (Person A → Person C)

Everything Person C needs from the load side lives in `models/load_pipeline.py`.

```python
from models.load_pipeline import warm_up, classify_and_forecast, forecast_with_model
from data_gen.synthetic_data import generate_service_traces

warm_up()                                   # call ONCE before the loop (~3 s)
traces = generate_service_traces(n_services=3, length=400)

# inside control_loop(), every 60 s tick t (t >= 150):
window = traces["svc-0"]["load"][t-150:t]
out = classify_and_forecast(window)
L = out["predicted_load"]                   # -> K = ceil((L*capacity - provisioned) / replica_size)
```

## `classify_and_forecast(window, timestamps=None, horizon=1, classifier="rules", fine_tune_epochs=0) -> dict`

| key | type | meaning |
|---|---|---|
| `pattern` | str | `"periodic"`, `"bursty"` or `"hybrid"` |
| `predicted_load` | float | forecast load for the next control window (the **L** in the K formula). If `horizon > 1`, this is the **peak** of the forecast |
| `model_used` | str | `"ARIMA"`, `"LSTM"` or `"XGBoost"` |
| `forecast` | list[float] | all `horizon` steps |
| `features` | dict | `acf_peaks`, `wavelet_ratio`, `cv`, `hurst` |
| `last_observed` | float | last value in the window |
| `backend` | str | exact model variant that ran (shows fallbacks honestly) |
| `latency_ms` | float | time for classify + forecast |
| `lower`, `upper` | list[float] | 80% interval (ARIMA only) |

Rules for the window: at least 48 points (150 recommended), evenly spaced, NaNs are forward-filled.
Typical latency is 30–180 ms per call, so 3 services per tick stays well under a second.

## For the LSTM-only baseline (`simulator/baselines.py`)

```python
out = forecast_with_model(window, "LSTM")   # same keys, no pattern/features
L = out["predicted_load"]
```

## Units
Load values are abstract "load units" (think CPU cores or req/s ÷ 10). Pick `capacity` per
replica in the same units, e.g. `replica_capacity = 5.0` means one replica handles 5 load units.

## Handy for the demo
`format_decision(out)` returns a one-line log string, for example:
`pattern=bursty -> LSTM | L= 16.14 (last= 15.87) | acf=0 wav=0.53 cv=0.55 H=0.38 | 86 ms`

## Note for Person B
`data_gen/synthetic_data.py` has a placeholder `generate_node_telemetry()` at the bottom. Replace
it with your generator; nothing in the load side depends on it.
