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

## Real traces (added for Review 3)

The calls above are unchanged. Running on the GenTD26 trace adds four things.

```python
from models.load_pipeline import (warm_up, rolling_forecasts, add_dynamic_selection,
                                  DynamicSelector, ForecastMargin)
from traces import gentd26

warm_up(real=True)                          # LSTM trained on the GenTD26 request log
load = gentd26.load().series("qps_total")   # 1441 windows x 57 s, anonymised QPS

table = rolling_forecasts(load, start=150, window=150, horizon=2)   # ~9 min, cached in results/real/
table = add_dynamic_selection(table)        # adds Dynamic, Static, routed_to
```

| name | what it is |
|---|---|
| `warm_up(real=True)` | activates `models/artifacts/lstm_gentd26.pt` (train it with `python -m models.train_real`) instead of the LSTM trained on synthetic bursty series |
| `rolling_forecasts(series, start, window, horizon)` | replays the control loop over a series and records every forecaster's prediction at every window, plus what then happened. The forecasters are slow, so the simulator reads this table instead of calling them |
| `DynamicSelector` | dynamic model selection with feedback. Starts from the routing table above, scores every model after each window, and blends them with weight proportional to (1 / tracked error)^4. `add_dynamic_selection(table)` applies it to a table without ever using a later outcome for an earlier choice |
| `ForecastMargin(quantile)` | safety buffer = that quantile of the forecaster's recent shortfalls (actual - predicted). Scale for `predicted_load + margin.value()` |

`classify_and_forecast` now returns the last value with `backend = "last value (flat window)"`
when the window is constant (an idle service), instead of failing on degenerate statistics.

### Units on the real trace
Load is anonymised gateway QPS (peak about 9.6). One replica's capacity is read off the trace,
`gentd26.load().replica_capacity()` = 0.34 QPS per saturated GPU pod, so the peak needs about 28 replicas.
