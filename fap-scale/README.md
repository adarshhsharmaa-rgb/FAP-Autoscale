# FAP-Scale — Failure-Aware Predictive Autoscaling (Review 2 prototype)

Team: Adarsh Sharma Badhai · Nirupam Raut · Aman Kumar Sah — Operating Systems, VIT (Guide: Dhivyaa C R)

A self-contained simulation of the FAP-Scale control loop. Every module is a real
implementation (real ARIMA / LSTM / XGBoost / LightGBM), but it runs on **synthetic data**
instead of the 15 GB Google Cluster Trace, and on a **Python-simulated cluster** instead of
Kubernetes. Both of those are planned future work.

```
fap-scale/
├── data_gen/synthetic_data.py      workload traces (A) + node telemetry (B)
├── models/
│   ├── pattern_classifier.py       A  4 features + rule-based / decision-tree classifier
│   ├── forecasters.py              A  ARIMA, LSTM, XGBoost
│   ├── load_pipeline.py            A  classify_and_forecast()  ← what C calls
│   ├── artifacts/lstm_bursty.keras A  pre-trained LSTM (auto-created if missing)
│   ├── failure_scorer.py           B  LightGBM, CPI, TTS
│   └── interference.py             B  co-location matrix + EMA
├── fusion/fusion_engine.py         C  S(i), K, control_loop()
├── simulator/baselines.py          C  reactive HPA, LSTM-only + round robin
├── results/evaluate_load_side.py   A  classifier + forecaster evaluation → results/load_side/
├── demo_load_side.py               A  live demo of the Predict-Load stage
├── tests/test_load_side.py         A  15 pytest checks
├── INTERFACE.md                    A→C function contract
└── main.py                         C  end-to-end demo
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q                 # 15 passed
python demo_load_side.py                  # live demo, load side
python -m results.evaluate_load_side      # regenerates results/load_side/ (~3 min)
```

Run everything from the repo root, since the modules use package imports.
If TensorFlow is missing, the LSTM falls back to an MLP and the output says `LSTM-lite`.

---

## Load side (Person A): what was built

**1. Synthetic workloads** (`data_gen/synthetic_data.py`)
- *Periodic* series use a sine wave plus a 2nd harmonic and noise, with a cycle of 12–30 steps.
- *Bursty* series use a drifting baseline with 5–10% of steps turned into decaying spikes of 1.5–3.5× the base.
- *Hybrid* series combine a periodic base with occasional smaller bursts.
- `generate_service_traces()` produces long per-service traces for the control loop.

**2. Pattern classifier** (`models/pattern_classifier.py`)
Features, as described in the paper:
- ACF peak count (statsmodels)
- wavelet fine-band energy ratio (PyWavelets, db1, 3 levels)
- coefficient of variation
- Hurst exponent (`hurst` package)

There are two interchangeable classifiers:
- **Rule-based**, the default and easy to explain. A window with no ACF peaks and either low Hurst or high CV is *bursty*. A window with at least 2 ACF peaks and a low wavelet ratio is *periodic*. Anything else is *hybrid*.
- **Decision tree** (depth 4), trained on 240 labelled synthetic windows.

**3. Forecasters and routing** (`models/forecasters.py`, `models/load_pipeline.py`)

| pattern | model | details |
|---|---|---|
| periodic | ARIMA(2,1,2) | plus Fourier regressors for the detected cycle (dynamic harmonic regression) and an 80% interval |
| bursty | LSTM | 2 layers × 64 units, dropout 0.2, Huber loss, 24-step lookback; pre-trained on 60 bursty series and cached; optional online fine-tuning (`fine_tune_epochs`) |
| hybrid | XGBoost | spike-robust median lags, a seasonal lag, and Fourier terms; pseudo-Huber loss |

## Results (synthetic, preliminary)

The test set has 45 unseen windows (15 per pattern). Each window gets 12 rolling one-step forecasts.
NRMSE is RMSE divided by mean load, so lower is better.

| pattern | naive (last value) | ARIMA only | LSTM only | XGBoost only | **FAP-Scale dynamic** |
|---|---|---|---|---|---|
| periodic | 0.141 | **0.043** | 0.250 | 0.057 | **0.043** |
| bursty | 0.405 | 0.461 | **0.370** | 0.418 | **0.371** |
| hybrid | 0.229 | 0.275 | 0.271 | 0.235 | **0.234** |
| **overall** | 0.259 | 0.259 | 0.297 | 0.237 | **0.216** |

- Classifier accuracy: **97.8%** rule-based and 95.6% decision tree. Inside the rolling loop, routing is correct **97.4%** of the time.
- **Dynamic selection is 8.8% better than the best single model** (XGBoost), and it matches oracle routing (0.216).
- The routed model is the best, or tied for best, model for every pattern. This supports the core claim of Section IV-A.
- Latency per forecast: XGBoost ≈ 50 ms, LSTM ≈ 130 ms, ARIMA ≈ 180 ms. All are far below the 60 s control window.

Plots are in `results/load_side/`: `forecast_error.png`, `forecast_examples.png`,
`classifier_confusion.png`, and `classifier_features.png`.

## Honest notes (say these if asked)

1. **The 8.8% improvement is below the paper's estimated 18–24%.** Report the simulation number and label it preliminary, or update the paper text.
2. **Two models were strengthened after a first evaluation.**
   - With plain ARIMA and plain lag-based XGBoost, dynamic routing was 3.6% *worse* than ARIMA-only.
   - Adding Fourier regressors to ARIMA and robust features to XGBoost fixed this.
   - The tuning used a different seed from the reported test set.
3. **Hybrid windows are hard to forecast one step ahead**, because their bursts are random. The routed XGBoost (0.234) is roughly tied with simply repeating the last value (0.229).
4. **Some choices were tuned only on synthetic data.** The rule thresholds and the LSTM training both need re-tuning on the Google trace.
5. **The LSTM is pre-trained offline.** Online fine-tuning exists but is off by default so the demo stays fast.

## Future work (load side)
- Replace the synthetic series with `resource_usage` windows from the Google Cluster Trace 2019.
- Re-fit the classifier thresholds and tree on real labelled windows.
- Use multi-step horizons that match real pod start-up time (60–120 s).
- Scale the LSTM training up and enable online fine-tuning.
