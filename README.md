# FAP-Scale: Failure-Aware Predictive Autoscaling Framework

FAP-Scale is an intelligent, failure-aware predictive autoscaling and workload placement framework designed for cloud-native environments. It combines workload pattern classification, predictive load forecasting, LightGBM-based node failure probability scoring, and a 15-bin co-location resource interference matrix into a unified fusion control loop.

---

## 📁 Repository Structure

```
fap-scale/
├── data_gen/
│   ├── __init__.py
│   └── synthetic_data.py           # Workload series (Person A) + Node Telemetry (Person B)
├── models/
│   ├── __init__.py
│   ├── pattern_classifier.py       # 4-feature Workload Pattern Classifier (Person A)
│   ├── forecasters.py              # ARIMA / LSTM / XGBoost Forecasters (Person A)
│   ├── load_pipeline.py            # Unified classify → route → forecast API (Person A)
│   ├── artifacts/
│   │   └── lstm_bursty.keras       # Pre-trained LSTM model artifact
│   └── node_health/
│       ├── __init__.py
│       ├── failure_scorer.py       # LightGBM Failure Scorer, CPI & TTS (Person B)
│       └── interference.py         # 15-bin Co-location Interference Matrix + EMA (Person B)
├── fusion/
│   ├── __init__.py
│   └── fusion_engine.py            # Fusion Engine S(i), K replicas, EMA feedback (Person C)
├── simulator/
│   ├── __init__.py
│   └── baselines.py                # Reactive HPA & LSTM-RoundRobin Baselines (Person C)
├── results/
│   ├── __init__.py
│   ├── evaluate.py                 # SLA violations, over-provisioning %, unhealthy placements
│   ├── evaluate_load_side.py       # Person A load-side evaluation & plots
│   └── load_side/                  # Pre-generated Person A evaluation artifacts
│       ├── classifier_*.png/csv
│       ├── forecast_*.png/csv
│       └── summary.json
├── tests/
│   ├── __init__.py
│   ├── test_person_b.py            # Person B unit tests (6 tests)
│   ├── test_person_c.py            # Person C unit + integration tests (42 tests)
│   └── test_load_side.py           # Person A load pipeline tests (15 tests)
├── main.py                         # End-to-end simulation entrypoint (Person C)
├── demo_load_side.py               # Person A live classification demo
├── INTERFACE.md                    # Person A → C interface contract
├── requirements.txt
└── README.md
```

---

## 👥 Module Ownership & Responsibilities

| Team Member | Module Files | Core Responsibilities & Deliverables |
| :--- | :--- | :--- |
| **Person A (Nirupam)** | `models/pattern_classifier.py`<br>`models/forecasters.py`<br>`models/load_pipeline.py` | • Compute 4 features (Autocorr peaks, Wavelet energy, CV, Hurst exp).<br>• Classify workload pattern (`periodic`, `bursty`, `hybrid`).<br>• ARIMA, LSTM, XGBoost load forecasting with dynamic routing.<br>• **Deliverable**: `classify_and_forecast(window) → dict` with `pattern`, `predicted_load`, `model_used` |
| **Person B (Adarsh)** | `data_gen/synthetic_data.py`<br>`models/node_health/failure_scorer.py`<br>`models/node_health/interference.py` | • Generate multi-dimensional per-node telemetry with failure trends.<br>• Compute CPI and TTS health indicators.<br>• Train LightGBM model to predict node failure probability P_fail.<br>• Maintain 15-bin Co-location Interference Matrix with EMA (α=0.1).<br>• **Deliverables**:<br>  `score_node_failure(node_id, telemetry) → Pfail`<br>  `get_interference(type_i, type_j) → IS` |
| **Person C (Aman)** | `fusion/fusion_engine.py`<br>`simulator/baselines.py`<br>`main.py` | • Compute composite node placement score S(i) = α(1−P_fail) + β(1−IS) + γ(1−U).<br>• Calculate target replicas K = ⌈predicted_load / capacity⌉.<br>• Execute 60s window control loop with EMA feedback.<br>• Implement Reactive HPA and LSTM Round-Robin baselines.<br>• End-to-end integration of all modules. |

---

## 🚀 Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Tests (63 tests)
```bash
python -m pytest tests/ -v
```

### 3. Run Person A Load-Side Demo
```bash
python demo_load_side.py
```

### 4. Run End-to-End Simulation
```bash
python main.py
```
This runs a simulated cluster environment across 20 control windows, comparing **FAP-Scale** against **Reactive-HPA** and **LSTM-RoundRobin** baselines.

### 5. Regenerate Person A Evaluation Plots
```bash
python results/evaluate_load_side.py
```

---

## 📊 Key Results

| Strategy | SLA Violations | Over-Provisioning % | Unhealthy Placements |
|:---|:---:|:---:|:---:|
| **FAP-Scale (Proposed)** | 7 | 19.8% | **0** |
| Reactive-HPA (Baseline 1) | 12 | 42.9% | 4 |
| LSTM-RoundRobin (Baseline 2) | 7 | 22.7% | 5 |

FAP-Scale achieves **zero unhealthy placements** by actively avoiding degraded nodes via the S(i) composite scoring function.
