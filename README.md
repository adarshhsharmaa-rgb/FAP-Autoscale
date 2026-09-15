# FAP-Scale: Failure-Aware Predictive Autoscaling Framework

FAP-Scale is an intelligent, failure-aware predictive autoscaling and workload placement framework designed for cloud-native environments. It combines workload pattern classification, predictive load forecasting, LightGBM-based node failure probability scoring, and a 15-bin co-location resource interference matrix into a unified fusion control loop.

---

## 📁 Repository Structure

```
fap-scale/
├── data_gen/
│   ├── __init__.py
│   └── synthetic_data.py       # Workload (Person A) & Node Telemetry Generator (Person B)
├── models/
│   ├── __init__.py
│   ├── pattern_classifier.py   # Workload Pattern Classifier (Person A)
│   ├── forecasters.py          # ARIMA / LSTM / XGBoost Forecasters (Person A)
│   ├── failure_scorer.py       # LightGBM Failure Scorer, CPI & TTS (Person B)
│   └── interference.py         # 15-bin Co-location Interference Matrix with EMA (Person B)
├── fusion/
│   ├── __init__.py
│   └── fusion_engine.py        # Fusion Engine S(i), K replicas, Control Loop (Person C)
├── simulator/
│   ├── __init__.py
│   └── baselines.py            # Reactive HPA & LSTM-RoundRobin Baselines (Person C)
├── results/
│   ├── __init__.py
│   └── evaluate.py             # SLA violations, over-provisioning %, plots
├── tests/
│   └── test_person_b.py        # Unit test suite for Person B & pipeline
├── main.py                     # Runnable end-to-end simulation entrypoint
└── README.md
```

---

## 👥 Module Ownership & Responsibilities

| Team Member | Module Files | Core Responsibilities & Deliverables |
| :--- | :--- | :--- |
| **Person A** | `models/pattern_classifier.py`<br>`models/forecasters.py` | • Compute 4 features (Autocorr peaks, Wavelet energy, CV, Hurst exp).<br>• Classify workload pattern (`periodic`, `bursty`, `hybrid`).<br>• ARIMA, LSTM, XGBoost load forecasting.<br>• **Deliverable**: `classify_and_forecast(window) -> (pattern_label, predicted_load)` |
| **Person B** | `data_gen/synthetic_data.py`<br>`models/failure_scorer.py`<br>`models/interference.py` | • Generate multi-dimensional per-node telemetry with failure trends.<br>• Compute CPI ($0.5U + 0.35Q + 0.15C$) and TTS ($\text{slope}/\text{TDP}$).<br>• Train LightGBM model to predict node failure probability $P_{fail}$.<br>• Maintain 15-bin Co-location Interference Matrix with EMA ($\alpha=0.1$).<br>• **Deliverables**:<br>  `score_node_failure(node_id, telemetry) -> Pfail`<br>  `get_interference(type_i, type_j) -> IS` |
| **Person C** | `fusion/fusion_engine.py`<br>`simulator/baselines.py`<br>`main.py` | • Compute composite node placement score $S(i) = \alpha(1-P_{fail}) + \beta(1-IS) + \gamma(1-U)$.<br>• Calculate target replicas $K = \lceil \text{predicted\_load} / \text{capacity} \rceil$.<br>• Execute 60s window control loop.<br>• Implement Reactive HPA and LSTM Round-Robin baselines. |

---

## 🚀 Quickstart & Execution

### 1. Run Unit Verification Tests
To run unit tests for Person B deliverables and repo imports:
```bash
python -m unittest tests/test_person_b.py
```

### 2. Run End-to-End Simulation
To execute the live demo simulation:
```bash
python main.py
```
This runs a simulated cluster environment across control windows, comparing **FAP-Scale** against **Reactive-HPA** and **LSTM-RoundRobin** baselines.
