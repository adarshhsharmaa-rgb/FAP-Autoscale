"""
FAP-Scale End-to-End Simulation Entrypoint (Person C integration).

Ties together all three team modules in a single runnable script:
  - Data Generation    : Person A (workload) + Person B (node telemetry)
  - Model Training     : Person B (LightGBM NodeFailureScorer + InterferenceMatrix)
  - Control Loop       : Person C (FusionEngine 60 s window ticks)
  - Baselines          : Person C (ReactiveHPA + LSTM-RoundRobin)
  - Evaluation & Plots : results/evaluate.py

Usage:
    python main.py

The simulation runs N control windows, logging per-window decisions for all
three strategies, then computes SLA violations, over-provisioning %, and
unhealthy placement counts.
"""

import os
from typing import Dict, Any
import numpy as np

# -- Data generation -----------------------------------------------------------
from data_gen.synthetic_data import (
    generate_workload_time_series,
    generate_node_telemetry,
    extract_single_node_telemetry,
)

# -- Person B models -----------------------------------------------------------
from models.node_health.failure_scorer import (
    NodeFailureScorer, score_node_failure, calculate_cpi, calculate_tts
)
from models.node_health.interference import InterferenceMatrix, get_interference

# -- Person C modules ----------------------------------------------------------
from fusion.fusion_engine import FusionEngine, compute_node_score, calculate_replicas
from simulator.baselines import ReactiveHPAScaler, LSTMRoundRobinScaler
from models.load_pipeline import warm_up

# -- Evaluation ----------------------------------------------------------------
from results.evaluate import evaluate_experiment_results, plot_evaluation_summary


def _divider(char="=", width=80):
    print(char * width)


def _section(title, step_num=None):
    print()
    _divider("-")
    prefix = f"[Step {step_num}] " if step_num else ""
    print(f"  {prefix}{title}")
    _divider("-")


def run_fap_scale_simulation(
    num_windows: int = 30,
    num_nodes: int = 10,
    pattern: str = "periodic",
) -> Dict[str, Any]:
    """
    Run end-to-end FAP-Scale control loop simulation with detailed output.
    """
    os.makedirs("results", exist_ok=True)

    print()
    _divider("=")
    print("  FAP-SCALE: Failure-Aware Predictive Autoscaling Framework")
    print(f"  Simulation Config: {num_windows} windows | {num_nodes} nodes | pattern={pattern.upper()}")
    _divider("=")

    # =========================================================================
    # STEP 1: SYNTHETIC DATA GENERATION
    # =========================================================================
    _section("SYNTHETIC DATA GENERATION (Person A + Person B)", 1)

    print("\n  [Person A] Generating workload time-series...")
    print(f"    Formula: request_rate(t) = base_load + amplitude * sin(2*pi*t/period) + noise")
    workload_df = generate_workload_time_series(
        num_windows=num_windows, pattern=pattern, seed=42
    )
    total_points = len(workload_df)
    print(f"    -> {total_points} data points across {num_windows} control windows")
    print(f"    -> Mean load: {workload_df['request_rate'].mean():.1f} req/s, "
          f"Peak: {workload_df['request_rate'].max():.1f} req/s")

    print("\n  [Person B] Generating per-node hardware telemetry...")
    print(f"    Metrics: CPU%, run-queue, context-switches, temperature, DIMM errors,")
    print(f"             disk health, network retransmits")
    print(f"    Failing nodes injected: node-03, node-08 (progressive degradation)")
    node_telemetry_df = generate_node_telemetry(
        num_nodes=num_nodes,
        num_windows=num_windows,
        fail_node_ids=["node-03", "node-08"],
        seed=42,
    )
    print(f"    -> {len(node_telemetry_df)} telemetry rows ({num_nodes} nodes x {num_windows} windows)")

    # =========================================================================
    # STEP 2: MODEL TRAINING (Person B)
    # =========================================================================
    _section("NODE HEALTH MODEL TRAINING (Person B)", 2)

    print("\n  Training LightGBM NodeFailureScorer...")
    print("    Features: [cpu_util, run_queue, cs_rate, CPI, TTS, dimm_errors, disk_health, net_retransmits]")
    print("    Label: will_fail (binary)")
    failure_scorer = NodeFailureScorer(random_state=42)
    failure_scorer.fit(node_telemetry_df)
    backend = "LightGBM" if getattr(failure_scorer, "model", None).__class__.__name__ == "LGBMClassifier" else "HistGradientBoosting"
    print(f"    -> Trained ({backend}) on {len(node_telemetry_df)} rows")

    print("\n  Initialising 15-bin Co-location Interference Matrix...")
    print("    EMA update rule: IS_new = (1 - alpha) * IS_old + alpha * observed_degradation")
    interference_matrix = InterferenceMatrix(alpha=0.1, seed=42)
    print("    -> 15x15 symmetric matrix initialised (alpha=0.1)")

    # Show CPI/TTS formulas with a sample calculation
    print("\n  Key Formulas (Person B):")
    print("    CPI = 0.5 * U_norm + 0.35 * Q_norm + 0.15 * C_norm")
    print("    TTS = linear_slope(last_10_temps) / TDP")
    print("    Pfail = LightGBM.predict_proba(features)[class=1]")
    sample_cpi = calculate_cpi(75.0, 8.0, 3000.0)
    sample_tts = calculate_tts([60, 62, 64, 66, 68, 70, 72, 74, 76, 78], 105.0)
    print(f"    Example: CPI(cpu=75%, queue=8, cs=3000) = {sample_cpi:.4f}")
    print(f"    Example: TTS(temps=[60..78], TDP=105)    = {sample_tts:.4f}")
    sample_is = get_interference("periodic", "bursty", interference_matrix)
    print(f"    Example: IS(periodic, bursty)            = {sample_is:.4f}")

    # =========================================================================
    # STEP 3: FORECASTER WARM-UP & STRATEGY INIT (Person A + Person C)
    # =========================================================================
    _section("FORECASTER WARM-UP & STRATEGY INITIALISATION (Person A + C)", 3)

    print("\n  [Person A] Warming up forecasters (ARIMA, LSTM, XGBoost)...")
    warm_up()
    print("    -> Forecasters ready")

    print("\n  [Person C] Initialising Fusion Engine...")
    print("    Scoring: S(i) = alpha*(1-Pfail) + beta*(1-IS) + gamma*(1-U)")
    print(f"    Weights: alpha=0.4, beta=0.35, gamma=0.25")
    print(f"    Replica formula: K = ceil(predicted_load / capacity)")
    fusion_engine = FusionEngine(
        alpha=0.4,
        beta=0.35,
        gamma=0.25,
        replica_capacity=20.0,
        failure_scorer=failure_scorer,
        interference_matrix=interference_matrix,
    )

    # Show formula with sample
    sample_score = compute_node_score(pfail=0.1, is_score=0.3, utilization=50.0)
    sample_k = calculate_replicas(predicted_load=75.0)
    print(f"    Example: S(Pfail=0.1, IS=0.3, U=50%) = {sample_score:.3f}")
    print(f"    Example: K(load=75 req/s, cap=20)     = {sample_k} replicas")

    print("\n  [Person C] Initialising Baselines...")
    print("    Baseline 1: Reactive HPA (scale based on avg CPU threshold 70%)")
    print("    Baseline 2: LSTM Round-Robin (LSTM forecast + blind round-robin placement)")
    hpa_baseline = ReactiveHPAScaler(target_cpu_threshold=70.0, replica_capacity=20.0)
    lstm_baseline = LSTMRoundRobinScaler(replica_capacity=20.0)

    fap_history = []
    hpa_history = []
    lstm_history = []
    load_buffer: list = []

    # =========================================================================
    # STEP 4: SIMULATION CONTROL LOOP
    # =========================================================================
    _section("SIMULATION CONTROL LOOP (60-second windows)", 4)

    print("\n  Running all 3 strategies in parallel across each window...")
    print("  FAP-Scale pipeline per window:")
    print("    1. Classify workload pattern (autocorrelation, wavelet, CV, Hurst)")
    print("    2. Forecast load (ARIMA/LSTM/XGBoost based on pattern)")
    print("    3. Calculate target replicas K")
    print("    4. Score each node: S(i) = alpha*(1-Pfail) + beta*(1-IS) + gamma*(1-U)")
    print("    5. Rank nodes by S(i), select top-K healthy nodes")
    print()

    header = (
        f"{'Win':<4} | {'Actual':>7} | {'Pattern':<9} | {'Pred':>7} | "
        f"{'K_fap':>5} | {'Top Node':<10} | {'Pfail':>6} | {'IS':>5} | {'S(i)':>5} | EMA"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for w in range(num_windows):
        win_workload = workload_df[workload_df["window_id"] == w]["request_rate"].values
        actual_load = float(np.mean(win_workload)) if len(win_workload) > 0 else 50.0

        load_buffer.extend(win_workload.tolist())
        rolling_window = load_buffer[-120:]

        curr_telemetry = []
        node_pfails: Dict[str, float] = {}
        for n in range(1, num_nodes + 1):
            nid = f"node-{n:02d}"
            t_dict = extract_single_node_telemetry(node_telemetry_df, nid, w)
            curr_telemetry.append(t_dict)
            node_pfails[nid] = score_node_failure(nid, t_dict, failure_scorer)

        # -- FAP-Scale Fusion Engine --
        fap_result = fusion_engine.evaluate_window(
            workload_window=rolling_window,
            cluster_telemetry=curr_telemetry,
            current_workload_type=pattern,
        )
        fap_result["actual_load"] = actual_load
        fap_result["node_pfails"] = node_pfails
        fap_history.append(fap_result)

        # -- Reactive HPA baseline --
        hpa_res = hpa_baseline.evaluate_window(actual_load, curr_telemetry)
        hpa_res["actual_load"] = actual_load
        hpa_res["node_pfails"] = node_pfails
        hpa_history.append(hpa_res)

        # -- LSTM Round-Robin baseline --
        lstm_res = lstm_baseline.evaluate_window(rolling_window, curr_telemetry)
        lstm_res["actual_load"] = actual_load
        lstm_res["node_pfails"] = node_pfails
        lstm_history.append(lstm_res)

        # -- Console logging (first 6 + every 5th + last) --
        if w < 6 or w % 5 == 0 or w == num_windows - 1:
            top = fap_result["ranked_nodes"][0] if fap_result["ranked_nodes"] else {}
            ema_val = fap_result.get("ema_update", 0.0)
            print(
                f"{w:<4} | {actual_load:>7.1f} | {fap_result['pattern_label']:<9} | "
                f"{fap_result['predicted_load']:>7.1f} | {fap_result['target_replicas']:>5} | "
                f"{top.get('node_id', '-'):<10} | {top.get('pfail', 0):>6.3f} | "
                f"{top.get('is_score', 0):>5.3f} | {top.get('score', 0):>5.3f} | "
                f"{ema_val:.3f}"
            )

    print("-" * len(header))

    # =========================================================================
    # STEP 5: FUSION ENGINE SUMMARY
    # =========================================================================
    _section("FUSION ENGINE SUMMARY STATISTICS", 5)

    stats = fusion_engine.summary_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # Show node health snapshot for the last window
    print("\n  Node Health Snapshot (Final Window):")
    print(f"    {'Node':<10} | {'Pfail':>8} | {'Status':<12}")
    print("    " + "-" * 35)
    for nid in sorted(node_pfails.keys()):
        pf = node_pfails[nid]
        status = "HEALTHY" if pf < 0.4 else "AT RISK" if pf < 0.7 else "FAILING"
        marker = "  " if pf < 0.4 else " !" if pf < 0.7 else " X"
        print(f"   {marker}{nid:<10} | {pf:>8.4f} | {status:<12}")

    # =========================================================================
    # STEP 6: EVALUATION & COMPARISON
    # =========================================================================
    _section("FINAL EVALUATION & BASELINE COMPARISON", 6)

    metrics = {
        "FAP-Scale (Proposed)":        evaluate_experiment_results(fap_history),
        "Reactive-HPA (Baseline 1)":   evaluate_experiment_results(hpa_history),
        "LSTM-RoundRobin (Baseline 2)": evaluate_experiment_results(lstm_history),
    }

    print()
    _divider("=")
    print(
        f"  {'Strategy':<32} | {'SLA Violations':>14} | "
        f"{'Over-Prov %':>11} | {'Unhealthy Placements':>20}"
    )
    _divider("=")
    for strat, m in metrics.items():
        print(
            f"  {strat:<32} | {m['sla_violations']:>14} | "
            f"{m['overprovisioning_percent']:>10.1f}% | {m['unhealthy_placements']:>20}"
        )
    _divider("=")

    # Interpretation
    fap_m = metrics["FAP-Scale (Proposed)"]
    hpa_m = metrics["Reactive-HPA (Baseline 1)"]
    lstm_m = metrics["LSTM-RoundRobin (Baseline 2)"]

    print("\n  Key Findings:")
    print(f"    - FAP-Scale SLA violations: {fap_m['sla_violations']} vs "
          f"HPA: {hpa_m['sla_violations']} vs LSTM-RR: {lstm_m['sla_violations']}")
    print(f"    - FAP-Scale over-provisioning: {fap_m['overprovisioning_percent']:.1f}% vs "
          f"HPA: {hpa_m['overprovisioning_percent']:.1f}% vs LSTM-RR: {lstm_m['overprovisioning_percent']:.1f}%")
    print(f"    - FAP-Scale unhealthy placements: {fap_m['unhealthy_placements']} vs "
          f"HPA: {hpa_m['unhealthy_placements']} vs LSTM-RR: {lstm_m['unhealthy_placements']}")

    if fap_m['unhealthy_placements'] <= min(hpa_m['unhealthy_placements'], lstm_m['unhealthy_placements']):
        print("    -> FAP-Scale successfully avoids placing workloads on failing nodes!")
    if fap_m['overprovisioning_percent'] <= min(hpa_m['overprovisioning_percent'], lstm_m['overprovisioning_percent']):
        print("    -> FAP-Scale achieves the lowest over-provisioning (best resource efficiency)!")

    print()
    save_path = "results/comparison_metrics.png"
    plot_evaluation_summary(metrics, save_path=save_path)

    return metrics


if __name__ == "__main__":
    run_fap_scale_simulation(num_windows=20, num_nodes=10, pattern="hybrid")
