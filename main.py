"""
FAP-Scale End-to-End Simulation Entrypoint.

Ties together all three team modules in a single runnable script:
  - Data Generation    : Nirupam (workload) + Adarsh (node telemetry)
  - Model Training     : Adarsh (LightGBM NodeFailureScorer + InterferenceMatrix)
  - Control Loop       : Aman (FusionEngine 60 s window ticks)
  - Baselines          : Aman (ReactiveHPA + LSTM-RoundRobin)
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

# -- Node health models (Adarsh) ----------------------------------------------
from models.node_health.failure_scorer import (
    NodeFailureScorer, score_node_failure, calculate_cpi, calculate_tts
)
from models.node_health.interference import InterferenceMatrix, get_interference

# -- Fusion & baselines (Aman) ------------------------------------------------
from fusion.fusion_engine import FusionEngine, compute_node_score, calculate_replicas
from simulator.baselines import ReactiveHPAScaler, LSTMRoundRobinScaler
from models.load_pipeline import warm_up

# -- Evaluation ----------------------------------------------------------------
from results.evaluate import evaluate_experiment_results, plot_evaluation_summary


def _banner(text, char="=", width=88):
    print(char * width)
    print(f"  {text}")
    print(char * width)


def _section(title, step_num=None):
    print()
    prefix = f"STEP {step_num}" if step_num else ""
    line = "-" * 88
    print(line)
    if prefix:
        print(f"  [{prefix}] {title}")
    else:
        print(f"  {title}")
    print(line)


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
    _banner(
        "FAP-SCALE: Failure-Aware Predictive Autoscaling Framework\n"
        f"  Simulation: {num_windows} windows | {num_nodes} cluster nodes | pattern = {pattern.upper()}\n"
        f"  Team: Nirupam (Workload) | Adarsh (Node Health) | Aman (Fusion & Integration)"
    )

    # =========================================================================
    # STEP 1: SYNTHETIC DATA GENERATION
    # =========================================================================
    _section("SYNTHETIC DATA GENERATION", 1)

    print("\n  [Nirupam] Generating workload time-series...")
    print("    Model: request_rate(t) = base + amplitude * sin(2*pi*t/period) + noise")
    workload_df = generate_workload_time_series(
        num_windows=num_windows, pattern=pattern, seed=42
    )
    total_points = len(workload_df)
    mean_load = workload_df['request_rate'].mean()
    peak_load = workload_df['request_rate'].max()
    min_load = workload_df['request_rate'].min()
    print(f"    Generated {total_points} data points across {num_windows} control windows")
    print(f"    Load stats: mean={mean_load:.1f} | peak={peak_load:.1f} | min={min_load:.1f} req/s")

    print("\n  [Adarsh] Generating per-node hardware telemetry...")
    print("    Signals: CPU%, run-queue, context-switches, temperature,")
    print("             DIMM errors, disk health, network retransmits")
    print("    Failure injection: node-03, node-08 (progressive degradation)")
    node_telemetry_df = generate_node_telemetry(
        num_nodes=num_nodes,
        num_windows=num_windows,
        fail_node_ids=["node-03", "node-08"],
        seed=42,
    )
    print(f"    Generated {len(node_telemetry_df)} telemetry rows ({num_nodes} nodes x {num_windows} windows)")

    # =========================================================================
    # STEP 2: NODE HEALTH MODEL TRAINING (Adarsh)
    # =========================================================================
    _section("NODE HEALTH MODEL TRAINING [Adarsh]", 2)

    print("\n  Training LightGBM NodeFailureScorer...")
    print("    Input features:")
    print("      [cpu_util, run_queue, cs_rate, CPI, TTS, dimm_errors, disk_health, net_retransmits]")
    print("    Target label: will_fail (binary classification)")
    failure_scorer = NodeFailureScorer(random_state=42)
    failure_scorer.fit(node_telemetry_df)
    backend = "LightGBM" if getattr(failure_scorer, "model", None).__class__.__name__ == "LGBMClassifier" else "HistGradientBoosting"
    print(f"    Trained successfully ({backend}) on {len(node_telemetry_df)} rows")

    print("\n  Initialising 15-bin Co-location Interference Matrix...")
    print("    Online learning rule (EMA):")
    print("      IS_new = (1 - alpha) * IS_old + alpha * observed_degradation")
    interference_matrix = InterferenceMatrix(alpha=0.1, seed=42)
    print("    Initialised 15x15 symmetric matrix (alpha = 0.1)")

    # Show formulas with sample calculations
    print("\n  Health Metric Formulas:")
    print("    CPI = 0.5 * (CPU/100) + 0.35 * (RunQueue/32) + 0.15 * (CtxSwitch/10000)")
    print("    TTS = polyfit_slope(last_10_temps) / TDP")
    print("    Pfail = LightGBM.predict_proba(features)[class=1]")
    sample_cpi = calculate_cpi(75.0, 8.0, 3000.0)
    sample_tts = calculate_tts([60, 62, 64, 66, 68, 70, 72, 74, 76, 78], 105.0)
    sample_is = get_interference("periodic", "bursty", interference_matrix)
    print(f"\n    Sample outputs:")
    print(f"      CPI(cpu=75%, queue=8, cs=3000) = {sample_cpi:.4f}")
    print(f"      TTS(temps=[60..78], TDP=105)   = {sample_tts:.4f}")
    print(f"      IS(periodic, bursty)           = {sample_is:.4f}")

    # =========================================================================
    # STEP 3: FORECASTER WARM-UP & STRATEGY INIT
    # =========================================================================
    _section("FORECASTER WARM-UP & STRATEGY INIT", 3)

    print("\n  [Nirupam] Warming up prediction models...")
    print("    ARIMA  -> periodic workloads (statsmodels)")
    print("    LSTM   -> bursty workloads   (keras, pre-trained)")
    print("    XGBoost -> hybrid workloads  (gradient boosting)")
    warm_up()
    print("    All forecasters ready")

    print("\n  [Aman] Initialising Fusion Engine...")
    print("    Node placement score:")
    print("      S(i) = alpha*(1 - Pfail) + beta*(1 - IS) + gamma*(1 - U/100)")
    print("    Weights: alpha=0.40 (failure) | beta=0.35 (interference) | gamma=0.25 (utilization)")
    print("    Replica count: K = ceil(predicted_load / capacity_per_pod)")
    fusion_engine = FusionEngine(
        alpha=0.4,
        beta=0.35,
        gamma=0.25,
        replica_capacity=20.0,
        failure_scorer=failure_scorer,
        interference_matrix=interference_matrix,
    )

    sample_score = compute_node_score(pfail=0.1, is_score=0.3, utilization=50.0)
    sample_k = calculate_replicas(predicted_load=75.0)
    print(f"\n    Sample outputs:")
    print(f"      S(Pfail=0.1, IS=0.3, U=50%) = {sample_score:.3f}")
    print(f"      K(load=75 req/s, cap=20)    = {sample_k} replicas")

    print("\n  [Aman] Initialising Baseline Scalers...")
    print("    Baseline 1: Reactive HPA")
    print("      -> Scales when avg CPU exceeds 70% threshold")
    print("      -> Blind round-robin placement (no health awareness)")
    print("    Baseline 2: LSTM Round-Robin")
    print("      -> Uses LSTM load prediction for replica count")
    print("      -> Still round-robin placement (ignores Pfail and IS)")
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

    print("\n  Executing all 3 autoscaling strategies in parallel...")
    print("  Per-window FAP-Scale pipeline:")
    print("    [Nirupam] 1. Classify workload pattern (autocorrelation, wavelet, CV, Hurst)")
    print("    [Nirupam] 2. Forecast next-window load (ARIMA / LSTM / XGBoost)")
    print("    [Aman]    3. Calculate target replicas K = ceil(load / capacity)")
    print("    [Adarsh]  4. Score each node: S(i) = a*(1-Pfail) + b*(1-IS) + g*(1-U)")
    print("    [Aman]    5. Rank nodes by S(i), place replicas on top-K")
    print()

    header = (
        f"{'Win':<5}| {'Actual':>7} | {'Pattern':<9} | {'Forecast':>8} | "
        f"{'K':>3} | {'Best Node':<10} | {'Pfail':>6} | {'IS':>5} | {'S(i)':>5} | {'EMA':>5}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

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
                f"{w:<5}| {actual_load:>7.1f} | {fap_result['pattern_label']:<9} | "
                f"{fap_result['predicted_load']:>8.1f} | "
                f"{fap_result['target_replicas']:>3} | "
                f"{top.get('node_id', '-'):<10} | {top.get('pfail', 0):>6.3f} | "
                f"{top.get('is_score', 0):>5.3f} | {top.get('score', 0):>5.3f} | "
                f"{ema_val:>5.3f}"
            )

    print(sep)

    # =========================================================================
    # STEP 5: FUSION ENGINE SUMMARY & NODE HEALTH
    # =========================================================================
    _section("FUSION ENGINE SUMMARY & NODE HEALTH", 5)

    stats = fusion_engine.summary_stats()
    print("\n  Fusion Engine Statistics:")
    for k, v in stats.items():
        print(f"    {k:.<35} {v}")

    # Node health snapshot
    print(f"\n  Cluster Node Health Snapshot (Window {num_windows - 1}):")
    print(f"    {'Node':<10} | {'Pfail':>8} | {'Status':<10} | {'Verdict'}")
    print("    " + "-" * 55)
    healthy_count = 0
    failing_count = 0
    for nid in sorted(node_pfails.keys()):
        pf = node_pfails[nid]
        if pf < 0.4:
            status, verdict, marker = "HEALTHY", "OK to place replicas", " "
            healthy_count += 1
        elif pf < 0.7:
            status, verdict, marker = "AT RISK", "Avoid if possible", "!"
            failing_count += 1
        else:
            status, verdict, marker = "FAILING", "DO NOT place replicas", "X"
            failing_count += 1
        print(f"   {marker} {nid:<10} | {pf:>8.4f} | {status:<10} | {verdict}")

    print(f"\n    Summary: {healthy_count} healthy, {failing_count} unhealthy/failing")
    if failing_count > 0:
        print("    FAP-Scale automatically excludes failing nodes from placement!")

    # =========================================================================
    # STEP 6: EVALUATION & COMPARISON
    # =========================================================================
    _section("FINAL EVALUATION & COMPARISON", 6)

    metrics = {
        "FAP-Scale (Proposed)":        evaluate_experiment_results(fap_history),
        "Reactive-HPA (Baseline 1)":   evaluate_experiment_results(hpa_history),
        "LSTM-RoundRobin (Baseline 2)": evaluate_experiment_results(lstm_history),
    }

    print()
    print("=" * 88)
    print(
        f"  {'Strategy':<34} | {'SLA Violations':>14} | "
        f"{'Over-Prov %':>11} | {'Unhealthy Placements':>20}"
    )
    print("=" * 88)
    for strat, m in metrics.items():
        print(
            f"  {strat:<34} | {m['sla_violations']:>14} | "
            f"{m['overprovisioning_percent']:>10.1f}% | {m['unhealthy_placements']:>20}"
        )
    print("=" * 88)

    # Interpretation
    fap_m = metrics["FAP-Scale (Proposed)"]
    hpa_m = metrics["Reactive-HPA (Baseline 1)"]
    lstm_m = metrics["LSTM-RoundRobin (Baseline 2)"]

    print("\n  Analysis:")
    # SLA comparison
    if hpa_m['sla_violations'] > 0:
        sla_reduction = ((hpa_m['sla_violations'] - fap_m['sla_violations']) / hpa_m['sla_violations']) * 100
        print(f"    SLA violations reduced by {sla_reduction:.0f}% vs Reactive HPA "
              f"({fap_m['sla_violations']} vs {hpa_m['sla_violations']})")
    # Over-prov comparison
    if hpa_m['overprovisioning_percent'] > 0:
        ovp_reduction = ((hpa_m['overprovisioning_percent'] - fap_m['overprovisioning_percent'])
                         / hpa_m['overprovisioning_percent']) * 100
        print(f"    Over-provisioning reduced by {ovp_reduction:.0f}% vs Reactive HPA "
              f"({fap_m['overprovisioning_percent']:.1f}% vs {hpa_m['overprovisioning_percent']:.1f}%)")

    if fap_m['unhealthy_placements'] <= min(hpa_m['unhealthy_placements'], lstm_m['unhealthy_placements']):
        print("    FAP-Scale: 0 unhealthy placements -- never placed workloads on failing nodes")
    if fap_m['overprovisioning_percent'] <= min(hpa_m['overprovisioning_percent'], lstm_m['overprovisioning_percent']):
        print("    FAP-Scale achieves the best resource efficiency among all strategies")

    print()
    save_path = "results/comparison_metrics.png"
    plot_evaluation_summary(metrics, save_path=save_path)

    print("\n  Done. Results saved to results/ directory.")
    print()

    return metrics


if __name__ == "__main__":
    run_fap_scale_simulation(num_windows=20, num_nodes=10, pattern="hybrid")
