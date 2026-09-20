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

# ── Data generation ──────────────────────────────────────────────────────────
from data_gen.synthetic_data import (
    generate_workload_time_series,
    generate_node_telemetry,
    extract_single_node_telemetry,
)

# ── Person B models ──────────────────────────────────────────────────────────
from models.node_health.failure_scorer import NodeFailureScorer, score_node_failure
from models.node_health.interference import InterferenceMatrix

# ── Person C modules ─────────────────────────────────────────────────────────
from fusion.fusion_engine import FusionEngine
from simulator.baselines import ReactiveHPAScaler, LSTMRoundRobinScaler

# ── Evaluation ───────────────────────────────────────────────────────────────
from results.evaluate import evaluate_experiment_results, plot_evaluation_summary


def run_fap_scale_simulation(
    num_windows: int = 30,
    num_nodes: int = 10,
    pattern: str = "periodic",
) -> Dict[str, Any]:
    """
    Run end-to-end FAP-Scale control loop simulation.

    Args:
        num_windows: Number of 60-second control windows to simulate.
        num_nodes:   Number of cluster nodes to simulate.
        pattern:     Workload pattern: 'periodic', 'bursty', or 'hybrid'.

    Returns:
        Metrics dict mapping strategy name → evaluation metrics.
    """
    # Ensure results output directory exists
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print(f"  FAP-SCALE FRAMEWORK DEMO SIMULATION (Pattern: {pattern.upper()})")
    print("=" * 70)

    # ── Step 1: Generate synthetic workload & node telemetry ─────────────────
    print("\n[Step 1] Generating synthetic workload & cluster node telemetry...")
    workload_df = generate_workload_time_series(
        num_windows=num_windows, pattern=pattern, seed=42
    )
    node_telemetry_df = generate_node_telemetry(
        num_nodes=num_nodes,
        num_windows=num_windows,
        fail_node_ids=["node-03", "node-08"],
        seed=42,
    )
    print(f"  - Generated {len(workload_df)} workload data points over {num_windows} control windows.")
    print(f"  - Generated telemetry for {num_nodes} nodes across {num_windows} control windows.")

    # ── Step 2: Train Person B models ────────────────────────────────────────
    print("\n[Step 2] Initialising & Training Person B Node Health Models...")
    failure_scorer = NodeFailureScorer(random_state=42)
    failure_scorer.fit(node_telemetry_df)
    backend = "LightGBM" if getattr(failure_scorer, "model", None).__class__.__name__ == "LGBMClassifier" else "HistGradientBoosting"
    print(f"  - NodeFailureScorer trained ({backend}) on {len(node_telemetry_df)} telemetry rows.")

    interference_matrix = InterferenceMatrix(alpha=0.1, seed=42)
    print("  - 15-bin Co-location Interference Matrix initialised (EMA α=0.1).")

    # ── Step 3: Initialise Person C strategies ───────────────────────────────
    print("\n[Step 3] Initialising FAP-Scale Fusion Engine & Baseline Scalers...")
    fusion_engine = FusionEngine(
        alpha=0.4,
        beta=0.35,
        gamma=0.25,
        replica_capacity=20.0,
        failure_scorer=failure_scorer,
        interference_matrix=interference_matrix,
    )
    hpa_baseline = ReactiveHPAScaler(target_cpu_threshold=70.0, replica_capacity=20.0)
    lstm_baseline = LSTMRoundRobinScaler(replica_capacity=20.0)

    fap_history = []
    hpa_history = []
    lstm_history = []

    # ── Step 4: 60-second window control loop ────────────────────────────────
    print("\n[Step 4] Executing Simulation Control Loop (60 s window ticks)...")
    header = (
        f"{'Win':<4} | {'Actual':>7} | {'Pattern':<9} | {'Pred':>7} | "
        f"{'K_fap':>5} | {'Top Node':<10} | {'Pfail':>6} | {'IS':>5} | {'S(i)':>5} | EMA"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for w in range(num_windows):
        # Current workload window (60 s slice)
        win_workload = workload_df[workload_df["window_id"] == w]["request_rate"].values
        actual_load = float(np.mean(win_workload)) if len(win_workload) > 0 else 50.0

        # Gather telemetry for all nodes at this window
        curr_telemetry = []
        node_pfails: Dict[str, float] = {}
        for n in range(1, num_nodes + 1):
            nid = f"node-{n:02d}"
            t_dict = extract_single_node_telemetry(node_telemetry_df, nid, w)
            curr_telemetry.append(t_dict)
            node_pfails[nid] = score_node_failure(nid, t_dict, failure_scorer)

        # ── FAP-Scale Fusion Engine ──────────────────────────────────────────
        fap_result = fusion_engine.evaluate_window(
            workload_window=list(win_workload),
            cluster_telemetry=curr_telemetry,
            current_workload_type=pattern,
        )
        fap_result["actual_load"] = actual_load
        fap_result["node_pfails"] = node_pfails
        fap_history.append(fap_result)

        # ── Reactive HPA baseline ────────────────────────────────────────────
        hpa_res = hpa_baseline.evaluate_window(actual_load, curr_telemetry)
        hpa_res["actual_load"] = actual_load
        hpa_res["node_pfails"] = node_pfails
        hpa_history.append(hpa_res)

        # ── LSTM Round-Robin baseline ────────────────────────────────────────
        lstm_res = lstm_baseline.evaluate_window(list(win_workload), curr_telemetry)
        lstm_res["actual_load"] = actual_load
        lstm_res["node_pfails"] = node_pfails
        lstm_history.append(lstm_res)

        # ── Console logging (first 5 + every 5th + last) ─────────────────────
        if w < 5 or w % 5 == 0 or w == num_windows - 1:
            top = fap_result["ranked_nodes"][0] if fap_result["ranked_nodes"] else {}
            print(
                f"{w:<4} | {actual_load:>7.1f} | {fap_result['pattern_label']:<9} | "
                f"{fap_result['predicted_load']:>7.1f} | {fap_result['target_replicas']:>5} | "
                f"{top.get('node_id', '-'):<10} | {top.get('pfail', 0):>6.3f} | "
                f"{top.get('is_score', 0):>5.3f} | {top.get('score', 0):>5.3f} | "
                f"{fap_result['ema_update']:.3f}"
            )

    print("-" * len(header))

    # ── Step 5: Summary stats from Fusion Engine ─────────────────────────────
    print("\n[Step 5] Fusion Engine Summary:")
    stats = fusion_engine.summary_stats()
    for k, v in stats.items():
        print(f"  - {k}: {v}")

    # ── Step 6: Evaluation & baseline comparison ──────────────────────────────
    print("\n[Step 6] Final Results & Baseline Comparison Metrics:")
    metrics = {
        "FAP-Scale (Proposed)":       evaluate_experiment_results(fap_history),
        "Reactive-HPA (Baseline 1)":  evaluate_experiment_results(hpa_history),
        "LSTM-RoundRobin (Baseline 2)": evaluate_experiment_results(lstm_history),
    }

    print("\n" + "=" * 80)
    print(
        f"{'Strategy':<32} | {'SLA Violations':>14} | "
        f"{'Over-Prov %':>11} | {'Unhealthy Placements':>20}"
    )
    print("=" * 80)
    for strat, m in metrics.items():
        print(
            f"{strat:<32} | {m['sla_violations']:>14} | "
            f"{m['overprovisioning_percent']:>10.1f}% | {m['unhealthy_placements']:>20}"
        )
    print("=" * 80 + "\n")

    save_path = "results/comparison_metrics.png"
    plot_evaluation_summary(metrics, save_path=save_path)

    return metrics


if __name__ == "__main__":
    run_fap_scale_simulation(num_windows=20, num_nodes=10, pattern="hybrid")
