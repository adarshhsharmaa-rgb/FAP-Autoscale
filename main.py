"""
Main integration script for FAP-Scale framework.

Ties together:
- Data Generation (Person A workload + Person B node telemetry)
- Model Training (Person B LightGBM Failure Scorer & Interference Matrix)
- Workload Pattern Classification & Forecasting (Person A)
- Fusion Control Loop & Node Placement (Person C)
- Baseline Comparison & Evaluation (Reactive HPA vs LSTM-Only vs FAP-Scale)
"""

from typing import Dict, Any
import numpy as np

# Local package imports
from data_gen.synthetic_data import (
    generate_workload_time_series,
    generate_node_telemetry,
    extract_single_node_telemetry
)
from models.failure_scorer import NodeFailureScorer, score_node_failure
from models.interference import InterferenceMatrix
from fusion.fusion_engine import FusionEngine
from simulator.baselines import ReactiveHPAScaler, LSTMRoundRobinScaler
from results.evaluate import evaluate_experiment_results, plot_evaluation_summary


def run_fap_scale_simulation(
    num_windows: int = 30,
    num_nodes: int = 10,
    pattern: str = "periodic"
) -> Dict[str, Any]:
    """
    Run end-to-end FAP-Scale control loop simulation.
    """
    print("=" * 70)
    print(f"  FAP-SCALE FRAMEWORK DEMO SIMULATION (Pattern: {pattern.upper()})")
    print("=" * 70)

    # Step 1: Generate synthetic workload and node telemetry
    print("\n[Step 1] Generating synthetic workload & cluster node telemetry...")
    workload_df = generate_workload_time_series(num_windows=num_windows, pattern=pattern)
    node_telemetry_df = generate_node_telemetry(num_nodes=num_nodes, num_windows=num_windows, fail_node_ids=["node-03", "node-08"])
    print(f"  - Generated {len(workload_df)} workload data points over {num_windows} control windows.")
    print(f"  - Generated telemetry for {num_nodes} nodes across {num_windows} control windows.")

    # Step 2: Initialize and train Person B's LightGBM Failure Scorer & Interference Matrix
    print("\n[Step 2] Initializing & Training Person B Node Health Models...")
    failure_scorer = NodeFailureScorer(random_state=42)
    failure_scorer.fit(node_telemetry_df)
    print("  - LightGBM NodeFailureScorer successfully trained on synthetic telemetry.")

    interference_matrix = InterferenceMatrix(alpha=0.1, seed=42)
    print("  - 15-bin Co-location Interference Matrix initialized with EMA alpha=0.1.")

    # Step 3: Initialize Person C's Fusion Engine
    fusion_engine = FusionEngine(
        alpha=0.4,
        beta=0.35,
        gamma=0.25,
        replica_capacity=20.0,
        failure_scorer=failure_scorer,
        interference_matrix=interference_matrix
    )

    # Initialize Baselines for comparison
    hpa_baseline = ReactiveHPAScaler(target_cpu_threshold=70.0, replica_capacity=20.0)
    lstm_baseline = LSTMRoundRobinScaler(replica_capacity=20.0)

    fap_history = []
    hpa_history = []
    lstm_history = []

    print("\n[Step 3] Executing Simulation Control Loop (60s window ticks)...")
    print("-" * 75)
    print(f"{'Win':<4} | {'Actual':<7} | {'Pattern':<9} | {'Pred':<7} | {'K_fap':<6} | {'Top Node':<10} | {'Pfail':<6} | {'IS':<5} | {'S(i)':<5}")
    print("-" * 75)

    for w in range(num_windows):
        # Extract workload window (last 60s)
        win_workload = workload_df[workload_df["window_id"] == w]["request_rate"].values
        actual_load = float(np.mean(win_workload)) if len(win_workload) > 0 else 50.0

        # Collect current telemetry for all cluster nodes
        curr_telemetry = []
        node_pfails = {}
        for n in range(1, num_nodes + 1):
            nid = f"node-{n:02d}"
            t_dict = extract_single_node_telemetry(node_telemetry_df, nid, w)
            curr_telemetry.append(t_dict)
            pf = score_node_failure(nid, t_dict, failure_scorer)
            node_pfails[nid] = pf

        # Run FAP-Scale Fusion Engine
        fap_result = fusion_engine.evaluate_window(
            workload_window=win_workload,
            cluster_telemetry=curr_telemetry,
            current_workload_type=pattern
        )
        fap_result["actual_load"] = actual_load
        fap_result["node_pfails"] = node_pfails
        fap_history.append(fap_result)

        # Run Baselines
        hpa_res = hpa_baseline.evaluate_window(actual_load, curr_telemetry)
        hpa_res["actual_load"] = actual_load
        hpa_res["node_pfails"] = node_pfails
        hpa_history.append(hpa_res)

        lstm_res = lstm_baseline.evaluate_window(win_workload, curr_telemetry)
        lstm_res["actual_load"] = actual_load
        lstm_res["node_pfails"] = node_pfails
        lstm_history.append(lstm_res)

        # Log sample output ticks
        if w < 5 or w % 5 == 0 or w == num_windows - 1:
            top_node_info = fap_result["ranked_nodes"][0]
            top_nid = top_node_info["node_id"]
            top_pf = top_node_info["pfail"]
            top_is = top_node_info["is_score"]
            top_s = top_node_info["score"]

            print(f"{w:<4} | {actual_load:<7.1f} | {fap_result['pattern_label']:<9} | "
                  f"{fap_result['predicted_load']:<7.1f} | {fap_result['target_replicas']:<6} | "
                  f"{top_nid:<10} | {top_pf:<6.2f} | {top_is:<5.2f} | {top_s:<5.2f}")

    print("-" * 75)

    # Step 4: Evaluate results & comparison
    print("\n[Step 4] Final Results & Baseline Comparison Metrics:")
    metrics = {
        "FAP-Scale (Proposed)": evaluate_experiment_results(fap_history),
        "Reactive-HPA (Baseline 1)": evaluate_experiment_results(hpa_history),
        "LSTM-RoundRobin (Baseline 2)": evaluate_experiment_results(lstm_history)
    }

    print("\n" + "=" * 75)
    print(f"{'Autoscaling Strategy':<30} | {'SLA Violations':<15} | {'Over-Prov %':<12} | {'Unhealthy Placements':<20}")
    print("=" * 75)
    for strat, m in metrics.items():
        print(f"{strat:<30} | {m['sla_violations']:<15} | {m['overprovisioning_percent']:<12.1f}% | {m['unhealthy_placements']:<20}")
    print("=" * 75 + "\n")

    plot_evaluation_summary(metrics, save_path="results/comparison_metrics.png")
    return metrics


if __name__ == "__main__":
    run_fap_scale_simulation(num_windows=20, num_nodes=10, pattern="hybrid")
