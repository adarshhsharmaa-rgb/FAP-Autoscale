"""
Evaluation & Metrics Module (Shared module across team).

Calculates:
1. SLA Violations (under-provisioned demand or placements on failed nodes)
2. Over-provisioning % (wasted capacity)
3. Unhealthy Placement Count (placing replicas on nodes with high Pfail)
4. Generates comparison plots across baselines and FAP-Scale framework.
"""

from typing import Dict, List, Any
import numpy as np


def evaluate_experiment_results(
    decision_history: List[Dict[str, Any]],
    replica_capacity: float = 20.0
) -> Dict[str, Any]:
    """
    Calculate performance metrics over a simulation run.

    Args:
        decision_history: List of window decision dictionaries.
        replica_capacity: Capacity per replica (req/sec). Default 20.0.

    Returns:
        Metrics summary dict:
        {
            'total_windows': int,
            'sla_violations': int,
            'overprovisioning_percent': float,
            'unhealthy_placements': int,
            'avg_replicas': float
        }
    """
    if not decision_history:
        return {
            "total_windows": 0,
            "sla_violations": 0,
            "overprovisioning_percent": 0.0,
            "unhealthy_placements": 0,
            "avg_replicas": 0.0
        }

    total_windows = len(decision_history)
    sla_violations = 0
    unhealthy_placements = 0
    overprov_list = []
    replica_counts = []

    for entry in decision_history:
        actual_load = entry.get("actual_load", 50.0)
        target_replicas = entry.get("target_replicas", 2)
        total_capacity = target_replicas * replica_capacity

        replica_counts.append(target_replicas)

        # SLA violation: demand exceeds allocated capacity
        if total_capacity < actual_load:
            sla_violations += 1

        # Over-provisioning %
        if total_capacity > actual_load:
            overprov = ((total_capacity - actual_load) / total_capacity) * 100.0
            overprov_list.append(overprov)

        # Unhealthy placement count
        selected = entry.get("selected_nodes", [])
        node_pfails = entry.get("node_pfails", {})
        for nid in selected:
            if node_pfails.get(nid, 0.0) > 0.4:
                unhealthy_placements += 1

    return {
        "total_windows": total_windows,
        "sla_violations": sla_violations,
        "overprovisioning_percent": float(np.mean(overprov_list)) if overprov_list else 0.0,
        "unhealthy_placements": unhealthy_placements,
        "avg_replicas": float(np.mean(replica_counts)) if replica_counts else 0.0
    }


def plot_evaluation_summary(metrics_by_strategy: Dict[str, Dict[str, Any]], save_path: str = "results_plot.png"):
    """
    Generate bar charts comparing metrics across autoscaling strategies.
    """
    try:
        import matplotlib.pyplot as plt

        strategies = list(metrics_by_strategy.keys())
        sla = [metrics_by_strategy[s]["sla_violations"] for s in strategies]
        overprov = [metrics_by_strategy[s]["overprovisioning_percent"] for s in strategies]
        unhealthy = [metrics_by_strategy[s]["unhealthy_placements"] for s in strategies]

        # Dynamic color palette that scales with number of strategies
        default_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]
        colors = (default_colors * ((len(strategies) // len(default_colors)) + 1))[:len(strategies)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        axes[0].bar(strategies, sla, color=colors)
        axes[0].set_title("SLA Violations (Lower is Better)")
        axes[0].set_ylabel("Count")

        axes[1].bar(strategies, overprov, color=colors)
        axes[1].set_title("Over-Provisioning % (Lower is Better)")
        axes[1].set_ylabel("Percentage (%)")

        axes[2].bar(strategies, unhealthy, color=colors)
        axes[2].set_title("Unhealthy Node Placements (Lower is Better)")
        axes[2].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Evaluation plot saved successfully to {save_path}")
    except Exception as e:
        print(f"Note: Matplotlib plot generation skipped ({e})")
