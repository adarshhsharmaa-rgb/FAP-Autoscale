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
    Handles overlapping labels, integer y-axes, and zero-value charts cleanly.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        strategies = list(metrics_by_strategy.keys())
        # Use short labels for x-axis, full names in legend/title
        short_labels = []
        for s in strategies:
            if "FAP" in s:
                short_labels.append("FAP-Scale")
            elif "HPA" in s:
                short_labels.append("Reactive HPA")
            elif "LSTM" in s or "RoundRobin" in s:
                short_labels.append("LSTM-RR")
            else:
                short_labels.append(s[:12])

        sla = [metrics_by_strategy[s]["sla_violations"] for s in strategies]
        overprov = [metrics_by_strategy[s]["overprovisioning_percent"] for s in strategies]
        unhealthy = [metrics_by_strategy[s]["unhealthy_placements"] for s in strategies]

        colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"]
        colors = (colors * ((len(strategies) // len(colors)) + 1))[:len(strategies)]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        fig.suptitle("FAP-Scale vs Baseline Autoscaling Strategies", fontsize=14, fontweight="bold", y=1.02)

        # --- Chart 1: SLA Violations ---
        bars1 = axes[0].bar(short_labels, sla, color=colors, edgecolor="white", linewidth=1.2)
        axes[0].set_title("SLA Violations\n(Lower is Better)", fontsize=11)
        axes[0].set_ylabel("Count")
        axes[0].set_ylim(0, max(max(sla) * 1.3, 1))
        # Add value labels on bars
        for bar, val in zip(bars1, sla):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         str(int(val)), ha="center", va="bottom", fontweight="bold", fontsize=11)

        # --- Chart 2: Over-Provisioning % ---
        bars2 = axes[1].bar(short_labels, overprov, color=colors, edgecolor="white", linewidth=1.2)
        axes[1].set_title("Over-Provisioning %\n(Lower is Better)", fontsize=11)
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_ylim(0, max(max(overprov) * 1.3, 1))
        for bar, val in zip(bars2, overprov):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

        # --- Chart 3: Unhealthy Placements ---
        bars3 = axes[2].bar(short_labels, unhealthy, color=colors, edgecolor="white", linewidth=1.2)
        axes[2].set_title("Unhealthy Node Placements\n(Lower is Better)", fontsize=11)
        axes[2].set_ylabel("Count")
        # Fix: when all values are 0, set a sensible y-axis range instead of 0.04
        max_unhealthy = max(unhealthy) if max(unhealthy) > 0 else 5
        axes[2].set_ylim(0, max_unhealthy * 1.3)
        axes[2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for bar, val in zip(bars3, unhealthy):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         str(int(val)), ha="center", va="bottom", fontweight="bold", fontsize=11)

        # Rotate x-axis labels slightly for readability
        for ax in axes:
            ax.tick_params(axis="x", rotation=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Evaluation plot saved successfully to {save_path}")
    except Exception as e:
        print(f"Note: Matplotlib plot generation skipped ({e})")
