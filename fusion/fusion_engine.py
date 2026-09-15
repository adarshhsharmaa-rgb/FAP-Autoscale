"""
Fusion Engine (Person C skeleton & implementation).

Computes composite node selection score:
S(i) = alpha * (1 - Pfail) + beta * (1 - IS) + gamma * (1 - U)

Calculates target replica count:
K = ceil(predicted_load / replica_capacity)

Ranks suitable nodes and logs joint autoscaling + placement decisions.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from models.failure_scorer import NodeFailureScorer, score_node_failure
from models.interference import InterferenceMatrix, get_interference
from models.forecasters import classify_and_forecast


def compute_node_score(
    pfail: float,
    is_score: float,
    utilization: float,
    alpha: float = 0.4,
    beta: float = 0.35,
    gamma: float = 0.25
) -> float:
    """
    Compute composite fusion placement score S(i):
    S(i) = alpha * (1 - Pfail) + beta * (1 - IS) + gamma * (1 - U)

    Args:
        pfail: Node failure probability in [0, 1]
        is_score: Co-location interference score in [0, 1]
        utilization: CPU utilization ratio in [0, 1]
        alpha: Weight for failure avoidance
        beta: Weight for interference minimization
        gamma: Weight for load balancing

    Returns:
        S(i): Composite score in range [0.0, 1.0] (higher is better)
    """
    u_ratio = float(np.clip(utilization / 100.0 if utilization > 1.0 else utilization, 0.0, 1.0))
    pf = float(np.clip(pfail, 0.0, 1.0))
    is_val = float(np.clip(is_score, 0.0, 1.0))

    score = alpha * (1.0 - pf) + beta * (1.0 - is_val) + gamma * (1.0 - u_ratio)
    return float(np.clip(score, 0.0, 1.0))


def calculate_replicas(predicted_load: float, replica_capacity: float = 20.0) -> int:
    """
    Calculate target replica count K:
    K = ceil(predicted_load / replica_capacity)

    Args:
        predicted_load: Forecasted workload request demand.
        replica_capacity: Single replica handling capacity.

    Returns:
        K: Target integer replica count (minimum 1).
    """
    if replica_capacity <= 0:
        return 1
    return max(1, math.ceil(predicted_load / replica_capacity))


class FusionEngine:
    """
    FAP-Scale Fusion Engine coordinator (Person C responsibility).
    Combines Person A's forecasting and Person B's node failure & interference scores.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.35,
        gamma: float = 0.25,
        replica_capacity: float = 20.0,
        failure_scorer: Optional[NodeFailureScorer] = None,
        interference_matrix: Optional[InterferenceMatrix] = None
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.replica_capacity = replica_capacity
        self.failure_scorer = failure_scorer
        self.interference_matrix = interference_matrix

    def evaluate_window(
        self,
        workload_window: List[float],
        cluster_telemetry: List[Dict[str, Any]],
        current_workload_type: str = "periodic"
    ) -> Dict[str, Any]:
        """
        Run fusion engine evaluation for a single control window.

        Args:
            workload_window: Recent workload history values.
            cluster_telemetry: List of telemetry dictionaries for all cluster nodes.
            current_workload_type: Category of current workload.

        Returns:
            Dictionary with decision log:
            {
                'pattern_label': str,
                'predicted_load': float,
                'target_replicas': int,
                'ranked_nodes': List[Tuple[str, float, float, float, float]]
            }
        """
        # Step 1: Classify workload & forecast load (Person A call)
        pattern_label, pred_load = classify_and_forecast(workload_window)

        # Step 2: Determine target replica count
        k_replicas = calculate_replicas(pred_load, self.replica_capacity)

        # Step 3: Score nodes using failure prediction & co-location interference (Person B calls)
        scored_nodes = []
        for telem in cluster_telemetry:
            nid = telem.get("node_id", "node-unknown")
            util = telem.get("cpu_utilization", 30.0)

            # Person B: Pfail
            pfail = score_node_failure(nid, telem, self.failure_scorer)

            # Person B: Interference score IS
            is_score = get_interference(current_workload_type, "hybrid", self.interference_matrix)

            # Fusion score S(i)
            s_i = compute_node_score(pfail, is_score, util, self.alpha, self.beta, self.gamma)

            scored_nodes.append({
                "node_id": nid,
                "score": s_i,
                "pfail": pfail,
                "is_score": is_score,
                "utilization": util
            })

        # Rank nodes descending by composite score S(i)
        scored_nodes.sort(key=lambda x: x["score"], reverse=True)

        return {
            "pattern_label": pattern_label,
            "predicted_load": pred_load,
            "target_replicas": k_replicas,
            "ranked_nodes": scored_nodes,
            "selected_nodes": [n["node_id"] for n in scored_nodes[:k_replicas]]
        }
