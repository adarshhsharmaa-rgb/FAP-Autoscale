"""
Baseline Autoscalers (Person C skeleton & implementation).

1. Reactive HPA Scaler: Scales reactively when CPU utilization crosses threshold (e.g. 70%).
2. LSTM-Only Round Robin Scaler: Predicts load using single LSTM, places replicas round-robin (blind to node failure & interference).
"""

import math
from typing import Dict, List, Any
import numpy as np

from models.forecasters import LSTMForecaster


class ReactiveHPAScaler:
    """
    Standard Reactive Horizontal Pod Autoscaler (HPA).
    Fires replica changes only after current CPU threshold (e.g. 70%) is exceeded.
    Placement selects nodes sequentially without health/interference awareness.
    """

    def __init__(self, target_cpu_threshold: float = 70.0, replica_capacity: float = 20.0):
        self.target_cpu_threshold = target_cpu_threshold
        self.replica_capacity = replica_capacity
        self.current_replicas = 2
        self.rr_index = 0

    def evaluate_window(
        self,
        current_load: float,
        cluster_telemetry: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        avg_cpu = float(np.mean([t.get("cpu_utilization", 30.0) for t in cluster_telemetry]))

        # Reactive formula: desired = ceil(current_replicas * (avg_cpu / target_cpu))
        if avg_cpu > self.target_cpu_threshold:
            desired = math.ceil(self.current_replicas * (avg_cpu / self.target_cpu_threshold))
        elif avg_cpu < 30.0:
            desired = max(1, math.floor(self.current_replicas * 0.8))
        else:
            desired = self.current_replicas

        self.current_replicas = max(1, desired)

        # Round-robin placement
        node_ids = [t.get("node_id", f"node-{i:02d}") for i, t in enumerate(cluster_telemetry)]
        selected = []
        for _ in range(self.current_replicas):
            selected.append(node_ids[self.rr_index % len(node_ids)])
            self.rr_index += 1

        return {
            "strategy": "Reactive-HPA",
            "target_replicas": self.current_replicas,
            "selected_nodes": selected
        }


class LSTMRoundRobinScaler:
    """
    Predictive LSTM-only scaler with Round-Robin Placement.
    Uses single LSTM model for forecasting, but ignores node failure and co-location interference.
    """

    def __init__(self, replica_capacity: float = 20.0):
        self.replica_capacity = replica_capacity
        self.forecaster = LSTMForecaster()
        self.rr_index = 0

    def evaluate_window(
        self,
        workload_window: List[float],
        cluster_telemetry: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        series = np.array(workload_window, dtype=float)
        pred_load = self.forecaster.predict_next(series)
        target_replicas = max(1, math.ceil(pred_load / self.replica_capacity))

        node_ids = [t.get("node_id", f"node-{i:02d}") for i, t in enumerate(cluster_telemetry)]
        selected = []
        for _ in range(target_replicas):
            selected.append(node_ids[self.rr_index % len(node_ids)])
            self.rr_index += 1

        return {
            "strategy": "LSTM-RoundRobin",
            "predicted_load": pred_load,
            "target_replicas": target_replicas,
            "selected_nodes": selected
        }
