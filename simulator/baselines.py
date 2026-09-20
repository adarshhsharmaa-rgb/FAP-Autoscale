"""
Baseline Autoscalers (Person C — full implementation).

Two comparison baselines for FAP-Scale evaluation:

1. ReactiveHPAScaler
   - Standard Kubernetes-style Horizontal Pod Autoscaler.
   - Scales *reactively* (after the fact) when observed avg CPU crosses a threshold (default 70 %).
   - Placement: blind round-robin over all nodes — no failure awareness, no interference scoring.

2. LSTMRoundRobinScaler
   - Predictive scaler using a lightweight LSTM / EMA forecaster (Person A module).
   - Forecasts next-window demand and sizes replicas proactively.
   - Placement: still round-robin — no Pfail or IS awareness.

Both baselines expose the same `evaluate_window()` interface as FusionEngine so they can
be dropped into the simulation control loop unchanged.
"""

import math
from typing import Dict, List, Any, Tuple
import numpy as np

from models.forecasters import LSTMForecaster


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1 — Reactive HPA
# ─────────────────────────────────────────────────────────────────────────────

class ReactiveHPAScaler:
    """
    Standard Reactive Horizontal Pod Autoscaler (HPA).

    Scaling logic:
        - If avg_cpu > target_cpu_threshold: scale UP → desired = ceil(replicas * avg_cpu / target)
        - If avg_cpu < 30 %: scale DOWN → desired = floor(replicas * 0.8)
        - Otherwise: hold current replica count

    Placement: round-robin across all nodes in the cluster (no health/interference checks).

    Attributes:
        target_cpu_threshold: CPU % trigger for scale-up (default 70.0).
        replica_capacity:     Max load a single replica handles (req/s).
        current_replicas:     Live replica count, updated each window.
        rr_index:             Round-robin cursor.
        history:              Per-window decision log.
    """

    def __init__(self, target_cpu_threshold: float = 70.0, replica_capacity: float = 20.0):
        self.target_cpu_threshold = target_cpu_threshold
        self.replica_capacity = replica_capacity
        self.current_replicas: int = 2
        self.rr_index: int = 0
        self.history: List[Dict[str, Any]] = []

    def evaluate_window(
        self,
        current_load: float,
        cluster_telemetry: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate one control window and return scaling + placement decision.

        Args:
            current_load:      Observed (actual) workload in this window.
            cluster_telemetry: List of node telemetry dicts.

        Returns:
            Decision dict: strategy, target_replicas, selected_nodes, avg_cpu.
        """
        if not cluster_telemetry:
            decision = {
                "strategy": "Reactive-HPA",
                "target_replicas": self.current_replicas,
                "selected_nodes": [],
                "avg_cpu": 0.0,
            }
            self.history.append(decision)
            return decision

        avg_cpu = float(np.mean([t.get("cpu_utilization", 30.0) for t in cluster_telemetry]))

        # ── Reactive scaling formula ─────────────────────────────────────────
        if avg_cpu > self.target_cpu_threshold:
            desired = math.ceil(self.current_replicas * (avg_cpu / self.target_cpu_threshold))
        elif avg_cpu < 30.0:
            desired = max(1, math.floor(self.current_replicas * 0.8))
        else:
            desired = self.current_replicas

        self.current_replicas = max(1, desired)

        # ── Round-robin placement (blind, no health awareness) ───────────────
        node_ids = [t.get("node_id", f"node-{i:02d}") for i, t in enumerate(cluster_telemetry)]
        selected: List[str] = []
        for _ in range(self.current_replicas):
            selected.append(node_ids[self.rr_index % len(node_ids)])
            self.rr_index += 1

        decision = {
            "strategy": "Reactive-HPA",
            "target_replicas": self.current_replicas,
            "selected_nodes": selected,
            "avg_cpu": round(avg_cpu, 2),
        }
        self.history.append(decision)
        return decision

    def reset(self):
        """Reset scaler state."""
        self.current_replicas = 2
        self.rr_index = 0
        self.history = []


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2 — LSTM-Only Round-Robin Scaler
# ─────────────────────────────────────────────────────────────────────────────

class LSTMRoundRobinScaler:
    """
    Predictive LSTM-only scaler with round-robin placement.

    Uses the lightweight LSTMForecaster from Person A's module for demand
    forecasting (proactive), but ignores node failure probability (Pfail) and
    co-location interference (IS) during placement — pure round-robin.

    Attributes:
        replica_capacity: Max load a single replica handles.
        forecaster:       LSTMForecaster instance.
        rr_index:         Round-robin cursor.
        history:          Per-window decision log.
    """

    def __init__(self, replica_capacity: float = 20.0):
        self.replica_capacity = replica_capacity
        self.forecaster = LSTMForecaster()
        self.rr_index: int = 0
        self.history: List[Dict[str, Any]] = []

    def evaluate_window(
        self,
        workload_window: List[float],
        cluster_telemetry: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate one control window and return scaling + placement decision.

        Args:
            workload_window:   Recent workload time-series values.
            cluster_telemetry: List of node telemetry dicts.

        Returns:
            Decision dict: strategy, predicted_load, target_replicas, selected_nodes.
        """
        if not cluster_telemetry:
            decision = {
                "strategy": "LSTM-RoundRobin",
                "predicted_load": 0.0,
                "target_replicas": 1,
                "selected_nodes": [],
            }
            self.history.append(decision)
            return decision

        # ── LSTM forecast ────────────────────────────────────────────────────
        series = np.array(workload_window, dtype=float)
        pred_load = self.forecaster.predict_next(series)
        target_replicas = max(1, math.ceil(pred_load / self.replica_capacity))

        # ── Round-robin placement (blind) ────────────────────────────────────
        node_ids = [t.get("node_id", f"node-{i:02d}") for i, t in enumerate(cluster_telemetry)]
        selected: List[str] = []
        for _ in range(target_replicas):
            selected.append(node_ids[self.rr_index % len(node_ids)])
            self.rr_index += 1

        decision = {
            "strategy": "LSTM-RoundRobin",
            "predicted_load": round(pred_load, 2),
            "target_replicas": target_replicas,
            "selected_nodes": selected,
        }
        self.history.append(decision)
        return decision

    def reset(self):
        """Reset scaler state."""
        self.rr_index = 0
        self.history = []
