"""
Fusion Engine (Person C — full implementation).

Computes composite node placement score:
    S(i) = alpha * (1 - Pfail) + beta * (1 - IS) + gamma * (1 - U)

Calculates target replica count:
    K = ceil(predicted_load / replica_capacity)

Executes 60-second window control loop:
- Classifies workload and forecasts next-window load (Person A)
- Scores every node using failure probability and co-location interference (Person B)
- Ranks nodes by S(i) and selects top-K for replica placement
- Feeds observed degradation back into the interference matrix (EMA update)
- Maintains full per-window decision history for downstream evaluation

Person C deliverables:
    FusionEngine.evaluate_window(workload_window, cluster_telemetry) -> decision_dict
    compute_node_score(pfail, is_score, utilization) -> S_i
    calculate_replicas(predicted_load, replica_capacity) -> K
"""

import math
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from models.node_health.failure_scorer import NodeFailureScorer, score_node_failure
from models.node_health.interference import InterferenceMatrix, get_interference
from models.forecasters import classify_and_forecast


# ─────────────────────────────────────────────
# Stateless helper functions (also exported for tests)
# ─────────────────────────────────────────────

def compute_node_score(
    pfail: float,
    is_score: float,
    utilization: float,
    alpha: float = 0.4,
    beta: float = 0.35,
    gamma: float = 0.25,
) -> float:
    """
    Compute composite fusion placement score S(i):
        S(i) = alpha * (1 - Pfail) + beta * (1 - IS) + gamma * (1 - U)

    Args:
        pfail:       Node failure probability in [0, 1]
        is_score:    Co-location interference score in [0, 1]
        utilization: CPU utilisation value (% or ratio — auto-normalised)
        alpha:       Weight for failure avoidance (default 0.4)
        beta:        Weight for interference minimisation (default 0.35)
        gamma:       Weight for load balancing (default 0.25)

    Returns:
        S(i) composite score in [0.0, 1.0]; higher is better for placement.
    """
    # Normalise utilisation: accept either 0-100 (%) or 0-1 ratio
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
        predicted_load:   Forecasted workload request demand (req/s or CPU units).
        replica_capacity: Single replica handling capacity (same units as predicted_load).

    Returns:
        K: Target integer replica count (minimum 1).
    """
    if replica_capacity <= 0:
        return 1
    return max(1, math.ceil(predicted_load / replica_capacity))


def _estimate_observed_degradation(
    scored_nodes: List[Dict[str, Any]],
    selected_indices: int,
) -> float:
    """
    Estimate observed interference degradation after placement.

    Uses average CPU utilisation of *selected* nodes as a proxy for
    actual resource contention — high utilisation implies higher co-location
    degradation. This keeps the EMA feedback loop data-driven without
    needing a separate profiler.

    Args:
        scored_nodes:     Full ranked node list (dicts with 'utilization').
        selected_indices: Number of nodes selected (top-K).

    Returns:
        Estimated degradation ratio in [0.0, 1.0].
    """
    if not scored_nodes or selected_indices == 0:
        return 0.1
    selected = scored_nodes[:selected_indices]
    avg_util = np.mean([n.get("utilization", 30.0) for n in selected])
    # Normalise 0–100 utilisation to 0–1 degradation range
    return float(np.clip(avg_util / 100.0, 0.05, 1.0))


# ─────────────────────────────────────────────
# Stateful Fusion Engine class
# ─────────────────────────────────────────────

class FusionEngine:
    """
    FAP-Scale Fusion Engine — Person C core component.

    Coordinates:
      * Person A: classify_and_forecast() for workload classification & load prediction
      * Person B: score_node_failure() for Pfail, InterferenceMatrix for IS

    Maintains per-window decision history and feeds observed degradation
    back into the interference matrix via EMA after every placement cycle.

    Attributes:
        alpha, beta, gamma: Scoring weights summing to 1.0 (failure, interference, utilisation).
        replica_capacity:   Max request rate a single replica can handle.
        failure_scorer:     Optional fitted NodeFailureScorer (Person B).
        interference_matrix: InterferenceMatrix instance (Person B).
        history:            List of all per-window decision dicts (for evaluation).
        window_count:       Running count of processed control windows.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.35,
        gamma: float = 0.25,
        replica_capacity: float = 20.0,
        failure_scorer: Optional[NodeFailureScorer] = None,
        interference_matrix: Optional[InterferenceMatrix] = None,
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, (
            f"Weights must sum to 1.0, got {alpha + beta + gamma:.4f}"
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.replica_capacity = replica_capacity
        self.failure_scorer = failure_scorer
        self.interference_matrix = interference_matrix if interference_matrix is not None else InterferenceMatrix()

        # Control-loop state
        self.history: List[Dict[str, Any]] = []
        self.window_count: int = 0
        self._prev_pattern: str = "periodic"  # carry last pattern for EMA pairing

    # ─── main per-window evaluation ───────────────────────────────────────────

    def evaluate_window(
        self,
        workload_window: List[float],
        cluster_telemetry: List[Dict[str, Any]],
        current_workload_type: str = "periodic",
    ) -> Dict[str, Any]:
        """
        Run fusion engine evaluation for a single 60-second control window.

        Pipeline:
            1. Classify workload & forecast next-window load  (Person A)
            2. Compute target replica count K
            3. Score every node: Pfail (Person B) + IS (Person B) → S(i)
            4. Rank nodes descending by S(i); select top-K
            5. Feed observed degradation back into interference matrix (EMA)
            6. Append decision to history and return decision dict

        Args:
            workload_window:      Recent workload history values (request rates).
            cluster_telemetry:    List of telemetry dicts for all cluster nodes.
            current_workload_type: Hint for interference lookup if classifier is uncertain.

        Returns:
            decision dict with keys:
            {
                'window_id':       int,
                'timestamp':       float  (Unix epoch),
                'pattern_label':   str,
                'predicted_load':  float,
                'target_replicas': int,
                'ranked_nodes':    List[Dict] — node_id, score, pfail, is_score, utilization,
                'selected_nodes':  List[str]  — top-K node IDs chosen for placement,
                'ema_update':      float       — degradation fed into matrix this window,
            }
        """
        window_id = self.window_count
        ts = time.time()

        # ── Step 1: Person A — Workload classification & load forecast ──────
        series = np.array(workload_window, dtype=float) if len(workload_window) > 0 else np.array([50.0])
        pattern_label, pred_load = classify_and_forecast(series)

        # ── Step 2: Target replica count ────────────────────────────────────
        k_replicas = calculate_replicas(pred_load, self.replica_capacity)

        # ── Step 3: Node scoring using Person B outputs ─────────────────────
        scored_nodes: List[Dict[str, Any]] = []
        for telem in cluster_telemetry:
            nid = telem.get("node_id", "node-unknown")
            util = float(telem.get("cpu_utilization", 30.0))

            # Person B: Failure probability
            pfail = score_node_failure(nid, telem, self.failure_scorer)

            # Person B: Co-location interference score
            # The incoming node's existing workload type vs new workload to be placed
            node_wl_type = telem.get("pattern_type", current_workload_type)
            is_score = get_interference(pattern_label, node_wl_type, self.interference_matrix)

            # Composite fusion score S(i)
            s_i = compute_node_score(pfail, is_score, util, self.alpha, self.beta, self.gamma)

            scored_nodes.append({
                "node_id": nid,
                "score": round(s_i, 4),
                "pfail": round(pfail, 4),
                "is_score": round(is_score, 4),
                "utilization": round(util, 2),
            })

        # ── Step 4: Rank nodes and select top-K ─────────────────────────────
        scored_nodes.sort(key=lambda x: x["score"], reverse=True)
        max_selectable = min(k_replicas, len(scored_nodes))
        selected_nodes = [n["node_id"] for n in scored_nodes[:max_selectable]]

        # ── Step 5: EMA feedback — update interference matrix ────────────────
        observed_degradation = _estimate_observed_degradation(scored_nodes, max_selectable)
        self.interference_matrix.update_ema(
            pattern_label,
            self._prev_pattern,
            observed_degradation,
        )
        self._prev_pattern = pattern_label  # carry forward for next window

        # ── Step 6: Record decision ──────────────────────────────────────────
        decision = {
            "window_id": window_id,
            "timestamp": ts,
            "pattern_label": pattern_label,
            "predicted_load": round(pred_load, 2),
            "target_replicas": k_replicas,
            "ranked_nodes": scored_nodes,
            "selected_nodes": selected_nodes,
            "ema_update": round(observed_degradation, 4),
        }
        self.history.append(decision)
        self.window_count += 1
        return decision

    # ─── utility ──────────────────────────────────────────────────────────────

    def get_history(self) -> List[Dict[str, Any]]:
        """Return full per-window decision history."""
        return self.history

    def reset(self):
        """Reset engine state (history, window counter, EMA carry-over)."""
        self.history = []
        self.window_count = 0
        self._prev_pattern = "periodic"

    def summary_stats(self) -> Dict[str, Any]:
        """
        Compute high-level summary statistics over all processed windows.

        Returns:
            Dict with avg_replicas, avg_top_node_score, avg_top_pfail,
            total_ema_updates, windows_processed.
        """
        if not self.history:
            return {"windows_processed": 0}

        avg_k = np.mean([d["target_replicas"] for d in self.history])
        top_scores = [d["ranked_nodes"][0]["score"] for d in self.history if d["ranked_nodes"]]
        top_pfails = [d["ranked_nodes"][0]["pfail"] for d in self.history if d["ranked_nodes"]]
        ema_vals = [d["ema_update"] for d in self.history]

        return {
            "windows_processed": self.window_count,
            "avg_replicas": round(float(avg_k), 2),
            "avg_top_node_score": round(float(np.mean(top_scores)), 4) if top_scores else 0.0,
            "avg_top_pfail": round(float(np.mean(top_pfails)), 4) if top_pfails else 0.0,
            "avg_ema_degradation": round(float(np.mean(ema_vals)), 4) if ema_vals else 0.0,
        }
