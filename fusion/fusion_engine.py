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

On real traces three optional pieces make scaling and placement one decision
instead of two (all off by default, so the plain formulas above still hold):

    pfail_gate    nodes whose Pfail is at or above the gate are not "healthy":
                  they are never chosen, and replicas already on them are
                  listed in `drain_nodes` so they can be moved before the node dies
    joint_sizing  K is the smallest number of replicas whose *expected*
                  capacity on the nodes they will actually run on,
                  sum over chosen nodes of capacity * (1 - IS(i)),
                  covers the demand. With IS = 0 this is exactly
                  K = ceil(load / capacity).
    margin        a ForecastMargin: demand = predicted_load + buffer learned
                  from the forecaster's recent misses

Person C deliverables:
    FusionEngine.evaluate_window(workload_window, cluster_telemetry) -> decision_dict
    compute_node_score(pfail, is_score, utilization) -> S_i
    calculate_replicas(predicted_load, replica_capacity) -> K
"""

import math
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from models.node_health.failure_scorer import NodeFailureScorer, score_node_failure, calculate_cpi
from models.node_health.interference import InterferenceMatrix, get_interference
from models.load_pipeline import classify_and_forecast, ForecastMargin


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
    u_ratio = min(1.0, max(0.0, utilization / 100.0 if utilization > 1.0 else utilization))
    pf = min(1.0, max(0.0, pfail))
    is_val = min(1.0, max(0.0, is_score))

    score = alpha * (1.0 - pf) + beta * (1.0 - is_val) + gamma * (1.0 - u_ratio)
    return float(min(1.0, max(0.0, score)))


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
        pfail_gate:         Pfail at or above which a node is not healthy (None = no gate).
        joint_sizing:       Size K from expected capacity on the chosen nodes.
        margin:             Optional ForecastMargin added to the predicted load.
        use_interference:   Set False to ablate the interference term (IS treated as 0).
        use_failure:        Set False to ablate failure awareness (Pfail treated as 0, no gate).
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
        pfail_gate: Optional[float] = None,
        joint_sizing: bool = False,
        margin: Optional[ForecastMargin] = None,
        use_interference: bool = True,
        use_failure: bool = True,
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
        self.pfail_gate = pfail_gate
        self.joint_sizing = joint_sizing
        self.margin = margin
        self.use_interference = use_interference
        self.use_failure = use_failure

        # Control-loop state
        self.history: List[Dict[str, Any]] = []
        self.window_count: int = 0
        self._prev_pattern: str = "periodic"  # carry last pattern for EMA pairing
        self._pfail_cache: Dict[Any, float] = {}     # hardware reading -> Pfail
        self._prev_profiles: Dict[str, Tuple[int, int]] = {}  # node -> (service bin, tenant bin)

    # ─── per-node signals ─────────────────────────────────────────────────────

    def _pfail(self, nid: str, telem: Dict[str, Any]) -> float:
        """Pfail for a node. Hardware readings change daily, so identical readings are scored once."""
        key = telem.get("hw_reading")
        if key is None:
            return score_node_failure(nid, telem, self.failure_scorer)
        if key not in self._pfail_cache:
            self._pfail_cache[key] = score_node_failure(nid, telem, self.failure_scorer)
        return self._pfail_cache[key]

    def prime_failure_cache(self, readings: List[Dict[str, Any]]) -> None:
        """
        Score a batch of hardware readings in one model call and remember the
        results. A replayed trace has a known, finite set of readings, so this
        replaces thousands of single-row predictions with one.
        """
        fresh = [r for r in readings if r.get("hw_reading") is not None
                 and r["hw_reading"] not in self._pfail_cache]
        if not fresh or self.failure_scorer is None:
            return
        for r, p in zip(fresh, self.failure_scorer.predict_many(fresh)):
            self._pfail_cache[r["hw_reading"]] = float(p)

    def _learn_interference(self, cluster_telemetry: List[Dict[str, Any]]) -> Optional[float]:
        """
        EMA feedback from the trace: each node reports the slowdown its tenant
        just experienced (`observed_degradation`). That is the outcome of the
        profile pairing the node had one window ago, so it updates that matrix
        entry. Nodes that share a pairing are averaged first, giving one EMA
        step per pairing per window.
        """
        observed: Dict[Tuple[int, int], List[float]] = {}
        for telem in cluster_telemetry:
            pair = self._prev_profiles.get(telem.get("node_id"))
            if pair is not None and "observed_degradation" in telem and telem.get("ready", True):
                observed.setdefault(pair, []).append(float(telem["observed_degradation"]))
        if not observed:
            return None
        for (svc_bin, tenant_bin), values in observed.items():
            self.interference_matrix.update_ema(svc_bin, tenant_bin, float(np.mean(values)))
        return float(np.mean([v for vals in observed.values() for v in vals]))

    # ─── main per-window evaluation ───────────────────────────────────────────

    def evaluate_window(
        self,
        workload_window: List[float],
        cluster_telemetry: List[Dict[str, Any]],
        current_workload_type: str = "periodic",
        forecast: Optional[Dict[str, Any]] = None,
        hosting: Optional[List[str]] = None,
        service_utilization: float = 50.0,
    ) -> Dict[str, Any]:
        """
        Run fusion engine evaluation for a single 60-second control window.

        Pipeline:
            1. Classify workload & forecast next-window load  (Person A)
            2. Score every node: Pfail (Person B) + IS (Person B) → S(i)
            3. Rank nodes descending by S(i), healthy nodes first
            4. Compute target replica count K and select the K nodes
            5. Feed observed degradation back into interference matrix (EMA)
            6. Append decision to history and return decision dict

        Args:
            workload_window:      Recent workload history values (request rates).
            cluster_telemetry:    List of telemetry dicts for all cluster nodes.
            current_workload_type: Hint for interference lookup if classifier is uncertain.
            forecast:             Optional precomputed {'pattern', 'predicted_load'} for this
                                  window (used when replaying a trace); skips step 1.
            hosting:              Node IDs that already run a replica. Replicas are not
                                  reshuffled every window: these nodes are kept (best
                                  first) and only the difference to K is added or removed.
            service_utilization:  Utilisation (%) of the service's own replicas, for its
                                  interference profile.

        Returns:
            decision dict with keys:
            {
                'window_id':       int,
                'timestamp':       float  (Unix epoch),
                'pattern_label':   str,
                'predicted_load':  float,
                'load_margin':     float       — buffer added to the forecast,
                'target_replicas': int,
                'ranked_nodes':    List[Dict] — node_id, score, pfail, is_score, utilization, healthy,
                'selected_nodes':  List[str]  — the K node IDs chosen for placement,
                'drain_nodes':     List[str]  — hosting nodes that failed the health gate,
                'expected_capacity': float     — capacity expected from the selected nodes,
                'ema_update':      float       — degradation fed into matrix this window,
            }
        """
        window_id = self.window_count
        ts = time.time()

        # ── Step 1: Person A — Workload classification & load forecast ──────
        if forecast is not None:
            pattern_label = forecast["pattern"]
            pred_load = float(forecast["predicted_load"])
        else:
            series = np.array(workload_window, dtype=float) if len(workload_window) > 0 else np.array([50.0])
            res = classify_and_forecast(series)
            pattern_label = res["pattern"]
            pred_load = res["predicted_load"]

        load_margin = self.margin.value(workload_window) if self.margin is not None else 0.0
        demand = pred_load + load_margin

        # ── Step 5 (first half): learn from what the last window's pairings did ──
        trace_degradation = self._learn_interference(cluster_telemetry)

        # ── Step 2: Node scoring using Person B outputs ─────────────────────
        scored_nodes: List[Dict[str, Any]] = []
        profiles: Dict[str, Tuple[int, int]] = {}
        svc_bin = InterferenceMatrix.profile_bin(pattern_label, service_utilization)
        for telem in cluster_telemetry:
            nid = telem.get("node_id", "node-unknown")
            util = float(telem.get("cpu_utilization", 30.0))

            # Person B: Failure probability
            pfail = self._pfail(nid, telem) if self.use_failure else 0.0

            # Person B: Co-location interference score
            if "tenant_pattern" in telem:
                # real trace: workload profiles = pattern x utilisation level
                tenant_bin = InterferenceMatrix.profile_bin(telem["tenant_pattern"], util)
                profiles[nid] = (svc_bin, tenant_bin)
                is_score = self.interference_matrix.get_interference(svc_bin, tenant_bin)
            else:
                # The incoming node's existing workload type vs new workload to be placed
                node_wl_type = telem.get("pattern_type", current_workload_type)
                is_score = get_interference(pattern_label, node_wl_type, self.interference_matrix)
            if not self.use_interference:
                is_score = 0.0

            # Composite fusion score S(i)
            s_i = compute_node_score(pfail, is_score, util, self.alpha, self.beta, self.gamma)

            healthy = bool(telem.get("ready", True)) and (self.pfail_gate is None or pfail < self.pfail_gate)
            node = {
                "node_id": nid,
                "score": round(s_i, 4),
                "pfail": round(pfail, 4),
                "is_score": round(is_score, 4),
                "utilization": round(util, 2),
                "healthy": healthy,
            }
            if "queue_ms" in telem:
                node["cpi"] = round(calculate_cpi(util, 32.0 * min(1.0, telem["queue_ms"] / 1000.0)), 4)
            scored_nodes.append(node)
        self._prev_profiles = profiles

        # ── Step 3: Rank nodes ──────────────────────────────────────────────
        scored_nodes.sort(key=lambda x: x["score"], reverse=True)
        # "top-K healthy nodes": nodes that are NotReady or fail the Pfail gate are
        # listed last and never selected (the sort is stable, so S(i) order is kept)
        scored_nodes.sort(key=lambda x: not x["healthy"])
        candidates = [n for n in scored_nodes if n["healthy"]]

        # ── Step 4: Target replica count and node selection ─────────────────
        hosting_set = set(hosting or [])
        if hosting is not None:
            # keep running replicas where they are (best node first), then new nodes
            order = ([n for n in candidates if n["node_id"] in hosting_set]
                     + [n for n in candidates if n["node_id"] not in hosting_set])
        else:
            order = candidates

        if self.joint_sizing:
            # smallest K whose expected capacity on its nodes covers the demand
            selected, expected = [], 0.0
            for n in order:
                if expected >= demand and selected:
                    break
                selected.append(n)
                expected += self.replica_capacity * (1.0 - n["is_score"])
            k_replicas = max(1, len(selected))
        else:
            k_replicas = calculate_replicas(demand, self.replica_capacity)
            selected = order[:min(k_replicas, len(order))]
            expected = sum(self.replica_capacity * (1.0 - n["is_score"]) for n in selected)
        selected_nodes = [n["node_id"] for n in selected]
        healthy_ids = {n["node_id"] for n in candidates}
        drain_nodes = [nid for nid in (hosting or []) if nid not in healthy_ids]

        # ── Step 5: EMA feedback — update interference matrix ────────────────
        if trace_degradation is not None:
            observed_degradation = trace_degradation
        else:
            observed_degradation = _estimate_observed_degradation(scored_nodes, len(selected))
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
            "load_margin": round(load_margin, 4),
            "target_replicas": k_replicas,
            "ranked_nodes": scored_nodes,
            "selected_nodes": selected_nodes,
            "drain_nodes": drain_nodes,
            "expected_capacity": round(expected, 4),
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
        self._pfail_cache = {}
        self._prev_profiles = {}
        if self.margin is not None:
            self.margin.reset()

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
