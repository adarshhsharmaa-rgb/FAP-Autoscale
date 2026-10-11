"""
simulator/policies.py  --  the three strategies as policies for ClusterSimulator.

    ReactiveHPAPolicy      Kubernetes HPA at a utilisation target, round-robin placement
    LSTMRoundRobinPolicy   LSTM forecast + safety margin, round-robin placement
    FAPScalePolicy         dynamic model selection + safety margin, Fusion Engine placement

Each wraps the class the rest of the project already uses (ReactiveHPAScaler,
LSTMRoundRobinScaler, FusionEngine) and translates between it and the
simulator's Observation / Decision.

The two predictive policies read their forecasts from a table produced by
models.load_pipeline.rolling_forecasts(), so the (slow) forecasters run once
per trace no matter how many cluster draws are simulated. A forecast made at
window t is only compared with what actually happened once those windows are
over, so the safety margin never sees the future.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from fusion.fusion_engine import FusionEngine
from models.load_pipeline import ForecastMargin
from models.node_health.failure_scorer import NodeFailureScorer
from models.node_health.interference import InterferenceMatrix
from simulator.baselines import LSTMRoundRobinScaler, ReactiveHPAScaler
from simulator.cluster import HISTORY_WINDOW, Decision, Observation


class _RoundRobin:
    """Blind placement: keep cycling through the nodes."""

    def __init__(self):
        self.cursor = 0

    def order(self, n_nodes: int) -> List[int]:
        return [(self.cursor + i) % n_nodes for i in range(n_nodes)]

    def advance(self, placed: List[int], n_nodes: int) -> None:
        if placed:
            self.cursor = (placed[-1] + 1) % n_nodes


class _ForecastFeed:
    """Forecast lookup plus honest, delayed scoring of past forecasts."""

    def __init__(self, forecasts: pd.DataFrame, column: str, margin: ForecastMargin, horizon: int):
        steps = forecasts["step"].to_numpy()
        self._pred: Dict[int, float] = dict(zip(steps, forecasts[column].to_numpy(dtype=float)))
        self._actual: Dict[int, float] = dict(zip(steps, forecasts["actual"].to_numpy(dtype=float)))
        self._pattern: Dict[int, str] = dict(zip(steps, forecasts["pattern"]))
        self.margin = margin
        self.horizon = horizon
        self._scored_to = None

    def reset_feed(self) -> None:
        self.margin.reset()
        self._scored_to = None

    def predicted(self, obs: Observation):
        """(forecast peak load over the horizon, pattern label) as of window obs.t."""
        if obs.t in self._pred:
            return self._pred[obs.t], self._pattern[obs.t]
        return float(obs.load_history[-1]), "hybrid"     # outside the table: persistence

    def score_past(self, obs: Observation) -> None:
        """Feed the margin every forecast whose horizon has fully elapsed by window obs.t."""
        last = obs.t - self.horizon
        first = last if self._scored_to is None else self._scored_to + 1
        for s in range(first, last + 1):
            if s in self._pred:
                self.margin.update(self._pred[s], self._actual[s])
        self._scored_to = max(last, self._scored_to if self._scored_to is not None else last)


# --------------------------------------------------------------------------- #
class ReactiveHPAPolicy:
    """Baseline 1: scale on the utilisation just observed, place round-robin."""

    name = "Reactive-HPA"

    def __init__(self, capacity: float, target_utilization: float = 70.0):
        self.scaler = ReactiveHPAScaler(target_cpu_threshold=target_utilization,
                                        replica_capacity=capacity)
        self.rr = _RoundRobin()
        self._n = 0

    def reset(self) -> None:
        self.scaler.reset()
        self.rr = _RoundRobin()

    def on_boot(self, replicas: int) -> None:
        self.scaler.current_replicas = replicas
        self.scaler._recommendations.clear()
        self.scaler._recommendations.append(replicas)

    def decide(self, obs: Observation) -> Decision:
        self._n = obs.scenario.n_nodes
        k = self.scaler.desired_replicas(obs.utilization, obs.ready_replicas, obs.total_replicas)
        return Decision(target_replicas=k, place_order=self.rr.order(self._n))

    def on_placed(self, nodes: List[int]) -> None:
        self.rr.advance(nodes, self._n)


# --------------------------------------------------------------------------- #
class LSTMRoundRobinPolicy(_ForecastFeed):
    """Baseline 2: one forecaster (LSTM) for every pattern, place round-robin."""

    name = "LSTM-RoundRobin"

    def __init__(self, capacity: float, forecasts: pd.DataFrame, quantile: float = 0.9,
                 horizon: int = 2, column: str = "LSTM"):
        super().__init__(forecasts, column, ForecastMargin(quantile=quantile), horizon)
        self.scaler = LSTMRoundRobinScaler(replica_capacity=capacity, margin=self.margin)
        self.rr = _RoundRobin()
        self._n = 0

    def reset(self) -> None:
        self.scaler.reset()
        self.reset_feed()
        self.rr = _RoundRobin()

    def on_boot(self, replicas: int) -> None:
        pass

    def decide(self, obs: Observation) -> Decision:
        self._n = obs.scenario.n_nodes
        self.score_past(obs)
        predicted, _ = self.predicted(obs)
        k = self.scaler.desired_replicas(predicted, obs.load_history[-HISTORY_WINDOW:])
        return Decision(target_replicas=k, place_order=self.rr.order(self._n),
                        info={"predicted_load": predicted})

    def on_placed(self, nodes: List[int]) -> None:
        self.rr.advance(nodes, self._n)


# --------------------------------------------------------------------------- #
class FAPScalePolicy(_ForecastFeed):
    """
    FAP-Scale: the Fusion Engine decides how many replicas and where, together.

    failure_aware / interference_aware switch the two awareness terms off for
    ablation runs. With both off this is a predictive scaler that only
    balances utilisation.
    """

    name = "FAP-Scale"

    def __init__(self, capacity: float, forecasts: pd.DataFrame,
                 failure_scorer: Optional[NodeFailureScorer] = None,
                 pfail_gate: Optional[float] = None, quantile: float = 0.9, horizon: int = 2,
                 column: str = "Dynamic", failure_aware: bool = True,
                 interference_aware: bool = True, name: Optional[str] = None,
                 alpha: float = 0.4, beta: float = 0.35, gamma: float = 0.25):
        super().__init__(forecasts, column, ForecastMargin(quantile=quantile), horizon)
        if name:
            self.name = name
        self.engine = FusionEngine(
            alpha=alpha, beta=beta, gamma=gamma,
            replica_capacity=capacity,
            failure_scorer=failure_scorer,
            # the matrix starts with no opinion and is learned from the trace
            interference_matrix=InterferenceMatrix(alpha=0.1, prior=0.0),
            pfail_gate=pfail_gate if failure_aware else None,
            joint_sizing=True,
            margin=self.margin,
            use_interference=interference_aware,
            use_failure=failure_aware,
        )

    def reset(self) -> None:
        self.engine.reset()
        self.engine.interference_matrix = InterferenceMatrix(alpha=0.1, prior=0.0)
        self.reset_feed()
        self._primed = False

    def on_boot(self, replicas: int) -> None:
        pass

    def decide(self, obs: Observation) -> Decision:
        sc = obs.scenario
        if not getattr(self, "_primed", False):
            self.engine.prime_failure_cache([r for rows in sc.hardware for r in rows])
            self._primed = True
        self.score_past(obs)
        predicted, pattern = self.predicted(obs)
        d = self.engine.evaluate_window(
            obs.load_history[-HISTORY_WINDOW:],
            obs.telemetry,
            forecast={"pattern": pattern, "predicted_load": predicted},
            hosting=[sc.node_ids[n] for n in obs.hosting],
            service_utilization=obs.utilization,
        )
        index = {nid: n for n, nid in enumerate(sc.node_ids)}
        score = {index[n["node_id"]]: n["score"] for n in d["ranked_nodes"]}
        return Decision(
            target_replicas=d["target_replicas"],
            place_order=[index[n["node_id"]] for n in d["ranked_nodes"] if n["healthy"]],
            drain=[index[nid] for nid in d["drain_nodes"]],
            remove_order=sorted(set(obs.hosting), key=lambda n: score.get(n, 0.0)),
            info=d,
        )

    def on_placed(self, nodes: List[int]) -> None:
        pass
