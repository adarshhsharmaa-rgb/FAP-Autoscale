"""
simulator/cluster.py  --  trace-driven cluster simulation on real data.

The cluster is assembled from two official traces (see data/README.md):

  Alibaba GenTD26 (2026)        what the cluster is asked to do, minute by minute
      load L(t)                 gateway QPS, the demand the autoscaler must cover
      node utilisation          each node hosts one tenant pod of the trace and
                                carries that pod's GPU duty cycle and memory use
      interference              the latency inflation that tenant experienced is
                                the throughput a replica sharing the node loses
  Backblaze Drive Stats Q2 2026  what happens to the hardware
      node health               each node replays the S.M.A.R.T. history of one
                                real drive
      node failure              a node goes down on the day its drive was
                                recorded as failed

Nothing in a run is drawn from a random distribution. A seed only decides which
drive is attached to which node and where in the run its history is placed.

Two modelling choices are worth knowing about:

  * Time scales. GenTD26 covers one day at 57 s resolution; drive telemetry is
    daily. The hardware history is therefore replayed faster than real time:
    `replay_days` days of S.M.A.R.T. readings are spread over the run, so a
    node's health reading changes every ~44 control windows. Failure counts
    per run are set by `n_failing`, i.e. this is failure injection with real
    failure trajectories, not a claim about how often nodes fail in a day.
  * Kubernetes mechanics, applied identically to every strategy. A new replica
    serves traffic one window after it is requested (cold start; GenTD26's
    base-model load time is 23 s median, 42 s p90). A node that dies is marked
    NotReady one window later, and replicas stranded on it are only evicted
    and replaced after the default 5-minute toleration (6 windows).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.node_health.failure_scorer import (FAILURE_HORIZON_DAYS, SMART_COUNTERS,
                                               THERMAL_LIMIT_C, TTS_READINGS,
                                               smart_feature_frame)
from models.pattern_classifier import classify_pattern

STARTUP_STEPS = 1      # windows from requesting a replica to it serving traffic
NOT_READY_STEPS = 1    # windows until a dead node is marked NotReady
EVICTION_STEPS = 6     # windows until replicas on a dead node are evicted (300 s toleration)
HISTORY_WINDOW = 150   # load history handed to the forecasters
RECLASSIFY_EVERY = 60  # windows between re-classifications of a tenant's load pattern


# --------------------------------------------------------------------------- #
# Scenario: the ground truth one run is played against
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    load: np.ndarray                 # [T] demand per window
    capacity: float                  # demand one replica serves at full speed
    step_seconds: int
    start: int                       # first evaluated window
    node_ids: List[str]
    tenant_util: np.ndarray          # [T, N] GPU duty cycle of the node's tenant, %
    tenant_mem: np.ndarray           # [T, N] tenant memory utilisation, 0-1
    tenant_queue_ms: np.ndarray      # [T, N] tenant gateway queue time, ms
    tenant_pattern: np.ndarray       # [T, N] load pattern of the tenant (str)
    slowdown: np.ndarray             # [T, N] throughput a co-located replica loses, 0-1
    hardware: List[List[Dict[str, Any]]]   # per node: one telemetry dict per replayed day
    day_of_step: np.ndarray          # [T] which replayed day a window falls in
    fail_step: np.ndarray            # [N] window in which the node dies, -1 = never
    steps_per_day: int
    seed: int = 0
    drive_ids: List[int] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.load)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def is_down(self, node: int, t: int) -> bool:
        return 0 <= self.fail_step[node] <= t

    def fails_soon(self, node: int, t: int) -> bool:
        """Ground truth: does this (still running) node die within the failure horizon?"""
        f = self.fail_step[node]
        return f > t and f - t <= FAILURE_HORIZON_DAYS * self.steps_per_day

    def hardware_reading(self, node: int, t: int) -> Dict[str, Any]:
        """Latest health reading taken before window t (the last one if the node is dead)."""
        rows = self.hardware[node]
        return rows[min(int(self.day_of_step[t]), len(rows) - 1)]

    def telemetry(self, t: int) -> List[Dict[str, Any]]:
        """What the control plane can see about every node at the end of window t."""
        out = []
        for n, nid in enumerate(self.node_ids):
            f = self.fail_step[n]
            known_dead = f >= 0 and t >= f + NOT_READY_STEPS
            reading = dict(self.hardware_reading(n, t))
            reading.update({
                "node_id": nid,
                "ready": not known_dead,
                "cpu_utilization": float(self.tenant_util[t, n]),
                "memory_utilization": float(self.tenant_mem[t, n]),
                "queue_ms": float(self.tenant_queue_ms[t, n]),
                "tenant_pattern": str(self.tenant_pattern[t, n]),
                "observed_degradation": float(self.slowdown[t, n]),
            })
            out.append(reading)
        return out


@lru_cache(maxsize=None)
def _tenant_patterns(n_steps: int, pod: int) -> tuple:
    """Load pattern of one tenant pod over the day, re-classified every RECLASSIFY_EVERY windows."""
    from traces import gentd26
    util = gentd26.load().gpu_util[:, pod]
    labels, current = [], "bursty"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # mostly-idle windows have degenerate statistics
        for t in range(n_steps):
            if t % RECLASSIFY_EVERY == 0 and t >= 48:
                window = util[max(0, t - HISTORY_WINDOW):t]
                # an idle GPU pod has no pattern yet; inference pods are bursty by nature
                current = classify_pattern(window)[0] if np.ptp(window) > 0 else "bursty"
            labels.append(current)
    return tuple(labels)


def _hardware_rows(drive: pd.DataFrame, first: int, last: int) -> List[Dict[str, Any]]:
    """Telemetry dicts for record rows first..last (inclusive) of one drive."""
    temps = drive["temperature_filled"].to_numpy(dtype=float)
    rows = []
    for i in range(first, last + 1):
        rec = drive.iloc[i]
        row = {
            "hw_reading": (int(rec["drive_id"]), i),
            "drive_model": rec["model"],
            "temperature_c": rec["temperature_c"],
            "temperature_history": [float(v) for v in temps[max(0, i - TTS_READINGS + 1): i + 1]],
            "thermal_limit_c": THERMAL_LIMIT_C,
            "power_on_hours": rec["power_on_hours"],
        }
        for c in SMART_COUNTERS:
            row[c] = rec[c]
            row[f"{c}_delta"] = rec[f"{c}_delta"]
        rows.append(row)
    return rows


def build_scenario(drive_days: pd.DataFrame, n_nodes: int = 72, n_failing: int = 7,
                   replay_days: int = 30, start: int = HISTORY_WINDOW, seed: int = 0,
                   load_series: str = "qps_total", min_history_days: int = 10) -> Scenario:
    """
    Assemble a cluster from the two traces.

    drive_days  drive-day table for the drives nodes may be built from. Pass
                only drives the failure model was *not* trained on.
    n_failing   nodes backed by a drive that really failed; each dies at a
                random point of the run, with at least `min_history_days` of its
                own history before that.
    """
    from traces import gentd26
    trace = gentd26.load()
    rng = np.random.default_rng(seed)
    T = trace.n_steps

    pool = trace.tenant_pool()
    if n_nodes > len(pool):
        raise ValueError(f"the trace has {len(pool)} usable tenant pods, asked for {n_nodes} nodes")
    pods = [int(p) for p in rng.permutation(pool)[:n_nodes]]

    df = drive_days if "tts" in drive_days.columns else smart_feature_frame(drive_days)
    by_drive = {d: g.reset_index(drop=True) for d, g in df.groupby("drive_id", sort=False)}
    failed_ids = [d for d, g in by_drive.items()
                  if g["failure"].iloc[-1] == 1 and len(g) - 1 >= min_history_days]
    healthy_ids = [d for d, g in by_drive.items()
                   if not g["will_fail"].iloc[0] and len(g) >= replay_days + TTS_READINGS]
    if len(failed_ids) < n_failing or len(healthy_ids) < n_nodes - n_failing:
        raise ValueError(f"not enough drives with enough history: "
                         f"{len(failed_ids)} failed / {len(healthy_ids)} healthy")

    steps_per_day = math.ceil((T - start) / replay_days)
    day_of_step = np.clip((np.arange(T) - start) // steps_per_day, 0, replay_days - 1)

    failing_nodes = set(int(i) for i in rng.choice(n_nodes, size=n_failing, replace=False))
    failed_pick = list(rng.choice(failed_ids, size=n_failing, replace=False))
    healthy_pick = list(rng.choice(healthy_ids, size=n_nodes - n_failing, replace=False))

    hardware, fail_step, drive_ids = [], np.full(n_nodes, -1, dtype=int), []
    for n in range(n_nodes):
        if n in failing_nodes:
            drive = by_drive[failed_pick.pop()]
            f_idx = len(drive) - 1                      # the row with failure == 1
            # day of the run on which the drive dies; it needs that many earlier readings
            d_fail = int(rng.integers(1, min(replay_days - 1, f_idx) + 1))
            # readings for days 0 .. d_fail-1; the failure-day row is the post-mortem record
            rows = _hardware_rows(drive, f_idx - d_fail, f_idx - 1)
            fail_step[n] = min(T - 1, start + d_fail * steps_per_day + int(rng.integers(0, steps_per_day)))
        else:
            drive = by_drive[healthy_pick.pop()]
            first = int(rng.integers(TTS_READINGS - 1, len(drive) - replay_days + 1))
            rows = _hardware_rows(drive, first, first + replay_days - 1)
        hardware.append(rows)
        drive_ids.append(int(drive["drive_id"].iloc[0]))

    queue = pd.DataFrame(trace.queue_rt_ms[:, pods]).ffill().fillna(0.0).to_numpy()
    patterns = np.array([_tenant_patterns(T, p) for p in pods], dtype=object).T
    return Scenario(
        load=trace.series(load_series),
        capacity=trace.replica_capacity(),
        step_seconds=trace.step_seconds,
        start=start,
        node_ids=[f"node-{n + 1:02d}" for n in range(n_nodes)],
        tenant_util=np.nan_to_num(trace.gpu_util[:, pods]),
        tenant_mem=np.nan_to_num(trace.mem_util[:, pods]),
        tenant_queue_ms=queue,
        tenant_pattern=patterns,
        slowdown=trace.slowdown()[:, pods],
        hardware=hardware,
        day_of_step=day_of_step,
        fail_step=fail_step,
        steps_per_day=steps_per_day,
        seed=seed,
        drive_ids=drive_ids,
    )


# --------------------------------------------------------------------------- #
# What a policy sees and what it returns
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    t: int
    load_history: np.ndarray         # demand up to and including window t
    utilization: float               # % of the serving replicas' capacity in use in window t
    ready_replicas: int              # replicas that served traffic in window t
    total_replicas: int              # replicas the controller counts (incl. starting, stranded)
    hosting: List[int]               # nodes that hold one of those replicas
    scenario: "Scenario" = field(repr=False, default=None)
    _telemetry: Optional[List[Dict[str, Any]]] = field(repr=False, default=None)

    @property
    def telemetry(self) -> List[Dict[str, Any]]:
        """Per-node telemetry as of the end of window t (built on first use)."""
        if self._telemetry is None:
            self._telemetry = self.scenario.telemetry(self.t)
        return self._telemetry


@dataclass
class Decision:
    target_replicas: int
    place_order: List[int]                    # nodes to try for new replicas, best first
    drain: List[int] = field(default_factory=list)   # nodes to move replicas off
    remove_order: Optional[List[int]] = None  # hosting nodes to give up first on scale-down
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Replica:
    node: int
    ready_at: int
    seq: int = 0                       # creation order
    dead_since: Optional[int] = None   # the node died under it
    retire_at: Optional[int] = None    # being drained; removed once its replacement serves


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #
class ClusterSimulator:
    """Plays one autoscaling policy against a Scenario and logs every window."""

    def __init__(self, scenario: Scenario, startup_steps: int = STARTUP_STEPS,
                 eviction_steps: int = EVICTION_STEPS):
        self.sc = scenario
        self.startup_steps = startup_steps
        self.eviction_steps = eviction_steps

    def run(self, policy, stop: Optional[int] = None) -> pd.DataFrame:
        """
        Returns one row per evaluated window:
            step, load, capacity (effective), nominal (paid for), replicas,
            serving, violation, unserved, lost (replicas killed by a node
            failure this window), placed, unhealthy_placed, drained
        """
        sc, c = self.sc, self.sc.capacity
        stop = sc.n_steps if stop is None else min(stop, sc.n_steps)
        replicas: List[_Replica] = []
        policy.reset()
        rows = []
        placed = unhealthy = drained = created = 0

        for t in range(sc.start - 1, stop):
            booting = t == sc.start - 1          # the window before the run: initial placement only
            lost = 0
            if booting:
                serving, capacity = [], 0.0
            else:
                # 1. a node that is down takes the replicas on it with it
                for r in replicas:
                    if r.dead_since is None and sc.is_down(r.node, t):
                        r.dead_since = t
                        lost += 1
                # 2. stranded replicas are evicted after the toleration; drained ones retire
                replicas = [r for r in replicas
                            if not (r.dead_since is not None and t - r.dead_since >= self.eviction_steps)
                            and not (r.retire_at is not None and r.retire_at <= t)]
                # 3. serve this window's demand
                serving = [r for r in replicas if r.dead_since is None and r.ready_at <= t]
                capacity = sum(c * (1.0 - sc.slowdown[t, r.node]) for r in serving)
                alive = sum(1 for r in replicas if r.dead_since is None)
                load = float(sc.load[t])
                rows.append({
                    "step": t, "load": load, "capacity": capacity, "nominal": c * alive,
                    "replicas": alive, "serving": len(serving),
                    "violation": capacity + 1e-9 < load, "unserved": max(0.0, load - capacity),
                    "lost": lost, "placed": placed, "unhealthy_placed": unhealthy, "drained": drained,
                })
                placed = unhealthy = drained = 0
                if t == stop - 1:
                    break

            # 4. the policy looks at the window that just ended and decides for the next
            counted = [r for r in replicas if r.retire_at is None]
            load_now = float(sc.load[t])
            util = 100.0 * min(1.0, load_now / capacity) if capacity > 0 else (100.0 if load_now > 0 else 0.0)
            obs = Observation(
                t=t,
                load_history=sc.load[:t + 1],
                utilization=util,
                ready_replicas=len(serving),
                total_replicas=len(counted),
                hosting=[r.node for r in counted],
                scenario=sc,
            )
            decision = policy.decide(obs)
            if booting:
                # every policy starts the run with the same number of warm replicas
                # (enough for the current load at 70 % utilisation); only where they
                # go is the policy's choice
                decision.target_replicas = max(1, math.ceil(load_now / (0.7 * c)))
                policy.on_boot(decision.target_replicas)

            # 5. reconcile towards the decision; changes take effect from window t+1
            ready_at = t + 1 if booting else t + 1 + self.startup_steps
            known_dead = {n for n in range(sc.n_nodes)
                          if sc.fail_step[n] >= 0 and t >= sc.fail_step[n] + NOT_READY_STEPS}
            for node in decision.drain:
                for r in replicas:
                    if r.node == node and r.retire_at is None:
                        drained += 1
                        # a replica that is still alive keeps serving until its replacement is up
                        r.retire_at = t if r.dead_since is not None else ready_at
            replicas = [r for r in replicas if not (r.retire_at is not None and r.retire_at <= t)]

            counted = [r for r in replicas if r.retire_at is None]
            need = max(1, int(decision.target_replicas)) - len(counted)
            new_nodes: List[int] = []
            if need > 0:
                occupied = {r.node for r in replicas}
                for node in decision.place_order:
                    if len(new_nodes) == need:
                        break
                    if node in occupied or node in known_dead:
                        continue
                    created += 1
                    replicas.append(_Replica(node, ready_at, seq=created))
                    occupied.add(node)
                    new_nodes.append(node)
                    unhealthy += int(sc.is_down(node, t) or sc.fails_soon(node, t))
                placed += len(new_nodes)
            elif need < 0:
                preference = {n: i for i, n in enumerate(decision.remove_order or [])}

                def removal_key(r: _Replica):
                    # Kubernetes order: replicas on NotReady nodes, then ones not serving yet,
                    # then the policy's choice (default: newest first)
                    stranded = r.dead_since is not None and r.node in known_dead
                    return (0 if stranded else 1, 0 if r.ready_at > t else 1,
                            preference.get(r.node, len(preference)), -r.seq)

                for r in sorted(counted, key=removal_key)[:-need]:
                    replicas.remove(r)
            policy.on_placed(new_nodes)

        log = pd.DataFrame(rows)
        log.attrs.update({"policy": policy.name, "seed": sc.seed, "capacity": c})
        return log
