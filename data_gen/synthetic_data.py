"""
data_gen/synthetic_data.py — Shared synthetic data generator.

Person A owns:  workload time-series utilities
Person B owns:  per-node hardware telemetry

Together these feed:
  • Person A's pattern classifier / forecasters (via generate_workload_series,
    generate_workload_windows, generate_service_traces)
  • Person C's control loop (via generate_workload_time_series,
    generate_node_telemetry, extract_single_node_telemetry)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PATTERNS: Tuple[str, ...] = ("periodic", "bursty", "hybrid")


# ═════════════════════════════════════════════════════════════════════════════
# PERSON A — workload generation
# ═════════════════════════════════════════════════════════════════════════════

def _periodic_signal(t: np.ndarray, rng: np.random.Generator,
                     base: float, amp: float,
                     period: float, noise: float) -> np.ndarray:
    phase = rng.uniform(0, 2 * np.pi)
    series = base + amp * np.sin(2 * np.pi * t / period + phase)
    # weak 2nd harmonic to look less like a textbook sine
    series += 0.25 * amp * np.sin(4 * np.pi * t / period + phase / 2)
    return series + rng.normal(0, noise, size=len(t))


def _add_bursts(series: np.ndarray, rng: np.random.Generator,
                base: float, burst_prob: float,
                burst_height: Tuple[float, float], max_len: int) -> np.ndarray:
    """Overlay short decaying spikes."""
    out = series.copy()
    n = len(out)
    i = 0
    while i < n:
        if rng.random() < burst_prob:
            length = int(rng.integers(1, max_len + 1))
            height = rng.uniform(*burst_height) * base
            decay = np.linspace(1.0, 0.4, length)
            end = min(n, i + length)
            out[i:end] += height * decay[:end - i]
            i = end
        else:
            i += 1
    return out


def generate_workload_series(pattern: str,
                             length: int = 200,
                             seed: Optional[int] = None) -> np.ndarray:
    """
    Return a 1-D load array of *length* control steps for the given pattern.

    Used by: Person A's load_pipeline warm_up / evaluation scripts, test_load_side.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}, got {pattern!r}")
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)

    if pattern == "periodic":
        base, amp = 50.0, 30.0
        period = rng.uniform(18, 30)
        s = _periodic_signal(t, rng, base, amp, period, noise=3.0)
    elif pattern == "bursty":
        base = rng.uniform(20, 40)
        drift = rng.uniform(-0.02, 0.02)
        s = base + drift * t + rng.normal(0, 3, length)
        s = _add_bursts(s, rng, base,
                        burst_prob=0.04,
                        burst_height=(1.2, 2.5),
                        max_len=4)
    else:  # hybrid
        base, amp = 40.0, 25.0
        period = rng.uniform(18, 30)
        s = _periodic_signal(t, rng, base, amp, period, noise=2.0)
        s = _add_bursts(s, rng, base,
                        burst_prob=0.02,
                        burst_height=(0.8, 1.8),
                        max_len=3)

    return np.clip(s, 1.0, 300.0)


def generate_workload_windows(
    n_windows: int = 30,
    length: int = 200,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, str]]:
    """
    Return a list of (window_array, true_label) pairs, cycling through all
    three patterns so every label appears ≈ n_windows/3 times.

    Used by: test_load_side, results/evaluate_load_side.py
    """
    rng = np.random.default_rng(seed)
    results: List[Tuple[np.ndarray, str]] = []
    labels = [PATTERNS[i % len(PATTERNS)] for i in range(n_windows)]
    rng.shuffle(labels)  # type: ignore[arg-type]
    for label in labels:
        sub_seed = int(rng.integers(0, 2**31))
        results.append((generate_workload_series(label, length, seed=sub_seed), label))
    return results


def generate_service_traces(
    n_services: int = 3,
    length: int = 400,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return a dict   { "svc-0": {"pattern": label, "load": np.ndarray}, … }

    Used by: demo_load_side.py, test_load_side.py
    """
    rng = np.random.default_rng(seed)
    traces: Dict[str, Dict[str, Any]] = {}
    for i in range(n_services):
        label = PATTERNS[i % len(PATTERNS)]
        sub_seed = int(rng.integers(0, 2**31))
        traces[f"svc-{i}"] = {
            "pattern": label,
            "load": generate_workload_series(label, length, seed=sub_seed),
        }
    return traces


def generate_workload_time_series(
    num_windows: int = 30,
    window_size: int = 60,
    pattern: str = "hybrid",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a DataFrame with per-second load rows grouped into control windows.

    Columns: window_id, timestamp, request_rate, cpu_demand, pattern_type

    Used by: main.py (Person C), test_person_b.py
    """
    np.random.seed(seed)
    total_points = num_windows * window_size
    t = np.arange(total_points)

    if pattern == "periodic":
        base_load = 50 + 30 * np.sin(2 * np.pi * t / (60 * 5))
        load = base_load + np.random.normal(0, 3, total_points)
    elif pattern == "bursty":
        base_load = np.random.uniform(20, 35, total_points)
        spikes = np.zeros(total_points)
        spike_locs = np.random.choice(total_points,
                                      size=int(total_points * 0.05), replace=False)
        spikes[spike_locs] = np.random.uniform(50, 100, len(spike_locs))
        load = base_load + spikes
    else:  # hybrid
        base_load = 40 + 25 * np.sin(2 * np.pi * t / (60 * 5))
        spikes = np.zeros(total_points)
        spike_locs = np.random.choice(total_points,
                                      size=int(total_points * 0.02), replace=False)
        spikes[spike_locs] = np.random.uniform(40, 80, len(spike_locs))
        load = base_load + spikes + np.random.normal(0, 2, total_points)

    load = np.clip(load, 5.0, 200.0)

    records = []
    for w in range(num_windows):
        start = w * window_size
        end = (w + 1) * window_size
        window_load = load[start:end]
        for idx, val in enumerate(window_load):
            records.append({
                "window_id": w,
                "timestamp": start + idx,
                "request_rate": float(val),
                "cpu_demand": float(val * 0.8 + np.random.normal(0, 2)),
                "pattern_type": pattern,
            })

    df = pd.DataFrame(records)
    df["cpu_demand"] = df["cpu_demand"].clip(5.0, 200.0)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# PERSON B — per-node hardware telemetry
# ═════════════════════════════════════════════════════════════════════════════

def generate_node_telemetry(
    num_nodes: int = 10,
    num_windows: int = 30,
    fail_node_ids: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic per-node hardware telemetry for *num_nodes* nodes over
    *num_windows* 60-second control windows.

    Columns:
        node_id, window_id, cpu_utilization, run_queue_length,
        context_switch_rate, temperature_history, tdp,
        dimm_errors, disk_health, net_retransmits, will_fail

    Used by: main.py, test_person_b.py
    """
    np.random.seed(seed)
    fail_node_ids = fail_node_ids or []
    records = []

    node_states: Dict[str, Any] = {}
    for n in range(num_nodes):
        nid = f"node-{n + 1:02d}"
        node_states[nid] = {
            "tdp": float(np.random.choice([65, 105, 125, 150])),
            "temp_history": [float(np.random.uniform(40, 55))] * 10,
            "base_cpu": float(np.random.uniform(20, 60)),
            "degrading": nid in fail_node_ids,
        }

    for w in range(num_windows):
        for nid, st in node_states.items():
            deg_factor = ((w / num_windows) ** 0.6) * 1.5 if st["degrading"] else 0.0

            cpu_util = float(np.clip(
                st["base_cpu"] + deg_factor * 40 + np.random.normal(0, 5),
                5, 98))
            run_q = float(np.clip(
                2 + deg_factor * 15 + np.random.exponential(1.5),
                0, 30))
            cs_rate = float(np.clip(
                1000 + deg_factor * 4000 + np.random.normal(0, 200),
                0, 10000))

            latest_temp = float(np.clip(
                st["temp_history"][-1] + deg_factor * 4 + np.random.normal(0, 1.5),
                35, 95))
            st["temp_history"].append(latest_temp)
            if len(st["temp_history"]) > 20:
                st["temp_history"].pop(0)

            dimm_errs = 0 if (np.random.rand() > 0.15 or not st["degrading"]) else 1
            disk_h = float(np.clip(1.0 - deg_factor * 0.5, 0.1, 1.0))
            net_ret = int(np.random.randint(0, max(2, int(5 + deg_factor * 25))))

            will_fail = st["degrading"] and (w > num_windows * 0.3) and (np.random.rand() > 0.3)

            records.append({
                "node_id": nid,
                "window_id": w,
                "cpu_utilization": cpu_util,
                "run_queue_length": run_q,
                "context_switch_rate": cs_rate,
                "temperature_history": list(st["temp_history"]),
                "tdp": st["tdp"],
                "dimm_errors": dimm_errs,
                "disk_health": disk_h,
                "net_retransmits": net_ret,
                "will_fail": bool(will_fail),
            })

    return pd.DataFrame(records)


def extract_single_node_telemetry(
    df: pd.DataFrame,
    node_id: str,
    window_id: int,
) -> Dict[str, Any]:
    """
    Return a telemetry dict for one (node_id, window_id) pair.
    Falls back to safe defaults if not found.
    """
    row = df[(df["node_id"] == node_id) & (df["window_id"] == window_id)]
    if row.empty:
        return {
            "node_id": node_id,
            "window_id": window_id,
            "cpu_utilization": 30.0,
            "run_queue_length": 2.0,
            "context_switch_rate": 1000.0,
            "temperature_history": [50.0] * 10,
            "tdp": 105.0,
            "dimm_errors": 0,
            "disk_health": 1.0,
            "net_retransmits": 0,
            "will_fail": False,
        }
    return row.iloc[0].to_dict()
