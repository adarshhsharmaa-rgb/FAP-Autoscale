"""
Synthetic Data Generator for FAP-Scale framework.

Person A: Workload time series generator (Periodic, Bursty, Hybrid).
Person B: Per-node hardware telemetry generator (CPU, Run-Queue, Context Switches, Thermal, Memory/Disk/Net faults).
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

PATTERNS = ("periodic", "bursty", "hybrid")

def generate_workload_time_series(
    num_windows: int = 30,
    window_size: int = 60,
    pattern: str = "periodic",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic workload time series (Person A responsibility).

    Args:
        num_windows: Number of 60s control windows.
        window_size: Number of seconds per window.
        pattern: 'periodic', 'bursty', or 'hybrid'.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ['window_id', 'timestamp', 'request_rate', 'cpu_demand', 'pattern_type']
    """
    np.random.seed(seed)
    total_points = num_windows * window_size
    t = np.arange(total_points)

    if pattern == "periodic":
        # Sine wave with noise
        base_load = 50 + 30 * np.sin(2 * np.pi * t / (60 * 5))
        noise = np.random.normal(0, 3, total_points)
        load = base_load + noise
    elif pattern == "bursty":
        # Random baseline with sharp spikes
        base_load = np.random.uniform(20, 35, total_points)
        spikes = np.zeros(total_points)
        spike_locs = np.random.choice(total_points, size=int(total_points * 0.05), replace=False)
        spikes[spike_locs] = np.random.uniform(50, 100, len(spike_locs))
        load = base_load + spikes
    else:  # hybrid
        # Sine wave with occasional bursty spikes
        base_load = 40 + 25 * np.sin(2 * np.pi * t / (60 * 5))
        spikes = np.zeros(total_points)
        spike_locs = np.random.choice(total_points, size=int(total_points * 0.02), replace=False)
        spikes[spike_locs] = np.random.uniform(40, 80, len(spike_locs))
        load = base_load + spikes + np.random.normal(0, 2, total_points)

    load = np.clip(load, 5.0, 200.0)

    records = []
    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = (w + 1) * window_size
        win_load = load[start_idx:end_idx]
        for sec, val in enumerate(win_load):
            records.append({
                "window_id": w,
                "timestamp": start_idx + sec,
                "request_rate": float(val),
                "cpu_demand": float(val * 0.8),
                "pattern_type": pattern
            })

    return pd.DataFrame(records)


def generate_node_telemetry(
    num_nodes: int = 10,
    num_windows: int = 50,
    fail_node_ids: Optional[List[str]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic per-node telemetry for simulated cluster nodes (Person B implementation).

    Simulates node metrics over time, injecting failure trends (rising CPU pressure,
    rising thermal trend score, DIMM memory errors, network retransmissions) for designated failing nodes.

    Args:
        num_nodes: Total number of nodes in cluster (e.g., 10 to 15).
        num_windows: Number of control windows to simulate.
        fail_node_ids: List of node IDs destined to fail (e.g. ['node-02', 'node-07']).
                       If None, defaults to ['node-02', 'node-07'].
        seed: Random seed for reproducibility.

    Returns:
        DataFrame containing telemetry per node per window with columns:
        ['node_id', 'window_id', 'cpu_utilization', 'run_queue_length',
         'context_switch_rate', 'temperature_history', 'tdp', 'dimm_errors',
         'disk_health', 'net_retransmits', 'will_fail']
    """
    np.random.seed(seed)
    if fail_node_ids is None:
        fail_node_ids = ["node-02", "node-07"]

    node_ids = [f"node-{i:02d}" for i in range(1, num_nodes + 1)]
    records = []

    # Initialize node baseline parameters
    node_states: Dict[str, Dict[str, Any]] = {}
    for nid in node_ids:
        node_states[nid] = {
            "base_cpu": np.random.uniform(20.0, 50.0),
            "base_temp": np.random.uniform(45.0, 55.0),
            "base_q": np.random.uniform(1.0, 4.0),
            "base_cs": np.random.uniform(800.0, 2000.0),
            "tdp": 105.0,
            "temp_history": list(np.random.uniform(45.0, 55.0, size=10)),
            "dimm_errors": 0,
            "disk_health": 1.0,
            "net_retransmits": np.random.randint(0, 3)
        }

    for w in range(num_windows):
        for nid in node_ids:
            st = node_states[nid]
            is_failing_node = (nid in fail_node_ids)

            # Determine if node is in failure trajectory window (e.g. failing in second half of simulation)
            in_failure_regime = is_failing_node and (w >= num_windows // 3)

            if in_failure_regime:
                # Progressive degradation factor (0.0 to 1.0)
                degrad = min(1.0, (w - num_windows // 3) / (num_windows * 0.5))
                cpu_util = min(99.0, st["base_cpu"] + degrad * 45.0 + np.random.normal(0, 3))
                run_q = st["base_q"] + degrad * 20.0 + np.random.normal(0, 1)
                cs_rate = st["base_cs"] + degrad * 6000.0 + np.random.normal(0, 200)

                # Thermal rise (up to TDP limit or thermal throttling threshold ~95°C)
                latest_temp = min(98.0, st["base_temp"] + degrad * 40.0 + np.random.normal(0, 1.5))
                st["temp_history"].pop(0)
                st["temp_history"].append(latest_temp)

                # Fault accumulation
                dimm_errs = int(np.random.poisson(degrad * 4))
                disk_h = max(0.1, 1.0 - degrad * 0.7)
                net_retrans = int(st["net_retransmits"] + np.random.poisson(degrad * 15))

                will_fail = True if degrad > 0.3 else False
            else:
                cpu_util = np.clip(st["base_cpu"] + np.random.normal(0, 5), 5.0, 95.0)
                run_q = max(0.5, st["base_q"] + np.random.normal(0, 0.5))
                cs_rate = max(100.0, st["base_cs"] + np.random.normal(0, 100))

                latest_temp = np.clip(st["base_temp"] + np.random.normal(0, 1.0), 30.0, 80.0)
                st["temp_history"].pop(0)
                st["temp_history"].append(latest_temp)

                dimm_errs = 0 if np.random.rand() > 0.05 else 1
                disk_h = 1.0
                net_retrans = np.random.randint(0, 5)

                will_fail = False

            records.append({
                "node_id": nid,
                "window_id": w,
                "cpu_utilization": float(cpu_util),
                "run_queue_length": float(run_q),
                "context_switch_rate": float(cs_rate),
                "temperature_history": list(st["temp_history"]),
                "tdp": float(st["tdp"]),
                "dimm_errors": int(dimm_errs),
                "disk_health": float(disk_h),
                "net_retransmits": int(net_retrans),
                "will_fail": will_fail
            })

    return pd.DataFrame(records)


def extract_single_node_telemetry(df: pd.DataFrame, node_id: str, window_id: int) -> Dict[str, Any]:
    """
    Utility function to fetch telemetry dictionary for a given node at a specific window.
    """
    row = df[(df["node_id"] == node_id) & (df["window_id"] == window_id)]
    if row.empty:
        # Fallback dummy telemetry with consistent schema
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
            "will_fail": False
        }
    rec = row.iloc[0].to_dict()
    return rec


def _periodic(t, rng, base, amp, period, noise):
    phase = rng.uniform(0, 2 * np.pi)
    series = base + amp * np.sin(2 * np.pi * t / period + phase)
    # a weak 2nd harmonic makes it look less like a textbook sine
    series += 0.25 * amp * np.sin(4 * np.pi * t / period + phase / 2)
    return series + rng.normal(0, noise, size=len(t))


def _add_bursts(series, rng, base, burst_prob, burst_height, max_len):
    """Overlay short spikes. Each burst lasts 1..max_len steps and decays."""
    out = series.copy()
    n = len(out)
    i = 0
    while i < n:
        if rng.random() < burst_prob:
            length = rng.integers(1, max_len + 1)
            height = rng.uniform(*burst_height) * base
            decay = np.linspace(1.0, 0.4, length)
            end = min(n, i + length)
            out[i:end] += height * decay[: end - i]
            i = end
        else:
            i += 1
    return out


def generate_workload_series(pattern: str, length: int = 200, seed: int | None = None) -> np.ndarray:
    """Generate one load series (e.g. requests/s or CPU cores) of a given pattern."""
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}")
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    base = rng.uniform(8, 20)

    if pattern == "periodic":
        series = _periodic(t, rng, base, amp=rng.uniform(0.35, 0.6) * base,
                           period=rng.integers(12, 31), noise=0.04 * base)

    elif pattern == "bursty":
        # baseline with slow random-walk drift, no seasonality
        drift = np.cumsum(rng.normal(0, 0.02 * base, size=length))
        drift -= np.linspace(0, drift[-1], length)          # keep it bounded
        series = base + drift + rng.normal(0, 0.05 * base, size=length)
        series = _add_bursts(series, rng, base, burst_prob=rng.uniform(0.05, 0.10),
                             burst_height=(1.5, 3.5), max_len=4)

    else:  # hybrid
        series = _periodic(t, rng, base, amp=rng.uniform(0.3, 0.5) * base,
                           period=rng.integers(12, 31), noise=0.05 * base)
        series = _add_bursts(series, rng, base, burst_prob=rng.uniform(0.03, 0.06),
                             burst_height=(1.0, 2.5), max_len=3)

    return np.clip(series, 0.1, None)  # load can never be negative


def generate_workload_windows(n_windows: int = 30, length: int = 200,
                              seed: int = 42) -> list[tuple[np.ndarray, str]]:
    """
    Balanced labelled dataset: list of (window_array, true_label).
    The true label is only used for training/evaluating the classifier;
    classify_and_forecast() never sees it.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_windows):
        label = PATTERNS[i % len(PATTERNS)]
        out.append((generate_workload_series(label, length, seed=int(rng.integers(1e9))), label))
    order = rng.permutation(len(out))
    return [out[k] for k in order]


def generate_service_traces(n_services: int = 3, length: int = 400,
                            seed: int = 7) -> dict[str, dict]:
    """
    Long per-service traces for Person C's control loop.
    Each tick the loop slides a window over trace['load'] and calls
    classify_and_forecast(trace['load'][t-W:t]).

    returns {service_id: {"pattern": true_label, "load": np.ndarray}}
    """
    rng = np.random.default_rng(seed)
    traces = {}
    for i in range(n_services):
        label = PATTERNS[i % len(PATTERNS)]
        traces[f"svc-{i}"] = {
            "pattern": label,
            "load": generate_workload_series(label, length, seed=int(rng.integers(1e9))),
        }
    return traces


