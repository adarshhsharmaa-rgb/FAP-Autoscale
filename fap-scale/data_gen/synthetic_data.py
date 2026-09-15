"""
data_gen/synthetic_data.py  --  WORKLOAD PART (Person A)

Synthetic stand-in for the Google Cluster Trace 2019 resource_usage table.
Generates load time series for three workload patterns:

    periodic : base + A*sin(2*pi*t/P) + small noise           (diurnal-like cycles)
    bursty   : flat/drifting baseline + short random spikes   (flash crowds)
    hybrid   : periodic base + occasional bursts on top       (mixed services)

One sample = one "control step" (60 s in the real system).
Person B owns the node-telemetry part of this file (see bottom).
"""
from __future__ import annotations

import numpy as np

PATTERNS = ("periodic", "bursty", "hybrid")


# --------------------------------------------------------------------------- #
# Low-level series builders
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# NODE TELEMETRY PART  --  owned by Person B (placeholder so imports don't break)
# --------------------------------------------------------------------------- #
def generate_node_telemetry(*args, **kwargs):
    raise NotImplementedError("Person B: node telemetry generator goes here")


if __name__ == "__main__":
    for w, lab in generate_workload_windows(n_windows=6):
        print(f"{lab:9s} len={len(w)} mean={w.mean():6.2f} std={w.std():6.2f} "
              f"min={w.min():6.2f} max={w.max():6.2f}")
