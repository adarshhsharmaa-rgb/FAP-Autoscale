"""
FAP-Scale End-to-End Simulation Entrypoint (real traces).

Ties together all three team modules in a single runnable script:
  - Datasets           : Alibaba GenTD26 (2026) + Backblaze Drive Stats Q2 2026
  - Model Training     : Adarsh (LightGBM NodeFailureScorer + InterferenceMatrix)
  - Load Forecasting   : Nirupam (pattern classifier + ARIMA / LSTM / XGBoost)
  - Control Loop       : Aman (FusionEngine 60 s window ticks)
  - Baselines          : Aman (ReactiveHPA + LSTM-RoundRobin)
  - Evaluation & Plots : results/evaluate.py

Usage:
    python main.py                      # one simulated cluster, the whole trace day
    python main.py --seed 3             # a different draw of node hardware
    python main.py --windows 300        # only the first 300 control windows

The load, the node utilisation and the interference come from GenTD26; node
health and node failures come from Backblaze drives the failure model never saw
in training. All three strategies face exactly the same cluster. For the
averaged numbers over 20 clusters run:  python -m results.evaluate_real

The original synthetic-data demo is still available as demo_synthetic.py.
"""

import argparse
import os
import warnings
from typing import Any, Dict

import numpy as np

warnings.filterwarnings("ignore")

# -- Datasets -------------------------------------------------------------------
from traces import backblaze, gentd26

# -- Node health models (Adarsh) ----------------------------------------------
from models.node_health import train_real as node_health
from models.node_health.failure_scorer import calculate_cpi, calculate_tts, THERMAL_LIMIT_C
from models.node_health.interference import InterferenceMatrix

# -- Load forecasting (Nirupam) -------------------------------------------------
from models.load_pipeline import ROUTING, add_dynamic_selection, DynamicSelector

# -- Fusion & baselines (Aman) ------------------------------------------------
from fusion.fusion_engine import compute_node_score, calculate_replicas
from simulator.cluster import ClusterSimulator, build_scenario, EVICTION_STEPS, STARTUP_STEPS
from simulator.policies import FAPScalePolicy, LSTMRoundRobinPolicy, ReactiveHPAPolicy

# -- Evaluation ----------------------------------------------------------------
from results.evaluate import summarize_run, plot_evaluation_summary
from results.evaluate_real import forecast_table, HPA_TARGET, QUANTILE


def _banner(text, char="=", width=96):
    print(char * width)
    print(f"  {text}")
    print(char * width)


def _section(title, step_num=None):
    print()
    prefix = f"STEP {step_num}" if step_num else ""
    line = "-" * 96
    print(line)
    if prefix:
        print(f"  [{prefix}] {title}")
    else:
        print(f"  {title}")
    print(line)


def run_fap_scale_simulation(seed: int = 0, num_windows: int = None) -> Dict[str, Any]:
    """
    Run end-to-end FAP-Scale control loop on the real traces with detailed output.
    """
    os.makedirs("results", exist_ok=True)

    print()
    _banner(
        "FAP-SCALE: Failure-Aware Predictive Autoscaling Framework\n"
        "  Trace-driven simulation on official 2026 datasets (no self-generated data)\n"
        "  Team: Nirupam (Workload) | Adarsh (Node Health) | Aman (Fusion & Integration)"
    )

    # =========================================================================
    # STEP 1: REAL DATASETS
    # =========================================================================
    _section("OFFICIAL DATASETS", 1)

    print("\n  [Nirupam] Workload trace: Alibaba GenTD26 (cluster-trace-v2026-GenAI)")
    print("    Source: github.com/alibaba/clusterdata  (Alibaba Open Cluster Trace Program)")
    gentd26.fetch()
    trace = gentd26.load()
    w = trace.workload
    print(f"    {trace.n_steps} control windows x {trace.step_seconds} s = "
          f"{trace.n_steps * trace.step_seconds / 3600:.1f} h of a production GenAI serving system")
    print(f"    Gateway load: mean={w.qps_total.mean():.2f} | peak={w.qps_total.max():.2f} QPS (anonymised units)")
    print(f"    {len(trace.pod_ids)} pods with GPU / memory / queue / latency telemetry "
          f"({len(trace.tenant_pool())} observed all day)")

    print("\n  [Adarsh] Hardware failures: Backblaze Drive Stats Q2 2026")
    print("    Source: backblaze.com/cloud-storage/resources/hard-drive-test-data")
    meta = backblaze.load_meta()
    print(f"    {meta['first_day']} to {meta['last_day']}: {meta['drive_days_scanned']:,} drive-days scanned, "
          f"fleet of {meta['fleet_size_last_day']:,} drives")
    print(f"    Kept all {meta['failed_drives']} drives that failed + {meta['healthy_drives_sampled']:,} "
          f"healthy drives ({meta['healthy_sample_rate']:.1%} sample)")
    print("    Signals: temperature, reallocated / pending / uncorrectable sectors,")
    print("             command timeouts, interface CRC errors, power-on hours")

    # =========================================================================
    # STEP 2: NODE HEALTH MODEL TRAINING (Adarsh)
    # =========================================================================
    _section("NODE HEALTH MODEL TRAINING [Adarsh]", 2)

    print("\n  Training LightGBM NodeFailureScorer on real drive failures...")
    print("    Label: the drive fails within the next 7 days")
    failure_scorer, gate, test_drives, health = node_health.train(verbose=True)
    backend = "LightGBM" if failure_scorer.model.__class__.__name__ == "LGBMClassifier" else "HistGradientBoosting"
    print(f"    Model: {backend}, {len(health['features'])} features, tested on drives never seen in training")

    print("\n  Initialising 15-bin Co-location Interference Matrix...")
    print("    15 bins = 3 load patterns x 5 utilisation levels; learned online from the trace")
    print("    Online learning rule (EMA):")
    print("      IS_new = (1 - alpha) * IS_old + alpha * observed_degradation   (alpha = 0.1)")
    print("    observed_degradation = throughput lost to latency inflation, measured in GenTD26")

    print("\n  Health Metric Formulas:")
    print("    CPI = 0.5 * (U/100) + 0.35 * (Queue/32)      [context switches are not in any public trace]")
    print(f"    TTS = polyfit_slope(last_10_temps) / {THERMAL_LIMIT_C:.0f} C")
    print("    Pfail = LightGBM.predict_proba(SMART features), Platt-calibrated")

    # =========================================================================
    # STEP 3: FORECASTERS & STRATEGIES
    # =========================================================================
    _section("LOAD FORECASTING & STRATEGY INIT", 3)

    print("\n  [Nirupam] Forecasters (LSTM trained on the real GenTD26 request log):")
    print("    ARIMA(2,1,2) | LSTM 2x64 | XGBoost, each replayed over the trace one window at a time")
    table = add_dynamic_selection(forecast_table())
    print(f"    Fixed routing from Review 2: " + ", ".join(f"{p} -> {m}" for p, m in ROUTING.items()))
    print("    Dynamic selection now scores every model after each window and weights")
    print("    them by tracked error, so the routing follows the real data")
    trusted = table["routed_to"].value_counts(normalize=True)
    print("    Most trusted model over the day: " + ", ".join(f"{m} {v:.0%}" for m, v in trusted.items()))

    print("\n  [Aman] Assembling the simulated cluster...")
    scenario = build_scenario(test_drives, seed=seed)
    capacity = scenario.capacity
    failing = np.flatnonzero(scenario.fail_step >= 0)
    stop = scenario.n_steps if num_windows is None else min(scenario.n_steps, scenario.start + num_windows)
    print(f"    {scenario.n_nodes} nodes, each hosting one GenTD26 tenant pod and backed by one real drive")
    print(f"    {len(failing)} nodes are backed by drives that really failed: "
          + ", ".join(scenario.node_ids[n] for n in failing))
    print(f"    Replica capacity read from the trace: {capacity:.3f} QPS per replica")
    print(f"    Cold start {STARTUP_STEPS} window | replicas on a dead node evicted after {EVICTION_STEPS} windows")

    print("\n  [Aman] Fusion Engine:")
    print("    S(i) = alpha*(1 - Pfail) + beta*(1 - IS) + gamma*(1 - U/100)")
    print("    Weights: alpha=0.40 (failure) | beta=0.35 (interference) | gamma=0.25 (utilization)")
    print(f"    Health gate: nodes with Pfail >= {gate:.3f} are never selected and are drained")
    print("    K = smallest replica count whose expected capacity,")
    print("        sum of capacity * (1 - IS(i)) over its nodes, covers forecast + margin")
    print(f"    Margin = {QUANTILE:.0%} quantile of the forecaster's recent misses")
    sample_score = compute_node_score(pfail=0.01, is_score=0.1, utilization=20.0)
    print(f"    Sample: S(Pfail=0.01, IS=0.10, U=20%) = {sample_score:.3f} | "
          f"K(load=5.0, cap={capacity:.2f}, IS=0) = {calculate_replicas(5.0, capacity)}")

    print("\n  [Aman] Baselines:")
    print(f"    Baseline 1: Reactive HPA  -> Kubernetes HPA algorithm, {HPA_TARGET:.0f}% utilisation target,")
    print("                                 5-minute scale-down stabilisation, round-robin placement")
    print("    Baseline 2: LSTM Round-Robin -> LSTM forecast + the same margin, round-robin placement")

    fap = FAPScalePolicy(capacity, table, failure_scorer, pfail_gate=gate, quantile=QUANTILE)
    hpa = ReactiveHPAPolicy(capacity, HPA_TARGET)
    lstm = LSTMRoundRobinPolicy(capacity, table, quantile=QUANTILE)

    # =========================================================================
    # STEP 4: SIMULATION CONTROL LOOP
    # =========================================================================
    _section(f"SIMULATION CONTROL LOOP ({scenario.step_seconds}-second windows)", 4)

    print("\n  Running all 3 strategies against the same cluster...")
    print("  Per-window FAP-Scale pipeline:")
    print("    [Nirupam] 1. Classify workload pattern (autocorrelation, wavelet, CV, Hurst)")
    print("    [Nirupam] 2. Forecast the next windows (ARIMA / LSTM / XGBoost, error-weighted)")
    print("    [Adarsh]  3. Score each node: Pfail from SMART telemetry, IS from the matrix")
    print("    [Aman]    4. Rank healthy nodes by S(i), size K on their expected capacity")
    print("    [Aman]    5. Place / remove / drain replicas, feed observed slowdown back (EMA)")
    print()

    simulator = ClusterSimulator(scenario)
    logs = {p.name: simulator.run(p, stop=stop) for p in (fap, hpa, lstm)}
    fap_log = logs[fap.name]
    decisions = fap.engine.get_history()      # decisions[i] was taken at the end of window start-1+i

    header = (
        f"{'Win':<5}| {'Load':>6} | {'Pattern':<9} | {'Forecast':>8} | {'Margin':>6} | "
        f"{'K':>3} | {'Capacity':>8} | {'Best Node':<9} | {'Pfail':>6} | {'IS':>5} | {'S(i)':>5} | {'SLA':<4}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    n_rows = len(fap_log)
    for i, row in enumerate(fap_log.itertuples(index=False)):
        if not (i < 6 or i % 100 == 0 or i == n_rows - 1):
            continue
        d = decisions[i]                       # the decision this window ran under
        top = d["ranked_nodes"][0] if d["ranked_nodes"] else {}
        print(
            f"{row.step:<5}| {row.load:>6.2f} | {d['pattern_label']:<9} | "
            f"{d['predicted_load']:>8.2f} | {d['load_margin']:>6.2f} | "
            f"{row.replicas:>3} | {row.capacity:>8.2f} | "
            f"{top.get('node_id', '-'):<9} | {top.get('pfail', 0):>6.3f} | "
            f"{top.get('is_score', 0):>5.3f} | {top.get('score', 0):>5.3f} | "
            f"{'MISS' if row.violation else 'ok':<4}"
        )
    print(sep)
    print("  Load / Capacity in QPS; Capacity = replicas that were up, minus what interference cost them")

    # =========================================================================
    # STEP 5: FUSION ENGINE SUMMARY & NODE HEALTH
    # =========================================================================
    _section("FUSION ENGINE SUMMARY & NODE HEALTH", 5)

    stats = fap.engine.summary_stats()
    print("\n  Fusion Engine Statistics:")
    for k, v in stats.items():
        print(f"    {k:.<35} {v}")

    matrix = fap.engine.interference_matrix
    levels = InterferenceMatrix.LEVEL_NAMES
    print("\n  Interference learned from the trace (IS by how busy the co-located tenant is):")
    for level, name in enumerate(levels):
        cols = [p * len(levels) + level for p in range(len(InterferenceMatrix.PATTERNS))]
        weight = matrix.updates[:, cols]
        if weight.sum() > 0:
            learned = float((matrix.matrix[:, cols] * weight).sum() / weight.sum())
            print(f"    tenant {name:<10} IS = {learned:.3f}")

    # Node health as FAP-Scale saw it, against what really happened
    last_seen = min(stop, scenario.n_steps) - 1
    print(f"\n  Nodes the health gate flagged (Pfail >= {gate:.3f}) or that failed:")
    print(f"    {'Node':<9} | {'Peak Pfail':>10} | {'Flagged':<8} | {'CPI':>5} | {'TTS':>8} | {'What really happened'}")
    print("    " + "-" * 78)
    caught = missed = false_alarm = 0
    for n, nid in enumerate(scenario.node_ids):
        readings = scenario.hardware[n][: int(scenario.day_of_step[last_seen]) + 1]
        peak = max(fap.engine._pfail_cache.get(r["hw_reading"], 0.0) for r in readings)
        flagged = peak >= gate
        fail_at = scenario.fail_step[n]
        died = 0 <= fail_at <= last_seen
        fails_later = fail_at > last_seen          # beyond the windows simulated in this run
        if not (flagged or died):
            continue
        caught += flagged and (died or fails_later)
        missed += died and not flagged
        false_alarm += flagged and fail_at < 0
        cpi = calculate_cpi(float(scenario.tenant_util[:last_seen + 1, n].mean()),
                            32.0 * min(1.0, float(scenario.tenant_queue_ms[:last_seen + 1, n].mean()) / 1000.0))
        tts = calculate_tts(readings[-1]["temperature_history"], THERMAL_LIMIT_C)
        if died:
            truth = f"FAILED at window {fail_at}"
        elif fails_later:
            truth = f"fails later, at window {fail_at}"
        else:
            truth = "kept running"
        marker = "X" if (died or fails_later) else "!"
        print(f"   {marker} {nid:<9} | {peak:>10.4f} | {'yes' if flagged else 'NO':<8} | "
              f"{cpi:>5.2f} | {tts:>8.5f} | {truth}")
    print(f"\n    Summary: {caught} failures flagged in advance, {missed} missed, "
          f"{false_alarm} healthy nodes flagged (false alarms: kept out of placement)")

    # =========================================================================
    # STEP 6: EVALUATION & COMPARISON
    # =========================================================================
    _section("FINAL EVALUATION & COMPARISON", 6)

    metrics = {
        "FAP-Scale (Proposed)":        summarize_run(logs[fap.name]),
        "Reactive-HPA (Baseline 1)":   summarize_run(logs[hpa.name]),
        "LSTM-RoundRobin (Baseline 2)": summarize_run(logs[lstm.name]),
    }

    print()
    print("=" * 110)
    print(
        f"  {'Strategy':<30} | {'SLA Violations':>14} | {'Over-Prov %':>11} | "
        f"{'Avg Replicas':>12} | {'Unhealthy Plc.':>14} | {'Replicas Lost':>13}"
    )
    print("=" * 110)
    for strat, m in metrics.items():
        print(
            f"  {strat:<30} | {m['sla_violations']:>6} ({m['sla_violation_percent']:>4.1f}%) | "
            f"{m['overprovisioning_percent']:>10.1f}% | {m['avg_replicas']:>12.2f} | "
            f"{m['unhealthy_placements']:>14} | {m['replicas_lost']:>13}"
        )
    print("=" * 110)
    print(f"  {metrics['FAP-Scale (Proposed)']['total_windows']} control windows on cluster draw seed={seed}")

    fap_m = metrics["FAP-Scale (Proposed)"]
    hpa_m = metrics["Reactive-HPA (Baseline 1)"]
    lstm_m = metrics["LSTM-RoundRobin (Baseline 2)"]

    def _versus(name, other):
        notes = []
        for key, label, unit in (("sla_violations", "SLA violations", ""),
                                 ("overprovisioning_percent", "over-provisioning", "%"),
                                 ("unhealthy_placements", "unhealthy placements", "")):
            a, b = fap_m[key], other[key]
            word = "fewer" if key != "overprovisioning_percent" else "lower"
            if a < b:
                notes.append(f"{word} {label} ({a:.4g}{unit} vs {b:.4g}{unit})")
            elif a > b:
                notes.append(f"MORE {label} ({a:.4g}{unit} vs {b:.4g}{unit})")
            else:
                notes.append(f"equal {label} ({a:.4g}{unit})")
        print(f"    vs {name}: " + "; ".join(notes))

    print("\n  Analysis (this cluster draw):")
    _versus("Reactive HPA", hpa_m)
    _versus("LSTM Round-Robin", lstm_m)
    print("    Over-provisioning and SLA violations trade off against each other, so read them")
    print("    together; results/real/tradeoff.png compares the strategies at equal cost.")

    print()
    save_path = "results/comparison_metrics.png"
    plot_evaluation_summary(metrics, save_path=save_path)

    print("\n  Done. Results saved to results/ directory.")
    print("  Averages over 20 cluster draws: python -m results.evaluate_real  (results/real/)")
    print()

    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FAP-Scale trace-driven simulation on real 2026 datasets.")
    ap.add_argument("--seed", type=int, default=0, help="which draw of node hardware to simulate")
    ap.add_argument("--windows", type=int, default=None, help="control windows to run (default: whole trace)")
    args = ap.parse_args()
    run_fap_scale_simulation(seed=args.seed, num_windows=args.windows)
