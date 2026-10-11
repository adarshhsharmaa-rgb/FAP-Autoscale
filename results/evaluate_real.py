"""
results/evaluate_real.py  --  the full evaluation on the real traces.

Everything written to results/real/ comes from Alibaba GenTD26 (2026) and
Backblaze Drive Stats Q2 2026; see data/README.md for provenance.

  node_health.json / .png      failure model on held-out drives
  forecast_raw.csv             every forecaster's prediction at every control window
  forecast_error.csv / .png    NRMSE per pattern: single models vs dynamic selection
  forecast_timeline.png        the real day with the dynamic forecast over it
  closed_loop_runs.csv         one row per (cluster draw, strategy)
  closed_loop.csv / .png       mean and spread over the cluster draws
  ablation.csv                 FAP-Scale with one part switched off at a time
  tradeoff.csv / .png          cost against SLA violations as each strategy's caution varies
  summary.json                 headline numbers

    python -m results.evaluate_real              # 20 cluster draws, ~10 min
    python -m results.evaluate_real --seeds 5    # quicker
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.load_pipeline import ROUTING, add_dynamic_selection, rolling_forecasts, warm_up
from models.node_health import train_real as node_health
from results.evaluate import summarize_run
from simulator.cluster import HISTORY_WINDOW, ClusterSimulator, build_scenario
from simulator.policies import FAPScalePolicy, LSTMRoundRobinPolicy, ReactiveHPAPolicy
from traces import backblaze, gentd26

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real")
FORECASTS = os.path.join(OUT, "forecast_raw.csv")

HORIZON = 2               # windows ahead a forecast must cover (next window + cold start)
QUANTILE = 0.95           # share of forecast misses the safety margin covers
HPA_TARGET = 70.0         # Reactive-HPA utilisation target, as in Review 2
SWEEP_Q = (0.5, 0.8, 0.9, 0.95, 0.98, 0.995)
SWEEP_HPA = (50.0, 60.0, 70.0, 80.0, 90.0)

FAP, HPA, LSTM = "FAP-Scale", "Reactive-HPA", "LSTM-RoundRobin"
METRICS = ["sla_violations", "sla_violation_percent", "unserved_percent",
           "overprovisioning_percent", "avg_replicas", "replicas_lost",
           "unhealthy_placements", "placements", "drained"]

# categorical palette, checked for colour-blind separation; a colour always means the same thing
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"
COLOR = {FAP: "#2a78d6", "Dynamic": "#2a78d6", HPA: "#eb6834", LSTM: "#1baf7a", "LSTM": "#1baf7a",
         "ARIMA": "#eda100", "XGBoost": "#6250d6", "Static": "#9a9994"}
PATTERN_COLOR = {"periodic": "#eda100", "bursty": "#eb6834", "hybrid": "#6250d6"}


def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def nrmse(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)) / np.mean(actual))


# --------------------------------------------------------------------------- #
# 1. Load side: forecasters on the real day
# --------------------------------------------------------------------------- #
def forecast_table(force: bool = False) -> pd.DataFrame:
    """Every forecaster replayed over the GenTD26 day (cached; ~9 min to rebuild)."""
    if os.path.exists(FORECASTS) and not force:
        return pd.read_csv(FORECASTS)
    print("  replaying the forecasters over the trace (about 9 minutes, done once)...")
    backend = warm_up(real=True)
    t0 = time.time()
    table = rolling_forecasts(
        gentd26.load().series("qps_total"), start=HISTORY_WINDOW, window=HISTORY_WINDOW,
        horizon=HORIZON,
        progress=lambda i, n: print(f"    {i}/{n} windows  {time.time() - t0:.0f}s", flush=True))
    table.insert(0, "series", "qps_total")
    table.to_csv(FORECASTS, index=False)
    print(f"  saved {FORECASTS}  [LSTM: {backend}]")
    return table


def evaluate_load_side(table: pd.DataFrame) -> dict:
    load = gentd26.load().series("qps_total")
    t = table.copy()
    t["Persistence_next"] = load[t["step"].to_numpy()]
    models = ["Persistence", "ARIMA", "LSTM", "XGBoost", "Static", "Dynamic"]
    groups = [(p, g) for p, g in t.groupby("pattern")] + [("overall", t)]
    err = pd.DataFrame({name: {m: nrmse(g["actual_next"], g[m + "_next"]) for m in models}
                        for name, g in groups}).T
    err.insert(0, "windows", [len(g) for _, g in groups])
    err.index.name = "pattern"
    err.round(4).to_csv(os.path.join(OUT, "forecast_error.csv"))

    # chart: single models against dynamic selection
    shown = ["ARIMA", "LSTM", "XGBoost", "Static", "Dynamic"]
    names = {"Static": "Fixed routing (Review 2)", "Dynamic": "Dynamic selection (FAP-Scale)"}
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x, width = np.arange(len(err)), 0.16
    for i, m in enumerate(shown):
        bars = ax.bar(x + (i - 2) * width, err[m], width - 0.02, color=COLOR[m],
                      label=names.get(m, m))
        if m == "Dynamic":
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                        f"{b.get_height():.3f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{p}\n({n} windows)" for p, n in zip(err.index, err["windows"])])
    ax.set_ylabel("NRMSE, next control window (lower is better)", color=MUTED)
    ax.set_title("Load forecast error on the real GenTD26 day", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forecast_error.png"), dpi=150)
    plt.close(fig)

    # chart: the day itself, with the forecast and the detected pattern
    fig, (top, strip) = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True,
                                     gridspec_kw={"height_ratios": [9, 1], "hspace": 0.08})
    minutes = t["step"] * gentd26.STEP_SECONDS / 3600.0
    top.plot(minutes + gentd26.STEP_SECONDS / 3600.0, t["actual_next"], color=MUTED, lw=0.9,
             label="actual gateway load")
    top.plot(minutes + gentd26.STEP_SECONDS / 3600.0, t["Dynamic_next"], color=COLOR["Dynamic"],
             lw=1.6, label="FAP-Scale forecast (dynamic selection)")
    top.set_ylabel("load (anonymised QPS)", color=MUTED)
    top.set_title("One production day of GenTD26, forecast one window ahead", loc="left",
                  color=INK, fontsize=12)
    top.legend(frameon=False, loc="upper right", fontsize=9)
    _style(top)
    for p, c in PATTERN_COLOR.items():
        sel = (t["pattern"] == p).to_numpy()
        strip.bar(minutes[sel], 1.0, width=gentd26.STEP_SECONDS / 3600.0, color=c, label=p)
    strip.set_yticks([])
    strip.set_xlabel("hours into the trace", color=MUTED)
    strip.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -1.9),
                 fontsize=9, title="detected pattern", title_fontsize=9)
    for side in ("top", "right", "left"):
        strip.spines[side].set_visible(False)
    strip.spines["bottom"].set_color(GRID)
    strip.tick_params(colors=MUTED, length=0)
    fig.savefig(os.path.join(OUT, "forecast_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    overall = err.loc["overall"]
    singles = {m: float(overall[m]) for m in ("ARIMA", "LSTM", "XGBoost")}
    best = min(singles, key=singles.get)
    return {
        "trace": "Alibaba GenTD26 gateway QPS, 57 s windows",
        "windows": int(len(t)),
        "patterns_detected": {k: int(v) for k, v in t["pattern"].value_counts().items()},
        "nrmse_next_window": {m: round(float(overall[m]), 4) for m in models},
        "best_single_model": best,
        "dynamic_vs_best_single_pct": round(100 * (1 - overall["Dynamic"] / singles[best]), 2),
        "dynamic_vs_fixed_routing_pct": round(100 * (1 - overall["Dynamic"] / overall["Static"]), 2),
        "dynamic_vs_lstm_pct": round(100 * (1 - overall["Dynamic"] / overall["LSTM"]), 2),
        "most_trusted_model_share": {k: round(float(v), 3) for k, v in
                                     t["routed_to"].value_counts(normalize=True).items()},
        "fixed_routing_table": dict(ROUTING),
    }


# --------------------------------------------------------------------------- #
# 2. Node side: failure model on held-out drives
# --------------------------------------------------------------------------- #
def evaluate_node_health():
    scorer, gate, test, metrics = node_health.train()
    rows = test[test["days_to_failure"] != 0].copy()
    rows["pfail"] = scorer.predict_frame(rows)
    failing = rows[rows["will_fail"] & (rows["days_to_failure"] <= 30)]
    by_day = failing.groupby("days_to_failure")["pfail"].median()
    healthy_median = float(rows.loc[~rows["will_fail"], "pfail"].median())
    points = pd.DataFrame(metrics["operating_points"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    a.plot(by_day.index, by_day.values, color=COLOR[FAP], lw=2, marker="o", ms=4,
           label="drives that went on to fail (median)")
    a.axhline(healthy_median, color=MUTED, lw=1.5, ls="--", label="healthy drives (median)")
    a.axhline(gate, color=INK, lw=1, ls=":")
    a.text(30, gate * 1.12, f"alert gate {gate:.3f}", ha="left", va="bottom", fontsize=8.5, color=INK)
    a.set_yscale("log")
    a.invert_xaxis()
    a.set_xlabel("days before the drive failed", color=MUTED)
    a.set_ylabel("Pfail (log scale)", color=MUTED)
    a.set_title("Pfail rises well before a real failure", loc="left", color=INK, fontsize=12)
    a.legend(frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.3))
    _style(a)

    x, width = np.arange(len(points)), 0.36
    caught = b.bar(x - width / 2, 100 * points["failing_drives_flagged"], width - 0.03,
                   color=COLOR[FAP], label="failing drives flagged in their last week")
    false = b.bar(x + width / 2, 100 * points["healthy_drives_ever_flagged"], width - 0.03,
                  color=COLOR[HPA], label="healthy drives ever flagged")
    for bars in (caught, false):
        for bar in bars:
            b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                   f"{bar.get_height():.0f}%", ha="center", fontsize=9, color=INK)
    b.set_xticks(x, [f"{100 * v:.0f}%" for v in points["false_alarm_budget"]])
    b.set_xlabel("false-alarm budget (share of healthy drive-days)", color=MUTED)
    b.set_ylabel("share of test drives", color=MUTED)
    b.set_ylim(0, 100)
    b.set_title("What the alert gate catches (held-out drives)", loc="left", color=INK, fontsize=12)
    b.legend(frameon=False, fontsize=9, loc="upper left")
    _style(b)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "node_health.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "node_health.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return scorer, gate, test, metrics


# --------------------------------------------------------------------------- #
# 3. Closed loop: the three strategies on the same simulated clusters
# --------------------------------------------------------------------------- #
def strategies(capacity, table, scorer, gate, quantile=QUANTILE, hpa_target=HPA_TARGET):
    return [
        FAPScalePolicy(capacity, table, scorer, pfail_gate=gate, quantile=quantile),
        ReactiveHPAPolicy(capacity, hpa_target),
        LSTMRoundRobinPolicy(capacity, table, quantile=quantile),
    ]


def ablations(capacity, table, scorer, gate, quantile=QUANTILE):
    kw = dict(failure_scorer=scorer, pfail_gate=gate, quantile=quantile)
    return [
        FAPScalePolicy(capacity, table, name="FAP-Scale without failure awareness",
                       failure_aware=False, **kw),
        FAPScalePolicy(capacity, table, name="FAP-Scale without interference awareness",
                       interference_aware=False, **kw),
        FAPScalePolicy(capacity, table, name="FAP-Scale with the LSTM forecast only",
                       column="LSTM", **kw),
    ]


def run_closed_loop(seeds, table, scorer, gate, test, with_sweep=True, sweep_seeds=10):
    runs, sweep = [], []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        scenario = build_scenario(test, seed=seed)
        sim, c = ClusterSimulator(scenario), scenario.capacity
        for kind, policies in (("main", strategies(c, table, scorer, gate)),
                               ("ablation", ablations(c, table, scorer, gate))):
            for policy in policies:
                m = summarize_run(sim.run(policy))
                m.update(strategy=policy.name, kind=kind, seed=seed)
                runs.append(m)
        if with_sweep and i < sweep_seeds:
            grid = ([(FAP, q, FAPScalePolicy(c, table, scorer, pfail_gate=gate, quantile=q)) for q in SWEEP_Q]
                    + [(LSTM, q, LSTMRoundRobinPolicy(c, table, quantile=q)) for q in SWEEP_Q]
                    + [(HPA, u, ReactiveHPAPolicy(c, u)) for u in SWEEP_HPA])
            for name, setting, policy in grid:
                m = summarize_run(sim.run(policy))
                m.update(strategy=name, setting=setting, seed=seed)
                sweep.append(m)
        print(f"    cluster draw {i + 1}/{len(seeds)}  {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(runs), pd.DataFrame(sweep)


def summarise_closed_loop(runs: pd.DataFrame) -> pd.DataFrame:
    stats = runs.groupby(["kind", "strategy"], sort=False)[METRICS].agg(["mean", "std"])
    stats.columns = [f"{m}_{s}" for m, s in stats.columns]
    return stats.reset_index()


def paired_wins(runs: pd.DataFrame, metric: str, other: str) -> str:
    """On how many cluster draws FAP-Scale is at least as good as `other` (lower is better)."""
    wide = runs[runs["kind"] == "main"].pivot(index="seed", columns="strategy", values=metric)
    return f"{int((wide[FAP] <= wide[other]).sum())}/{len(wide)}"


def plot_closed_loop(runs: pd.DataFrame) -> None:
    main = runs[runs["kind"] == "main"]
    order = [FAP, HPA, LSTM]
    panels = [("sla_violation_percent", "SLA violations", "% of control windows", "{:.1f}%"),
              ("overprovisioning_percent", "Over-provisioning", "% of paid capacity unused", "{:.1f}%"),
              ("unhealthy_placements", "Unhealthy placements", "replicas put on a node about to fail", "{:.1f}"),
              ("replicas_lost", "Replicas lost", "replicas on a node when it died", "{:.2f}")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    for ax, (metric, title, unit, fmt) in zip(axes, panels):
        g = main.groupby("strategy")[metric]
        mean, std = g.mean().reindex(order), g.std().reindex(order).fillna(0)
        bars = ax.bar(range(3), mean, 0.62, color=[COLOR[s] for s in order],
                      yerr=std, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
        top = float((mean + std).max()) or 1.0
        for bar, v, s in zip(bars, mean, std):
            ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.03 * top, fmt.format(v),
                    ha="center", fontsize=10, color=INK, fontweight="bold")
        ax.set_ylim(0, top * 1.2)
        ax.set_xticks(range(3), ["FAP-Scale", "Reactive\nHPA", "LSTM\nround-robin"], color=INK)
        ax.set_title(f"{title}\n", loc="left", color=INK, fontsize=12)
        ax.text(0, 1.02, f"{unit} (lower is better)", transform=ax.transAxes, fontsize=8.5, color=MUTED)
        _style(ax)
    n = main["seed"].nunique()
    fig.suptitle(f"FAP-Scale against the baselines on real traces: mean of {n} simulated clusters, "
                 f"bars show one standard deviation", x=0.01, ha="left", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "closed_loop.png"), dpi=150)
    plt.close(fig)


def summarise_tradeoff(sweep: pd.DataFrame) -> pd.DataFrame:
    return (sweep.groupby(["strategy", "setting"], sort=False)
            [["avg_replicas", "sla_violation_percent", "sla_violations", "overprovisioning_percent",
              "unhealthy_placements", "replicas_lost"]].mean().reset_index())


def violations_at_cost(tradeoff: pd.DataFrame, strategy: str, replicas: float) -> float:
    """SLA-violation % a strategy reaches at a given average replica count (interpolated)."""
    curve = tradeoff[tradeoff["strategy"] == strategy].sort_values("avg_replicas")
    return float(np.interp(replicas, curve["avg_replicas"], curve["sla_violation_percent"]))


def plot_tradeoff(tradeoff: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    labels = {FAP: "FAP-Scale", HPA: "Reactive HPA", LSTM: "LSTM round-robin"}
    defaults = {FAP: QUANTILE, LSTM: QUANTILE, HPA: HPA_TARGET}
    for name in (HPA, LSTM, FAP):
        curve = tradeoff[tradeoff["strategy"] == name].sort_values("avg_replicas")
        ax.plot(curve["avg_replicas"], curve["sla_violation_percent"], color=COLOR[name], lw=2,
                marker="o", ms=5, markeredgecolor="white", markeredgewidth=1, label=labels[name])
        chosen = curve[np.isclose(curve["setting"], defaults[name])]
        ax.scatter(chosen["avg_replicas"], chosen["sla_violation_percent"], s=150, color=COLOR[name],
                   edgecolor=INK, linewidth=1.4, zorder=5)
        end = curve.iloc[0]
        ax.annotate(labels[name], (end["avg_replicas"], end["sla_violation_percent"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9.5, color=INK)
    ax.scatter([], [], s=150, color="white", edgecolor=INK, linewidth=1.4,
               label=f"default setting (margin q={QUANTILE}, HPA target {HPA_TARGET:.0f}%)")
    ax.set_xlabel("average replicas running (cost)", color=MUTED)
    ax.set_ylabel("SLA violations, % of control windows", color=MUTED)
    ax.set_title("Cost against SLA violations as each strategy is made more cautious",
                 loc="left", color=INK, fontsize=12, pad=22)
    ax.text(0, 1.02, "lower-left is better; each point is one setting, averaged over cluster draws",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tradeoff.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(n_seeds: int = 20, sweep_seeds: int = 10) -> dict:
    os.makedirs(OUT, exist_ok=True)
    gentd26.fetch()
    trace = gentd26.load()

    print("1/3 node failure model (Backblaze Drive Stats Q2 2026)")
    scorer, gate, test, health = evaluate_node_health()
    print(f"    ROC-AUC {health['roc_auc']:.3f}; gate {gate:.3f} flags "
          f"{health['failed_drives_caught']:.0%} of failing test drives in their last week")

    print("2/3 load forecasting (Alibaba GenTD26)")
    table = add_dynamic_selection(forecast_table())
    table.to_csv(FORECASTS, index=False)
    load_side = evaluate_load_side(table)
    print("    NRMSE " + "  ".join(f"{k}={v}" for k, v in load_side["nrmse_next_window"].items()))

    print(f"3/3 closed loop: {n_seeds} cluster draws")
    runs, sweep = run_closed_loop(range(n_seeds), table, scorer, gate, test, sweep_seeds=sweep_seeds)
    runs.to_csv(os.path.join(OUT, "closed_loop_runs.csv"), index=False)
    stats = summarise_closed_loop(runs)
    stats[stats["kind"] == "main"].drop(columns="kind").round(3).to_csv(
        os.path.join(OUT, "closed_loop.csv"), index=False)
    stats[stats["kind"] == "ablation"].drop(columns="kind").round(3).to_csv(
        os.path.join(OUT, "ablation.csv"), index=False)
    plot_closed_loop(runs)
    tradeoff = summarise_tradeoff(sweep)
    tradeoff.round(3).to_csv(os.path.join(OUT, "tradeoff.csv"), index=False)
    plot_tradeoff(tradeoff)

    mean = runs[runs["kind"] == "main"].groupby("strategy")[METRICS].mean()
    equal_cost = {}
    for ref in (HPA, LSTM):
        cost = float(mean.loc[ref, "avg_replicas"])
        equal_cost[f"at {ref}'s cost"] = {
            "avg_replicas": round(cost, 2),
            **{s: round(violations_at_cost(tradeoff, s, cost), 2) for s in (FAP, HPA, LSTM)}}
    scenario = build_scenario(test, seed=0)
    summary = {
        "datasets": {
            "workload": f"Alibaba GenTD26 (cluster-trace-v2026-GenAI), commit {gentd26.COMMIT[:10]}",
            "failures": health["dataset"],
        },
        "setup": {
            "cluster_draws": n_seeds, "nodes": scenario.n_nodes,
            "failing_nodes_per_draw": int((scenario.fail_step >= 0).sum()),
            "control_windows": int(scenario.n_steps - scenario.start),
            "window_seconds": trace.step_seconds,
            "replica_capacity_qps": round(scenario.capacity, 4),
            "margin_quantile": QUANTILE, "hpa_target_utilization": HPA_TARGET,
            "forecast_horizon_windows": HORIZON,
        },
        "node_health": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in health.items() if k not in ("features", "operating_points")},
        "load_forecasting": load_side,
        "closed_loop_mean": {s: {m: round(float(mean.loc[s, m]), 3) for m in METRICS}
                             for s in (FAP, HPA, LSTM)},
        "fap_at_least_as_good_on_draws": {
            f"{metric} vs {other}": paired_wins(runs, metric, other)
            for metric in ("sla_violations", "overprovisioning_percent", "unhealthy_placements",
                           "replicas_lost")
            for other in (HPA, LSTM)},
        "sla_violation_percent_at_equal_cost": equal_cost,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    show = ["sla_violations", "sla_violation_percent", "overprovisioning_percent", "avg_replicas",
            "unhealthy_placements", "replicas_lost"]
    print("\n" + mean.loc[[FAP, HPA, LSTM], show].round(2).to_string())
    print("\nablation:\n" + runs[runs["kind"] == "ablation"].groupby("strategy", sort=False)[show]
          .mean().round(2).to_string())
    print("\nSLA violation % at equal cost: " + json.dumps(equal_cost))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate FAP-Scale on the real traces.")
    ap.add_argument("--seeds", type=int, default=20, help="cluster draws to simulate")
    ap.add_argument("--sweep-seeds", type=int, default=10, help="draws used for the trade-off curves")
    args = ap.parse_args()
    main(args.seeds, args.sweep_seeds)
