"""
results/evaluate_load_side.py  (Person A)

Evaluates the load side on unseen synthetic windows and writes to results/load_side/:

  classifier_confusion.png   rule-based vs decision-tree confusion matrices
  classifier_features.png    feature scatter coloured by true pattern
  forecast_error.png         NRMSE per pattern: single models vs dynamic routing
  forecast_examples.png      one example per pattern with rolling forecasts
  forecast_error.csv         per-pattern / per-model NRMSE table
  summary.json               headline numbers for the slides

Error metric: NRMSE = RMSE / mean(actual), so series with different base loads
are comparable. Each window is forecast one step at a time over its last
TEST_STEPS points (rolling origin), exactly as the control loop would.

Run:  python -m results.evaluate_load_side
"""
from __future__ import annotations

import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_gen.synthetic_data import PATTERNS, generate_workload_windows
from models.forecasters import FORECASTERS
from models.load_pipeline import ROUTING, warm_up
from models.pattern_classifier import (FEATURE_NAMES, classify_pattern,
                                       extract_features, get_trained_tree)

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "load_side")
N_WINDOWS = 45          # 15 per pattern
WINDOW_LEN = 200
TEST_STEPS = 12
SEED = 2026             # different from every training seed
COLORS = {"periodic": "#3b6fd8", "bursty": "#e0663a", "hybrid": "#2f9e6e"}


def nrmse(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(np.sqrt(np.mean((actual - pred) ** 2)) / np.mean(actual))


# --------------------------------------------------------------------------- #
def evaluate_classifier(data):
    rows = []
    for w, label in data:
        f = extract_features(w)
        rows.append({**f, "true": label,
                     "rules": classify_pattern(w, "rules")[0],
                     "tree": classify_pattern(w, "tree")[0]})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    acc = {}
    for ax, method in zip(axes, ("rules", "tree")):
        cm = pd.crosstab(df["true"], df[method]).reindex(index=PATTERNS, columns=PATTERNS, fill_value=0)
        acc[method] = float((df["true"] == df[method]).mean())
        ax.imshow(cm.values, cmap="Blues")
        for i in range(3):
            for j in range(3):
                v = cm.values[i, j]
                ax.text(j, i, v, ha="center", va="center",
                        color="white" if v > cm.values.max() / 2 else "black", fontsize=12)
        ax.set_xticks(range(3), PATTERNS)
        ax.set_yticks(range(3), PATTERNS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        name = "Rule-based" if method == "rules" else "Decision tree"
        ax.set_title(f"{name}  (accuracy {acc[method]:.0%})")
    fig.suptitle(f"Workload pattern classifier on {len(df)} unseen windows")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "classifier_confusion.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for p in PATTERNS:
        d = df[df["true"] == p]
        axes[0].scatter(d["hurst"], d["cv"], c=COLORS[p], label=p, alpha=0.8)
        axes[1].scatter(d["acf_peaks"] + np.random.uniform(-.15, .15, len(d)),
                        d["wavelet_ratio"], c=COLORS[p], label=p, alpha=0.8)
    axes[0].set(xlabel="Hurst exponent", ylabel="Coefficient of variation")
    axes[1].set(xlabel="ACF peak count (jittered)", ylabel="Wavelet energy ratio")
    axes[0].legend()
    fig.suptitle("Classifier feature space")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "classifier_features.png"), dpi=150)
    plt.close(fig)

    df.to_csv(os.path.join(OUT, "classifier_predictions.csv"), index=False)
    return df, acc


# --------------------------------------------------------------------------- #
def evaluate_forecasters(data):
    """Rolling one-step forecasts with every model; dynamic = routed choice."""
    records = []
    examples = {}
    timing = {m: [] for m in FORECASTERS}
    for wi, (w, label) in enumerate(data):
        for step in range(TEST_STEPS):
            end = len(w) - TEST_STEPS + step
            hist, actual = w[:end], w[end]
            preds = {}
            for m, fn in FORECASTERS.items():
                t0 = time.perf_counter()
                preds[m] = float(fn(hist)["forecast"][0])
                timing[m].append(time.perf_counter() - t0)
            routed = ROUTING[classify_pattern(hist, "rules")[0]]
            rec = {"window": wi, "pattern": label, "actual": actual,
                   "naive_last": float(hist[-1]), **preds,
                   "Dynamic (FAP-Scale)": preds[routed],
                   "Oracle routing": preds[ROUTING[label]], "routed_to": routed}
            records.append(rec)
        if label not in examples:
            examples[label] = wi
        print(f"  window {wi + 1:2d}/{len(data)} ({label}) done")
    return pd.DataFrame(records), examples, {m: 1000 * np.mean(v) for m, v in timing.items()}


def summarise_errors(df):
    models = ["naive_last", "ARIMA", "LSTM", "XGBoost", "Dynamic (FAP-Scale)", "Oracle routing"]
    rows = []
    for p in list(PATTERNS) + ["overall"]:
        d = df if p == "overall" else df[df["pattern"] == p]
        # NRMSE computed per window, then averaged (each window weighted equally)
        row = {"pattern": p}
        for m in models:
            row[m] = float(np.mean([nrmse(g["actual"], g[m]) for _, g in d.groupby("window")]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("pattern")[models]


def plot_errors(tab):
    models = ["ARIMA", "LSTM", "XGBoost", "Dynamic (FAP-Scale)"]
    colors = ["#9aa5b8", "#b8a39a", "#9ab8a6", "#1b2a6b"]
    groups = list(tab.index)
    x = np.arange(len(groups))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (m, c) in enumerate(zip(models, colors)):
        bars = ax.bar(x + (i - 1.5) * width, tab[m], width, label=m, color=c)
        if m.startswith("Dynamic"):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                        f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, groups)
    ax.set_ylabel("NRMSE (lower is better)")
    ax.set_title("One-step load forecast error: single models vs dynamic model selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forecast_error.png"), dpi=150)
    plt.close(fig)


def plot_examples(data, df, examples):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    for ax, p in zip(axes, PATTERNS):
        wi = examples[p]
        w = data[wi][0]
        d = df[df["window"] == wi]
        t_test = np.arange(len(w) - TEST_STEPS, len(w))
        ax.plot(np.arange(len(w) - 70, len(w)), w[-70:], color="#444", lw=1.4, label="actual load")
        routed = d["routed_to"].mode()[0]
        ax.plot(t_test, d["Dynamic (FAP-Scale)"], "o-", color=COLORS[p], lw=2,
                label=f"FAP-Scale forecast (routed to {routed})")
        for m, ls in (("ARIMA", ":"), ("LSTM", "--"), ("XGBoost", "-.")):
            if m != routed:
                ax.plot(t_test, d[m], ls, color="#999", lw=1, label=f"{m} (not selected)")
        ax.axvline(len(w) - TEST_STEPS - 0.5, color="#bbb", lw=0.8)
        ax.set_title(f"{p} workload")
        ax.set_ylabel("load")
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("control step (60 s each)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forecast_examples.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    print("warming up (loads cached LSTM, trains decision tree)...")
    warm_up()
    get_trained_tree()
    data = generate_workload_windows(N_WINDOWS, WINDOW_LEN, seed=SEED)

    print("1/2 classifier...")
    cdf, acc = evaluate_classifier(data)
    print(f"    rules accuracy={acc['rules']:.1%}  tree accuracy={acc['tree']:.1%}")

    print("2/2 forecasters (rolling one-step)...")
    fdf, examples, lat = evaluate_forecasters(data)
    fdf.to_csv(os.path.join(OUT, "forecast_raw.csv"), index=False)
    tab = summarise_errors(fdf)
    tab.round(4).to_csv(os.path.join(OUT, "forecast_error.csv"))
    plot_errors(tab)
    plot_examples(data, fdf, examples)

    overall = tab.loc["overall"]
    best_single = min(("ARIMA", "LSTM", "XGBoost"), key=lambda m: overall[m])
    summary = {
        "n_test_windows": N_WINDOWS,
        "rolling_steps_per_window": TEST_STEPS,
        "classifier_accuracy": {k: round(v, 4) for k, v in acc.items()},
        "overall_nrmse": {k: round(float(v), 4) for k, v in overall.items()},
        "best_single_model": best_single,
        "dynamic_vs_best_single_improvement_pct":
            round(100 * (1 - overall["Dynamic (FAP-Scale)"] / overall[best_single]), 1),
        "routing_accuracy_in_loop": round(float(
            (fdf["routed_to"] == fdf["pattern"].map(ROUTING)).mean()), 4),
        "mean_latency_ms": {k: round(v, 1) for k, v in lat.items()},
        "note": "Synthetic data; preliminary simulation results, not Google trace numbers.",
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nNRMSE by pattern:\n" + tab.round(4).to_string())
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
