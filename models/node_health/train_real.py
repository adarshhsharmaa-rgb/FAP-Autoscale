"""
models/node_health/train_real.py  (Person B)

Train and evaluate the node failure model on real hardware failures:
Backblaze Drive Stats Q2 2026 (traces/backblaze.py).

Protocol
    * Drives are split 60 / 20 / 20 into train / calibration / test, failed and
      healthy drives separately so each part has the same mix. A drive is in
      exactly one part, so the model is always tested on hardware it has never
      seen.
    * Label: the drive fails within the next 7 days. The failure day itself is
      excluded; the point is to know beforehand.
    * LightGBM is fitted on train, mapped to probabilities on calibration
      (Platt scaling), and the alert gate is the lowest Pfail that flags at
      most 5 % of the drive-days of healthy calibration drives. The budget is
      generous on purpose: a false alarm costs one replica migration with no
      downtime, a miss costs the replica.
    * Everything reported comes from the test drives. The simulator also draws
      its nodes from the test drives only.

The sample keeps every failed drive but only 1.2 % of healthy ones, so failures
are over-represented about 80-fold. Ranking metrics (ROC-AUC, recall, false
alarm rate) are unaffected; precision is reported both on the sample and
re-weighted to the real fleet.

    python -m models.node_health.train_real
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from models.node_health.failure_scorer import (FAILURE_HORIZON_DAYS, SMART_FEATURES,
                                               NodeFailureScorer, failure_label,
                                               smart_feature_frame)
from traces import backblaze

SPLIT = (0.6, 0.2, 0.2)
MAX_FALSE_ALARM = 0.05     # share of healthy drives' drive-days the gate may flag


def split_drives(features: pd.DataFrame, seed: int = 2026
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drive-disjoint train / calibration / test split, stratified by outcome."""
    rng = np.random.default_rng(seed)
    part = {}
    outcome = features.groupby("drive_id")["will_fail"].first()
    for flag in (True, False):
        ids = rng.permutation(np.sort(outcome.index[outcome == flag].to_numpy()))
        a, b = int(SPLIT[0] * len(ids)), int((SPLIT[0] + SPLIT[1]) * len(ids))
        part.update({d: 0 for d in ids[:a]})
        part.update({d: 1 for d in ids[a:b]})
        part.update({d: 2 for d in ids[b:]})
    which = features["drive_id"].map(part)
    return features[which == 0], features[which == 1], features[which == 2]


def alert_gate(scorer: NodeFailureScorer, calibration: pd.DataFrame,
               max_false_alarm: float = MAX_FALSE_ALARM) -> float:
    """Lowest Pfail gate that flags at most `max_false_alarm` of healthy drives' drive-days."""
    healthy = calibration[~calibration["will_fail"]]
    return float(np.quantile(scorer.predict_frame(healthy), 1.0 - max_false_alarm))


def operating_points(scorer: NodeFailureScorer, calibration: pd.DataFrame, test: pd.DataFrame,
                     budgets=(0.01, 0.02, 0.05, 0.10)) -> pd.DataFrame:
    """Detection versus false alarms on the test drives for several alert budgets."""
    rows = test[test["days_to_failure"] != 0].copy()
    rows["pfail"] = scorer.predict_frame(rows)
    failing, healthy = rows[rows["will_fail"]], rows[~rows["will_fail"]]
    last_week = failing[failing["days_to_failure"].between(1, FAILURE_HORIZON_DAYS)]
    out = []
    for budget in budgets:
        gate = alert_gate(scorer, calibration, budget)
        out.append({
            "false_alarm_budget": budget,
            "gate": gate,
            "failing_drives_flagged": float((last_week["pfail"] >= gate).groupby(last_week["drive_id"]).any().mean()),
            "healthy_drive_days_flagged": float((healthy["pfail"] >= gate).mean()),
            "healthy_drives_ever_flagged": float((healthy["pfail"] >= gate).groupby(healthy["drive_id"]).any().mean()),
        })
    return pd.DataFrame(out)


def evaluate(scorer: NodeFailureScorer, test: pd.DataFrame, gate: float,
             healthy_sample_rate: float) -> Dict[str, float]:
    """Held-out metrics, per drive-day and per drive."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = test[test["days_to_failure"] != 0].copy()
    y = failure_label(rows)
    p = scorer.predict_frame(rows)
    rows["pfail"], rows["flag"] = p, p >= gate

    # A drive-day is a hit if the drive fails within the horizon, a false alarm if
    # the drive never fails. Flags on a failing drive more than a week early are
    # early warnings and count as neither.
    is_healthy = ~rows["will_fail"].to_numpy()
    tp = float(np.sum(rows["flag"].to_numpy() & (y == 1)))
    fp = float(np.sum(rows["flag"].to_numpy() & is_healthy))
    # healthy drives were sub-sampled: weight them back up for fleet-level numbers
    weight = np.where(is_healthy, 1.0 / healthy_sample_rate, 1.0)

    # per drive: was a drive that failed flagged on any of its last 7 days before failing?
    failed = rows[rows["will_fail"]]
    last_week = failed[failed["days_to_failure"].between(1, FAILURE_HORIZON_DAYS)]
    caught = last_week.groupby("drive_id")["flag"].any()
    first_flag = failed[failed["flag"]].groupby("drive_id")["days_to_failure"].max()
    lead = first_flag[first_flag.index.isin(caught.index[caught])]
    healthy_flagged = rows[is_healthy].groupby("drive_id")["flag"].any()
    errors = ["realloc_sectors", "pending_sectors", "offline_uncorrectable", "reported_uncorrectable"]
    has_errors = rows[errors].fillna(0).sum(axis=1) > 0

    return {
        "test_drives": int(test["drive_id"].nunique()),
        "test_failed_drives": int(test.loc[test["will_fail"], "drive_id"].nunique()),
        "test_drive_days": int(len(rows)),
        "positive_drive_days": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_sample": float(average_precision_score(y, p)),
        "pr_auc_fleet": float(average_precision_score(y, p, sample_weight=weight)),
        "gate": float(gate),
        "recall_drive_days": tp / max(1.0, float(y.sum())),
        "false_alarm_drive_days": fp / max(1.0, float(is_healthy.sum())),
        "precision_sample": tp / max(1.0, tp + fp),
        "precision_fleet": tp / max(1.0, tp + fp / healthy_sample_rate),
        "failed_drives_with_warning_window": int(len(caught)),
        "failed_drives_caught": float(caught.mean()) if len(caught) else 0.0,
        "median_lead_days": float(lead.median()) if len(lead) else 0.0,
        "healthy_drives_ever_flagged": float(healthy_flagged.mean()) if len(healthy_flagged) else 0.0,
        "error_counters_nonzero_day_before_failure":
            float(has_errors[(rows["days_to_failure"] == 1).to_numpy()].mean()),
        "error_counters_nonzero_healthy": float(has_errors[is_healthy].mean()),
    }


def train(seed: int = 2026, verbose: bool = False):
    """
    Returns (scorer, gate, test_features, metrics).

    `test_features` is the drive-day table (with SMART features) of the drives
    held out from training and calibration: the only drives the simulator may use.
    metrics["operating_points"] lists detection against false alarms for other
    alert budgets.
    """
    drive_days = backblaze.load_drive_days()
    meta = backblaze.load_meta()
    features = smart_feature_frame(drive_days)
    train_df, calib_df, test_df = split_drives(features, seed)

    scorer = NodeFailureScorer(random_state=seed).fit(train_df, calibration_df=calib_df)
    gate = alert_gate(scorer, calib_df)
    metrics = evaluate(scorer, test_df, gate, meta["healthy_sample_rate"])
    metrics["operating_points"] = operating_points(scorer, calib_df, test_df).to_dict("records")
    metrics.update({
        "train_drives": int(train_df["drive_id"].nunique()),
        "train_drive_days": int((train_df["days_to_failure"] != 0).sum()),
        "calibration_drives": int(calib_df["drive_id"].nunique()),
        "features": list(SMART_FEATURES),
        "horizon_days": FAILURE_HORIZON_DAYS,
        "dataset": f"Backblaze Drive Stats {meta['quarter']} "
                   f"({meta['first_day']} to {meta['last_day']})",
    })
    if verbose:
        print(f"  {metrics['dataset']}")
        print(f"  drives: {metrics['train_drives']:,} train / {metrics['calibration_drives']:,} "
              f"calibration / {metrics['test_drives']:,} test "
              f"({metrics['test_failed_drives']} of the test drives failed)")
        print(f"  ROC-AUC {metrics['roc_auc']:.3f} | PR-AUC {metrics['pr_auc_sample']:.3f} "
              f"| alert gate Pfail >= {gate:.3f}")
        print(f"  at the gate: {metrics['failed_drives_caught']:.0%} of failing drives flagged "
              f"in their last week (first flag a median {metrics['median_lead_days']:.0f} d "
              f"ahead), {metrics['false_alarm_drive_days']:.1%} of healthy drive-days flagged")
    return scorer, gate, test_df, metrics


if __name__ == "__main__":
    scorer, gate, test_df, metrics = train(verbose=True)
    for k, v in metrics.items():
        if k not in ("features", "operating_points"):
            print(f"    {k:.<44} {v}")
    print(pd.DataFrame(metrics["operating_points"]).round(3).to_string(index=False))
