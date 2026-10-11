"""
Tests for the real-data path (the dataset loaders have their own file, test_traces.py):
  - S.M.A.R.T. features and the failure model on real drives
  - interference profiles, dynamic model selection, forecast margin
  - Kubernetes HPA algorithm, Fusion Engine health gate and joint sizing
  - the trace-driven cluster simulator

Tests that need the GenTD26 files are skipped until they have been downloaded
(python -m traces.gentd26); everything else runs offline from the committed
Backblaze sample or from small hand-built scenarios.
"""

import math
import unittest
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fusion.fusion_engine import FusionEngine, calculate_replicas
from models.load_pipeline import DynamicSelector, ForecastMargin, ROUTING, add_dynamic_selection
from models.node_health import train_real as node_health
from models.node_health.failure_scorer import (SMART_COUNTERS, SMART_FEATURES, THERMAL_LIMIT_C,
                                               NodeFailureScorer, calculate_cpi, calculate_tts,
                                               failure_label, smart_feature_frame)
from models.node_health.interference import InterferenceMatrix
from results.evaluate import summarize_run
from simulator.baselines import ReactiveHPAScaler
from simulator.cluster import (EVICTION_STEPS, STARTUP_STEPS, ClusterSimulator, Decision, Scenario,
                               build_scenario)
from simulator.policies import FAPScalePolicy, LSTMRoundRobinPolicy, ReactiveHPAPolicy
from traces import backblaze, gentd26

HAVE_GENTD26 = gentd26.is_available()
NEEDS_GENTD26 = "GenTD26 not downloaded yet: run  python -m traces.gentd26"


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures (built once)
# ─────────────────────────────────────────────────────────────────────────────

_CACHE = {}


def drive_features() -> pd.DataFrame:
    """SMART features for a fixed subset of the committed sample (keeps the suite fast)."""
    if "features" not in _CACHE:
        dd = backblaze.load_drive_days()
        failed = np.sort(dd.loc[dd["will_fail"], "drive_id"].unique())[:260]
        healthy = np.sort(dd.loc[~dd["will_fail"], "drive_id"].unique())[:700]
        subset = dd[dd["drive_id"].isin(np.concatenate([failed, healthy]))]
        _CACHE["features"] = smart_feature_frame(subset)
    return _CACHE["features"]


def trained():
    """(scorer, gate, test drives) trained on the subset above."""
    if "trained" not in _CACHE:
        train, calib, test = node_health.split_drives(drive_features())
        scorer = NodeFailureScorer(random_state=0).fit(train, calibration_df=calib)
        _CACHE["trained"] = (scorer, node_health.alert_gate(scorer, calib), test)
    return _CACHE["trained"]


def healthy_reading(**overrides) -> dict:
    reading = {"temperature_history": [30.0] * 10, "power_on_hours": 20000.0}
    reading.update({c: 0.0 for c in SMART_COUNTERS})
    reading.update({f"{c}_delta": 0.0 for c in SMART_COUNTERS})
    reading.update(overrides)
    return reading


def node(nid, util=10.0, pattern="bursty", ready=True, **hw) -> dict:
    telem = healthy_reading(**hw)
    telem.update({"node_id": nid, "cpu_utilization": util, "tenant_pattern": pattern,
                  "ready": ready, "queue_ms": 200.0})
    return telem


# ─────────────────────────────────────────────────────────────────────────────
# Node health on real telemetry
# ─────────────────────────────────────────────────────────────────────────────

class TestSmartFeatures(unittest.TestCase):

    def test_cpi_without_context_switches_is_renormalised(self):
        self.assertAlmostEqual(calculate_cpi(100.0, 32.0), 1.0)
        self.assertAlmostEqual(calculate_cpi(50.0, 0.0), 0.5 * 0.5 / 0.85)
        # with all three terms the Review-2 formula is unchanged
        self.assertAlmostEqual(calculate_cpi(50.0, 16.0, 5000.0), 0.5 * 0.5 + 0.35 * 0.5 + 0.15 * 0.5)

    def test_frame_tts_equals_calculate_tts(self):
        f = drive_features()
        one = f[f["drive_id"] == f["drive_id"].iloc[0]].reset_index(drop=True)
        for i in (1, 5, 25, len(one) - 1):
            temps = one["temperature_filled"].iloc[max(0, i - 9): i + 1].tolist()
            self.assertAlmostEqual(one["tts"].iloc[i], calculate_tts(temps, THERMAL_LIMIT_C), places=9)

    def test_deltas_only_look_backwards(self):
        f = drive_features()
        for c in SMART_COUNTERS:
            self.assertTrue((f[f"{c}_delta"].dropna() >= 0).all())
        first_rows = f.groupby("drive_id").head(1)
        self.assertTrue((first_rows[[f"{c}_delta" for c in SMART_COUNTERS]].fillna(0) == 0).all().all())

    def test_label_excludes_the_failure_day(self):
        f = drive_features()
        y = failure_label(f)
        self.assertEqual(int(y[(f["days_to_failure"] == 0).to_numpy()].sum()), 0)
        self.assertTrue((y[(f["days_to_failure"] == 3).to_numpy()] == 1).all())
        self.assertEqual(int(y[(~f["will_fail"]).to_numpy()].sum()), 0)


class TestFailureScorerOnRealDrives(unittest.TestCase):

    def test_split_is_drive_disjoint(self):
        train, calib, test = node_health.split_drives(drive_features())
        ids = [set(part["drive_id"]) for part in (train, calib, test)]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertTrue(all(part["will_fail"].any() for part in (train, calib, test)))

    def test_model_separates_failing_from_healthy_drives(self):
        from sklearn.metrics import roc_auc_score
        scorer, gate, test = trained()
        self.assertEqual(scorer.schema, "smart")
        rows = test[test["days_to_failure"] != 0]
        p = scorer.predict_frame(rows)
        self.assertTrue(((p > 0) & (p < 1)).all())
        self.assertGreater(roc_auc_score(failure_label(rows), p), 0.75)

    def test_dict_and_frame_paths_agree(self):
        scorer, _, test = trained()
        f = drive_features()
        row = test[test["will_fail"] & (test["days_to_failure"] == 2)].iloc[0]
        history = f[(f["drive_id"] == row["drive_id"]) & (f["date"] <= row["date"])]
        reading = {"temperature_history": history["temperature_filled"].tolist()[-10:],
                   "temperature_c": row["temperature_c"], "power_on_hours": row["power_on_hours"]}
        reading.update({c: row[c] for c in SMART_COUNTERS})
        reading.update({f"{c}_delta": row[f"{c}_delta"] for c in SMART_COUNTERS})
        frame_p = float(scorer.predict_frame(pd.DataFrame([row]))[0])
        self.assertAlmostEqual(scorer.predict_pfail(reading), frame_p, places=6)
        self.assertAlmostEqual(float(scorer.predict_many([reading])[0]), frame_p, places=6)

    def test_media_errors_raise_pfail(self):
        scorer, _, _ = trained()
        clean = scorer.predict_pfail(healthy_reading())
        dying = scorer.predict_pfail(healthy_reading(
            pending_sectors=400.0, pending_sectors_delta=200.0,
            offline_uncorrectable=400.0, offline_uncorrectable_delta=200.0,
            realloc_sectors=3000.0, realloc_sectors_delta=800.0))
        self.assertGreater(dying, clean)
        self.assertGreater(dying, 5 * clean)

    def test_gate_respects_the_false_alarm_budget(self):
        scorer, _, _ = trained()
        _, calib, _ = node_health.split_drives(drive_features())
        gate = node_health.alert_gate(scorer, calib, 0.05)
        healthy = calib[~calib["will_fail"]]
        self.assertLessEqual((scorer.predict_frame(healthy) > gate).mean(), 0.05 + 1e-9)

    def test_legacy_schema_still_works(self):
        from data_gen.synthetic_data import generate_node_telemetry
        scorer = NodeFailureScorer().fit(generate_node_telemetry(6, 15, ["node-02"]))
        self.assertEqual(scorer.schema, "legacy")
        self.assertEqual(len(scorer.extract_features({"cpu_utilization": 50.0})), 8)


class TestInterferenceProfiles(unittest.TestCase):

    def test_fifteen_distinct_profile_bins(self):
        bins = {InterferenceMatrix.profile_bin(p, u)
                for p in InterferenceMatrix.PATTERNS for u in (0.0, 5.0, 20.0, 45.0, 90.0)}
        self.assertEqual(bins, set(range(15)))
        self.assertEqual(InterferenceMatrix.profile_name(InterferenceMatrix.profile_bin("bursty", 90.0)),
                         "bursty/saturated")

    def test_prior_matrix_learns_by_ema(self):
        m = InterferenceMatrix(alpha=0.1, prior=0.0)
        self.assertEqual(float(m.matrix.max()), 0.0)
        i, j = InterferenceMatrix.profile_bin("hybrid", 30.0), InterferenceMatrix.profile_bin("bursty", 70.0)
        for _ in range(60):
            m.update_ema(i, j, 0.3)
        self.assertAlmostEqual(m.get_interference(i, j), 0.3, delta=0.01)
        self.assertEqual(m.get_interference(i, j), m.get_interference(j, i))
        self.assertEqual(int(m.updates[i, j]), 60)


# ─────────────────────────────────────────────────────────────────────────────
# Load side: dynamic selection and the safety margin
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicSelector(unittest.TestCase):

    def test_starts_from_the_review2_routing_table(self):
        self.assertEqual(DynamicSelector().table(), ROUTING)

    def test_weight_moves_to_the_model_that_is_right(self):
        s = DynamicSelector(min_scores=5)
        for _ in range(30):
            s.update("hybrid", {"ARIMA": 0.1, "LSTM": 1.0, "XGBoost": 2.0})
        w = s.weights("hybrid")
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertEqual(s.choose("hybrid"), "ARIMA")
        self.assertGreater(w["ARIMA"], 0.99)
        self.assertEqual(s.choose("bursty"), ROUTING["bursty"])      # other patterns untouched

    def test_blend_lies_between_the_models(self):
        s = DynamicSelector(min_scores=1)
        s.update("periodic", {"ARIMA": 1.0, "LSTM": 1.0, "XGBoost": 1.0})
        self.assertAlmostEqual(s.blend("periodic", {"ARIMA": 3.0, "LSTM": 6.0, "XGBoost": 9.0}), 6.0)

    def test_selection_never_uses_the_future(self):
        n = 60
        table = pd.DataFrame({"step": np.arange(n), "pattern": ["hybrid"] * n,
                              "actual_next": np.ones(n), "actual": np.ones(n)})
        for m, v in (("ARIMA", 1.0), ("LSTM", 5.0), ("XGBoost", 9.0)):   # ARIMA is always right
            table[m] = table[m + "_next"] = v
        out = add_dynamic_selection(table, DynamicSelector(min_scores=10))
        self.assertEqual(out["routed_to"].iloc[0], ROUTING["hybrid"])   # no evidence yet
        self.assertEqual(out["routed_to"].iloc[-1], "ARIMA")
        self.assertTrue((out["Static"] == 9.0).all())
        # changing a later outcome must not change an earlier selection
        table2 = table.copy()
        table2.loc[40:, "actual_next"] = 100.0
        out2 = add_dynamic_selection(table2, DynamicSelector(min_scores=10))
        self.assertTrue((out["Dynamic"].iloc[:41] == out2["Dynamic"].iloc[:41]).all())


class TestForecastMargin(unittest.TestCase):

    def test_margin_is_the_quantile_of_recent_shortfalls(self):
        m = ForecastMargin(quantile=0.9, window=100, min_samples=10)
        for shortfall in np.linspace(-1, 1, 101):
            m.update(predicted=5.0, actual=5.0 + shortfall)
        self.assertAlmostEqual(m.value(), float(np.quantile(np.linspace(-1, 1, 101)[-100:], 0.9)), places=6)

    def test_never_negative_and_falls_back_before_it_has_a_record(self):
        m = ForecastMargin(min_samples=5)
        self.assertEqual(m.value(), 0.0)
        self.assertGreater(m.value(history=[1, 2, 4, 3, 6, 5, 9]), 0.0)
        for _ in range(10):
            m.update(predicted=10.0, actual=2.0)       # always over-forecasting
        self.assertEqual(m.value(), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: Kubernetes HPA algorithm
# ─────────────────────────────────────────────────────────────────────────────

class TestKubernetesHPA(unittest.TestCase):

    def test_scale_up_formula(self):
        hpa = ReactiveHPAScaler(target_cpu_threshold=70.0)
        self.assertEqual(hpa.desired_replicas(90.0, 10, 10), math.ceil(10 * 90 / 70))

    def test_tolerance_band_holds(self):
        hpa = ReactiveHPAScaler(target_cpu_threshold=70.0)
        self.assertEqual(hpa.desired_replicas(75.0, 10, 10), 10)     # within +-10 % of target

    def test_capacity_already_starting_is_not_requested_twice(self):
        hpa = ReactiveHPAScaler(target_cpu_threshold=70.0)
        self.assertEqual(hpa.desired_replicas(90.0, 10, 13), 13)     # 3 replicas on their way

    def test_scale_down_waits_out_the_stabilisation_window(self):
        hpa = ReactiveHPAScaler(target_cpu_threshold=70.0, stabilization_windows=5)
        self.assertEqual(hpa.desired_replicas(90.0, 10, 10), 13)
        held = [hpa.desired_replicas(20.0, 13, 13) for _ in range(4)]
        self.assertEqual(held, [13, 13, 13, 13])
        self.assertLess(hpa.desired_replicas(20.0, 13, 13), 13)      # the high recommendation aged out


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Engine: health gate and joint sizing
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionEngineOnRealTelemetry(unittest.TestCase):

    def setUp(self):
        self.scorer, self.gate, _ = trained()
        self.sick = dict(pending_sectors=400.0, pending_sectors_delta=200.0,
                         offline_uncorrectable=400.0, offline_uncorrectable_delta=200.0,
                         realloc_sectors=3000.0, realloc_sectors_delta=800.0)
        self.forecast = {"pattern": "hybrid", "predicted_load": 4.0}

    def engine(self, **kw):
        return FusionEngine(replica_capacity=1.0, failure_scorer=self.scorer,
                            interference_matrix=InterferenceMatrix(prior=0.0), **kw)

    def test_gate_keeps_replicas_off_a_failing_node(self):
        cluster = [node("node-01", **self.sick)] + [node(f"node-{i:02d}") for i in range(2, 9)]
        d = self.engine(pfail_gate=self.gate, joint_sizing=True).evaluate_window(
            [4.0] * 60, cluster, forecast=self.forecast)
        self.assertNotIn("node-01", d["selected_nodes"])
        self.assertEqual(d["ranked_nodes"][-1]["node_id"], "node-01")
        self.assertFalse(d["ranked_nodes"][-1]["healthy"])

    def test_replica_on_a_failing_or_dead_node_is_drained(self):
        cluster = ([node("node-01", **self.sick), node("node-02", ready=False)]
                   + [node(f"node-{i:02d}") for i in range(3, 9)])
        d = self.engine(pfail_gate=self.gate, joint_sizing=True).evaluate_window(
            [4.0] * 60, cluster, forecast=self.forecast, hosting=["node-01", "node-02", "node-03"])
        self.assertEqual(set(d["drain_nodes"]), {"node-01", "node-02"})
        self.assertIn("node-03", d["selected_nodes"])                # healthy replicas stay put

    def test_without_failure_awareness_the_sick_node_is_fair_game(self):
        cluster = [node("node-01", util=0.0, **self.sick)] + [node(f"node-{i:02d}", util=40.0) for i in range(2, 6)]
        d = self.engine(pfail_gate=self.gate, joint_sizing=True, use_failure=False).evaluate_window(
            [4.0] * 60, cluster, forecast=self.forecast)
        self.assertEqual(d["selected_nodes"][0], "node-01")          # least utilised, risk ignored

    def test_joint_sizing_reduces_to_the_plain_formula_without_interference(self):
        cluster = [node(f"node-{i:02d}") for i in range(1, 13)]
        d = self.engine(joint_sizing=True).evaluate_window([4.0] * 60, cluster, forecast=self.forecast)
        self.assertEqual(d["target_replicas"], calculate_replicas(4.0, 1.0))

    def test_joint_sizing_adds_replicas_when_nodes_are_noisy(self):
        engine = self.engine(joint_sizing=True)
        svc = InterferenceMatrix.profile_bin("hybrid", 50.0)
        engine.interference_matrix.matrix[svc, :] = 0.2              # every placement loses 20 %
        cluster = [node(f"node-{i:02d}") for i in range(1, 13)]
        d = engine.evaluate_window([4.0] * 60, cluster, forecast=self.forecast)
        self.assertEqual(d["target_replicas"], math.ceil(4.0 / 0.8))
        self.assertGreaterEqual(d["expected_capacity"], 4.0)

    def test_margin_is_added_to_the_forecast(self):
        margin = ForecastMargin(quantile=0.9, min_samples=5)
        for _ in range(10):
            margin.update(predicted=4.0, actual=6.0)
        cluster = [node(f"node-{i:02d}") for i in range(1, 13)]
        d = self.engine(joint_sizing=True, margin=margin).evaluate_window(
            [4.0] * 60, cluster, forecast=self.forecast)
        self.assertAlmostEqual(d["load_margin"], 2.0)
        self.assertEqual(d["target_replicas"], 6)

    def test_matrix_learns_from_observed_degradation(self):
        engine = self.engine(joint_sizing=True)
        busy = [dict(node(f"node-{i:02d}", util=80.0), observed_degradation=0.4) for i in range(1, 7)]
        for _ in range(40):
            engine.evaluate_window([4.0] * 60, busy, forecast=self.forecast)
        svc = InterferenceMatrix.profile_bin("hybrid", 50.0)
        tenant = InterferenceMatrix.profile_bin("bursty", 80.0)
        self.assertAlmostEqual(engine.interference_matrix.get_interference(svc, tenant), 0.4, delta=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# Cluster simulator mechanics (hand-built scenario: no datasets needed)
# ─────────────────────────────────────────────────────────────────────────────

def tiny_scenario(steps=40, nodes=6, load=2.0, fail=None, slowdown=0.0, start=2) -> Scenario:
    fail_step = np.full(nodes, -1)
    for n, t in (fail or {}).items():
        fail_step[n] = t
    return Scenario(
        load=np.full(steps, load), capacity=1.0, step_seconds=57, start=start,
        node_ids=[f"node-{n + 1:02d}" for n in range(nodes)],
        tenant_util=np.zeros((steps, nodes)), tenant_mem=np.zeros((steps, nodes)),
        tenant_queue_ms=np.zeros((steps, nodes)),
        tenant_pattern=np.full((steps, nodes), "bursty", dtype=object),
        slowdown=np.full((steps, nodes), slowdown),
        hardware=[[dict(healthy_reading(), hw_reading=(n, 0))] for n in range(nodes)],
        day_of_step=np.zeros(steps, dtype=int), fail_step=fail_step, steps_per_day=steps,
    )


class _FixedPolicy:
    """Always asks for the same replica count, on the lowest-numbered free nodes."""
    name = "fixed"

    def __init__(self, replicas, drain_at=None):
        self.replicas, self.drain_at = replicas, drain_at or {}

    def reset(self): pass
    def on_boot(self, replicas): pass
    def on_placed(self, nodes): pass

    def decide(self, obs):
        return Decision(self.replicas, list(range(obs.scenario.n_nodes)),
                        drain=self.drain_at.get(obs.t, []))


class TestClusterSimulator(unittest.TestCase):

    def test_steady_state_meets_demand(self):
        log = ClusterSimulator(tiny_scenario(load=2.0)).run(_FixedPolicy(3))
        self.assertEqual(int(log["violation"].sum()), 0)
        self.assertTrue((log["capacity"].iloc[3:] == 3.0).all())
        m = summarize_run(log)
        self.assertAlmostEqual(m["overprovisioning_percent"], 100 * (3 - 2) / 3, places=1)

    def test_interference_costs_capacity(self):
        log = ClusterSimulator(tiny_scenario(load=2.0, slowdown=0.5)).run(_FixedPolicy(3))
        self.assertTrue((log["capacity"].iloc[3:] == 1.5).all())
        self.assertTrue(log["violation"].iloc[3:].all())             # 1.5 < 2.0 though 3 replicas are paid for

    def test_node_failure_strands_a_replica_until_eviction(self):
        fail_at = 12
        log = ClusterSimulator(tiny_scenario(load=2.5, fail={0: fail_at})).run(_FixedPolicy(3)).set_index("step")
        self.assertEqual(int(log["lost"].sum()), 1)
        self.assertEqual(log.loc[fail_at - 1, "capacity"], 3.0)
        # capacity is short from the failure until the replacement is up: the stranded
        # replica is evicted EVICTION_STEPS windows later, its replacement is requested
        # at the end of that window and serves after the cold start
        back = fail_at + EVICTION_STEPS + 1 + STARTUP_STEPS
        self.assertTrue((log.loc[fail_at:back - 1, "capacity"] == 2.0).all())
        self.assertTrue(log.loc[fail_at:back - 1, "violation"].all())
        self.assertEqual(log.loc[back, "capacity"], 3.0)

    def test_draining_before_the_failure_loses_nothing(self):
        fail_at = 12
        policy = _FixedPolicy(3, drain_at={fail_at - 4: [0]})
        log = ClusterSimulator(tiny_scenario(load=2.5, fail={0: fail_at})).run(policy)
        self.assertEqual(int(log["lost"].sum()), 0)
        self.assertEqual(int(log["violation"].sum()), 0)             # replacement was up before the old one left
        self.assertEqual(int(log["drained"].sum()), 1)

    def test_placing_on_a_node_about_to_fail_is_counted(self):
        log = ClusterSimulator(tiny_scenario(load=2.5, fail={0: 20})).run(_FixedPolicy(3))
        self.assertEqual(summarize_run(log)["unhealthy_placements"], 1)

    def test_dead_nodes_are_not_reused(self):
        sc = tiny_scenario(steps=40, load=2.5, fail={0: 10})
        log = ClusterSimulator(sc).run(_FixedPolicy(3)).set_index("step")
        self.assertEqual(log.loc[39, "capacity"], 3.0)
        self.assertEqual(summarize_run(log.reset_index())["unhealthy_placements"], 1)   # only the initial one


@unittest.skipUnless(HAVE_GENTD26, NEEDS_GENTD26)
class TestTraceDrivenScenario(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scorer, cls.gate, cls.test = trained()
        cls.scenario = build_scenario(cls.test, n_nodes=24, n_failing=3, seed=4)

    def test_scenario_is_reproducible_and_real(self):
        again = build_scenario(self.test, n_nodes=24, n_failing=3, seed=4)
        self.assertTrue(np.array_equal(self.scenario.fail_step, again.fail_step))
        self.assertEqual(self.scenario.drive_ids, again.drive_ids)
        self.assertEqual(int((self.scenario.fail_step >= 0).sum()), 3)
        self.assertTrue(np.array_equal(self.scenario.load, gentd26.load().series("qps_total")))
        # failing nodes are backed by drives that really failed, healthy ones by drives that did not
        outcome = self.test.groupby("drive_id")["will_fail"].first()
        for n, drive in enumerate(self.scenario.drive_ids):
            self.assertEqual(bool(outcome[drive]), self.scenario.fail_step[n] >= 0)

    def test_telemetry_hides_the_failure_until_it_is_detected(self):
        n = int(np.flatnonzero(self.scenario.fail_step >= 0)[0])
        f = int(self.scenario.fail_step[n])
        self.assertTrue(self.scenario.telemetry(f - 1)[n]["ready"])
        self.assertTrue(self.scenario.telemetry(f)[n]["ready"])       # died this window, not marked yet
        self.assertFalse(self.scenario.telemetry(f + 1)[n]["ready"])

    def test_three_strategies_run_on_the_trace(self):
        steps = np.arange(self.scenario.start, self.scenario.start + 120)
        load = self.scenario.load
        table = pd.DataFrame({"step": steps, "pattern": "hybrid",
                              "actual": np.maximum(load[steps + 1], load[steps + 2])})
        table["LSTM"] = table["Dynamic"] = load[steps]              # persistence stands in for the models
        sim, c = ClusterSimulator(self.scenario), self.scenario.capacity
        stop = self.scenario.start + 120
        for policy in (ReactiveHPAPolicy(c), LSTMRoundRobinPolicy(c, table),
                       FAPScalePolicy(c, table, self.scorer, pfail_gate=self.gate)):
            m = summarize_run(sim.run(policy, stop=stop))
            self.assertEqual(m["total_windows"], 120)
            self.assertGreaterEqual(m["avg_replicas"], 1.0)
            self.assertTrue(0.0 <= m["overprovisioning_percent"] <= 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
