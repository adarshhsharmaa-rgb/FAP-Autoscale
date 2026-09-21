"""
Test suite for Person C deliverables:
  - FusionEngine (fusion/fusion_engine.py)
  - ReactiveHPAScaler & LSTMRoundRobinScaler (simulator/baselines.py)
  - End-to-end integration with Person A & B modules

Covers:
  1. compute_node_score() S(i) formula correctness
  2. calculate_replicas() ceiling arithmetic
  3. FusionEngine.evaluate_window() returns correct schema
  4. FusionEngine EMA feedback (interference matrix mutates between windows)
  5. FusionEngine weight-sum validation
  6. FusionEngine summary_stats()
  7. FusionEngine.history records all windows
  8. ReactiveHPAScaler — scale-up and scale-down logic
  9. LSTMRoundRobinScaler — proactive replica count from LSTM forecast
 10. Full end-to-end integration smoke test (Person A + B + C pipeline)
"""

import unittest
import numpy as np

from data_gen.synthetic_data import generate_workload_time_series, generate_node_telemetry, extract_single_node_telemetry
from models.node_health.failure_scorer import NodeFailureScorer
from models.node_health.interference import InterferenceMatrix
from fusion.fusion_engine import FusionEngine, compute_node_score, calculate_replicas
from simulator.baselines import ReactiveHPAScaler, LSTMRoundRobinScaler


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_telemetry(node_id: str = "node-01", cpu: float = 30.0) -> dict:
    return {
        "node_id": node_id,
        "cpu_utilization": cpu,
        "run_queue_length": 2.0,
        "context_switch_rate": 1000.0,
        "temperature_history": [50.0] * 10,
        "tdp": 105.0,
        "dimm_errors": 0,
        "disk_health": 1.0,
        "net_retransmits": 0,
    }


def _make_cluster(n: int = 5) -> list:
    return [_make_telemetry(f"node-{i:02d}", cpu=30.0 + i * 5) for i in range(1, n + 1)]


def _make_workload(size: int = 60, val: float = 60.0) -> list:
    return [val + float(np.sin(i / 5.0) * 5) for i in range(size)]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeNodeScore(unittest.TestCase):
    """Tests for compute_node_score() S(i) formula."""

    def test_output_in_range(self):
        score = compute_node_score(pfail=0.1, is_score=0.2, utilization=40.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_perfect_node_scores_high(self):
        """A node with Pfail=0, IS=0, U=0 should score 1.0."""
        score = compute_node_score(pfail=0.0, is_score=0.0, utilization=0.0)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_worst_node_scores_low(self):
        """A node with Pfail=1, IS=1, U=1 should score 0.0."""
        score = compute_node_score(pfail=1.0, is_score=1.0, utilization=1.0)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_utilization_normalised_from_percent(self):
        """Passing utilization as 50.0 (%) or 0.5 (ratio) should give the same result."""
        s_pct = compute_node_score(pfail=0.2, is_score=0.3, utilization=50.0)
        s_ratio = compute_node_score(pfail=0.2, is_score=0.3, utilization=0.5)
        self.assertAlmostEqual(s_pct, s_ratio, places=4)

    def test_weights_respected(self):
        """Higher pfail should reduce score proportionally."""
        s_low = compute_node_score(pfail=0.1, is_score=0.2, utilization=30.0)
        s_high = compute_node_score(pfail=0.9, is_score=0.2, utilization=30.0)
        self.assertGreater(s_low, s_high)


class TestCalculateReplicas(unittest.TestCase):
    """Tests for calculate_replicas() ceiling arithmetic."""

    def test_basic_ceiling(self):
        # 75 / 20 = 3.75 → ceil → 4
        self.assertEqual(calculate_replicas(75.0, 20.0), 4)

    def test_exact_division(self):
        # 60 / 20 = 3.0 → ceil → 3
        self.assertEqual(calculate_replicas(60.0, 20.0), 3)

    def test_minimum_one_replica(self):
        self.assertEqual(calculate_replicas(0.0, 20.0), 1)
        self.assertEqual(calculate_replicas(1.0, 20.0), 1)

    def test_zero_capacity_fallback(self):
        self.assertEqual(calculate_replicas(100.0, 0.0), 1)

    def test_large_load(self):
        k = calculate_replicas(1000.0, 20.0)
        self.assertEqual(k, 50)


class TestFusionEngineInit(unittest.TestCase):
    """Tests for FusionEngine initialisation."""

    def test_default_weights_valid(self):
        fe = FusionEngine()
        self.assertAlmostEqual(fe.alpha + fe.beta + fe.gamma, 1.0, places=6)

    def test_invalid_weights_raise(self):
        with self.assertRaises(AssertionError):
            FusionEngine(alpha=0.5, beta=0.5, gamma=0.5)

    def test_default_interference_matrix_created(self):
        fe = FusionEngine()
        self.assertIsInstance(fe.interference_matrix, InterferenceMatrix)


class TestFusionEngineEvaluateWindow(unittest.TestCase):
    """Tests for FusionEngine.evaluate_window() output schema and correctness."""

    def setUp(self):
        self.engine = FusionEngine(alpha=0.4, beta=0.35, gamma=0.25)
        self.cluster = _make_cluster(n=6)
        self.workload = _make_workload(size=60, val=60.0)

    def test_output_keys_present(self):
        result = self.engine.evaluate_window(self.workload, self.cluster)
        for key in ["window_id", "pattern_label", "predicted_load",
                    "target_replicas", "ranked_nodes", "selected_nodes", "ema_update"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_pattern_label_valid(self):
        result = self.engine.evaluate_window(self.workload, self.cluster)
        self.assertIn(result["pattern_label"], ["periodic", "bursty", "hybrid"])

    def test_predicted_load_positive(self):
        result = self.engine.evaluate_window(self.workload, self.cluster)
        self.assertGreater(result["predicted_load"], 0.0)

    def test_target_replicas_positive_int(self):
        result = self.engine.evaluate_window(self.workload, self.cluster)
        self.assertIsInstance(result["target_replicas"], int)
        self.assertGreaterEqual(result["target_replicas"], 1)

    def test_ranked_nodes_sorted_descending(self):
        result = self.engine.evaluate_window(self.workload, self.cluster)
        scores = [n["score"] for n in result["ranked_nodes"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_selected_nodes_bounded_by_cluster_size(self):
        result = self.engine.evaluate_window(self.workload, self.cluster)
        self.assertLessEqual(len(result["selected_nodes"]), len(self.cluster))

    def test_node_scores_in_range(self):
        result = self.engine.evaluate_window(self.workload, self.cluster)
        for n in result["ranked_nodes"]:
            self.assertGreaterEqual(n["score"], 0.0)
            self.assertLessEqual(n["score"], 1.0)
            self.assertGreaterEqual(n["pfail"], 0.0)
            self.assertLessEqual(n["pfail"], 1.0)

    def test_window_id_increments(self):
        r0 = self.engine.evaluate_window(self.workload, self.cluster)
        r1 = self.engine.evaluate_window(self.workload, self.cluster)
        self.assertEqual(r0["window_id"], 0)
        self.assertEqual(r1["window_id"], 1)

    def test_ema_update_in_range(self):
        result = self.engine.evaluate_window(self.workload, self.cluster)
        self.assertGreaterEqual(result["ema_update"], 0.0)
        self.assertLessEqual(result["ema_update"], 1.0)


class TestFusionEngineEMAFeedback(unittest.TestCase):
    """Test that interference matrix is updated by evaluate_window (EMA feedback)."""

    def test_matrix_changes_after_window(self):
        engine = FusionEngine(alpha=0.4, beta=0.35, gamma=0.25)
        cluster = _make_cluster(n=4)
        workload = _make_workload(size=60, val=60.0)

        # Snapshot matrix before
        mat_before = engine.interference_matrix.matrix.copy()
        engine.evaluate_window(workload, cluster)
        mat_after = engine.interference_matrix.matrix.copy()

        # At least one value should have changed
        self.assertFalse(np.allclose(mat_before, mat_after),
                         "Interference matrix should be mutated after EMA feedback")


class TestFusionEngineHistory(unittest.TestCase):
    """Test that FusionEngine maintains per-window history correctly."""

    def test_history_grows_with_windows(self):
        engine = FusionEngine()
        cluster = _make_cluster(n=4)
        workload = _make_workload()
        for _ in range(5):
            engine.evaluate_window(workload, cluster)
        self.assertEqual(len(engine.get_history()), 5)

    def test_reset_clears_history(self):
        engine = FusionEngine()
        cluster = _make_cluster(n=4)
        workload = _make_workload()
        engine.evaluate_window(workload, cluster)
        engine.reset()
        self.assertEqual(len(engine.get_history()), 0)
        self.assertEqual(engine.window_count, 0)


class TestFusionEngineSummaryStats(unittest.TestCase):
    """Test FusionEngine.summary_stats() output."""

    def test_summary_stats_keys(self):
        engine = FusionEngine()
        cluster = _make_cluster(n=4)
        workload = _make_workload()
        for _ in range(3):
            engine.evaluate_window(workload, cluster)
        stats = engine.summary_stats()
        for key in ["windows_processed", "avg_replicas", "avg_top_node_score",
                    "avg_top_pfail", "avg_ema_degradation"]:
            self.assertIn(key, stats)

    def test_summary_stats_windows_count(self):
        engine = FusionEngine()
        cluster = _make_cluster(n=4)
        workload = _make_workload()
        for _ in range(7):
            engine.evaluate_window(workload, cluster)
        self.assertEqual(engine.summary_stats()["windows_processed"], 7)


class TestReactiveHPAScaler(unittest.TestCase):
    """Tests for ReactiveHPAScaler."""

    def setUp(self):
        self.cluster = _make_cluster(n=5)

    def test_output_schema(self):
        scaler = ReactiveHPAScaler()
        res = scaler.evaluate_window(50.0, self.cluster)
        for key in ["strategy", "target_replicas", "selected_nodes"]:
            self.assertIn(key, res)
        self.assertEqual(res["strategy"], "Reactive-HPA")

    def test_scale_up_on_high_cpu(self):
        """High CPU cluster should trigger scale-up."""
        high_cpu_cluster = [_make_telemetry(f"node-{i:02d}", cpu=90.0) for i in range(1, 6)]
        scaler = ReactiveHPAScaler(target_cpu_threshold=70.0)
        res = scaler.evaluate_window(80.0, high_cpu_cluster)
        self.assertGreater(res["target_replicas"], 2)  # started at 2, should go up

    def test_scale_down_on_low_cpu(self):
        """Very low CPU cluster should trigger scale-down."""
        low_cpu_cluster = [_make_telemetry(f"node-{i:02d}", cpu=10.0) for i in range(1, 6)]
        scaler = ReactiveHPAScaler()
        scaler.current_replicas = 6  # start high
        res = scaler.evaluate_window(10.0, low_cpu_cluster)
        self.assertLess(res["target_replicas"], 6)

    def test_minimum_one_replica(self):
        scaler = ReactiveHPAScaler()
        scaler.current_replicas = 1
        low_cpu_cluster = [_make_telemetry(f"node-{i:02d}", cpu=5.0) for i in range(1, 3)]
        res = scaler.evaluate_window(5.0, low_cpu_cluster)
        self.assertGreaterEqual(res["target_replicas"], 1)

    def test_selected_nodes_count_matches_replicas(self):
        scaler = ReactiveHPAScaler()
        res = scaler.evaluate_window(50.0, self.cluster)
        self.assertEqual(len(res["selected_nodes"]), res["target_replicas"])

    def test_empty_cluster(self):
        scaler = ReactiveHPAScaler()
        res = scaler.evaluate_window(50.0, [])
        self.assertEqual(res["selected_nodes"], [])

    def test_reset(self):
        scaler = ReactiveHPAScaler()
        scaler.evaluate_window(50.0, self.cluster)
        scaler.evaluate_window(80.0, self.cluster)
        scaler.reset()
        self.assertEqual(scaler.current_replicas, 2)
        self.assertEqual(scaler.rr_index, 0)
        self.assertEqual(len(scaler.history), 0)


class TestLSTMRoundRobinScaler(unittest.TestCase):
    """Tests for LSTMRoundRobinScaler."""

    def setUp(self):
        self.cluster = _make_cluster(n=5)
        self.workload = _make_workload(size=60, val=80.0)

    def test_output_schema(self):
        scaler = LSTMRoundRobinScaler()
        res = scaler.evaluate_window(self.workload, self.cluster)
        for key in ["strategy", "predicted_load", "target_replicas", "selected_nodes"]:
            self.assertIn(key, res)
        self.assertEqual(res["strategy"], "LSTM-RoundRobin")

    def test_predicted_load_positive(self):
        scaler = LSTMRoundRobinScaler()
        res = scaler.evaluate_window(self.workload, self.cluster)
        self.assertGreater(res["predicted_load"], 0.0)

    def test_replica_count_matches_forecast(self):
        import math
        scaler = LSTMRoundRobinScaler(replica_capacity=20.0)
        res = scaler.evaluate_window(self.workload, self.cluster)
        expected_k = max(1, math.ceil(res["predicted_load"] / 20.0))
        self.assertEqual(res["target_replicas"], expected_k)

    def test_selected_nodes_count_matches_replicas(self):
        scaler = LSTMRoundRobinScaler()
        res = scaler.evaluate_window(self.workload, self.cluster)
        self.assertEqual(len(res["selected_nodes"]), res["target_replicas"])

    def test_empty_cluster(self):
        scaler = LSTMRoundRobinScaler()
        res = scaler.evaluate_window(self.workload, [])
        self.assertEqual(res["selected_nodes"], [])

    def test_reset(self):
        scaler = LSTMRoundRobinScaler()
        scaler.evaluate_window(self.workload, self.cluster)
        scaler.reset()
        self.assertEqual(scaler.rr_index, 0)
        self.assertEqual(len(scaler.history), 0)


class TestEndToEndIntegration(unittest.TestCase):
    """
    Smoke test for the full Person A + B + C pipeline.
    Runs N windows and asserts the overall pipeline doesn't crash and produces
    sane metric values.
    """

    def test_full_pipeline_runs(self):
        import os

        NUM_WINDOWS = 10
        NUM_NODES = 6

        # Generate data (Person A + B)
        workload_df = generate_workload_time_series(num_windows=NUM_WINDOWS, pattern="hybrid", seed=99)
        telemetry_df = generate_node_telemetry(num_nodes=NUM_NODES, num_windows=NUM_WINDOWS,
                                               fail_node_ids=["node-02"], seed=99)

        # Train Person B models
        scorer = NodeFailureScorer(random_state=99)
        scorer.fit(telemetry_df)
        imat = InterferenceMatrix(alpha=0.1, seed=99)

        # Person C Fusion Engine
        engine = FusionEngine(alpha=0.4, beta=0.35, gamma=0.25,
                              replica_capacity=20.0,
                              failure_scorer=scorer,
                              interference_matrix=imat)
        hpa = ReactiveHPAScaler()
        lstm = LSTMRoundRobinScaler()

        fap_history, hpa_history, lstm_history = [], [], []

        for w in range(NUM_WINDOWS):
            win_wl = workload_df[workload_df["window_id"] == w]["request_rate"].values
            actual = float(np.mean(win_wl)) if len(win_wl) > 0 else 50.0

            curr_telemetry = []
            pfails = {}
            for n in range(1, NUM_NODES + 1):
                nid = f"node-{n:02d}"
                t = extract_single_node_telemetry(telemetry_df, nid, w)
                curr_telemetry.append(t)
                pfails[nid] = scorer.predict_pfail(t)

            fap_r = engine.evaluate_window(list(win_wl), curr_telemetry)
            fap_r["actual_load"] = actual
            fap_r["node_pfails"] = pfails
            fap_history.append(fap_r)

            hpa_r = hpa.evaluate_window(actual, curr_telemetry)
            hpa_r["actual_load"] = actual
            hpa_r["node_pfails"] = pfails
            hpa_history.append(hpa_r)

            lstm_r = lstm.evaluate_window(list(win_wl), curr_telemetry)
            lstm_r["actual_load"] = actual
            lstm_r["node_pfails"] = pfails
            lstm_history.append(lstm_r)

        # Basic sanity checks
        self.assertEqual(len(fap_history), NUM_WINDOWS)
        self.assertEqual(len(hpa_history), NUM_WINDOWS)
        self.assertEqual(len(lstm_history), NUM_WINDOWS)

        # All decisions should have valid replicas
        for entry in fap_history:
            self.assertGreaterEqual(entry["target_replicas"], 1)
        for entry in hpa_history:
            self.assertGreaterEqual(entry["target_replicas"], 1)
        for entry in lstm_history:
            self.assertGreaterEqual(entry["target_replicas"], 1)

        # EMA should have mutated the interference matrix
        stats = engine.summary_stats()
        self.assertEqual(stats["windows_processed"], NUM_WINDOWS)

    def test_fap_scale_beats_rr_on_unhealthy_placements(self):
        """
        FAP-Scale should place fewer replicas on high-Pfail nodes than
        round-robin because it actively avoids them.
        Over a simulation with obviously degrading nodes, FAP unhealthy count
        should be ≤ LSTM-RR unhealthy count.
        """
        from results.evaluate import evaluate_experiment_results

        NUM_WINDOWS = 15
        NUM_NODES = 8

        workload_df = generate_workload_time_series(num_windows=NUM_WINDOWS, pattern="periodic", seed=0)
        telemetry_df = generate_node_telemetry(num_nodes=NUM_NODES, num_windows=NUM_WINDOWS,
                                               fail_node_ids=["node-01", "node-02", "node-03"], seed=0)

        scorer = NodeFailureScorer(random_state=0)
        scorer.fit(telemetry_df)
        imat = InterferenceMatrix(alpha=0.1, seed=0)

        engine = FusionEngine(alpha=0.4, beta=0.35, gamma=0.25,
                              replica_capacity=20.0,
                              failure_scorer=scorer,
                              interference_matrix=imat)
        lstm = LSTMRoundRobinScaler()

        fap_hist, lstm_hist = [], []

        for w in range(NUM_WINDOWS):
            win_wl = workload_df[workload_df["window_id"] == w]["request_rate"].values
            actual = float(np.mean(win_wl)) if len(win_wl) > 0 else 50.0

            curr_telemetry = []
            pfails = {}
            for n in range(1, NUM_NODES + 1):
                nid = f"node-{n:02d}"
                t = extract_single_node_telemetry(telemetry_df, nid, w)
                curr_telemetry.append(t)
                pfails[nid] = scorer.predict_pfail(t)

            fr = engine.evaluate_window(list(win_wl), curr_telemetry)
            fr["actual_load"] = actual
            fr["node_pfails"] = pfails
            fap_hist.append(fr)

            lr = lstm.evaluate_window(list(win_wl), curr_telemetry)
            lr["actual_load"] = actual
            lr["node_pfails"] = pfails
            lstm_hist.append(lr)

        fap_metrics = evaluate_experiment_results(fap_hist)
        lstm_metrics = evaluate_experiment_results(lstm_hist)

        # FAP should have fewer or equal unhealthy placements than blind RR
        self.assertLessEqual(
            fap_metrics["unhealthy_placements"],
            lstm_metrics["unhealthy_placements"],
            msg=f"FAP unhealthy={fap_metrics['unhealthy_placements']} "
                f"should be <= LSTM-RR unhealthy={lstm_metrics['unhealthy_placements']}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
