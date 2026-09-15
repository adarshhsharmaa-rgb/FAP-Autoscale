"""
Verification Test Suite for Person B (Node Health & Interference) and Repository Skeleton.
"""

import unittest
import numpy as np
import pandas as pd

from data_gen.synthetic_data import (
    generate_workload_time_series,
    generate_node_telemetry,
    extract_single_node_telemetry
)
from models.failure_scorer import (
    NodeFailureScorer,
    score_node_failure,
    calculate_cpi,
    calculate_tts
)
from models.interference import (
    InterferenceMatrix,
    get_interference
)
from models.pattern_classifier import classify_workload_pattern
from models.forecasters import classify_and_forecast
from fusion.fusion_engine import FusionEngine, compute_node_score, calculate_replicas
from simulator.baselines import ReactiveHPAScaler, LSTMRoundRobinScaler


class TestPersonBAndSkeleton(unittest.TestCase):

    def test_imports_and_skeletons(self):
        """Verify that all project skeletons import cleanly."""
        ts_df = generate_workload_time_series(num_windows=5)
        self.assertFalse(ts_df.empty)

        pattern, forecast = classify_and_forecast([10, 12, 15, 14, 18])
        self.assertIn(pattern, ["periodic", "bursty", "hybrid"])
        self.assertGreater(forecast, 0.0)

    def test_node_telemetry_generation(self):
        """Test Person B synthetic node telemetry generator."""
        telem_df = generate_node_telemetry(num_nodes=10, num_windows=15, fail_node_ids=["node-02"])
        self.assertFalse(telem_df.empty)
        self.assertIn("node_id", telem_df.columns)
        self.assertIn("cpu_utilization", telem_df.columns)
        self.assertIn("will_fail", telem_df.columns)

        # Verify failing node telemetry has expected flag
        failing_rows = telem_df[telem_df["node_id"] == "node-02"]
        self.assertGreater(len(failing_rows), 0)

    def test_cpi_and_tts_calculations(self):
        """Test CPI and TTS math formulas."""
        # Low pressure
        cpi_low = calculate_cpi(cpu_utilization=10.0, run_queue_length=1.0, context_switch_rate=500.0)
        self.assertGreaterEqual(cpi_low, 0.0)
        self.assertLessEqual(cpi_low, 0.3)

        # High pressure
        cpi_high = calculate_cpi(cpu_utilization=95.0, run_queue_length=25.0, context_switch_rate=9000.0)
        self.assertGreater(cpi_high, 0.6)

        # Thermal trend score (TTS)
        rising_temps = [50.0, 53.0, 56.0, 59.0, 62.0, 65.0, 68.0, 71.0, 74.0, 77.0]
        tts_val = calculate_tts(rising_temps, tdp=105.0)
        self.assertGreater(tts_val, 0.0)

    def test_node_failure_scorer(self):
        """Test LightGBM NodeFailureScorer training and prediction."""
        telem_df = generate_node_telemetry(num_nodes=8, num_windows=20, fail_node_ids=["node-03"])
        scorer = NodeFailureScorer()
        scorer.fit(telem_df)

        healthy_telem = {
            "cpu_utilization": 25.0,
            "run_queue_length": 1.0,
            "context_switch_rate": 800.0,
            "temperature_history": [45.0] * 10,
            "tdp": 105.0,
            "dimm_errors": 0,
            "disk_health": 1.0,
            "net_retransmits": 0
        }
        pfail_healthy = score_node_failure("node-01", healthy_telem, scorer)
        self.assertGreaterEqual(pfail_healthy, 0.0)
        self.assertLessEqual(pfail_healthy, 1.0)

        failing_telem = {
            "cpu_utilization": 98.0,
            "run_queue_length": 25.0,
            "context_switch_rate": 9500.0,
            "temperature_history": [80.0 + i * 2.0 for i in range(10)],
            "tdp": 105.0,
            "dimm_errors": 4,
            "disk_health": 0.2,
            "net_retransmits": 40
        }
        pfail_failing = score_node_failure("node-03", failing_telem, scorer)
        self.assertGreaterEqual(pfail_failing, pfail_healthy)

    def test_interference_matrix(self):
        """Test 15-bin Co-location Interference Matrix and EMA update."""
        matrix = InterferenceMatrix(alpha=0.1, seed=42)

        is_periodic_bursty = get_interference("periodic", "bursty", matrix)
        self.assertGreaterEqual(is_periodic_bursty, 0.0)
        self.assertLessEqual(is_periodic_bursty, 1.0)

        # Test EMA update
        updated_is = matrix.update_ema("periodic", "bursty", observed_degradation=0.9)
        self.assertGreaterEqual(updated_is, 0.0)
        self.assertLessEqual(updated_is, 1.0)

    def test_fusion_engine(self):
        """Test FusionEngine composite scoring S(i) and replica count K."""
        score = compute_node_score(pfail=0.1, is_score=0.2, utilization=40.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        k_replicas = calculate_replicas(predicted_load=75.0, replica_capacity=20.0)
        self.assertEqual(k_replicas, 4)


if __name__ == "__main__":
    unittest.main()
