"""
Node Health Failure Scorer (Person B implementation).

Computes composite health metrics (CPU Pressure Index - CPI, Thermal Trend Score - TTS)
and predicts node failure probability Pfail(i) using LightGBM (or sklearn HistGradientBoosting fallback).
"""

from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd

# LightGBM import with graceful fallback
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LIGHTGBM = False


def calculate_cpi(
    cpu_utilization: float,
    run_queue_length: float,
    context_switch_rate: float
) -> float:
    """
    Compute CPU Pressure Index (CPI):
    CPI = 0.5 * U_norm + 0.35 * Q_norm + 0.15 * C_norm

    Args:
        cpu_utilization: CPU % in [0, 100]
        run_queue_length: Number of threads waiting in run queue
        context_switch_rate: Context switches per second

    Returns:
        CPI float value bounded in [0.0, 1.0]
    """
    u_norm = np.clip(cpu_utilization / 100.0, 0.0, 1.0)
    q_norm = np.clip(run_queue_length / 32.0, 0.0, 1.0)
    c_norm = np.clip(context_switch_rate / 10000.0, 0.0, 1.0)

    cpi = 0.5 * u_norm + 0.35 * q_norm + 0.15 * c_norm
    return float(np.clip(cpi, 0.0, 1.0))


def calculate_tts(
    temperature_history: List[float],
    tdp: float = 105.0
) -> float:
    """
    Compute Thermal Trend Score (TTS):
    TTS = slope of last 10 temperature readings / TDP

    Args:
        temperature_history: List of last N temperature readings in °C.
        tdp: Thermal Design Power limit in °C (default 105.0).

    Returns:
        TTS float value representing rate of thermal increase per step normalized by TDP.
    """
    if len(temperature_history) < 2:
        return 0.0

    temps = np.array(temperature_history[-10:], dtype=float)
    x = np.arange(len(temps))

    # Linear regression slope = cov(x, y) / var(x)
    if np.var(x) == 0:
        slope = 0.0
    else:
        slope = np.cov(x, temps)[0, 1] / np.var(x)

    # Normalize by TDP
    tts = slope / max(tdp, 1.0)
    return float(tts)


class NodeFailureScorer:
    """
    LightGBM-backed Node Failure Scorer.
    Extracts CPI, TTS, and raw signals to output node failure probability Pfail in [0, 1].
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names = [
            "cpu_utilization",
            "run_queue_length",
            "context_switch_rate",
            "cpi",
            "tts",
            "dimm_errors",
            "disk_health",
            "net_retransmits"
        ]

        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state,
                verbosity=-1
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=50,
                max_depth=4,
                random_state=self.random_state
            )

    def extract_features(self, telemetry: Dict[str, Any]) -> List[float]:
        """
        Extract feature vector from telemetry dictionary.
        """
        cpu_util = float(telemetry.get("cpu_utilization", 30.0))
        run_q = float(telemetry.get("run_queue_length", 2.0))
        cs_rate = float(telemetry.get("context_switch_rate", 1000.0))
        temp_hist = telemetry.get("temperature_history", [50.0] * 10)
        tdp = float(telemetry.get("tdp", 105.0))
        dimm_errs = float(telemetry.get("dimm_errors", 0))
        disk_h = float(telemetry.get("disk_health", 1.0))
        net_ret = float(telemetry.get("net_retransmits", 0))

        cpi = calculate_cpi(cpu_util, run_q, cs_rate)
        tts = calculate_tts(temp_hist, tdp)

        return [cpu_util, run_q, cs_rate, cpi, tts, dimm_errs, disk_h, net_ret]

    def fit(self, telemetry_df: pd.DataFrame) -> "NodeFailureScorer":
        """
        Train the LightGBM classifier on synthetic node telemetry DataFrame.
        """
        X = []
        y = []

        for _, row in telemetry_df.iterrows():
            telem_dict = row.to_dict()
            feat_vec = self.extract_features(telem_dict)
            X.append(feat_vec)
            y.append(int(row["will_fail"]))

        X_mat = np.array(X)
        y_vec = np.array(y)

        self.model.fit(X_mat, y_vec)
        self.is_fitted = True
        return self

    def predict_pfail(self, telemetry: Dict[str, Any]) -> float:
        """
        Predict node failure probability Pfail in range [0.0, 1.0].
        If model is not yet fitted, uses CPI and TTS heuristic for graceful cold-start.
        """
        cpi = calculate_cpi(
            telemetry.get("cpu_utilization", 30.0),
            telemetry.get("run_queue_length", 2.0),
            telemetry.get("context_switch_rate", 1000.0)
        )
        tts = calculate_tts(
            telemetry.get("temperature_history", [50.0] * 10),
            telemetry.get("tdp", 105.0)
        )
        dimm_errs = telemetry.get("dimm_errors", 0)
        disk_h = telemetry.get("disk_health", 1.0)

        if not self.is_fitted:
            # Heuristic fallback before model training
            heuristic_score = 0.4 * cpi + 0.3 * min(1.0, max(0.0, tts * 5.0)) + 0.2 * (dimm_errs / 5.0) + 0.1 * (1.0 - disk_h)
            return float(np.clip(heuristic_score, 0.01, 0.99))

        feat_vec = np.array([self.extract_features(telemetry)])
        probs = self.model.predict_proba(feat_vec)[0]
        # Return probability of class 1 (failure)
        pfail = probs[1] if len(probs) > 1 else probs[0]
        return float(np.clip(pfail, 0.001, 0.999))


# Global default scorer instance for quick convenience function calls
_GLOBAL_FAILURE_SCORER = NodeFailureScorer()


def score_node_failure(node_id: str, telemetry: Dict[str, Any], scorer: Optional[NodeFailureScorer] = None) -> float:
    """
    Deliverable function for Person B:
    score_node_failure(node_id, telemetry) -> Pfail

    Args:
        node_id: Unique string identifier for node.
        telemetry: Dictionary containing node hardware telemetry metrics.
        scorer: Optional fitted NodeFailureScorer instance.

    Returns:
        Pfail: Failure probability float in range [0.0, 1.0].
    """
    active_scorer = scorer if scorer is not None else _GLOBAL_FAILURE_SCORER
    return active_scorer.predict_pfail(telemetry)
