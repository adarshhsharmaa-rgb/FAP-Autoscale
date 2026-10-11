"""
Node Health Failure Scorer (Person B implementation).

Computes composite health metrics (CPU Pressure Index - CPI, Thermal Trend Score - TTS)
and predicts node failure probability Pfail(i) using LightGBM (or sklearn HistGradientBoosting fallback).

Two telemetry schemas are supported:

  "smart"   real hardware telemetry from Backblaze Drive Stats Q2 2026
            (traces/backblaze.py): drive temperature history and S.M.A.R.T.
            error counters, labelled with real failures. This is what the
            simulation and the reported results use.
  "legacy"  the eight synthetic signals from data_gen/synthetic_data.py, kept
            so the original unit tests still exercise the model end to end.

`NodeFailureScorer.fit()` picks the schema from the columns it is given.
"""

from typing import Dict, List, Any, Optional
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
    context_switch_rate: Optional[float] = None
) -> float:
    """
    Compute CPU Pressure Index (CPI):
    CPI = 0.5 * U_norm + 0.35 * Q_norm + 0.15 * C_norm

    Args:
        cpu_utilization: CPU % in [0, 100]
        run_queue_length: Number of threads waiting in run queue
        context_switch_rate: Context switches per second. Pass None when the
            trace does not record it (no public cluster trace does): the
            remaining two terms are re-weighted to sum to 1.

    Returns:
        CPI float value bounded in [0.0, 1.0]
    """
    u_norm = min(1.0, max(0.0, cpu_utilization / 100.0))
    q_norm = min(1.0, max(0.0, run_queue_length / 32.0))

    if context_switch_rate is None:
        cpi = (0.5 * u_norm + 0.35 * q_norm) / 0.85
    else:
        c_norm = min(1.0, max(0.0, context_switch_rate / 10000.0))
        cpi = 0.5 * u_norm + 0.35 * q_norm + 0.15 * c_norm
    return float(min(1.0, max(0.0, cpi)))


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

    # Linear regression slope via least-squares polyfit
    slope = np.polyfit(x, temps, 1)[0]

    # Normalize by TDP
    tts = slope / max(tdp, 1.0)
    return float(tts)


# ─────────────────────────────────────────────────────────────────────────────
# Real hardware telemetry: S.M.A.R.T. features (Backblaze Drive Stats)
# ─────────────────────────────────────────────────────────────────────────────

# Cumulative error counters. The first five are the attributes Backblaze reports
# as predictive of drive failure; crc_errors counts interface/link errors.
SMART_COUNTERS = [
    "realloc_sectors",         # SMART 5
    "pending_sectors",         # SMART 197
    "offline_uncorrectable",   # SMART 198
    "reported_uncorrectable",  # SMART 187
    "command_timeouts",        # SMART 188
    "crc_errors",              # SMART 199
]
THERMAL_LIMIT_C = 60.0   # upper operating temperature of an enterprise drive
TTS_READINGS = 10        # TTS looks at the last 10 temperature readings
TREND_DAYS = 7           # counter growth is measured over the last 7 days
FAILURE_HORIZON_DAYS = 7  # Pfail = P(node fails within the next 7 days)

SMART_FEATURES = (
    ["temperature_c", "tts", "power_on_years", "disk_health"]
    + [f"log_{c}" for c in SMART_COUNTERS]
    + [f"{c}_delta" for c in SMART_COUNTERS]
)

LEGACY_FEATURES = [
    "cpu_utilization", "run_queue_length", "context_switch_rate", "cpi", "tts",
    "dimm_errors", "disk_health", "net_retransmits",
]


def disk_health_score(realloc, pending, offline_uncorrectable, reported_uncorrectable):
    """
    Disk health in [0, 1] (1 = no recorded media errors).

    Each counter contributes on a log scale, so the first few bad sectors cost
    much more than the thousandth. Pending and uncorrectable sectors weigh
    more than reallocated ones, which the drive has already repaired.
    """
    damage = (0.15 * np.log1p(np.nan_to_num(realloc, nan=0.0))
              + 0.30 * np.log1p(np.nan_to_num(pending, nan=0.0))
              + 0.30 * np.log1p(np.nan_to_num(offline_uncorrectable, nan=0.0))
              + 0.25 * np.log1p(np.nan_to_num(reported_uncorrectable, nan=0.0)))
    return np.exp(-damage)


def _slope_weights(n: int) -> np.ndarray:
    """Weights w with dot(w, y) == least-squares slope of y over 0..n-1."""
    x = np.arange(n, dtype=float)
    x -= x.mean()
    return x / np.dot(x, x)


_SLOPE_W = {n: _slope_weights(n) for n in range(2, TTS_READINGS + 1)}


def _rolling_slope(values: np.ndarray) -> float:
    n = len(values)
    return float(np.dot(_SLOPE_W[n], values)) if n >= 2 else 0.0


def smart_feature_frame(drive_days: pd.DataFrame) -> pd.DataFrame:
    """
    Add the SMART_FEATURES columns to a drive-day table (one row per drive per
    day, as returned by traces.backblaze.load_drive_days). Every feature on a
    given day uses only that day and earlier days of the same drive.
    """
    df = drive_days.sort_values(["drive_id", "date"]).reset_index(drop=True)
    by_drive = df.groupby("drive_id", sort=False)

    temp = df["temperature_c"].astype(float)
    df["temperature_c"] = temp
    # gaps in the temperature log are bridged with the neighbouring reading for the
    # trend only; the temperature feature itself stays missing where it was missing
    filled = by_drive["temperature_c"].transform(lambda s: s.ffill().bfill()).fillna(THERMAL_LIMIT_C / 2)
    df["temperature_filled"] = filled
    slope = (filled.groupby(df["drive_id"], sort=False)
             .rolling(TTS_READINGS, min_periods=1).apply(_rolling_slope, raw=True)
             .reset_index(level=0, drop=True))
    df["tts"] = slope / THERMAL_LIMIT_C
    df["power_on_years"] = df["power_on_hours"].astype(float) / 8766.0

    for c in SMART_COUNTERS:
        raw = df[c].astype(float)
        df[f"log_{c}"] = np.log1p(raw)
        # counters only grow, so the window minimum is the value TREND_DAYS ago
        floor = (raw.groupby(df["drive_id"], sort=False)
                 .rolling(TREND_DAYS + 1, min_periods=1).min().reset_index(level=0, drop=True))
        df[f"{c}_delta"] = (raw - floor).clip(lower=0)

    df["disk_health"] = disk_health_score(
        df["realloc_sectors"].astype(float), df["pending_sectors"].astype(float),
        df["offline_uncorrectable"].astype(float), df["reported_uncorrectable"].astype(float))
    return df


def failure_label(drive_days: pd.DataFrame, horizon: int = FAILURE_HORIZON_DAYS) -> np.ndarray:
    """1 where the drive fails within the next `horizon` days (not counting today)."""
    d = drive_days["days_to_failure"]
    return ((d >= 1) & (d <= horizon)).to_numpy(dtype=int)


class NodeFailureScorer:
    """
    LightGBM-backed Node Failure Scorer.
    Extracts CPI, TTS, and raw signals to output node failure probability Pfail in [0, 1].
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.is_fitted = False
        self.schema = "legacy"
        self.feature_names = list(LEGACY_FEATURES)
        self._single_class = None
        self._calibrator = None

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

    # ── feature extraction ────────────────────────────────────────────────

    def extract_features(self, telemetry: Dict[str, Any]) -> List[float]:
        """
        Extract feature vector from telemetry dictionary.
        """
        if self.schema == "smart":
            return self._extract_smart(telemetry)

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

    @staticmethod
    def _extract_smart(telemetry: Dict[str, Any]) -> List[float]:
        """Feature vector for one node from its hardware telemetry dict."""
        def num(key):
            v = telemetry.get(key)
            return float("nan") if v is None or pd.isna(v) else float(v)

        temps = [t for t in telemetry.get("temperature_history", []) if t is not None and not pd.isna(t)]
        if "temperature_c" in telemetry:
            temperature = num("temperature_c")
        else:
            temperature = temps[-1] if temps else float("nan")
        tts = calculate_tts(temps, THERMAL_LIMIT_C)
        counters = [num(c) for c in SMART_COUNTERS]
        health = float(disk_health_score(*[np.float64(v) for v in counters[:4]]))
        return ([temperature, tts, num("power_on_hours") / 8766.0, health]
                + [float(np.log1p(v)) for v in counters]
                + [num(f"{c}_delta") for c in SMART_COUNTERS])

    # ── training ──────────────────────────────────────────────────────────

    def fit(self, telemetry_df: pd.DataFrame, calibration_df: Optional[pd.DataFrame] = None
            ) -> "NodeFailureScorer":
        """
        Train the LightGBM classifier.

        A table with S.M.A.R.T. columns (traces.backblaze.load_drive_days) is
        treated as real hardware telemetry: features come from
        smart_feature_frame() and the label is "fails within the next 7 days".
        Pass a held-out `calibration_df` to map scores onto observed failure
        frequencies (Platt scaling). Any other table is the legacy
        synthetic telemetry with a `will_fail` column.
        """
        if "realloc_sectors" in telemetry_df.columns:
            return self._fit_smart(telemetry_df, calibration_df)

        self.schema = "legacy"
        self.feature_names = list(LEGACY_FEATURES)
        X = []
        y = []

        for _, row in telemetry_df.iterrows():
            telem_dict = row.to_dict()
            feat_vec = self.extract_features(telem_dict)
            X.append(feat_vec)
            y.append(int(row["will_fail"]))

        X_mat = np.array(X)
        y_vec = np.array(y)

        # Guard against single-class data (model can't learn a boundary)
        unique_classes = np.unique(y_vec)
        if len(unique_classes) < 2:
            self._single_class = int(unique_classes[0])
            self.is_fitted = True
            return self

        self._single_class = None
        self.model.fit(X_mat, y_vec)
        self.is_fitted = True
        return self

    def _fit_smart(self, drive_days: pd.DataFrame, calibration_df: Optional[pd.DataFrame]
                   ) -> "NodeFailureScorer":
        self.schema = "smart"
        self.feature_names = list(SMART_FEATURES)
        if HAS_LIGHTGBM:
            # more capacity than the 50-tree synthetic model: 16 features, ~10^5 rows
            self.model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=5,
                min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=5.0, random_state=self.random_state, verbosity=-1)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.03, max_depth=5, random_state=self.random_state)

        train = self._training_rows(drive_days)
        y = failure_label(train)
        if len(np.unique(y)) < 2:
            self._single_class = int(y[0])
            self.is_fitted = True
            return self
        self._single_class = None
        self.model.fit(train[SMART_FEATURES].to_numpy(dtype=float), y)
        self.is_fitted = True

        self._calibrator = None
        if calibration_df is not None and len(calibration_df):
            # Platt scaling: a logistic fit on the model's log-odds. Strictly
            # monotone, so it never changes how nodes rank against each other.
            from sklearn.linear_model import LogisticRegression
            cal = self._training_rows(calibration_df)
            raw = self._raw_scores(cal[SMART_FEATURES].to_numpy(dtype=float))
            self._calibrator = LogisticRegression(C=1e6).fit(self._logit(raw), failure_label(cal))
        return self

    @staticmethod
    def _logit(scores: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(s / (1.0 - s)).reshape(-1, 1)

    def _calibrated(self, scores: np.ndarray) -> np.ndarray:
        if self._calibrator is None:
            return scores
        return self._calibrator.predict_proba(self._logit(scores))[:, 1]

    @staticmethod
    def _training_rows(drive_days: pd.DataFrame) -> pd.DataFrame:
        df = drive_days if "tts" in drive_days.columns else smart_feature_frame(drive_days)
        # the failure day itself is the outcome, not something to predict from
        return df[df["days_to_failure"] != 0]

    # ── prediction ────────────────────────────────────────────────────────

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        return probs[:, classes.index(1)] if 1 in classes else np.zeros(len(X))

    def predict_frame(self, drive_days: pd.DataFrame) -> np.ndarray:
        """Pfail for every row of a drive-day table (vectorised; smart schema)."""
        df = drive_days if "tts" in drive_days.columns else smart_feature_frame(drive_days)
        if self._single_class is not None:
            return np.full(len(df), 0.99 if self._single_class == 1 else 0.01)
        scores = self._calibrated(self._raw_scores(df[SMART_FEATURES].to_numpy(dtype=float)))
        return np.clip(scores, 0.001, 0.999)

    def predict_many(self, telemetry_list: List[Dict[str, Any]]) -> np.ndarray:
        """Pfail for many telemetry dicts in one model call (same values as predict_pfail)."""
        if (self.schema != "smart" or not self.is_fitted or self._single_class is not None
                or not telemetry_list):
            return np.array([self.predict_pfail(t) for t in telemetry_list], dtype=float)
        X = np.array([self._extract_smart(t) for t in telemetry_list], dtype=float)
        return np.clip(self._calibrated(self._raw_scores(X)), 0.001, 0.999)

    def predict_pfail(self, telemetry: Dict[str, Any]) -> float:
        """
        Predict node failure probability Pfail in range [0.0, 1.0].
        If model is not yet fitted, uses CPI and TTS heuristic for graceful cold-start.
        """
        if self.schema == "smart" or (not self.is_fitted and "realloc_sectors" in telemetry):
            return self._predict_smart(telemetry)

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

        # Handle single-class training data (no model was fitted)
        if getattr(self, "_single_class", None) is not None:
            return 0.99 if self._single_class == 1 else 0.01

        feat_vec = np.array([self.extract_features(telemetry)])
        probs = self.model.predict_proba(feat_vec)[0]
        # Map to failure class (1) using model.classes_ for correctness
        classes = list(self.model.classes_)
        if 1 in classes:
            pfail = probs[classes.index(1)]
        else:
            pfail = 0.0
        return float(np.clip(pfail, 0.001, 0.999))

    def _predict_smart(self, telemetry: Dict[str, Any]) -> float:
        feats = self._extract_smart(telemetry)
        if not self.is_fitted:
            # cold start: lost disk health plus a rising temperature
            heuristic = 0.8 * (1.0 - feats[3]) + 0.2 * min(1.0, max(0.0, feats[1] * 20.0))
            return float(np.clip(heuristic, 0.01, 0.99))
        if self._single_class is not None:
            return 0.99 if self._single_class == 1 else 0.01
        score = self._calibrated(self._raw_scores(np.array([feats], dtype=float)))
        return float(np.clip(score[0], 0.001, 0.999))


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
