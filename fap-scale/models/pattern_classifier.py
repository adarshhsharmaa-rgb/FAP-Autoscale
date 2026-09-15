"""
models/pattern_classifier.py  (Person A)

Workload pattern classifier for FAP-Scale Section IV-A.
Four features per window (as in the Review-1 paper):

    acf_peaks      autocorrelation peak count      -> periodicity
    wavelet_ratio  high-freq / total wavelet energy -> spikiness
    cv             coefficient of variation         -> dispersion
    hurst          Hurst exponent (R/S)             -> long-range memory / trend

Two classifiers are provided:
    classify_rule_based()  explainable thresholds (default; no training needed)
    PatternClassifier      sklearn DecisionTree trained on labelled synthetic windows
"""
from __future__ import annotations

import numpy as np

try:
    from statsmodels.tsa.stattools import acf as _sm_acf
except ImportError:  # pragma: no cover
    _sm_acf = None

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None

try:
    from hurst import compute_Hc
except ImportError:  # pragma: no cover
    compute_Hc = None

FEATURE_NAMES = ["acf_peaks", "wavelet_ratio", "cv", "hurst"]
ACF_PEAK_THRESHOLD = 0.3


# --------------------------------------------------------------------------- #
# Feature 1: autocorrelation peak count
# --------------------------------------------------------------------------- #
def _acf(x, nlags):
    if _sm_acf is not None:
        return _sm_acf(x, nlags=nlags, fft=True)
    x = x - x.mean()
    denom = np.dot(x, x) or 1.0
    return np.array([np.dot(x[: len(x) - k], x[k:]) / denom for k in range(nlags + 1)])


def autocorr_peak_count(window, threshold=ACF_PEAK_THRESHOLD, max_lag=72):
    """Number of local maxima in the ACF (lag >= 2) that exceed `threshold`."""
    x = np.asarray(window, dtype=float)
    nlags = int(min(max_lag, len(x) // 2))
    r = _acf(x, nlags)
    peaks = 0
    for k in range(2, len(r) - 1):
        if r[k] > threshold and r[k] >= r[k - 1] and r[k] >= r[k + 1]:
            peaks += 1
    return peaks


# --------------------------------------------------------------------------- #
# Feature 2: wavelet energy ratio
# --------------------------------------------------------------------------- #
def wavelet_energy_ratio(window, wavelet="db1", level=3):
    """
    Share of signal energy in the two finest detail bands (cD1, cD2).
    Spikes put energy in fine scales; smooth cycles keep it in cA/cD3.
    The mean is removed first so the DC level doesn't dominate.
    """
    x = np.asarray(window, dtype=float)
    x = x - x.mean()
    if pywt is not None:
        coeffs = pywt.wavedec(x, wavelet, level=level)  # [cA3, cD3, cD2, cD1]
        energies = [float(np.sum(c ** 2)) for c in coeffs]
        fine = energies[-1] + energies[-2]
    else:  # Haar fallback: first differences approximate the finest details
        d1 = np.diff(x)[::2] / np.sqrt(2)
        d2 = np.diff(x[::2])[::2] / 2
        fine = float(np.sum(d1 ** 2) + np.sum(d2 ** 2))
        energies = [float(np.sum(x ** 2))]
    total = sum(energies) if pywt is not None else energies[0]
    return fine / total if total > 0 else 0.0


# --------------------------------------------------------------------------- #
# Feature 3: coefficient of variation
# --------------------------------------------------------------------------- #
def coefficient_of_variation(window):
    x = np.asarray(window, dtype=float)
    m = x.mean()
    return float(x.std() / m) if m != 0 else 0.0


# --------------------------------------------------------------------------- #
# Feature 4: Hurst exponent
# --------------------------------------------------------------------------- #
def _hurst_rs(x):
    """Plain rescaled-range estimate (fallback when `hurst` isn't installed)."""
    n = len(x)
    sizes = np.unique(np.floor(np.logspace(np.log10(8), np.log10(n // 2), 8)).astype(int))
    rs = []
    for s in sizes:
        vals = []
        for start in range(0, n - s + 1, s):
            seg = x[start:start + s]
            dev = np.cumsum(seg - seg.mean())
            sd = seg.std()
            if sd > 0:
                vals.append((dev.max() - dev.min()) / sd)
        if vals:
            rs.append(np.mean(vals))
    slope, _ = np.polyfit(np.log(sizes[: len(rs)]), np.log(rs), 1)
    return float(slope)


def hurst_exponent(window):
    x = np.asarray(window, dtype=float)
    if compute_Hc is not None and len(x) >= 100:
        try:
            H, _, _ = compute_Hc(x, kind="change", simplified=True)
            return float(H)
        except Exception:
            pass
    return _hurst_rs(np.diff(x))


# --------------------------------------------------------------------------- #
# Feature vector
# --------------------------------------------------------------------------- #
def extract_features(window) -> dict:
    x = np.asarray(window, dtype=float)
    if len(x) < 32:
        raise ValueError("window needs at least 32 points for stable features")
    return {
        "acf_peaks": autocorr_peak_count(x),
        "wavelet_ratio": wavelet_energy_ratio(x),
        "cv": coefficient_of_variation(x),
        "hurst": hurst_exponent(x),
    }


def features_to_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=float)


# --------------------------------------------------------------------------- #
# Classifier 1: rule-based (explainable, zero training)
# Thresholds were chosen from feature distributions on 300 synthetic windows
# (generate_workload_windows(300, seed=1)); tested on separate seeds.
# --------------------------------------------------------------------------- #
RULES = {
    "bursty_hurst_max": 0.68,      # bursty series have little long-range memory
    "bursty_cv_min": 0.50,         # ...and high dispersion
    "periodic_wavelet_max": 0.30,  # smooth cycles keep energy in coarse bands
    "periodic_min_peaks": 2,       # at least two clear ACF peaks
}


def classify_rule_based(feats: dict) -> str:
    r = RULES
    no_cycle = feats["acf_peaks"] == 0
    if no_cycle and (feats["hurst"] < r["bursty_hurst_max"] or feats["cv"] >= r["bursty_cv_min"]):
        return "bursty"
    if feats["acf_peaks"] >= r["periodic_min_peaks"] and feats["wavelet_ratio"] < r["periodic_wavelet_max"]:
        return "periodic"
    if no_cycle and feats["hurst"] < r["bursty_hurst_max"] + 0.1:
        return "bursty"
    return "hybrid"


# --------------------------------------------------------------------------- #
# Classifier 2: decision tree trained on labelled synthetic windows
# --------------------------------------------------------------------------- #
class PatternClassifier:
    """Thin wrapper around sklearn's DecisionTreeClassifier."""

    def __init__(self, max_depth: int = 4, random_state: int = 0):
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.fitted = False

    def fit(self, windows, labels):
        X = np.vstack([features_to_vector(extract_features(w)) for w in windows])
        self.model.fit(X, list(labels))
        self.fitted = True
        return self

    def predict_features(self, feats: dict) -> str:
        return str(self.model.predict(features_to_vector(feats).reshape(1, -1))[0])

    def predict(self, window) -> str:
        return self.predict_features(extract_features(window))

    def rules_text(self) -> str:
        from sklearn.tree import export_text
        return export_text(self.model, feature_names=FEATURE_NAMES)


_TREE: PatternClassifier | None = None


def get_trained_tree(n_train: int = 240, seed: int = 123) -> PatternClassifier:
    """Train once on synthetic data and cache (so the control loop stays fast)."""
    global _TREE
    if _TREE is None:
        from data_gen.synthetic_data import generate_workload_windows
        data = generate_workload_windows(n_train, seed=seed)
        _TREE = PatternClassifier().fit([w for w, _ in data], [l for _, l in data])
    return _TREE


def classify_pattern(window, method: str = "rules") -> tuple[str, dict]:
    """
    Public entry point.
    method: "rules" (default) or "tree"
    returns (pattern_label, features_dict)
    """
    feats = extract_features(window)
    if method == "tree":
        return get_trained_tree().predict_features(feats), feats
    return classify_rule_based(feats), feats


if __name__ == "__main__":
    from data_gen.synthetic_data import generate_workload_windows
    test = generate_workload_windows(30, seed=999)
    for method in ("rules", "tree"):
        correct = sum(classify_pattern(w, method)[0] == l for w, l in test)
        print(f"{method:5s} accuracy on 30 unseen windows: {correct}/30")
    w, l = test[0]
    lab, f = classify_pattern(w)
    print(f"\nexample: true={l} predicted={lab} features="
          + ", ".join(f"{k}={v:.3f}" for k, v in f.items()))
