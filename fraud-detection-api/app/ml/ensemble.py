"""
Ensemble combiner: XGBoost + Isolation Forest.

Final fraud score = 0.7 * xgb_probability + 0.3 * iso_binary_score
"""

import os
from typing import Any

import joblib
import numpy as np

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
XGB_PATH = os.path.join(ARTIFACTS_DIR, "xgboost_model.pkl")
ISO_PATH = os.path.join(ARTIFACTS_DIR, "isolation_forest.pkl")

XGB_WEIGHT = 0.7
ISO_WEIGHT = 0.3


class FraudEnsemble:
    def __init__(self) -> None:
        self.xgb_model: Any = None
        self.iso_model: Any = None

    def load_models(self) -> None:
        """Load both saved .pkl files from the artifacts directory."""
        if not os.path.exists(XGB_PATH):
            raise FileNotFoundError(f"XGBoost model not found at {XGB_PATH}. Run train.py first.")
        if not os.path.exists(ISO_PATH):
            raise FileNotFoundError(f"Isolation Forest not found at {ISO_PATH}. Run train.py first.")
        self.xgb_model = joblib.load(XGB_PATH)
        self.iso_model = joblib.load(ISO_PATH)
        print("[ensemble] Models loaded successfully.")

    def predict(self, feature_vector: np.ndarray) -> dict:
        """
        Run ensemble inference on a single feature vector or batch.

        Args:
            feature_vector: 1-D array (single sample) or 2-D array (batch).

        Returns:
            dict with keys:
              fraud_score  – combined ensemble score (0–1)
              is_fraud     – bool, True if fraud_score >= threshold (0.5)
              xgb_prob     – raw XGBoost fraud probability
              iso_score    – Isolation Forest anomaly flag (0 = normal, 1 = anomaly)
        """
        if self.xgb_model is None or self.iso_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        X = np.atleast_2d(feature_vector)

        xgb_prob: float = float(self.xgb_model.predict_proba(X)[0, 1])

        iso_raw: int = int(self.iso_model.predict(X)[0])
        iso_score: float = 0.0 if iso_raw == 1 else 1.0

        fraud_score: float = XGB_WEIGHT * xgb_prob + ISO_WEIGHT * iso_score

        threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

        return {
            "fraud_score": round(fraud_score, 6),
            "is_fraud": fraud_score >= threshold,
            "xgb_prob": round(xgb_prob, 6),
            "iso_score": iso_score,
        }


_ensemble: FraudEnsemble | None = None


def get_ensemble() -> FraudEnsemble:
    """Singleton accessor — loads models once on first call."""
    global _ensemble
    if _ensemble is None:
        _ensemble = FraudEnsemble()
        _ensemble.load_models()
    return _ensemble


def load_models() -> FraudEnsemble:
    """Convenience wrapper: create and load an ensemble instance."""
    e = FraudEnsemble()
    e.load_models()
    return e


def predict(feature_vector: np.ndarray) -> dict:
    """Module-level predict using the singleton ensemble."""
    return get_ensemble().predict(feature_vector)
