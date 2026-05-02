"""
Training pipeline for the Fraud Detection models.

Usage:
    python -m app.ml.train                 (from fraud-detection-api/)
    python fraud-detection-api/app/ml/train.py

Expects /data/creditcard.csv with columns:
    Time, V1–V28, Amount, Class  (0=legit, 1=fraud)
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
# Default: fraud-detection-api/data/creditcard.csv  (override with CREDITCARD_CSV env var)
_DEFAULT_DATA = os.path.join(HERE, "..", "..", "data", "creditcard.csv")
DATA_PATH = os.environ.get("CREDITCARD_CSV", _DEFAULT_DATA)
XGB_PATH = os.path.join(ARTIFACTS_DIR, "xgboost_model.pkl")
ISO_PATH = os.path.join(ARTIFACTS_DIR, "isolation_forest.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

sys.path.insert(0, os.path.join(HERE, "..", ".."))
from app.ml.features import engineer_features, get_feature_names


def load_data(path: str) -> pd.DataFrame:
    print(f"[train] Loading dataset from {path} …")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it at /data/creditcard.csv"
        )
    df = pd.read_csv(path)
    print(f"[train] Loaded {len(df):,} rows — fraud rate: {df['Class'].mean():.4%}")
    return df


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[XGBClassifier, dict]:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos
    print(f"[train] XGBoost — scale_pos_weight={scale_pos_weight:.2f} (neg={neg:,}, pos={pos:,})")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    t0 = time.time()
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elapsed = time.time() - t0
    print(f"[train] XGBoost trained in {elapsed:.1f}s")

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("\n[train] === XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"  AUC-ROC : {auc:.4f}")

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "fraud_train": int(y_train.sum()),
        "fraud_test": int(y_test.sum()),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "training_time_seconds": round(elapsed, 2),
    }

    return xgb, metrics


def train_isolation_forest(X_full: np.ndarray) -> IsolationForest:
    contamination = float(os.environ.get("ANOMALY_CONTAMINATION", "0.01"))
    print(f"\n[train] Isolation Forest — contamination={contamination}, samples={len(X_full):,}")

    t0 = time.time()
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        n_estimators=100,
    )
    iso.fit(X_full)
    elapsed = time.time() - t0
    print(f"[train] Isolation Forest trained in {elapsed:.1f}s")
    return iso


def save_metrics(metrics: dict) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[train] Metrics saved → {METRICS_PATH}")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    labels = df["Class"].values

    print("\n[train] Engineering features …")
    X_engineered = engineer_features(df, fit=True)
    feature_names = get_feature_names()
    X = X_engineered.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"[train] Split — train: {len(X_train):,}  test: {len(X_test):,}")

    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)

    iso_model = train_isolation_forest(X)

    print("\n[train] Saving model artifacts …")
    joblib.dump(xgb_model, XGB_PATH)
    print(f"  XGBoost        → {XGB_PATH}")
    joblib.dump(iso_model, ISO_PATH)
    print(f"  Isolation Forest → {ISO_PATH}")

    metrics = {
        "model": "xgboost_v1",
        "feature_names": feature_names,
        "xgboost": xgb_metrics,
        "isolation_forest": {
            "contamination": float(os.environ.get("ANOMALY_CONTAMINATION", "0.01")),
            "training_samples": len(X),
        },
    }
    save_metrics(metrics)

    print("\n[train] ✓ Training complete.\n")
    print(f"  Artifacts directory : {ARTIFACTS_DIR}")
    print(f"  xgboost_model.pkl   : {os.path.getsize(XGB_PATH) / 1024:.1f} KB")
    print(f"  isolation_forest.pkl: {os.path.getsize(ISO_PATH) / 1024:.1f} KB")
    print(f"  metrics.json        : {os.path.getsize(METRICS_PATH)} bytes")


if __name__ == "__main__":
    main()
