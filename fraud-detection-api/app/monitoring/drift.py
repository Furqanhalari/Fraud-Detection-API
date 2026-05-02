"""
Drift monitoring — Population Stability Index (PSI) computation and snapshot logic.

PSI = Σ (actual% - expected%) * ln(actual% / expected%)
Interpretation:
  PSI < 0.1  → No significant drift
  0.1–0.2    → Moderate drift, worth monitoring
  PSI > 0.2  → Significant drift, model may need retraining
"""

import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts")
TRAIN_DIST_PATH = os.path.join(ARTIFACTS_DIR, "train_distributions.json")
DRIFT_THRESHOLD = 0.2
_EPSILON = 1e-8


# ── PSI core ─────────────────────────────────────────────────────────────────

def compute_psi(
    train_distribution: np.ndarray,
    recent_distribution: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between training and recent distributions.

    Args:
        train_distribution:  1-D array of training values (expected).
        recent_distribution: 1-D array of recent values   (actual).
        bins:                Number of histogram bins.

    Returns:
        PSI score (float). Flag as drift if > 0.2.
    """
    if len(recent_distribution) == 0:
        return 0.0

    # Use training data to determine bin edges
    _, bin_edges = np.histogram(train_distribution, bins=bins)

    # Clamp values to bin range so nothing falls outside
    lo, hi = bin_edges[0], bin_edges[-1]
    train_clipped  = np.clip(train_distribution,  lo, hi)
    recent_clipped = np.clip(recent_distribution, lo, hi)

    train_counts,  _ = np.histogram(train_clipped,  bins=bin_edges)
    recent_counts, _ = np.histogram(recent_clipped, bins=bin_edges)

    # Convert to proportions, guard against zero bins
    train_pct  = (train_counts  + _EPSILON) / (train_counts.sum()  + _EPSILON * bins)
    recent_pct = (recent_counts + _EPSILON) / (recent_counts.sum() + _EPSILON * bins)

    psi = float(np.sum((recent_pct - train_pct) * np.log(recent_pct / train_pct)))
    return round(max(psi, 0.0), 6)


# ── Training distributions ────────────────────────────────────────────────────

def save_train_distributions(X: np.ndarray, feature_names: list[str], bins: int = 10) -> None:
    """
    Persist training feature distributions to train_distributions.json.
    Called once at the end of train.py after feature engineering.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    distributions: dict = {}

    for i, name in enumerate(feature_names):
        values = X[:, i].astype(float)
        counts, edges = np.histogram(values, bins=bins)
        distributions[name] = {
            "mean": round(float(np.mean(values)), 6),
            "std":  round(float(np.std(values)),  6),
            "bin_edges":  [round(float(e), 6) for e in edges],
            "bin_counts": [int(c) for c in counts],
        }

    payload = {
        "features": feature_names,
        "bins": bins,
        "training_samples": int(X.shape[0]),
        "saved_at": datetime.utcnow().isoformat(),
        "distributions": distributions,
    }

    with open(TRAIN_DIST_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[drift] Training distributions saved → {TRAIN_DIST_PATH}")


def load_train_distributions() -> dict:
    if not os.path.exists(TRAIN_DIST_PATH):
        raise FileNotFoundError(
            f"train_distributions.json not found at {TRAIN_DIST_PATH}. "
            "Run train.py first."
        )
    with open(TRAIN_DIST_PATH) as f:
        return json.load(f)


# ── Snapshot runner ───────────────────────────────────────────────────────────

def run_drift_snapshot(model_version_id: str, db: Session) -> dict:
    """
    Compute per-feature PSI between training distributions and the last 7 days
    of scored transactions, then persist one drift_snapshots row per feature.

    Returns:
        {features_flagged: int, worst_feature: str, worst_psi: float}
    """
    from app.models.db_models import DriftSnapshot, Transaction  # avoid circular imports

    # ── Load training distributions ──────────────────────────────────────────
    train_data = load_train_distributions()
    feature_names: list[str] = train_data["features"]
    distributions: dict       = train_data["distributions"]

    # ── Query last 7 days of transactions ────────────────────────────────────
    cutoff = datetime.utcnow() - timedelta(days=7)
    recent_rows = (
        db.query(Transaction)
        .filter(Transaction.created_at >= cutoff)
        .filter(Transaction.raw_features.isnot(None))
        .all()
    )

    # Extract feature vectors from raw_features JSON
    vectors: list[list[float]] = []
    for row in recent_rows:
        fv = (row.raw_features or {}).get("feature_vector")
        if fv and len(fv) == len(feature_names):
            vectors.append(fv)

    today = date.today()
    snapshot_rows = []
    psi_by_feature: dict[str, float] = {}

    for i, feature in enumerate(feature_names):
        dist_info = distributions[feature]

        # Reconstruct training sample from saved histogram
        bin_edges  = np.array(dist_info["bin_edges"])
        bin_counts = np.array(dist_info["bin_counts"])
        bin_mids   = (bin_edges[:-1] + bin_edges[1:]) / 2
        train_vals = np.repeat(bin_mids, bin_counts)

        # Recent values for this feature
        if vectors:
            recent_vals = np.array([v[i] for v in vectors])
        else:
            recent_vals = np.array([])

        psi = compute_psi(train_vals, recent_vals, bins=len(bin_counts))
        flagged = psi > DRIFT_THRESHOLD
        psi_by_feature[feature] = psi

        snapshot_rows.append(
            DriftSnapshot(
                id=str(uuid.uuid4()),
                model_version_id=model_version_id,
                snapshot_date=today,
                feature_name=feature,
                psi_score=psi,
                mean_train=dist_info["mean"],
                mean_recent=round(float(np.mean(recent_vals)), 6) if len(recent_vals) > 0 else None,
                drift_flagged=flagged,
                created_at=datetime.utcnow(),
            )
        )

    db.add_all(snapshot_rows)
    db.commit()

    flagged_features = [f for f, p in psi_by_feature.items() if p > DRIFT_THRESHOLD]
    worst_feature    = max(psi_by_feature, key=psi_by_feature.get)
    worst_psi        = psi_by_feature[worst_feature]

    return {
        "features_checked": len(feature_names),
        "features_flagged": len(flagged_features),
        "flagged_features": flagged_features,
        "worst_feature": worst_feature,
        "worst_psi": round(worst_psi, 6),
        "recent_transactions_used": len(vectors),
        "snapshot_date": today.isoformat(),
    }
