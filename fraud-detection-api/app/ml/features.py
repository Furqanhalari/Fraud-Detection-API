"""
Feature engineering for the Credit Card Fraud Detection dataset.

Input columns: Time, V1–V28, Amount, Class
Output columns: V1–V28, amount_scaled, time_of_day_sin, time_of_day_cos
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

FEATURE_COLS = (
    [f"V{i}" for i in range(1, 29)]
    + ["amount_scaled", "time_of_day_sin", "time_of_day_cos"]
)


def _cyclical_time(time_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert raw Time (seconds) into cyclical sin/cos features (24-hour cycle)."""
    seconds_in_day = 24 * 3600
    angle = (time_series % seconds_in_day) / seconds_in_day * 2 * np.pi
    return np.sin(angle), np.cos(angle)


def fit_scaler(df: pd.DataFrame) -> RobustScaler:
    """Fit a RobustScaler on the Amount column and save it to disk."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    scaler = RobustScaler()
    scaler.fit(df[["Amount"]])
    joblib.dump(scaler, SCALER_PATH)
    print(f"[features] Scaler saved → {SCALER_PATH}")
    return scaler


def load_scaler() -> RobustScaler:
    """Load the fitted RobustScaler from disk."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run train.py first."
        )
    return joblib.load(SCALER_PATH)


def engineer_features(
    df: pd.DataFrame,
    scaler: RobustScaler | None = None,
    fit: bool = False,
) -> pd.DataFrame:
    """
    Apply feature engineering to a DataFrame.

    Args:
        df:     Raw DataFrame with columns Time, V1–V28, Amount (and optionally Class).
        scaler: Pre-fitted RobustScaler. If None and fit=False, one is loaded from disk.
        fit:    If True, fit and save a new scaler from df.

    Returns:
        DataFrame with engineered feature columns only (no Time, Amount, or Class).
    """
    df = df.copy()

    if fit:
        scaler = fit_scaler(df)
    elif scaler is None:
        scaler = load_scaler()

    df["amount_scaled"] = scaler.transform(df[["Amount"]])

    sin_t, cos_t = _cyclical_time(df["Time"])
    df["time_of_day_sin"] = sin_t
    df["time_of_day_cos"] = cos_t

    df.drop(columns=["Time", "Amount"], inplace=True)
    if "Class" in df.columns:
        df.drop(columns=["Class"], inplace=True)

    return df[FEATURE_COLS]


def get_feature_names() -> list[str]:
    return FEATURE_COLS
