import uuid
from datetime import datetime, date

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.sqlite import CHAR
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(CHAR(36), primary_key=True, default=_uuid)
    transaction_id = Column(String, unique=True, nullable=False)
    amount = Column(Float, nullable=False)
    merchant_category = Column(String)
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    user_id = Column(String, index=True)
    ip_address = Column(String)
    device_fingerprint = Column(String)
    raw_features = Column(JSON)
    fraud_score = Column(Float)
    is_fraud_predicted = Column(Boolean)
    is_fraud_actual = Column(Boolean, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_transactions_user_id", "user_id"),
        Index("ix_transactions_created_at", "created_at"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(CHAR(36), primary_key=True, default=_uuid)
    version_tag = Column(String, unique=True)
    trained_at = Column(DateTime)
    training_rows = Column(Integer)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    threshold_used = Column(Float)
    is_active = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    drift_snapshots = relationship("DriftSnapshot", back_populates="model_version")
    backtest_runs = relationship("BacktestRun", back_populates="model_version")


class DriftSnapshot(Base):
    __tablename__ = "drift_snapshots"

    id = Column(CHAR(36), primary_key=True, default=_uuid)
    model_version_id = Column(CHAR(36), ForeignKey("model_versions.id"), nullable=False)
    snapshot_date = Column(Date, index=True)
    feature_name = Column(String)
    psi_score = Column(Float)
    mean_train = Column(Float)
    mean_recent = Column(Float)
    drift_flagged = Column(Boolean)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    model_version = relationship("ModelVersion", back_populates="drift_snapshots")

    __table_args__ = (
        Index("ix_drift_snapshots_snapshot_date", "snapshot_date"),
        Index("ix_drift_snapshots_model_version_id", "model_version_id"),
    )


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(CHAR(36), primary_key=True, default=_uuid)
    model_version_id = Column(CHAR(36), ForeignKey("model_versions.id"), nullable=False)
    threshold = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)
    false_positive_rate = Column(Float)
    true_positive_rate = Column(Float)
    dataset_label = Column(String)
    run_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    model_version = relationship("ModelVersion", back_populates="backtest_runs")
