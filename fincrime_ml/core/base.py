"""
core/base.py
============
Abstract base classes for all FinCrime-ML pipelines and scorers.

All fraud and AML pipeline implementations inherit from BasePipeline.
All scoring modules inherit from BaseScorer.

Design rationale: enforcing a common interface across fraud and AML modules
ensures consistent behaviour for model training, prediction, explanation,
and audit logging — a requirement under FCA SYSC 6.3 and SR 11-7 model
risk management guidance.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration shared across all pipeline implementations.

    Attributes:
        random_state: Seed for reproducibility. Fixed seed is required for
            regulatory model validation (SR 11-7 §4).
        test_size: Proportion of data reserved for holdout evaluation.
        primary_metric: Metric used for model selection. AUC-PR is preferred
            over ROC-AUC for imbalanced financial crime datasets.
        audit_log_enabled: Whether to write a decision audit trail. Should be
            True in any production or regulatory context.
        version: Model version string, included in all audit log entries.
    """

    random_state: int = 42
    test_size: float = 0.20
    primary_metric: str = "average_precision"  # AUC-PR
    audit_log_enabled: bool = True
    version: str = "0.1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class BasePipeline(abc.ABC):
    """Abstract base class for all FinCrime-ML domain pipelines.

    Subclasses must implement:
        - prepare_features(df): Feature engineering specific to the domain.
        - train(df): Train the model on a labelled dataset.
        - predict(df): Return a DataFrame with risk scores and metadata.
        - explain(df): Return SHAP-based explanations for predictions.

    All implementations should call self._log_audit() at prediction time
    to maintain a traceable decision record.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.model: Any = None
        self.feature_names: list[str] = []
        self._is_fitted: bool = False
        self._audit_log: list[dict] = []
        logger.info(
            "Initialised %s pipeline v%s",
            self.__class__.__name__,
            self.config.version,
        )

    @abc.abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw transaction data.

        Args:
            df: Raw transaction DataFrame conforming to FinCrime-ML schema.

        Returns:
            DataFrame with engineered features ready for model input.
        """

    @abc.abstractmethod
    def train(self, df: pd.DataFrame, label_col: str = "is_fraud") -> "BasePipeline":
        """Fit the pipeline on labelled training data.

        Args:
            df: Labelled transaction DataFrame.
            label_col: Name of the binary target column.

        Returns:
            Self (for method chaining).
        """

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score transactions and return risk assessments.

        Args:
            df: Unlabelled transaction DataFrame.

        Returns:
            DataFrame with columns: transaction_id, risk_score, risk_tier,
            model_version, scored_at.
        """

    @abc.abstractmethod
    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate SHAP-based explanations for scored transactions.

        Args:
            df: Transaction DataFrame (should be the same data passed to predict).

        Returns:
            DataFrame with SHAP values and top-N reason codes per transaction.
        """

    def _check_fitted(self) -> None:
        """Raise if the pipeline has not been trained."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. Call .train() first."
            )

    def _log_audit(self, event: str, metadata: dict | None = None) -> None:
        """Append an entry to the in-memory audit log.

        In production, this should be persisted to a database or append-only
        log store. The audit trail satisfies FCA SYSC 10A record-keeping
        requirements for automated decision systems.

        Args:
            event: Human-readable event description.
            metadata: Optional additional context (model version, record count, etc.).
        """
        if not self.config.audit_log_enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline": self.__class__.__name__,
            "version": self.config.version,
            "event": event,
            **(metadata or {}),
        }
        self._audit_log.append(entry)
        logger.debug("Audit: %s", entry)

    @property
    def audit_log(self) -> list[dict]:
        """Return the full audit log for this pipeline instance."""
        return list(self._audit_log)


class BaseScorer(abc.ABC):
    """Abstract base class for risk scorers.

    Scorers combine one or more pipeline outputs into a final risk score.
    The unified FinCrime risk scorer (core/scorer.py) inherits from this.
    """

    @abc.abstractmethod
    def score(self, fraud_output: pd.DataFrame, aml_output: pd.DataFrame) -> pd.DataFrame:
        """Combine fraud and AML signals into a unified risk score.

        Args:
            fraud_output: Output DataFrame from a fraud pipeline's predict().
            aml_output: Output DataFrame from an AML pipeline's predict().

        Returns:
            DataFrame with unified_risk_score, dominant_signal, and risk_tier.
        """

    @staticmethod
    def _assign_risk_tier(score: float) -> str:
        """Map a continuous [0, 1] score to a categorical risk tier.

        Tier definitions are aligned to typical UK bank alert thresholds:
            - LOW:      score < 0.30
            - MEDIUM:   0.30 <= score < 0.65
            - HIGH:     0.65 <= score < 0.85
            - CRITICAL: score >= 0.85

        Args:
            score: Continuous risk score in [0, 1].

        Returns:
            Risk tier label string.
        """
        if score >= 0.85:
            return "CRITICAL"
        elif score >= 0.65:
            return "HIGH"
        elif score >= 0.30:
            return "MEDIUM"
        return "LOW"
