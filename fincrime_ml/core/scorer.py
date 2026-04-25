"""
core/scorer.py
===============
Unified FinCrime risk scorer — configurable fraud + AML signal fusion.

Purpose
    Fraud and AML detection pipelines each produce a domain-specific risk
    score. In practice, an analyst or automated triage system needs a single
    unified risk signal that reflects both dimensions: a transaction may score
    low for fraud but high for AML typology patterns, or vice versa.

    This module provides ``FinCrimeScorer``, a stateless fusion layer that
    combines fraud_score and aml_score columns into a unified_risk_score
    using one of three configurable strategies:

    =================== ==================================================
    Strategy            Formula
    =================== ==================================================
    weighted_average    w_f * fraud + w_a * aml  (default, w sum to 1.0)
    max                 max(fraud, aml)
    harmonic_mean       2 * fraud * aml / (fraud + aml)
    =================== ==================================================

    The scorer degrades gracefully when only one signal is available: if
    fraud_score is absent it uses aml_score exclusively (and vice versa),
    logging a warning to alert the operator.

Signal fusion rationale
    Weighted average is the default. It is interpretable, auditable, and
    allows the operator to express domain priority — for example, a firm
    with a high AML risk appetite (JMLSG Part I para 5.3.1) might set
    aml_weight=0.7 to prioritise AML signals in the unified queue.

    The ``max`` strategy is appropriate when either signal alone should
    escalate the transaction (conservative; higher alert volume).

    ``harmonic_mean`` rewards consistency across both domains: a transaction
    must score meaningfully on both fraud and AML to rank highly, penalising
    partial evidence.

Risk tier mapping
    The unified risk score is mapped to the same four-tier taxonomy used
    across all FinCrime-ML modules:

    ===================== ==========
    Risk Score            Risk Tier
    ===================== ==========
    >= 0.85               CRITICAL
    >= 0.65 and < 0.85    HIGH
    >= 0.40 and < 0.65    MEDIUM
    < 0.40                LOW
    ===================== ==========

Regulatory alignment
    FCA SYSC 6.3   — Integrated financial crime monitoring; firms are expected
                     to consider fraud and AML signals holistically.
    JMLSG Part II  — Sector-specific transaction monitoring guidance.
    PRA SS1/23     — Model risk: blended models must have documented fusion
                     methodology and configurable weights.
    FCA SYSC 10A   — Audit trail requirement for automated scoring decisions.

Architecture note
    FinCrimeScorer lives in core/ — the shared layer. It does NOT import
    from fincrime_ml.fraud or fincrime_ml.aml (ADR 001). It operates only
    on pre-computed numeric score columns passed in a DataFrame.

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk tier thresholds — consistent across all FinCrime-ML modules
# ---------------------------------------------------------------------------

_CRITICAL_THRESHOLD: float = 0.85
_HIGH_THRESHOLD: float = 0.65
_MEDIUM_THRESHOLD: float = 0.40

# ---------------------------------------------------------------------------
# Valid fusion strategies
# ---------------------------------------------------------------------------

FUSION_STRATEGIES: frozenset[str] = frozenset({"weighted_average", "max", "harmonic_mean"})

# ---------------------------------------------------------------------------
# Output column list
# ---------------------------------------------------------------------------

UNIFIED_SCORE_COLS: list[str] = [
    "transaction_id",
    "fraud_score",
    "aml_score",
    "unified_risk_score",
    "risk_tier",
    "fusion_strategy",
    "model_version",
    "scored_at",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FusionConfig:
    """Configuration for the FinCrimeScorer.

    Attributes:
        fraud_weight: Weight applied to fraud_score in weighted_average fusion.
            Must sum to 1.0 with aml_weight (default: 0.5).
        aml_weight: Weight applied to aml_score in weighted_average fusion.
            Must sum to 1.0 with fraud_weight (default: 0.5).
        strategy: Fusion strategy. One of: 'weighted_average', 'max',
            'harmonic_mean'. Default is 'weighted_average'.
        fraud_score_col: Input column name for fraud scores.
        aml_score_col: Input column name for AML scores.
        unified_score_col: Output column name for the fused score.
        version: Scorer version string, included in audit log entries.
        audit_log_enabled: Whether to write a decision audit trail.
    """

    fraud_weight: float = 0.5
    aml_weight: float = 0.5
    strategy: str = "weighted_average"
    fraud_score_col: str = "fraud_score"
    aml_score_col: str = "aml_score"
    unified_score_col: str = "unified_risk_score"
    version: str = "0.1.0"
    audit_log_enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        if self.strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"FusionConfig.strategy must be one of {sorted(FUSION_STRATEGIES)}, "
                f"got '{self.strategy}'."
            )
        if self.strategy == "weighted_average":
            total = self.fraud_weight + self.aml_weight
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"fraud_weight + aml_weight must sum to 1.0, got {total:.6f}.")
        if not (0.0 <= self.fraud_weight <= 1.0):
            raise ValueError(f"fraud_weight must be in [0, 1], got {self.fraud_weight}.")
        if not (0.0 <= self.aml_weight <= 1.0):
            raise ValueError(f"aml_weight must be in [0, 1], got {self.aml_weight}.")


# ---------------------------------------------------------------------------
# Unified scorer
# ---------------------------------------------------------------------------


class FinCrimeScorer:
    """Unified FinCrime risk scorer — configurable fraud + AML signal fusion.

    Accepts a DataFrame containing pre-computed fraud and/or AML risk scores
    and returns a DataFrame with a unified risk score and risk tier.

    The scorer is stateless — it can be called repeatedly on different
    batches without re-initialisation. It does not import from
    fincrime_ml.fraud or fincrime_ml.aml.

    Example::

        from fincrime_ml.core.scorer import FinCrimeScorer, FusionConfig

        cfg = FusionConfig(fraud_weight=0.6, aml_weight=0.4, strategy="weighted_average")
        scorer = FinCrimeScorer(config=cfg)

        # scored_df contains 'fraud_score' and 'aml_score' columns
        result = scorer.score(scored_df)
        print(result[["transaction_id", "unified_risk_score", "risk_tier"]])

    Attributes:
        config: FusionConfig instance.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._audit_log: list[dict] = []
        logger.info(
            "FinCrimeScorer v%s | strategy=%s | weights=(fraud=%.2f, aml=%.2f)",
            self.config.version,
            self.config.strategy,
            self.config.fraud_weight,
            self.config.aml_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fuse fraud and AML scores into a unified risk score.

        At least one of fraud_score_col or aml_score_col must be present.
        If only one signal is available, it is used directly as the unified
        score and a warning is logged.

        Args:
            df: DataFrame containing at least one of the configured score
                columns. A ``transaction_id`` column is included in the
                output if present in the input.

        Returns:
            DataFrame with columns defined in ``UNIFIED_SCORE_COLS``.
            Sorted by unified_risk_score descending (highest risk first).

        Raises:
            KeyError: If neither fraud_score_col nor aml_score_col is present.
        """
        fraud_col = self.config.fraud_score_col
        aml_col = self.config.aml_score_col
        has_fraud = fraud_col in df.columns
        has_aml = aml_col in df.columns

        if not has_fraud and not has_aml:
            raise KeyError(
                f"FinCrimeScorer.score: neither '{fraud_col}' nor '{aml_col}' "
                "found in DataFrame. At least one score column is required."
            )

        fraud_scores = df[fraud_col].astype(float).values if has_fraud else None
        aml_scores = df[aml_col].astype(float).values if has_aml else None

        if not has_fraud:
            logger.warning("fraud_score column '%s' absent — using AML score only.", fraud_col)
            unified = aml_scores.copy()
            fraud_out = np.zeros(len(df))
            aml_out = aml_scores.copy()
        elif not has_aml:
            logger.warning("aml_score column '%s' absent — using fraud score only.", aml_col)
            unified = fraud_scores.copy()
            fraud_out = fraud_scores.copy()
            aml_out = np.zeros(len(df))
        else:
            fraud_out = fraud_scores.copy()
            aml_out = aml_scores.copy()
            unified = self._fuse(fraud_scores, aml_scores)

        unified = np.clip(unified, 0.0, 1.0)
        scored_at = datetime.utcnow().isoformat()

        result_rows: dict[str, Any] = {
            "fraud_score": np.round(fraud_out, 6),
            "aml_score": np.round(aml_out, 6),
            self.config.unified_score_col: np.round(unified, 6),
            "risk_tier": [_assign_risk_tier(s) for s in unified],
            "fusion_strategy": self.config.strategy,
            "model_version": self.config.version,
            "scored_at": scored_at,
        }

        if "transaction_id" in df.columns:
            result_rows["transaction_id"] = df["transaction_id"].values
        else:
            result_rows["transaction_id"] = [str(i) for i in df.index]

        out_cols = [c for c in UNIFIED_SCORE_COLS if c in result_rows]
        result = (
            pd.DataFrame(result_rows)[out_cols]
            .sort_values(self.config.unified_score_col, ascending=False)
            .reset_index(drop=True)
        )

        self._log_audit(
            "score",
            {
                "n_transactions": len(df),
                "has_fraud_signal": has_fraud,
                "has_aml_signal": has_aml,
                "n_critical": int((result["risk_tier"] == "CRITICAL").sum()),
                "n_high": int((result["risk_tier"] == "HIGH").sum()),
            },
        )
        logger.info(
            "FinCrimeScorer.score: %d transactions | strategy=%s | " "CRITICAL=%d HIGH=%d",
            len(df),
            self.config.strategy,
            (result["risk_tier"] == "CRITICAL").sum(),
            (result["risk_tier"] == "HIGH").sum(),
        )
        return result

    @property
    def audit_log(self) -> list[dict]:
        """Return an immutable copy of the audit log."""
        return list(self._audit_log)

    # ------------------------------------------------------------------
    # Private: fusion strategies
    # ------------------------------------------------------------------

    def _fuse(self, fraud: np.ndarray, aml: np.ndarray) -> np.ndarray:
        """Dispatch to the configured fusion strategy.

        Args:
            fraud: Fraud score array.
            aml: AML score array.

        Returns:
            Unified score array.
        """
        if self.config.strategy == "weighted_average":
            return self._weighted_average(fraud, aml)
        if self.config.strategy == "max":
            return self._max_fusion(fraud, aml)
        if self.config.strategy == "harmonic_mean":
            return self._harmonic_mean(fraud, aml)
        raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def _weighted_average(self, fraud: np.ndarray, aml: np.ndarray) -> np.ndarray:
        return self.config.fraud_weight * fraud + self.config.aml_weight * aml

    @staticmethod
    def _max_fusion(fraud: np.ndarray, aml: np.ndarray) -> np.ndarray:
        return np.maximum(fraud, aml)

    @staticmethod
    def _harmonic_mean(fraud: np.ndarray, aml: np.ndarray) -> np.ndarray:
        denom = fraud + aml
        return np.where(denom > 0, 2.0 * fraud * aml / denom, 0.0)

    # ------------------------------------------------------------------
    # Private: audit and validation
    # ------------------------------------------------------------------

    def _log_audit(self, event: str, metadata: dict | None = None) -> None:
        if not self.config.audit_log_enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "scorer": self.__class__.__name__,
            "version": self.config.version,
            "event": event,
            **(metadata or {}),
        }
        self._audit_log.append(entry)
        logger.debug("Audit: %s", entry)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _assign_risk_tier(score: float) -> str:
    """Map a unified risk score to a four-tier risk label.

    Args:
        score: Unified risk score in [0, 1].

    Returns:
        One of: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'.
    """
    if score >= _CRITICAL_THRESHOLD:
        return "CRITICAL"
    if score >= _HIGH_THRESHOLD:
        return "HIGH"
    if score >= _MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"
