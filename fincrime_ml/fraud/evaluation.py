"""
fraud/evaluation.py
====================
Evaluation suite for the fraud detection pipeline.

This module provides three complementary evaluation tools that address the
specific challenges of fraud model assessment in a UK regulatory context:

Threshold analysis
    Fraud models produce continuous risk scores; the operating threshold
    determines the precision/recall tradeoff. Different thresholds suit
    different operational contexts: a real-time card authorisation system
    might tolerate a 5% false positive rate to catch 95% of fraud, whereas
    a manual review queue with limited capacity might require higher precision.
    threshold_analysis() sweeps a range of thresholds and returns precision,
    recall, F1, and the number of alerts generated at each point.

False positive cost matrix
    False positives have real costs in a fraud context: declined legitimate
    transactions damage customer experience, trigger complaints, and may
    breach FCA PROD sourcebook obligations (treating customers fairly).
    False negatives allow fraud losses through. cost_matrix() computes the
    expected monetary cost of each error type at a given threshold using
    configurable cost parameters.

Model comparison
    compare_models() accepts predictions from multiple models and returns
    a side-by-side DataFrame of AUC-PR, ROC-AUC, and optionally threshold-
    specific metrics. Designed to produce the champion/challenger comparison
    required by PRA SS1/23 for model validation.

Regulatory context
    PRA SS1/23 Section 5: model validation must include out-of-sample
        performance testing with documented metrics.
    FCA PROD 2.1: firms must consider the impact of product decisions
        (including automated fraud decisions) on retail customers.
    SR 11-7 Section IV: model performance documentation must cover the
        range of conditions under which the model is expected to operate,
        including threshold sensitivity.

Architecture note
    This module imports only from fincrime_ml.core. No imports from
    fincrime_ml.aml are permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default cost parameters (GBP)
# ---------------------------------------------------------------------------

#: Default cost of a false positive (declined legitimate transaction).
#: Includes estimated customer friction cost and complaint handling overhead.
DEFAULT_FP_COST: float = 15.0

#: Default cost of a false negative (missed fraud). Estimated average fraud
#: loss per transaction in UK card fraud (UK Finance 2023 average).
DEFAULT_FN_COST: float = 250.0

#: Default threshold sweep range.
_THRESHOLD_STEPS: int = 100


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ThresholdMetrics:
    """Metrics at a single decision threshold.

    Attributes:
        threshold: The score cutoff applied.
        precision: Fraction of flagged transactions that are genuine fraud.
        recall: Fraction of genuine fraud transactions that are flagged.
        f1: Harmonic mean of precision and recall.
        n_alerts: Number of transactions flagged (predicted positive).
        n_true_positives: Correctly flagged fraud transactions.
        n_false_positives: Legitimate transactions incorrectly flagged.
        n_false_negatives: Fraud transactions missed.
        alert_rate: Fraction of all transactions flagged.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    n_alerts: int
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    alert_rate: float


@dataclass
class CostMatrix:
    """Monetary cost breakdown for a given threshold.

    Attributes:
        threshold: The score cutoff applied.
        fp_cost_per_case: Cost of one false positive (GBP).
        fn_cost_per_case: Cost of one false negative (GBP).
        total_fp_cost: Total false positive cost across all transactions.
        total_fn_cost: Total false negative cost (fraud losses passed through).
        total_cost: Sum of FP and FN costs.
        n_false_positives: Count of false positives.
        n_false_negatives: Count of false negatives.
    """

    threshold: float
    fp_cost_per_case: float
    fn_cost_per_case: float
    total_fp_cost: float
    total_fn_cost: float
    total_cost: float
    n_false_positives: int
    n_false_negatives: int


# ---------------------------------------------------------------------------
# Main evaluation class
# ---------------------------------------------------------------------------


class FraudEvaluator:
    """Evaluation suite for fraud detection models.

    All methods operate on arrays of true labels and predicted scores; they
    are model-agnostic and can be used with any classifier that produces
    probability estimates.

    Example::

        from fincrime_ml.fraud.evaluation import FraudEvaluator

        evaluator = FraudEvaluator()
        summary = evaluator.threshold_analysis(y_true, y_scores)
        print(summary.head(10))

        costs = evaluator.cost_matrix(y_true, y_scores, threshold=0.5)
        print(costs)

        comparison = evaluator.compare_models(
            y_true,
            {"xgb": xgb_scores, "logistic": lr_scores},
        )
        print(comparison)
    """

    def __init__(
        self,
        fp_cost: float = DEFAULT_FP_COST,
        fn_cost: float = DEFAULT_FN_COST,
    ) -> None:
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def threshold_analysis(
        self,
        y_true: np.ndarray | pd.Series,
        y_scores: np.ndarray | pd.Series,
        thresholds: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Sweep thresholds and return precision, recall, F1, and alert volume.

        Args:
            y_true: Binary ground truth labels (1 = fraud, 0 = legitimate).
            y_scores: Continuous fraud risk scores in [0, 1].
            thresholds: Array of threshold values to evaluate. If None, uses
                100 evenly spaced values between the min and max score.

        Returns:
            DataFrame with columns: threshold, precision, recall, f1, n_alerts,
            n_true_positives, n_false_positives, n_false_negatives, alert_rate.
            One row per threshold, sorted by threshold ascending.

        Raises:
            ValueError: If y_true contains non-binary values or lengths differ.
        """
        y_t, y_s = self._validate(y_true, y_scores)

        if thresholds is None:
            thresholds = np.linspace(y_s.min(), y_s.max(), _THRESHOLD_STEPS)

        rows = []
        for t in thresholds:
            m = self._metrics_at_threshold(y_t, y_s, float(t))
            rows.append(
                {
                    "threshold": round(float(t), 6),
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
                    "n_alerts": m.n_alerts,
                    "n_true_positives": m.n_true_positives,
                    "n_false_positives": m.n_false_positives,
                    "n_false_negatives": m.n_false_negatives,
                    "alert_rate": round(m.alert_rate, 4),
                }
            )

        result = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
        logger.info(
            "threshold_analysis: %d thresholds evaluated, score range [%.4f, %.4f]",
            len(thresholds),
            y_s.min(),
            y_s.max(),
        )
        return result

    def cost_matrix(
        self,
        y_true: np.ndarray | pd.Series,
        y_scores: np.ndarray | pd.Series,
        threshold: float,
        fp_cost: float | None = None,
        fn_cost: float | None = None,
    ) -> CostMatrix:
        """Compute monetary cost breakdown at a specific threshold.

        Args:
            y_true: Binary ground truth labels.
            y_scores: Continuous fraud risk scores.
            threshold: Decision threshold to apply.
            fp_cost: Cost per false positive in GBP. Defaults to self.fp_cost.
            fn_cost: Cost per false negative in GBP. Defaults to self.fn_cost.

        Returns:
            CostMatrix dataclass with per-case and total cost figures.

        Raises:
            ValueError: If y_true contains non-binary values or lengths differ.
        """
        y_t, y_s = self._validate(y_true, y_scores)
        fp_cost = fp_cost if fp_cost is not None else self.fp_cost
        fn_cost = fn_cost if fn_cost is not None else self.fn_cost

        m = self._metrics_at_threshold(y_t, y_s, threshold)
        total_fp_cost = m.n_false_positives * fp_cost
        total_fn_cost = m.n_false_negatives * fn_cost

        result = CostMatrix(
            threshold=round(threshold, 6),
            fp_cost_per_case=fp_cost,
            fn_cost_per_case=fn_cost,
            total_fp_cost=round(total_fp_cost, 2),
            total_fn_cost=round(total_fn_cost, 2),
            total_cost=round(total_fp_cost + total_fn_cost, 2),
            n_false_positives=m.n_false_positives,
            n_false_negatives=m.n_false_negatives,
        )
        logger.info(
            "cost_matrix: threshold=%.4f, total_cost=%.2f (FP=%.2f, FN=%.2f)",
            threshold,
            result.total_cost,
            result.total_fp_cost,
            result.total_fn_cost,
        )
        return result

    def optimal_threshold(
        self,
        y_true: np.ndarray | pd.Series,
        y_scores: np.ndarray | pd.Series,
        strategy: str = "f1",
        fp_cost: float | None = None,
        fn_cost: float | None = None,
    ) -> float:
        """Find the threshold that optimises a given strategy.

        Strategies:
            "f1": maximise F1 score.
            "cost": minimise total monetary cost (FP + FN).
            "recall_at_precision": maximise recall subject to precision >= 0.5.

        Args:
            y_true: Binary ground truth labels.
            y_scores: Continuous fraud risk scores.
            strategy: Optimisation strategy (see above).
            fp_cost: Cost per FP (used when strategy="cost").
            fn_cost: Cost per FN (used when strategy="cost").

        Returns:
            Optimal threshold value as float.

        Raises:
            ValueError: On invalid strategy or input validation failure.
        """
        valid_strategies = {"f1", "cost", "recall_at_precision"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"optimal_threshold: strategy must be one of {valid_strategies}, "
                f"got '{strategy}'."
            )

        y_t, y_s = self._validate(y_true, y_scores)
        thresholds = np.linspace(y_s.min(), y_s.max(), _THRESHOLD_STEPS)

        if strategy == "f1":
            best_t = max(thresholds, key=lambda t: self._metrics_at_threshold(y_t, y_s, t).f1)

        elif strategy == "cost":
            fp_c = fp_cost if fp_cost is not None else self.fp_cost
            fn_c = fn_cost if fn_cost is not None else self.fn_cost

            def total_cost(t: float) -> float:
                m = self._metrics_at_threshold(y_t, y_s, t)
                return m.n_false_positives * fp_c + m.n_false_negatives * fn_c

            best_t = min(thresholds, key=total_cost)

        else:  # recall_at_precision
            candidates = [
                t for t in thresholds if self._metrics_at_threshold(y_t, y_s, t).precision >= 0.5
            ]
            if not candidates:
                best_t = float(thresholds[-1])
            else:
                best_t = max(
                    candidates,
                    key=lambda t: self._metrics_at_threshold(y_t, y_s, t).recall,
                )

        logger.info("optimal_threshold: strategy=%s, threshold=%.4f", strategy, best_t)
        return float(best_t)

    def compare_models(
        self,
        y_true: np.ndarray | pd.Series,
        model_scores: dict[str, np.ndarray | pd.Series],
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Compare multiple models on AUC-PR, ROC-AUC, and optional threshold metrics.

        Args:
            y_true: Binary ground truth labels.
            model_scores: Dict mapping model name to score array.
            threshold: If provided, also computes precision, recall, F1 at this
                threshold for each model.

        Returns:
            DataFrame with one row per model and columns: model, auc_pr, roc_auc,
            and optionally precision, recall, f1, n_alerts at the given threshold.
            Sorted by auc_pr descending.

        Raises:
            ValueError: On input validation failure.
        """
        y_t, _ = self._validate(y_true, next(iter(model_scores.values())))

        rows = []
        for model_name, scores in model_scores.items():
            _, y_s = self._validate(y_true, scores)
            row: dict[str, Any] = {
                "model": model_name,
                "auc_pr": round(float(average_precision_score(y_t, y_s)), 4),
                "roc_auc": round(float(roc_auc_score(y_t, y_s)), 4),
            }
            if threshold is not None:
                m = self._metrics_at_threshold(y_t, y_s, threshold)
                row["precision"] = round(m.precision, 4)
                row["recall"] = round(m.recall, 4)
                row["f1"] = round(m.f1, 4)
                row["n_alerts"] = m.n_alerts
            rows.append(row)

        result = pd.DataFrame(rows).sort_values("auc_pr", ascending=False).reset_index(drop=True)
        logger.info("compare_models: %d models compared", len(model_scores))
        return result

    def pr_curve(
        self,
        y_true: np.ndarray | pd.Series,
        y_scores: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """Return precision-recall curve as a DataFrame.

        Args:
            y_true: Binary ground truth labels.
            y_scores: Continuous fraud risk scores.

        Returns:
            DataFrame with columns: threshold, precision, recall. AUC-PR is
            available via sklearn.metrics.average_precision_score separately.
        """
        y_t, y_s = self._validate(y_true, y_scores)
        precision, recall, thresholds = precision_recall_curve(y_t, y_s)
        return pd.DataFrame(
            {
                "threshold": np.append(thresholds, np.nan),
                "precision": precision,
                "recall": recall,
            }
        )

    def roc_curve(
        self,
        y_true: np.ndarray | pd.Series,
        y_scores: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """Return ROC curve as a DataFrame.

        Args:
            y_true: Binary ground truth labels.
            y_scores: Continuous fraud risk scores.

        Returns:
            DataFrame with columns: threshold, fpr, tpr.
        """
        y_t, y_s = self._validate(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_t, y_s)
        return pd.DataFrame({"threshold": thresholds, "fpr": fpr, "tpr": tpr})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _metrics_at_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float,
    ) -> ThresholdMetrics:
        """Compute confusion-matrix-derived metrics at a single threshold."""
        y_pred = (y_scores >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        n_alerts = int(y_pred.sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        alert_rate = n_alerts / len(y_true) if len(y_true) > 0 else 0.0

        return ThresholdMetrics(
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1=f1,
            n_alerts=n_alerts,
            n_true_positives=tp,
            n_false_positives=fp,
            n_false_negatives=fn,
            alert_rate=alert_rate,
        )

    @staticmethod
    def _validate(
        y_true: np.ndarray | pd.Series,
        y_scores: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert inputs to arrays and validate."""
        y_t = np.asarray(y_true, dtype=int)
        y_s = np.asarray(y_scores, dtype=float)

        if len(y_t) != len(y_s):
            raise ValueError(
                f"y_true and y_scores must have the same length. "
                f"Got y_true={len(y_t)}, y_scores={len(y_s)}."
            )
        non_binary = set(np.unique(y_t)) - {0, 1}
        if non_binary:
            raise ValueError(f"y_true must be binary (0/1). Found unexpected values: {non_binary}.")
        return y_t, y_s
