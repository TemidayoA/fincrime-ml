"""
aml/evaluation.py
==================
Alert fatigue metrics — false positive rate at configurable sensitivity levels.

Purpose
    Transaction monitoring systems face an inherent tension: increasing
    sensitivity (recall) catches more genuine suspicious activity, but
    simultaneously generates more false positive alerts. Compliance analysts
    can review only a finite number of alerts per day; excessive false
    positives cause alert fatigue, degrade investigator performance, and
    ultimately increase the risk of genuine SAR cases being missed.

    This module quantifies that tension. Given a set of model scores and
    binary labels, it computes:

    1. FPR (False Positive Rate) at each configurable sensitivity target —
       the core alert fatigue metric. At 90% recall, what fraction of
       legitimate transactions are incorrectly flagged?

    2. Alert volume profile — precision, recall, FPR, F1, and fatigue index
       across the full threshold range.

    3. Sensitivity curve — the FPR vs sensitivity trade-off for visual review
       by an MLRO or model validation team.

    4. Optimal threshold recommendation — the score cut-off that maximises
       F1 while optionally constraining minimum sensitivity.

Key metrics
    ================== ====================================================
    Metric             Definition
    ================== ====================================================
    FPR                FP / (FP + TN) — fraction of clean transactions
                       incorrectly escalated to the alert queue.
    Sensitivity        TP / (TP + FN) — fraction of suspicious transactions
                       correctly detected. Also called recall.
    Alert rate         (TP + FP) / N — fraction of all transactions
                       generating an alert.
    Fatigue index      FP / (TP + FP) = 1 - precision — fraction of alerts
                       that are false positives (analyst wasted effort rate).
    AUC-PR             Area under the Precision-Recall curve. Primary metric
                       for imbalanced AML datasets (JMLSG Part I para 5.3.1).
    ROC AUC            Area under the ROC curve. Secondary metric.
    ================== ====================================================

Regulatory alignment
    FCA SYSC 6.3 — Transaction monitoring alert management and review.
    JMLSG Part I Ch.5 para 5.3.1 — Tuning guidance for automated systems.
    MLR 2017 Reg 19 — Staff training and alert review capacity obligations.
    PRA SS1/23 — Model risk and performance monitoring expectations.

Architecture note
    AlertFatigueEvaluator is a stateless evaluation utility — no training
    or state is maintained between calls. It is compatible with scores
    from any pipeline: GraphScorer, AMLIsolationForest, SARScorer, or
    external systems. No imports from fincrime_ml.fraud (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

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
# Constants
# ---------------------------------------------------------------------------

#: Default sensitivity targets for FPR analysis (recall levels to evaluate).
DEFAULT_SENSITIVITY_TARGETS: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 0.99)

#: Number of threshold grid points for alert_volume_profile().
_N_THRESHOLDS: int = 200


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AlertFatigueConfig:
    """Configuration for the AlertFatigueEvaluator.

    Attributes:
        sensitivity_targets: Recall levels at which FPR and alert rate are
            computed. Each value must be in (0, 1].
        min_sensitivity: Minimum acceptable recall. The recommended threshold
            will not fall below this level (default: 0.80).
        alert_col_label: Display label for the alert column in profile
            DataFrames.
        version: Evaluator version string, included in report metadata.
    """

    sensitivity_targets: tuple[float, ...] = DEFAULT_SENSITIVITY_TARGETS
    min_sensitivity: float = 0.80
    alert_col_label: str = "alert_flag"
    version: str = "0.1.0"
    created_at: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class AlertFatigueEvaluator:
    """AML alert fatigue evaluator — FPR at configurable sensitivity levels.

    Evaluates a set of continuous risk scores against binary ground-truth
    labels. Returns a comprehensive suite of alert fatigue metrics suitable
    for MLRO review, model validation reports, and FCA SYSC 6.3 audit.

    The evaluator is stateless between calls — the same instance can be
    reused across different score batches.

    Example::

        from fincrime_ml.aml.evaluation import AlertFatigueEvaluator

        evaluator = AlertFatigueEvaluator()
        report = evaluator.evaluate(y_true, risk_scores)
        print(report["sensitivity_analysis"][0.90]["fpr"])

    Attributes:
        config: AlertFatigueConfig instance.
    """

    def __init__(self, config: AlertFatigueConfig | None = None) -> None:
        self.config = config or AlertFatigueConfig()
        logger.info(
            "AlertFatigueEvaluator v%s | sensitivity targets: %s",
            self.config.version,
            self.config.sensitivity_targets,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: Sequence[int],
        scores: Sequence[float],
        sensitivity_targets: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """Run the full alert fatigue evaluation suite.

        Args:
            y_true: Binary ground-truth labels (0=legitimate, 1=suspicious).
            scores: Continuous risk scores in [0, 1], one per transaction.
            sensitivity_targets: Override config sensitivity targets for this
                call. If None, uses config.sensitivity_targets.

        Returns:
            Dict with keys:
                n_positives, n_negatives, base_rate,
                auc_pr, roc_auc,
                sensitivity_analysis (nested dict keyed by target float),
                optimal_threshold (dict with threshold + metrics).

        Raises:
            ValueError: If y_true has fewer than 2 unique classes, or
                scores are not the same length as y_true.
        """
        y_arr, s_arr = self._validate_inputs(y_true, scores)
        targets = sensitivity_targets or self.config.sensitivity_targets

        n_pos = int(y_arr.sum())
        n_neg = int((y_arr == 0).sum())
        base_rate = n_pos / len(y_arr)

        auc_pr = float(average_precision_score(y_arr, s_arr))
        roc_auc = float(roc_auc_score(y_arr, s_arr))

        sens_analysis: dict[float, dict[str, float]] = {}
        for target in targets:
            thresh = self.threshold_at_sensitivity(y_arr, s_arr, target)
            metrics = self._metrics_at_threshold(y_arr, s_arr, thresh)
            sens_analysis[target] = {
                "threshold": round(float(thresh), 6),
                **metrics,
            }

        opt_thresh = self._optimal_threshold(y_arr, s_arr)
        opt_metrics = self._metrics_at_threshold(y_arr, s_arr, opt_thresh)

        report = {
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "base_rate": round(base_rate, 6),
            "auc_pr": round(auc_pr, 6),
            "roc_auc": round(roc_auc, 6),
            "sensitivity_analysis": sens_analysis,
            "optimal_threshold": {
                "value": round(float(opt_thresh), 6),
                **opt_metrics,
            },
        }
        logger.info(
            "AlertFatigueEvaluator.evaluate: n=%d pos=%d AUC-PR=%.4f ROC-AUC=%.4f",
            len(y_arr),
            n_pos,
            auc_pr,
            roc_auc,
        )
        return report

    def fpr_at_sensitivity(
        self,
        y_true: Sequence[int],
        scores: Sequence[float],
        sensitivity: float,
    ) -> float:
        """Compute FPR at a target sensitivity (recall) level.

        Finds the lowest score threshold that achieves at least ``sensitivity``
        recall, then returns the false positive rate at that threshold.

        Args:
            y_true: Binary ground-truth labels.
            scores: Continuous risk scores in [0, 1].
            sensitivity: Target recall level in (0, 1].

        Returns:
            FPR (False Positive Rate) at the threshold that achieves the
            target sensitivity. Returns 1.0 if target cannot be achieved.

        Raises:
            ValueError: If inputs are invalid.
        """
        y_arr, s_arr = self._validate_inputs(y_true, scores)
        thresh = self.threshold_at_sensitivity(y_arr, s_arr, sensitivity)
        return self._metrics_at_threshold(y_arr, s_arr, thresh)["fpr"]

    def threshold_at_sensitivity(
        self,
        y_true: Sequence[int],
        scores: Sequence[float],
        sensitivity: float,
    ) -> float:
        """Find the score threshold that achieves a minimum recall level.

        Searches the precision-recall curve for the highest threshold
        (most restrictive alert gate) at which recall >= ``sensitivity``.

        Args:
            y_true: Binary ground-truth labels.
            scores: Continuous risk scores in [0, 1].
            sensitivity: Minimum acceptable recall, in (0, 1].

        Returns:
            Score threshold (float). Returns 0.0 if no threshold achieves
            the target (i.e. the model's maximum recall is below target).

        Raises:
            ValueError: If inputs are invalid.
        """
        y_arr, s_arr = self._validate_inputs(y_true, scores)
        _, recall_arr, thresh_arr = precision_recall_curve(y_arr, s_arr)
        # thresh_arr has length len(recall_arr) - 1; last recall point has no threshold
        # Recall decreases as threshold increases; find last index where recall >= target
        qualifying = np.where(recall_arr[:-1] >= sensitivity)[0]
        if len(qualifying) == 0:
            return 0.0
        # Highest threshold that still achieves the recall target
        idx = qualifying[-1]
        return float(thresh_arr[idx])

    def alert_volume_profile(
        self,
        y_true: Sequence[int],
        scores: Sequence[float],
        thresholds: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        """Compute alert metrics across a range of score thresholds.

        For each threshold, calculates: alert count, alert rate, precision,
        recall (sensitivity), FPR, F1, and fatigue index.

        Args:
            y_true: Binary ground-truth labels.
            scores: Continuous risk scores in [0, 1].
            thresholds: Explicit threshold grid. If None, uses a uniform
                grid of _N_THRESHOLDS points over [0, 1].

        Returns:
            DataFrame with columns: threshold, n_alerts, alert_rate,
            precision, recall, fpr, f1, fatigue_index. Sorted by
            threshold ascending.

        Raises:
            ValueError: If inputs are invalid.
        """
        y_arr, s_arr = self._validate_inputs(y_true, scores)

        if thresholds is None:
            thresh_grid = np.linspace(0.0, 1.0, _N_THRESHOLDS)
        else:
            thresh_grid = np.array(thresholds, dtype=float)

        rows = []
        for t in thresh_grid:
            metrics = self._metrics_at_threshold(y_arr, s_arr, t)
            rows.append({"threshold": round(float(t), 6), **metrics})

        return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    def sensitivity_curve(
        self,
        y_true: Sequence[int],
        scores: Sequence[float],
    ) -> pd.DataFrame:
        """Return a sensitivity (recall) vs FPR curve for plotting or review.

        Based on the ROC curve. Each row represents one operating point
        on the threshold continuum.

        Args:
            y_true: Binary ground-truth labels.
            scores: Continuous risk scores in [0, 1].

        Returns:
            DataFrame with columns: threshold, sensitivity (recall), fpr,
            specificity. Sorted by sensitivity ascending.

        Raises:
            ValueError: If inputs are invalid.
        """
        y_arr, s_arr = self._validate_inputs(y_true, scores)
        fpr_arr, tpr_arr, thresh_arr = roc_curve(y_arr, s_arr)

        rows = []
        for i, (fpr_val, tpr_val) in enumerate(zip(fpr_arr, tpr_arr)):
            thresh_val = float(thresh_arr[i]) if i < len(thresh_arr) else 0.0
            rows.append(
                {
                    "threshold": round(thresh_val, 6),
                    "sensitivity": round(float(tpr_val), 6),
                    "fpr": round(float(fpr_val), 6),
                    "specificity": round(1.0 - float(fpr_val), 6),
                }
            )

        return pd.DataFrame(rows).sort_values("sensitivity").reset_index(drop=True)

    def fatigue_index(
        self,
        y_true: Sequence[int],
        scores: Sequence[float],
        threshold: float,
    ) -> float:
        """Compute the alert fatigue index at a given threshold.

        The fatigue index is 1 - precision = FP / (TP + FP), representing
        the fraction of analyst-reviewed alerts that are false positives.
        A fatigue index of 0.90 means 9 in 10 alerts are wasted reviews.

        Args:
            y_true: Binary ground-truth labels.
            scores: Continuous risk scores in [0, 1].
            threshold: Score threshold above which an alert is generated.

        Returns:
            Fatigue index in [0, 1]. Returns 0.0 if no alerts are generated.

        Raises:
            ValueError: If inputs are invalid.
        """
        y_arr, s_arr = self._validate_inputs(y_true, scores)
        return self._metrics_at_threshold(y_arr, s_arr, threshold)["fatigue_index"]

    def pr_auc(
        self,
        y_true: Sequence[int],
        scores: Sequence[float],
    ) -> float:
        """Compute the area under the Precision-Recall curve.

        Primary metric for AML model performance on imbalanced datasets.

        Args:
            y_true: Binary ground-truth labels.
            scores: Continuous risk scores in [0, 1].

        Returns:
            AUC-PR in [0, 1].
        """
        y_arr, s_arr = self._validate_inputs(y_true, scores)
        return float(average_precision_score(y_arr, s_arr))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        y_true: Sequence[int],
        scores: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and coerce inputs; raise ValueError on bad data.

        Returns:
            (y_arr, s_arr) as numpy float/int arrays.

        Raises:
            ValueError: If arrays differ in length, have fewer than 2
                samples, or y_true contains only one class.
        """
        y_arr = np.asarray(y_true, dtype=int)
        s_arr = np.asarray(scores, dtype=float)

        if len(y_arr) != len(s_arr):
            raise ValueError(f"y_true length ({len(y_arr)}) != scores length ({len(s_arr)})")
        if len(y_arr) < 2:
            raise ValueError("AlertFatigueEvaluator requires at least 2 samples.")
        if len(np.unique(y_arr)) < 2:
            raise ValueError(
                "y_true must contain both positive (1) and negative (0) examples. "
                f"Got unique values: {np.unique(y_arr).tolist()}"
            )
        return y_arr, s_arr

    @staticmethod
    def _metrics_at_threshold(
        y_arr: np.ndarray,
        s_arr: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        """Compute confusion-matrix-derived metrics at a single threshold.

        Args:
            y_arr: Binary label array.
            s_arr: Score array.
            threshold: Score cut-off for alert generation.

        Returns:
            Dict with keys: n_alerts, alert_rate, precision, recall,
            fpr, f1, fatigue_index.
        """
        predicted = (s_arr >= threshold).astype(int)
        tp = int(((predicted == 1) & (y_arr == 1)).sum())
        fp = int(((predicted == 1) & (y_arr == 0)).sum())
        fn = int(((predicted == 0) & (y_arr == 1)).sum())
        tn = int(((predicted == 0) & (y_arr == 0)).sum())

        n_alerts = tp + fp
        n = len(y_arr)

        precision = tp / n_alerts if n_alerts > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fatigue_index = 1.0 - precision if n_alerts > 0 else 0.0

        return {
            "n_alerts": n_alerts,
            "alert_rate": round(n_alerts / n, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "fpr": round(fpr, 6),
            "f1": round(f1, 6),
            "fatigue_index": round(fatigue_index, 6),
        }

    def _optimal_threshold(
        self,
        y_arr: np.ndarray,
        s_arr: np.ndarray,
    ) -> float:
        """Find the threshold that maximises F1, subject to min_sensitivity.

        Searches the precision-recall curve for the operating point with
        the highest F1 score that also achieves config.min_sensitivity.
        Falls back to the threshold that maximises F1 unconstrained if
        no threshold meets the sensitivity constraint.

        Args:
            y_arr: Binary label array.
            s_arr: Score array.

        Returns:
            Optimal score threshold.
        """
        prec_arr, recall_arr, thresh_arr = precision_recall_curve(y_arr, s_arr)
        # thresh_arr is 1 element shorter than prec_arr / recall_arr
        prec_arr = prec_arr[:-1]
        recall_arr = recall_arr[:-1]

        denom = prec_arr + recall_arr
        f1_arr = np.where(denom > 0, 2 * prec_arr * recall_arr / denom, 0.0)

        # Constrain to min_sensitivity
        mask = recall_arr >= self.config.min_sensitivity
        if mask.any():
            best_idx = int(np.argmax(f1_arr[mask]))
            constrained_idxs = np.where(mask)[0]
            return float(thresh_arr[constrained_idxs[best_idx]])

        # Fallback: unconstrained F1 maximisation
        return float(thresh_arr[int(np.argmax(f1_arr))])
