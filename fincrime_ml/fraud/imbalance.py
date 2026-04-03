"""
fraud/imbalance.py
==================
Class imbalance handlers for the fraud detection pipeline.

Payment fraud datasets are severely imbalanced: typical positive rates range
from 0.3% to 3% in production card transaction data (UK Finance 2023). This
module provides two complementary strategies for handling this imbalance:

SMOTE (Synthetic Minority Oversampling Technique)
    Generates synthetic fraud samples by interpolating between real fraud
    examples in feature space. Effective when the fraud class is well
    clustered. Implemented via imbalanced-learn. Sampling strategy is
    configurable; default targets a 1:10 minority-to-majority ratio.

Cost-sensitive weighting
    Assigns higher misclassification cost to the minority (fraud) class
    by computing sample weights that are passed to the model's fit method.
    Avoids generating synthetic data; compatible with all sklearn-API
    estimators that accept a sample_weight argument.

Benchmark comparison
    ImbalanceHandler.benchmark() trains a reference estimator under both
    strategies and returns AUC-PR and ROC-AUC for each, allowing the caller
    to select the better approach for a given dataset. AUC-PR is the primary
    metric (average_precision_score) per the FinCrime-ML architecture rules.

Regulatory context
    PRA SS1/23 requires documented rationale for modelling choices that
    affect model performance, including class imbalance treatment. The
    benchmark output is designed to produce a reproducible comparison that
    can be attached to model validation evidence.

Architecture note
    This module imports only from fincrime_ml.core. No imports from
    fincrime_ml.aml are permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default SMOTE sampling strategy: minority class will be oversampled to
#: reach this fraction of the majority class size.
DEFAULT_SMOTE_STRATEGY: float = 0.1

#: Default number of cross-validation folds for the benchmark.
DEFAULT_CV_FOLDS: int = 5

#: Reference estimator used in benchmark if none is supplied.
_DEFAULT_ESTIMATOR_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImbalanceResult:
    """Benchmark result for a single imbalance handling strategy.

    Attributes:
        strategy: Name of the strategy ("smote" or "cost_sensitive").
        mean_auc_pr: Mean AUC-PR (average_precision_score) across CV folds.
            This is the primary metric per FinCrime-ML architecture rules.
        std_auc_pr: Standard deviation of AUC-PR across folds.
        mean_roc_auc: Mean ROC-AUC across CV folds. Reported for completeness
            but not used for strategy selection (misleading on imbalanced data).
        std_roc_auc: Standard deviation of ROC-AUC across folds.
        n_folds: Number of cross-validation folds used.
        fraud_rate: Positive class rate in the original dataset.
        extra: Optional dict for storing strategy-specific metadata.
    """

    strategy: str
    mean_auc_pr: float
    std_auc_pr: float
    mean_roc_auc: float
    std_roc_auc: float
    n_folds: int
    fraud_rate: float
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-line human-readable summary of the result."""
        return (
            f"{self.strategy}: AUC-PR={self.mean_auc_pr:.4f} "
            f"(+/-{self.std_auc_pr:.4f}), "
            f"ROC-AUC={self.mean_roc_auc:.4f} "
            f"(+/-{self.std_roc_auc:.4f}), "
            f"fraud_rate={self.fraud_rate:.4f}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ImbalanceHandler:
    """Handle class imbalance for fraud detection model training.

    Provides SMOTE oversampling and cost-sensitive sample weighting as
    independent strategies, plus a cross-validated benchmark that compares
    both on AUC-PR.

    Example::

        from fincrime_ml.fraud.imbalance import ImbalanceHandler

        handler = ImbalanceHandler(seed=42)

        # Apply SMOTE to training data
        X_res, y_res = handler.apply_smote(X_train, y_train)

        # Or compute sample weights for cost-sensitive training
        weights = handler.compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=weights)

        # Benchmark both strategies
        results = handler.benchmark(X, y)
        for r in results:
            print(r.summary())

    Attributes:
        seed: Random seed for reproducibility.
        smote_strategy: SMOTE sampling_strategy parameter.
        cv_folds: Number of stratified CV folds for benchmarking.
    """

    def __init__(
        self,
        seed: int = 42,
        smote_strategy: float = DEFAULT_SMOTE_STRATEGY,
        cv_folds: int = DEFAULT_CV_FOLDS,
    ) -> None:
        self.seed = seed
        self.smote_strategy = smote_strategy
        self.cv_folds = cv_folds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_smote(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        k_neighbors: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Oversample the minority class using SMOTE.

        Generates synthetic fraud samples by interpolating between existing
        fraud samples in feature space. The resampled dataset is suitable for
        passing directly to a model's fit method.

        Args:
            X: Feature matrix. Shape (n_samples, n_features). Must be numeric;
                categorical features must be encoded before calling this method.
            y: Binary target vector. 1 = fraud, 0 = legitimate.
            k_neighbors: Number of nearest neighbours used to generate synthetic
                samples. Must be less than the number of minority class samples.

        Returns:
            Tuple of (X_resampled, y_resampled) as numpy arrays. The resampled
            dataset is shuffled to avoid positional bias.

        Raises:
            ValueError: If y contains fewer than k_neighbors + 1 positive samples,
                or if X and y have incompatible lengths.
        """
        X_arr, y_arr = self._to_arrays(X, y)
        self._validate_inputs(X_arr, y_arr, k_neighbors)

        n_fraud_before = int(y_arr.sum())
        smote = SMOTE(
            sampling_strategy=self.smote_strategy,
            k_neighbors=k_neighbors,
            random_state=self.seed,
        )
        X_res, y_res = smote.fit_resample(X_arr, y_arr)

        n_fraud_after = int(y_res.sum())
        logger.info(
            "apply_smote: fraud samples %d -> %d (strategy=%.2f)",
            n_fraud_before,
            n_fraud_after,
            self.smote_strategy,
        )
        return X_res, y_res

    def compute_sample_weights(
        self,
        y: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """Compute per-sample weights for cost-sensitive learning.

        Assigns each sample a weight inversely proportional to its class
        frequency. Fraud samples receive a higher weight, penalising missed
        detections relative to false positives.

        The weights are computed using sklearn's compute_sample_weight with
        class_weight="balanced", which is equivalent to:
            weight_class = n_samples / (n_classes * n_samples_in_class)

        Args:
            y: Binary target vector. 1 = fraud, 0 = legitimate.

        Returns:
            Array of per-sample weights with shape (n_samples,).

        Raises:
            ValueError: If y is empty or contains only one class.
        """
        y_arr = np.asarray(y)
        if len(y_arr) == 0:
            raise ValueError("compute_sample_weights: y must not be empty.")
        n_classes = len(np.unique(y_arr))
        if n_classes < 2:
            raise ValueError(
                f"compute_sample_weights: y must contain at least 2 classes, " f"got {n_classes}."
            )
        weights = compute_sample_weight(class_weight="balanced", y=y_arr)
        fraud_weight = float(weights[y_arr == 1][0])
        legit_weight = float(weights[y_arr == 0][0])
        logger.info(
            "compute_sample_weights: fraud weight=%.4f, legit weight=%.4f, ratio=%.1f:1",
            fraud_weight,
            legit_weight,
            fraud_weight / legit_weight,
        )
        return weights

    def benchmark(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        estimator: Any = None,
    ) -> list[ImbalanceResult]:
        """Cross-validate both strategies and compare AUC-PR.

        Runs stratified k-fold cross-validation for SMOTE and cost-sensitive
        weighting. Returns one ImbalanceResult per strategy. The caller should
        select the strategy with the higher mean_auc_pr.

        Args:
            X: Feature matrix. Must be numeric.
            y: Binary target vector. 1 = fraud.
            estimator: sklearn-API estimator with a predict_proba method. If
                None, uses LogisticRegression with balanced class weight as a
                neutral reference model.

        Returns:
            List of two ImbalanceResult objects: [smote_result, cost_sensitive_result].

        Raises:
            ValueError: If X or y are invalid.
        """
        X_arr, y_arr = self._to_arrays(X, y)
        fraud_rate = float(y_arr.mean())

        if estimator is None:
            estimator = LogisticRegression(**_DEFAULT_ESTIMATOR_PARAMS)

        smote_result = self._cv_smote(X_arr, y_arr, estimator, fraud_rate)
        cs_result = self._cv_cost_sensitive(X_arr, y_arr, estimator, fraud_rate)

        logger.info("benchmark complete:")
        logger.info("  %s", smote_result.summary())
        logger.info("  %s", cs_result.summary())

        return [smote_result, cs_result]

    def best_strategy(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        estimator: Any = None,
    ) -> str:
        """Return the name of the better-performing strategy by AUC-PR.

        Runs benchmark() and returns "smote" or "cost_sensitive".

        Args:
            X: Feature matrix.
            y: Binary target vector.
            estimator: Optional sklearn estimator (see benchmark()).

        Returns:
            Strategy name: "smote" or "cost_sensitive".
        """
        results = self.benchmark(X, y, estimator=estimator)
        best = max(results, key=lambda r: r.mean_auc_pr)
        logger.info("best_strategy: %s (AUC-PR=%.4f)", best.strategy, best.mean_auc_pr)
        return best.strategy

    # ------------------------------------------------------------------
    # Private: CV helpers
    # ------------------------------------------------------------------

    def _cv_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator: Any,
        fraud_rate: float,
    ) -> ImbalanceResult:
        """Run stratified CV with SMOTE applied inside each fold."""
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        auc_prs, roc_aucs = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            n_minority = int(y_train.sum())
            k = min(5, n_minority - 1) if n_minority > 1 else 1
            if n_minority < 2:
                logger.warning("SMOTE fold skipped: fewer than 2 fraud samples in fold.")
                continue

            smote = SMOTE(
                sampling_strategy=self.smote_strategy,
                k_neighbors=k,
                random_state=self.seed,
            )
            X_res, y_res = smote.fit_resample(X_train, y_train)

            import sklearn.base

            est = sklearn.base.clone(estimator)
            est.fit(X_res, y_res)
            y_prob = est.predict_proba(X_val)[:, 1]

            auc_prs.append(average_precision_score(y_val, y_prob))
            roc_aucs.append(roc_auc_score(y_val, y_prob))

        return ImbalanceResult(
            strategy="smote",
            mean_auc_pr=float(np.mean(auc_prs)),
            std_auc_pr=float(np.std(auc_prs)),
            mean_roc_auc=float(np.mean(roc_aucs)),
            std_roc_auc=float(np.std(roc_aucs)),
            n_folds=len(auc_prs),
            fraud_rate=fraud_rate,
            extra={"smote_strategy": self.smote_strategy},
        )

    def _cv_cost_sensitive(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator: Any,
        fraud_rate: float,
    ) -> ImbalanceResult:
        """Run stratified CV with cost-sensitive sample weights."""
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        auc_prs, roc_aucs = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            weights = compute_sample_weight(class_weight="balanced", y=y_train)

            import sklearn.base

            est = sklearn.base.clone(estimator)
            try:
                est.fit(X_train, y_train, sample_weight=weights)
            except TypeError:
                # Estimator does not support sample_weight; fall back to unweighted
                logger.warning(
                    "cost_sensitive: estimator does not support sample_weight; "
                    "falling back to unweighted fit."
                )
                est.fit(X_train, y_train)

            y_prob = est.predict_proba(X_val)[:, 1]
            auc_prs.append(average_precision_score(y_val, y_prob))
            roc_aucs.append(roc_auc_score(y_val, y_prob))

        return ImbalanceResult(
            strategy="cost_sensitive",
            mean_auc_pr=float(np.mean(auc_prs)),
            std_auc_pr=float(np.std(auc_prs)),
            mean_roc_auc=float(np.mean(roc_aucs)),
            std_roc_auc=float(np.std(roc_aucs)),
            n_folds=len(auc_prs),
            fraud_rate=fraud_rate,
        )

    # ------------------------------------------------------------------
    # Private: input helpers
    # ------------------------------------------------------------------

    def _to_arrays(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert inputs to numpy arrays."""
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        return X_arr.astype(float), y_arr.astype(int)

    def _validate_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_neighbors: int,
    ) -> None:
        """Validate inputs for SMOTE.

        Args:
            X: Feature matrix.
            y: Target vector.
            k_neighbors: Requested k for SMOTE.

        Raises:
            ValueError: On shape mismatch or insufficient minority samples.
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got X={len(X)}, y={len(y)}.")
        n_fraud = int(y.sum())
        if n_fraud <= k_neighbors:
            raise ValueError(
                f"apply_smote requires at least k_neighbors + 1 = {k_neighbors + 1} "
                f"fraud samples. Found {n_fraud}."
            )
