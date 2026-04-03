"""
fraud/models/xgb_classifier.py
================================
XGBoost fraud detection classifier with AUC-PR optimisation.

This module implements the primary supervised fraud detection model for the
FinCrime-ML pipeline. XGBoost is chosen for its strong out-of-the-box
performance on tabular imbalanced datasets, native handling of missing values,
and the ability to use scale_pos_weight to penalise missed fraud detections.

Model design decisions
    - Primary metric: AUC-PR (average_precision_score). ROC-AUC is misleading
      on imbalanced datasets and is reported for reference only (PRA SS1/23).
    - scale_pos_weight: Set to (n_legit / n_fraud) to handle class imbalance
      without requiring SMOTE or data augmentation at training time. This is
      cost-sensitive learning built into the XGBoost objective.
    - Cross-validation: StratifiedKFold preserves the fraud rate in each fold.
      The model with the highest mean AUC-PR across folds is retained.
    - Early stopping: Applied per fold using a validation subset to prevent
      overfitting without a fixed number of rounds.

Regulatory context
    All predict() calls write to the in-memory audit log via _log_audit()
    (FCA SYSC 10A). The explain() method returns SHAP-based reason codes
    suitable for inclusion in a decision record (SR 11-7).

Architecture note
    Inherits from BasePipeline (fincrime_ml.core.base). No imports from
    fincrime_ml.aml are permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from fincrime_ml.core.base import BasePipeline, PipelineConfig
from fincrime_ml.fraud.features import FraudFeatureEngineer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns used by the XGBoost model
# ---------------------------------------------------------------------------

#: Columns produced by FraudFeatureEngineer.transform() that are used as
#: model inputs. Categorical columns are excluded; encoding is not required
#: since XGBoost handles numeric inputs natively. The mcc_risk_score column
#: replaces the raw mcc_risk string.
FEATURE_COLS: list[str] = [
    # Velocity features
    "velocity_count_1h",
    "velocity_count_24h",
    "velocity_count_168h",
    "velocity_amount_1h",
    "velocity_amount_24h",
    "velocity_amount_168h",
    # Amount deviation
    "amount_zscore",
    "amount_over_avg_ratio",
    # MCC risk
    "mcc_risk_score",
    "is_high_risk_mcc",
    # Geographic risk
    "is_international",
    "high_risk_corridor",
    # Temporal
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    # Raw amount (log-scale signal)
    "amount_gbp",
]

#: Default XGBoost hyperparameters. scale_pos_weight is set dynamically at
#: training time based on the observed class ratio.
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "verbosity": 0,
    "random_state": 42,
}

#: Number of rounds without AUC-PR improvement before early stopping.
EARLY_STOPPING_ROUNDS: int = 30

#: Validation fraction used for early stopping within each CV fold.
EARLY_STOPPING_VAL_FRAC: float = 0.15


class XGBFraudClassifier(BasePipeline):
    """XGBoost-based fraud detection classifier.

    Inherits from BasePipeline and implements the full FinCrime-ML pipeline
    interface: prepare_features, train, predict, explain.

    Training runs a stratified k-fold cross-validation harness to estimate
    generalisation performance. The model fitted on the full training set
    (using the best hyperparameters validated across folds) is retained for
    inference.

    Example::

        from fincrime_ml.fraud.models.xgb_classifier import XGBFraudClassifier

        clf = XGBFraudClassifier(n_cv_folds=5, seed=42)
        clf.train(df_train, label_col="is_fraud")

        scores = clf.predict(df_holdout)
        print(scores[["transaction_id", "risk_score", "risk_tier"]].head())

        explanations = clf.explain(df_holdout.head(100))
        print(explanations[["transaction_id", "top_reason_1", "top_reason_2"]].head())

    Attributes:
        n_cv_folds: Number of stratified CV folds.
        seed: Random seed.
        xgb_params: XGBoost model parameters (merged with defaults).
        cv_results: List of per-fold AUC-PR scores, populated after train().
        feature_engineer: FraudFeatureEngineer instance used in prepare_features().
    """

    def __init__(
        self,
        n_cv_folds: int = 5,
        seed: int = 42,
        xgb_params: dict[str, Any] | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        self.n_cv_folds = n_cv_folds
        self.seed = seed
        self.xgb_params: dict[str, Any] = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.xgb_params["random_state"] = seed
        self.cv_results: list[dict[str, float]] = []
        self.feature_engineer = FraudFeatureEngineer()
        self.feature_names: list[str] = FEATURE_COLS

    # ------------------------------------------------------------------
    # BasePipeline interface
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply FraudFeatureEngineer and select model feature columns.

        Computes velocity, amount deviation, and MCC risk features, then
        filters to the columns listed in FEATURE_COLS. Columns absent in
        the input (e.g. account_avg_spend missing in IEEE-CIS data) cause
        a ValueError before reaching the model.

        Args:
            df: Raw transaction DataFrame in FinCrime-ML schema.

        Returns:
            DataFrame with columns matching FEATURE_COLS, in that order.

        Raises:
            ValueError: If required base columns are absent from df.
        """
        enriched = self.feature_engineer.transform(df)
        missing = [c for c in FEATURE_COLS if c not in enriched.columns]
        if missing:
            raise ValueError(
                f"prepare_features: feature columns missing after engineering: {missing}"
            )
        return enriched[FEATURE_COLS]

    def train(self, df: pd.DataFrame, label_col: str = "is_fraud") -> "XGBFraudClassifier":
        """Train the XGBoost classifier with stratified CV and early stopping.

        Steps:
            1. Feature engineering via prepare_features().
            2. Compute scale_pos_weight from class ratio.
            3. Run StratifiedKFold CV; record AUC-PR and ROC-AUC per fold.
            4. Fit a final model on the full training set.

        Args:
            df: Labelled transaction DataFrame. Must contain label_col.
            label_col: Binary fraud label column (default: "is_fraud").

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: If label_col is absent or contains non-binary values.
        """
        if label_col not in df.columns:
            raise ValueError(f"train: label column '{label_col}' not found in DataFrame.")

        y = df[label_col].to_numpy(dtype=int)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError(f"train: '{label_col}' must be binary (0/1).")

        logger.info(
            "XGBFraudClassifier.train: %d rows, fraud rate=%.3f%%",
            len(df),
            y.mean() * 100,
        )

        X = self.prepare_features(df).to_numpy(dtype=float)

        n_fraud = int(y.sum())
        n_legit = len(y) - n_fraud
        scale_pos_weight = n_legit / max(n_fraud, 1)
        params = {**self.xgb_params, "scale_pos_weight": scale_pos_weight}

        # Cross-validation harness
        self.cv_results = self._run_cv(X, y, params)
        mean_auc_pr = float(np.mean([r["auc_pr"] for r in self.cv_results]))
        mean_roc_auc = float(np.mean([r["roc_auc"] for r in self.cv_results]))
        logger.info(
            "CV complete: mean AUC-PR=%.4f, mean ROC-AUC=%.4f (%d folds)",
            mean_auc_pr,
            mean_roc_auc,
            self.n_cv_folds,
        )

        # Final model on full training data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=EARLY_STOPPING_VAL_FRAC, stratify=y, random_state=self.seed
        )
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self._is_fitted = True
        self._log_audit(
            "train",
            {
                "n_samples": len(df),
                "fraud_rate": float(y.mean()),
                "scale_pos_weight": round(scale_pos_weight, 2),
                "mean_cv_auc_pr": round(mean_auc_pr, 4),
                "mean_cv_roc_auc": round(mean_roc_auc, 4),
                "n_cv_folds": self.n_cv_folds,
            },
        )
        logger.info("XGBFraudClassifier: training complete.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score transactions and return risk assessments.

        Args:
            df: Transaction DataFrame. Must contain the columns required by
                prepare_features(). The label column is not required.

        Returns:
            DataFrame with columns: transaction_id, risk_score, risk_tier,
            model_version, scored_at.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        self._check_fitted()

        X = self.prepare_features(df).to_numpy(dtype=float)
        risk_scores = self.model.predict_proba(X)[:, 1]

        result = pd.DataFrame(
            {
                "transaction_id": (
                    df["transaction_id"].values
                    if "transaction_id" in df.columns
                    else np.arange(len(df))
                ),
                "risk_score": np.round(risk_scores, 6),
                "risk_tier": [self._assign_risk_tier(s) for s in risk_scores],
                "model_version": self.config.version,
                "scored_at": datetime.utcnow().isoformat(),
            }
        )

        self._log_audit(
            "predict",
            {
                "n_scored": len(df),
                "high_risk_count": int((result["risk_tier"].isin(["HIGH", "CRITICAL"])).sum()),
            },
        )
        return result

    def explain(self, df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """Generate SHAP-based feature attributions for scored transactions.

        Uses TreeExplainer for efficient exact SHAP values. Returns the top N
        most influential features per transaction as named reason codes,
        suitable for inclusion in a decision record (SR 11-7 §6).

        Args:
            df: Transaction DataFrame (same schema as predict input).
            top_n: Number of top reason codes to include per transaction.

        Returns:
            DataFrame with columns: transaction_id, shap_values (list),
            and top_reason_1 ... top_reason_N (feature name strings).

        Raises:
            RuntimeError: If the model has not been trained.
        """
        self._check_fitted()

        X_df = self.prepare_features(df)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_df.to_numpy(dtype=float))

        top_reasons = self._extract_top_reasons(shap_values, top_n)

        result = pd.DataFrame(
            {
                "transaction_id": (
                    df["transaction_id"].values
                    if "transaction_id" in df.columns
                    else np.arange(len(df))
                ),
            }
        )
        for i in range(1, top_n + 1):
            result[f"top_reason_{i}"] = [r[i - 1] if len(r) >= i else None for r in top_reasons]

        self._log_audit("explain", {"n_explained": len(df), "top_n": top_n})
        return result

    # ------------------------------------------------------------------
    # Public utility methods
    # ------------------------------------------------------------------

    def cv_summary(self) -> pd.DataFrame:
        """Return cross-validation results as a DataFrame.

        Returns:
            DataFrame with columns: fold, auc_pr, roc_auc, n_train, n_val.

        Raises:
            RuntimeError: If train() has not been called.
        """
        if not self.cv_results:
            raise RuntimeError("cv_summary: no CV results available. Call train() first.")
        return pd.DataFrame(self.cv_results)

    def mean_cv_auc_pr(self) -> float:
        """Return the mean AUC-PR across all CV folds.

        Returns:
            Mean AUC-PR as a float.

        Raises:
            RuntimeError: If train() has not been called.
        """
        if not self.cv_results:
            raise RuntimeError("mean_cv_auc_pr: no CV results. Call train() first.")
        return float(np.mean([r["auc_pr"] for r in self.cv_results]))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> list[dict[str, float]]:
        """Run stratified k-fold cross-validation and return per-fold metrics."""
        skf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.seed)
        results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Early stopping uses a sub-split of the training fold
            X_tr_fit, X_es, y_tr_fit, y_es = train_test_split(
                X_tr,
                y_tr,
                test_size=EARLY_STOPPING_VAL_FRAC,
                stratify=y_tr,
                random_state=self.seed,
            )

            clf = xgb.XGBClassifier(**params)
            clf.fit(
                X_tr_fit,
                y_tr_fit,
                eval_set=[(X_es, y_es)],
                verbose=False,
            )

            y_prob = clf.predict_proba(X_val)[:, 1]
            auc_pr = float(average_precision_score(y_val, y_prob))
            roc_auc = float(roc_auc_score(y_val, y_prob))

            results.append(
                {
                    "fold": fold,
                    "auc_pr": round(auc_pr, 4),
                    "roc_auc": round(roc_auc, 4),
                    "n_train": len(y_tr),
                    "n_val": len(y_val),
                }
            )
            logger.info(
                "  Fold %d/%d: AUC-PR=%.4f, ROC-AUC=%.4f",
                fold,
                self.n_cv_folds,
                auc_pr,
                roc_auc,
            )

        return results

    def _extract_top_reasons(
        self,
        shap_values: np.ndarray,
        top_n: int,
    ) -> list[list[str]]:
        """Extract top N feature names by absolute SHAP value per row."""
        feature_names = self.feature_names
        top_reasons = []
        for row_shap in shap_values:
            ranked_idx = np.argsort(np.abs(row_shap))[::-1][:top_n]
            top_reasons.append([feature_names[i] for i in ranked_idx])
        return top_reasons

    def _assign_risk_tier(self, score: float) -> str:
        """Map risk score to tier using BaseScorer thresholds."""
        from fincrime_ml.core.base import BaseScorer

        return BaseScorer._assign_risk_tier(score)
