"""
fraud/models/logistic_baseline.py
===================================
Logistic regression baseline for fraud detection.

Purpose
    Every production ML system should maintain an interpretable baseline
    model alongside more complex approaches. Logistic regression serves
    three functions in FinCrime-ML:

    1. Performance floor: if the XGBoost classifier cannot beat logistic
       regression on AUC-PR, the feature engineering or data quality is
       the bottleneck, not the model architecture.

    2. Regulatory interpretability: logistic regression coefficients provide
       a direct, auditable link between input features and score. This aligns
       to PRA SS1/23 expectations for model risk documentation and SR 11-7
       guidance on model transparency.

    3. Feature importance comparison: coefficient magnitudes (standardised)
       can be compared against XGBoost SHAP values to validate that both
       models agree on which features drive fraud risk. Disagreement
       signals potential overfitting or data leakage in the complex model.

Model design
    - Solver: lbfgs (supports L2 regularisation, stable on small datasets)
    - class_weight: "balanced" for built-in cost-sensitive handling
    - Features are standardised via StandardScaler before fitting, which
      is required for interpretable coefficient comparison
    - Cross-validation uses the same StratifiedKFold harness as XGBFraudClassifier
      to ensure comparable AUC-PR estimates

Architecture note
    Inherits from BasePipeline. No imports from fincrime_ml.aml (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from fincrime_ml.core.base import BasePipeline, PipelineConfig
from fincrime_ml.fraud.features import FraudFeatureEngineer
from fincrime_ml.fraud.models.xgb_classifier import FEATURE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_LR_PARAMS: dict[str, Any] = {
    "penalty": "l2",
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}

_HOLDOUT_FRAC: float = 0.15


class LogisticFraudBaseline(BasePipeline):
    """Logistic regression baseline fraud classifier.

    Inherits from BasePipeline and implements the full pipeline interface.
    Features are standardised before fitting; coefficients are stored and
    exposed for interpretability and comparison against XGBoost SHAP values.

    Example::

        from fincrime_ml.fraud.models.logistic_baseline import LogisticFraudBaseline

        baseline = LogisticFraudBaseline(n_cv_folds=5, seed=42)
        baseline.train(df_train)

        scores = baseline.predict(df_holdout)
        importance = baseline.feature_importance()
        print(importance.head(10))

    Attributes:
        n_cv_folds: Number of stratified CV folds.
        seed: Random seed.
        lr_params: LogisticRegression parameters (merged with defaults).
        scaler: Fitted StandardScaler instance (available after train()).
        cv_results: Per-fold CV metrics (available after train()).
    """

    def __init__(
        self,
        n_cv_folds: int = 5,
        seed: int = 42,
        lr_params: dict[str, Any] | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        self.n_cv_folds = n_cv_folds
        self.seed = seed
        self.lr_params: dict[str, Any] = {**DEFAULT_LR_PARAMS, **(lr_params or {})}
        self.lr_params["random_state"] = seed
        self.scaler: StandardScaler = StandardScaler()
        self.cv_results: list[dict[str, float]] = []
        self.feature_engineer = FraudFeatureEngineer()
        self.feature_names: list[str] = FEATURE_COLS

    # ------------------------------------------------------------------
    # BasePipeline interface
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply FraudFeatureEngineer and select FEATURE_COLS.

        Uses the same feature set as XGBFraudClassifier to ensure
        a fair comparison between the two models.

        Args:
            df: Raw transaction DataFrame in FinCrime-ML schema.

        Returns:
            DataFrame with columns matching FEATURE_COLS.

        Raises:
            ValueError: If required columns are absent after engineering.
        """
        enriched = self.feature_engineer.transform(df)
        missing = [c for c in FEATURE_COLS if c not in enriched.columns]
        if missing:
            raise ValueError(
                f"prepare_features: feature columns missing after engineering: {missing}"
            )
        return enriched[FEATURE_COLS]

    def train(self, df: pd.DataFrame, label_col: str = "is_fraud") -> "LogisticFraudBaseline":
        """Fit the logistic regression baseline with stratified CV.

        Features are standardised via StandardScaler before fitting. The
        scaler is fitted on the training data only; it is applied to
        validation/test data at predict time using transform (not fit_transform).

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
            "LogisticFraudBaseline.train: %d rows, fraud rate=%.3f%%",
            len(df),
            y.mean() * 100,
        )

        X_raw = self.prepare_features(df).to_numpy(dtype=float)

        self.cv_results = self._run_cv(X_raw, y)
        mean_auc_pr = float(np.mean([r["auc_pr"] for r in self.cv_results]))
        mean_roc_auc = float(np.mean([r["roc_auc"] for r in self.cv_results]))
        logger.info(
            "CV complete: mean AUC-PR=%.4f, mean ROC-AUC=%.4f (%d folds)",
            mean_auc_pr,
            mean_roc_auc,
            self.n_cv_folds,
        )

        # Final model: fit scaler and model on full training data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_raw, y, test_size=_HOLDOUT_FRAC, stratify=y, random_state=self.seed
        )
        X_scaled = self.scaler.fit_transform(X_tr)
        self.model = LogisticRegression(**self.lr_params)
        self.model.fit(X_scaled, y_tr)

        self._is_fitted = True
        self._log_audit(
            "train",
            {
                "n_samples": len(df),
                "fraud_rate": float(y.mean()),
                "mean_cv_auc_pr": round(mean_auc_pr, 4),
                "mean_cv_roc_auc": round(mean_roc_auc, 4),
                "n_cv_folds": self.n_cv_folds,
            },
        )
        logger.info("LogisticFraudBaseline: training complete.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score transactions using the fitted logistic regression model.

        Args:
            df: Transaction DataFrame in FinCrime-ML schema.

        Returns:
            DataFrame with columns: transaction_id, risk_score, risk_tier,
            model_version, scored_at.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        self._check_fitted()

        X_raw = self.prepare_features(df).to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X_raw)
        risk_scores = self.model.predict_proba(X_scaled)[:, 1]

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

        self._log_audit("predict", {"n_scored": len(df)})
        return result

    def explain(self, df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """Return top contributing features per transaction using coefficients.

        For logistic regression, feature importance is the absolute value of
        the standardised coefficient. This method returns the top N features
        per transaction ranked by coefficient magnitude, which is constant
        across all transactions (unlike SHAP values which vary per row).

        Args:
            df: Transaction DataFrame.
            top_n: Number of top reason codes to include.

        Returns:
            DataFrame with transaction_id and top_reason_1 ... top_reason_N.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        self._check_fitted()

        coefficients = self.model.coef_[0]
        ranked_idx = np.argsort(np.abs(coefficients))[::-1][:top_n]
        top_features = [self.feature_names[i] for i in ranked_idx]

        n = len(df)
        result = pd.DataFrame(
            {
                "transaction_id": (
                    df["transaction_id"].values if "transaction_id" in df.columns else np.arange(n)
                ),
            }
        )
        for i, feat in enumerate(top_features, start=1):
            result[f"top_reason_{i}"] = feat

        self._log_audit("explain", {"n_explained": n, "top_n": top_n})
        return result

    # ------------------------------------------------------------------
    # Public utility methods
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of features ranked by standardised coefficient magnitude.

        Provides the primary interpretability output for the baseline model.
        Suitable for inclusion in model documentation (PRA SS1/23) and for
        comparison against XGBoost SHAP feature importances.

        Returns:
            DataFrame with columns: feature, coefficient, abs_coefficient,
            rank. Sorted by abs_coefficient descending.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        self._check_fitted()
        coefficients = self.model.coef_[0]
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": np.round(coefficients, 6),
                "abs_coefficient": np.round(np.abs(coefficients), 6),
            }
        ).sort_values("abs_coefficient", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)
        return importance_df.reset_index(drop=True)

    def compare_with_xgb(self, xgb_clf: Any) -> pd.DataFrame:
        """Compare feature importance rankings between this baseline and an XGBoost model.

        Useful for detecting disagreements that could indicate overfitting or
        data leakage in the more complex model. Both models must be fitted and
        must share the same feature set (FEATURE_COLS).

        Args:
            xgb_clf: A fitted XGBFraudClassifier instance.

        Returns:
            DataFrame with columns: feature, lr_rank, xgb_rank, rank_delta.
            Sorted by lr_rank ascending.

        Raises:
            RuntimeError: If either model has not been trained.
            ValueError: If xgb_clf does not have feature_importances_ available.
        """
        self._check_fitted()


        if not hasattr(xgb_clf, "model") or xgb_clf.model is None:
            raise ValueError("compare_with_xgb: xgb_clf has no fitted model.")

        lr_importance = self.feature_importance()[["feature", "rank"]].rename(
            columns={"rank": "lr_rank"}
        )

        xgb_importances = xgb_clf.model.feature_importances_
        xgb_df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "xgb_importance": xgb_importances,
                }
            )
            .sort_values("xgb_importance", ascending=False)
            .reset_index(drop=True)
        )
        xgb_df["xgb_rank"] = range(1, len(xgb_df) + 1)

        comparison = lr_importance.merge(xgb_df[["feature", "xgb_rank"]], on="feature")
        comparison["rank_delta"] = (comparison["lr_rank"] - comparison["xgb_rank"]).abs()
        return comparison.sort_values("lr_rank").reset_index(drop=True)

    def cv_summary(self) -> pd.DataFrame:
        """Return cross-validation results as a DataFrame.

        Raises:
            RuntimeError: If train() has not been called.
        """
        if not self.cv_results:
            raise RuntimeError("cv_summary: no CV results available. Call train() first.")
        return pd.DataFrame(self.cv_results)

    def mean_cv_auc_pr(self) -> float:
        """Return the mean AUC-PR across all CV folds.

        Raises:
            RuntimeError: If train() has not been called.
        """
        if not self.cv_results:
            raise RuntimeError("mean_cv_auc_pr: no CV results. Call train() first.")
        return float(np.mean([r["auc_pr"] for r in self.cv_results]))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_cv(self, X: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
        """Run stratified k-fold CV with StandardScaler fitted per fold."""
        skf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.seed)
        results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            clf = LogisticRegression(**self.lr_params)
            clf.fit(X_tr_scaled, y_tr)

            y_prob = clf.predict_proba(X_val_scaled)[:, 1]
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

    def _assign_risk_tier(self, score: float) -> str:
        from fincrime_ml.core.base import BaseScorer

        return BaseScorer._assign_risk_tier(score)
