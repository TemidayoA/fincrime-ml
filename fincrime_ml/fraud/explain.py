"""
fraud/explain.py
================
SHAP explainability layer for the FinCrime-ML fraud detection pipeline.

Purpose
    Model explainability is a regulatory requirement, not an optional feature.
    FCA expectations under SYSC 6.3, the PRA SS1/23 model risk guidance, and
    the Federal Reserve SR 11-7 letter all require that automated decision
    systems can produce intelligible explanations for individual decisions.
    In a fraud context this means: for any transaction scored as high-risk,
    a reviewer must be able to see which features drove that score.

    This module provides a dedicated explainability layer that operates on
    any fitted sklearn-API classifier and produces:

    1. Raw SHAP values per feature per transaction.
    2. Named reason codes: the top N features ranked by absolute SHAP value,
       formatted as human-readable strings suitable for a fraud analyst.
    3. A summary DataFrame: feature name, mean absolute SHAP value across a
       cohort, and rank — used for model documentation and validation reports.

SHAP method selection
    TreeExplainer: used for tree-based models (XGBoost, LightGBM, RandomForest).
        Exact SHAP values; fast on tree ensembles.
    LinearExplainer: used for linear models (LogisticRegression, LinearSVC).
        Exact SHAP values via analytic solution.
    KernelExplainer: model-agnostic fallback; approximation via sampling.
        Slow; only used when the model type is not recognised.

Regulatory alignment
    SR 11-7 (Federal Reserve / OCC): requires model documentation to include
        a description of model inputs and how they influence outputs (Section IV).
    PRA SS1/23 Section 4.6: model risk management must include documentation
        of model logic sufficient for a senior reviewer to challenge assumptions.
    FCA SYSC 6.3.1R: firms must maintain adequate systems and controls for
        financial crime; explainability supports challenge of automated alerts.

Architecture note
    This module imports only from fincrime_ml.core. No imports from
    fincrime_ml.aml are permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of top reason codes returned per transaction.
DEFAULT_TOP_N: int = 3

#: Number of background samples used when falling back to KernelExplainer.
_KERNEL_BACKGROUND_SAMPLES: int = 50


# ---------------------------------------------------------------------------
# Explainer class
# ---------------------------------------------------------------------------


class FraudExplainer:
    """SHAP-based explainability layer for fraud detection models.

    Wraps a fitted sklearn-API classifier and provides per-transaction
    SHAP values, named reason codes, and cohort-level feature importance
    summaries.

    The explainer auto-selects the SHAP method based on the model type:
        - XGBoost / LightGBM / RandomForest: TreeExplainer
        - LogisticRegression / linear models: LinearExplainer
        - Other: KernelExplainer (slow; sample background required)

    Example::

        from fincrime_ml.fraud.explain import FraudExplainer

        explainer = FraudExplainer(clf.model, feature_names=FEATURE_COLS)
        explainer.fit(X_background)

        reason_codes = explainer.reason_codes(X_scored, top_n=3)
        summary = explainer.feature_summary(X_scored)

    Attributes:
        model: The fitted classifier.
        feature_names: List of feature names matching the model's input columns.
        shap_explainer: The underlying shap.Explainer instance (set after fit()).
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer: Any = None
        self._model_type: str = self._detect_model_type(model)
        logger.info(
            "FraudExplainer initialised: model_type=%s, n_features=%d",
            self._model_type,
            len(feature_names),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_background: np.ndarray | pd.DataFrame | None = None,
    ) -> "FraudExplainer":
        """Initialise the underlying SHAP explainer.

        For TreeExplainer and LinearExplainer no background data is required.
        For KernelExplainer a background dataset must be supplied.

        Args:
            X_background: Background dataset used by KernelExplainer. Ignored
                for TreeExplainer and LinearExplainer. Should be a representative
                sample of legitimate transactions (50-200 rows is sufficient).

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: If model_type is "kernel" and X_background is None.
        """
        if self._model_type == "tree":
            self.shap_explainer = shap.TreeExplainer(self.model)
        elif self._model_type == "linear":
            if X_background is not None:
                bg = self._to_array(X_background)
                self.shap_explainer = shap.LinearExplainer(self.model, bg)
            else:
                self.shap_explainer = shap.LinearExplainer(
                    self.model, shap.maskers.Independent(np.zeros((1, len(self.feature_names))))
                )
        else:
            if X_background is None:
                raise ValueError(
                    "FraudExplainer.fit: KernelExplainer requires X_background. "
                    "Pass a representative sample of background transactions."
                )
            bg = self._to_array(X_background)
            bg_sample = bg[: min(_KERNEL_BACKGROUND_SAMPLES, len(bg))]
            self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)

        logger.info("FraudExplainer.fit: %s explainer initialised.", self._model_type)
        return self

    def shap_values(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Compute SHAP values for each transaction in X.

        Args:
            X: Feature matrix. Shape (n_samples, n_features).

        Returns:
            SHAP value array of shape (n_samples, n_features). Each value
            represents the contribution of that feature to the model's log-odds
            prediction for the positive (fraud) class.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        self._check_fitted()
        X_arr = self._to_array(X)

        if self._model_type == "tree":
            values = self.shap_explainer.shap_values(X_arr)
        elif self._model_type == "linear":
            values = self.shap_explainer.shap_values(X_arr)
        else:
            values = self.shap_explainer.shap_values(X_arr, nsamples=100)

        # TreeExplainer on binary classifiers may return a list [neg_class, pos_class]
        if isinstance(values, list):
            values = values[1]

        return np.array(values)

    def reason_codes(
        self,
        X: np.ndarray | pd.DataFrame,
        transaction_ids: list | np.ndarray | None = None,
        top_n: int = DEFAULT_TOP_N,
    ) -> pd.DataFrame:
        """Return top N reason codes per transaction ranked by absolute SHAP value.

        Each reason code is the feature name of the highest-contributing feature.
        Positive SHAP values indicate the feature pushed the score toward fraud;
        negative values indicate it pushed toward legitimate. Both are included
        in the ranking by absolute value — a large negative SHAP value is still
        an important feature for the decision.

        Args:
            X: Feature matrix. Shape (n_samples, n_features).
            transaction_ids: Optional list of transaction identifiers. If None,
                integer indices are used.
            top_n: Number of top features to include per transaction.

        Returns:
            DataFrame with columns: transaction_id, top_reason_1 ... top_reason_N,
            shap_1 ... shap_N (SHAP value for the corresponding feature),
            direction_1 ... direction_N ("increases_risk" or "reduces_risk").

        Raises:
            RuntimeError: If fit() has not been called.
        """
        self._check_fitted()
        values = self.shap_values(X)
        n = len(values)

        ids = list(transaction_ids) if transaction_ids is not None else list(range(n))

        rows = []
        for i, row_shap in enumerate(values):
            ranked_idx = np.argsort(np.abs(row_shap))[::-1][:top_n]
            row = {"transaction_id": ids[i]}
            for rank, feat_idx in enumerate(ranked_idx, start=1):
                feat_name = self.feature_names[feat_idx]
                shap_val = float(row_shap[feat_idx])
                row[f"top_reason_{rank}"] = feat_name
                row[f"shap_{rank}"] = round(shap_val, 6)
                row[f"direction_{rank}"] = "increases_risk" if shap_val > 0 else "reduces_risk"
            rows.append(row)

        result = pd.DataFrame(rows)
        logger.debug("reason_codes: %d transactions explained, top_n=%d", n, top_n)
        return result

    def feature_summary(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute mean absolute SHAP value per feature across a cohort.

        Used in model validation reports to document which features have the
        most influence on the model's fraud scores across a representative sample.
        Aligns to PRA SS1/23 Section 4.6 requirements for model documentation.

        Args:
            X: Feature matrix. Shape (n_samples, n_features).

        Returns:
            DataFrame with columns: rank, feature, mean_abs_shap. Sorted by
            mean_abs_shap descending.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        self._check_fitted()
        values = self.shap_values(X)

        mean_abs = np.abs(values).mean(axis=0)
        summary = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": np.round(mean_abs, 6),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        summary["rank"] = range(1, len(summary) + 1)
        return summary[["rank", "feature", "mean_abs_shap"]]

    def explain_single(
        self,
        X: np.ndarray | pd.DataFrame,
        idx: int = 0,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:
        """Return a detailed explanation dict for a single transaction.

        Designed for display in an alert review UI or audit trail entry.
        Provides feature name, raw value, SHAP contribution, and direction
        for each of the top N features.

        Args:
            X: Feature matrix containing at least idx + 1 rows.
            idx: Row index of the transaction to explain.
            top_n: Number of features to include.

        Returns:
            Dict with keys: transaction_idx, top_features (list of dicts with
            feature, raw_value, shap_value, direction).

        Raises:
            RuntimeError: If fit() has not been called.
            IndexError: If idx is out of range.
        """
        self._check_fitted()
        X_arr = self._to_array(X)

        if idx >= len(X_arr):
            raise IndexError(
                f"explain_single: idx={idx} is out of range for X with {len(X_arr)} rows."
            )

        single = X_arr[idx : idx + 1]
        row_shap = self.shap_values(single)[0]
        ranked_idx = np.argsort(np.abs(row_shap))[::-1][:top_n]

        top_features = []
        for feat_idx in ranked_idx:
            top_features.append(
                {
                    "feature": self.feature_names[feat_idx],
                    "raw_value": float(X_arr[idx, feat_idx]),
                    "shap_value": round(float(row_shap[feat_idx]), 6),
                    "direction": ("increases_risk" if row_shap[feat_idx] > 0 else "reduces_risk"),
                }
            )

        return {"transaction_idx": idx, "top_features": top_features}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_model_type(self, model: Any) -> str:
        """Detect whether to use tree, linear, or kernel SHAP explainer."""
        class_name = type(model).__name__.lower()
        if any(k in class_name for k in ("xgb", "lgbm", "lightgbm", "forest", "tree", "boost")):
            return "tree"
        if any(k in class_name for k in ("logistic", "linear", "ridge", "lasso", "svc")):
            return "linear"
        return "kernel"

    def _check_fitted(self) -> None:
        if self.shap_explainer is None:
            raise RuntimeError("FraudExplainer has not been fitted. Call .fit() before explaining.")

    @staticmethod
    def _to_array(X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def build_explainer(
    model: Any,
    feature_names: list[str],
    X_background: np.ndarray | pd.DataFrame | None = None,
) -> FraudExplainer:
    """Build and fit a FraudExplainer in one call.

    Args:
        model: Fitted sklearn-API classifier.
        feature_names: Feature names matching the model's input columns.
        X_background: Background data for KernelExplainer (ignored for tree/linear).

    Returns:
        Fitted FraudExplainer instance.
    """
    explainer = FraudExplainer(model=model, feature_names=feature_names)
    explainer.fit(X_background=X_background)
    return explainer


def top_reason_codes(
    model: Any,
    feature_names: list[str],
    X: np.ndarray | pd.DataFrame,
    X_background: np.ndarray | pd.DataFrame | None = None,
    transaction_ids: list | np.ndarray | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """One-shot reason code generation without instantiating FraudExplainer directly.

    Args:
        model: Fitted sklearn-API classifier.
        feature_names: Feature names.
        X: Feature matrix to explain.
        X_background: Background data (for KernelExplainer).
        transaction_ids: Optional transaction identifiers.
        top_n: Number of top reason codes per transaction.

    Returns:
        DataFrame with transaction_id, top_reason_1..N, shap_1..N, direction_1..N.
    """
    explainer = build_explainer(model, feature_names, X_background)
    return explainer.reason_codes(X, transaction_ids=transaction_ids, top_n=top_n)
