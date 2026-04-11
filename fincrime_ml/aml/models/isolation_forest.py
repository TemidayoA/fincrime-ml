"""
aml/models/isolation_forest.py
===============================
Unsupervised AML baseline using Isolation Forest anomaly detection.

Purpose
    Supervised AML models require labelled training data — a scarce resource
    in practice, since SARs are filed for a small, biased sample of suspicious
    activity and the vast majority of transactions are never reviewed. This
    module provides an unsupervised baseline that requires no labels at fit
    time, making it deployable on raw transaction streams from day one.

    Isolation Forest (Liu et al., 2008) isolates anomalies by recursively
    partitioning the feature space with random splits. Anomalous transactions
    require fewer splits to isolate — they occupy sparse regions of the feature
    space. This is well-suited to AML detection, where structuring, rapid
    movement, and unusual-hour transactions are inherently sparse relative to
    the bulk of legitimate activity.

Feature set (transaction level)
    Core: amount_gbp, hour_of_day, day_of_week, layering_depth,
          structuring_flag, rapid_movement_flag
    Derived: log_amount_gbp (log1p transform — stabilises the right-skewed
             amount distribution)
    Optional (used when present): is_mule_sender, is_mule_receiver

    Features are selected automatically from what is available in the input
    DataFrame, so the scorer accepts both SyntheticAMLGenerator output and
    PaySim-harmonised data without schema adjustment.

Score normalisation
    sklearn's IsolationForest.score_samples() returns negative average path
    lengths (more negative = more anomalous). Scores are converted to a [0, 1]
    risk score using min-max normalisation anchored on the training distribution,
    so that the training population median maps to approximately 0.5 and the
    most anomalous training observations approach 1.0. This makes scores
    interpretable as a relative anomaly percentile.

Regulatory alignment
    MLR 2017 Reg 28 requires proportionate customer due diligence. An
    unsupervised baseline addresses the cold-start problem: new transaction
    monitoring programmes can produce ranked anomaly alerts before any SAR
    data is available to train supervised models.

    JMLSG Part I para 5.3.4 recognises that statistical anomaly detection
    is a valid component of a transaction monitoring framework, complementing
    rule-based screens. The ``explain()`` method produces SHAP-based reason
    codes per transaction, satisfying FCA SYSC 10A record-keeping requirements
    for automated decision systems.

    PRA SS1/23 (model risk) requires documentation of model assumptions.
    The unsupervised nature of this model is flagged explicitly in the audit
    log and docstrings — downstream users must validate against labelled data
    before relying solely on this scorer.

Architecture note
    Imports only from fincrime_ml.core. No imports from fincrime_ml.fraud
    permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from fincrime_ml.core.base import BasePipeline, PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

#: Core transaction-level features always used when present.
CORE_FEATURE_COLS: list[str] = [
    "amount_gbp",
    "log_amount_gbp",  # derived inside prepare_features
    "hour_of_day",
    "day_of_week",
    "layering_depth",
    "structuring_flag",
    "rapid_movement_flag",
]

#: Optional features included only when the column exists in the input DataFrame.
OPTIONAL_FEATURE_COLS: list[str] = [
    "is_mule_sender",
    "is_mule_receiver",
]

#: Risk tier thresholds (consistent with GraphScorer and BaseScorer conventions).
_RISK_TIERS: list[tuple[float, str]] = [
    (0.85, "CRITICAL"),
    (0.65, "HIGH"),
    (0.30, "MEDIUM"),
    (0.0, "LOW"),
]


def _assign_risk_tier(score: float) -> str:
    """Map a [0, 1] anomaly score to a categorical risk tier.

    Args:
        score: Normalised anomaly score in [0, 1].

    Returns:
        Risk tier string: CRITICAL, HIGH, MEDIUM, or LOW.
    """
    for threshold, tier in _RISK_TIERS:
        if score >= threshold:
            return tier
    return "LOW"


# ---------------------------------------------------------------------------
# AMLIsolationForest
# ---------------------------------------------------------------------------


class AMLIsolationForest(BasePipeline):
    """Unsupervised AML transaction anomaly detector using Isolation Forest.

    Fits on unlabelled transaction data and scores each transaction by how
    anomalous it is relative to the observed population. No ``is_suspicious``
    labels are required for training, making this the recommended baseline
    for cold-start AML deployments and for benchmarking supervised models.

    When ground-truth labels are available, pass them to ``evaluate()`` to
    compute AUC-PR and ROC-AUC against the unsupervised scores.

    Label convention (AML): ``is_suspicious`` (not ``is_fraud``). The
    ``train()`` method accepts an optional ``label_col`` argument for
    interface compatibility with BasePipeline, but labels are never used
    during fitting.

    Example::

        from fincrime_ml.aml.models.isolation_forest import AMLIsolationForest
        from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

        gen = SyntheticAMLGenerator(n_accounts=1000, seed=42)
        df = gen.generate(n_transactions=10_000, suspicious_rate=0.05)

        model = AMLIsolationForest()
        model.train(df)          # no labels needed

        scores = model.predict(df)
        explanations = model.explain(df)

        # Optional evaluation against labels
        metrics = model.evaluate(df, label_col="is_suspicious")

    Attributes:
        config: PipelineConfig instance.
        model: Fitted sklearn IsolationForest (None before train()).
        feature_names: Ordered list of feature columns used for training.
        n_estimators: Number of isolation trees.
        contamination: Expected proportion of anomalies (used for threshold).
        max_features: Feature subsampling per tree.
    """

    LABEL_COL: str = "is_suspicious"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        n_estimators: int = 200,
        contamination: float | str = "auto",
        max_features: float = 1.0,
        max_samples: int | str = "auto",
    ) -> None:
        super().__init__(config)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.max_samples = max_samples

        # Score normalisation anchors fitted from training data
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._shap_explainer: Any = None

    # ------------------------------------------------------------------
    # Public API — BasePipeline implementation
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer transaction-level features for anomaly detection.

        Computes ``log_amount_gbp`` (log1p transform of amount_gbp) and selects
        all available core and optional feature columns from the input DataFrame.
        Missing optional columns are silently omitted; missing core columns raise
        a KeyError.

        Args:
            df: Transaction DataFrame. Must contain ``amount_gbp``,
                ``hour_of_day``, ``day_of_week``, ``layering_depth``,
                ``structuring_flag``, and ``rapid_movement_flag``.

        Returns:
            Feature DataFrame with one row per transaction and columns matching
            those stored in ``feature_names`` after training.

        Raises:
            KeyError: If any required core column (except ``log_amount_gbp``,
                which is derived) is absent from df.
        """
        required_input_cols = [c for c in CORE_FEATURE_COLS if c != "log_amount_gbp"]
        missing = [c for c in required_input_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"AMLIsolationForest.prepare_features: required columns missing: {missing}"
            )

        feat = df[required_input_cols].copy()
        feat["log_amount_gbp"] = np.log1p(
            pd.to_numeric(df["amount_gbp"], errors="coerce").fillna(0.0)
        )

        for opt_col in OPTIONAL_FEATURE_COLS:
            if opt_col in df.columns:
                feat[opt_col] = df[opt_col].fillna(0)

        # Enforce consistent column ordering
        ordered = [c for c in CORE_FEATURE_COLS if c in feat.columns] + [
            c for c in OPTIONAL_FEATURE_COLS if c in feat.columns
        ]
        return feat[ordered].reset_index(drop=True)

    def train(
        self,
        df: pd.DataFrame,
        label_col: str = "is_suspicious",
    ) -> "AMLIsolationForest":
        """Fit the Isolation Forest on unlabelled transaction data.

        Labels are not used during fitting. The ``label_col`` argument exists
        solely for interface compatibility with BasePipeline and is ignored.
        This design is intentional: the unsupervised nature of this model means
        it can be deployed without any prior SAR data (PRA SS1/23 cold-start
        scenario).

        Score normalisation anchors (min/max of training anomaly scores) are
        fitted here and applied in ``predict()`` to produce consistent [0, 1]
        risk scores.

        Args:
            df: Transaction DataFrame (labelled or unlabelled — labels ignored).
            label_col: Ignored. Present for BasePipeline interface compatibility.

        Returns:
            Self (for method chaining).

        Raises:
            KeyError: If required feature columns are absent from df.
            ValueError: If df has fewer than 2 rows.
        """
        if len(df) < 2:
            raise ValueError("AMLIsolationForest.train: DataFrame must have at least 2 rows.")

        feat_df = self.prepare_features(df)
        self.feature_names = list(feat_df.columns)

        x_train = feat_df.values

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            max_samples=self.max_samples,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(x_train)
        self._is_fitted = True

        # Fit score normalisation anchors on training data
        raw_scores = self.model.score_samples(x_train)
        # score_samples returns negative anomaly score: more negative = more anomalous
        # Invert so higher = more anomalous, then store normalisation anchors
        inverted = -raw_scores
        self._score_min = float(inverted.min())
        self._score_max = float(inverted.max())

        # Pre-compute SHAP TreeExplainer
        self._shap_explainer = shap.TreeExplainer(self.model)

        self._log_audit(
            "train",
            {
                "n_transactions": len(df),
                "n_features": len(self.feature_names),
                "n_estimators": self.n_estimators,
                "contamination": str(self.contamination),
                "supervised": False,
                "label_col_ignored": label_col,
            },
        )
        logger.info(
            "AMLIsolationForest.train: %d transactions, %d features, unsupervised",
            len(df),
            len(self.feature_names),
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score transactions by anomaly level.

        Applies the fitted Isolation Forest and normalises scores to [0, 1]
        using min-max anchors from the training distribution. Scores near 1.0
        indicate highly anomalous transactions; scores near 0.0 indicate
        typical behaviour.

        Args:
            df: Transaction DataFrame (labelled or unlabelled).

        Returns:
            DataFrame with one row per transaction, sorted descending by
            risk_score. Columns: transaction_id (if present in df), risk_score,
            risk_tier, model_version, scored_at.

        Raises:
            RuntimeError: If train() has not been called.
        """
        self._check_fitted()

        feat_df = self.prepare_features(df)
        x_score = feat_df.values

        raw_scores = self.model.score_samples(x_score)
        inverted = -raw_scores

        score_range = self._score_max - self._score_min
        if score_range > 0:
            risk_scores = np.clip((inverted - self._score_min) / score_range, 0.0, 1.0)
        else:
            risk_scores = np.zeros(len(df))

        n_high = int(np.sum(risk_scores >= 0.65))

        output: dict = {
            "risk_score": np.round(risk_scores, 4),
            "risk_tier": [_assign_risk_tier(float(s)) for s in risk_scores],
            "model_version": self.config.version,
            "scored_at": datetime.utcnow().isoformat(),
        }

        if "transaction_id" in df.columns:
            output["transaction_id"] = df["transaction_id"].values

        results = pd.DataFrame(output)

        # Move transaction_id to front if present
        if "transaction_id" in results.columns:
            cols = ["transaction_id"] + [c for c in results.columns if c != "transaction_id"]
            results = results[cols]

        results = results.sort_values("risk_score", ascending=False).reset_index(drop=True)

        self._log_audit(
            "predict",
            {
                "n_transactions_scored": len(results),
                "n_high_risk": n_high,
                "n_critical": int((results["risk_tier"] == "CRITICAL").sum()),
            },
        )
        logger.info(
            "AMLIsolationForest.predict: %d transactions scored, %d HIGH or above",
            len(results),
            n_high,
        )
        return results

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate SHAP-based feature explanations for each scored transaction.

        Computes SHAP values using the pre-fitted TreeExplainer. Returns per-
        transaction SHAP values and the top-3 reason codes ranked by absolute
        SHAP magnitude. For IsolationForest, SHAP values represent each
        feature's contribution to the anomaly score — positive values push
        the transaction toward anomalous, negative toward normal.

        This output satisfies FCA SYSC 10A record-keeping requirements for
        automated decision systems and supports MLRO review under MLR 2017.

        Args:
            df: Transaction DataFrame (same data passed to predict()).

        Returns:
            DataFrame with columns: transaction_id (if present), shap_{feature}
            for each feature, top_reason_1..3, top_shap_1..3.

        Raises:
            RuntimeError: If train() has not been called.
        """
        self._check_fitted()

        feat_df = self.prepare_features(df)
        x_explain = feat_df.values

        raw_shap = self._shap_explainer.shap_values(x_explain)
        # IsolationForest SHAP: single output — shape (n_samples, n_features)
        # or 3D array depending on SHAP version
        shap_arr = np.array(raw_shap)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 0]

        shap_cols = [f"shap_{f}" for f in self.feature_names]
        shap_df = pd.DataFrame(shap_arr, columns=shap_cols)

        if "transaction_id" in df.columns:
            shap_df.insert(0, "transaction_id", df["transaction_id"].values)

        # Top-3 reason codes by absolute SHAP magnitude
        abs_shap = np.abs(shap_arr)
        top_idx = np.argsort(-abs_shap, axis=1)[:, :3]

        for rank in range(3):
            shap_df[f"top_reason_{rank + 1}"] = [
                self.feature_names[top_idx[i, rank]] for i in range(len(top_idx))
            ]
            shap_df[f"top_shap_{rank + 1}"] = [
                round(float(shap_arr[i, top_idx[i, rank]]), 4) for i in range(len(top_idx))
            ]

        self._log_audit("explain", {"n_transactions_explained": len(shap_df)})
        logger.info("AMLIsolationForest.explain: %d transactions explained", len(shap_df))
        return shap_df

    def evaluate(
        self,
        df: pd.DataFrame,
        label_col: str = "is_suspicious",
    ) -> dict[str, float]:
        """Evaluate unsupervised anomaly scores against ground-truth labels.

        Provides post-hoc evaluation when labels are available. Computes
        AUC-PR (primary) and ROC-AUC (supplementary) against the normalised
        anomaly scores. This is the recommended approach for validating an
        unsupervised baseline before deploying a supervised replacement (PRA
        SS1/23 champion/challenger framework).

        Args:
            df: Labelled transaction DataFrame.
            label_col: Binary target column. Defaults to ``is_suspicious``.

        Returns:
            Dict with keys ``auc_pr`` and ``roc_auc``.

        Raises:
            RuntimeError: If train() has not been called.
            KeyError: If label_col is absent from df.
        """
        self._check_fitted()
        if label_col not in df.columns:
            raise KeyError(
                f"AMLIsolationForest.evaluate: label column '{label_col}' not in DataFrame."
            )

        from sklearn.metrics import average_precision_score, roc_auc_score

        scores = self.predict(df)["risk_score"].values
        labels = df[label_col].values

        auc_pr = float(average_precision_score(labels, scores))
        roc_auc = float(roc_auc_score(labels, scores))

        self._log_audit(
            "evaluate",
            {"auc_pr": round(auc_pr, 4), "roc_auc": round(roc_auc, 4), "label_col": label_col},
        )
        logger.info(
            "AMLIsolationForest.evaluate: AUC-PR=%.4f, ROC-AUC=%.4f",
            auc_pr,
            roc_auc,
        )
        return {"auc_pr": auc_pr, "roc_auc": roc_auc}
