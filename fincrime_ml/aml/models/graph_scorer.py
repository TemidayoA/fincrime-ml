"""
aml/models/graph_scorer.py
==========================
Graph-based AML anomaly scorer using centrality and flow deviation features.

Purpose
    Transaction-level rule engines and ML classifiers catch isolated suspicious
    events, but miss the structural patterns that reveal money laundering at the
    network level. This scorer operates on the entity graph produced by
    TransactionGraphBuilder: it extracts per-node features (betweenness
    centrality, PageRank, pass-through ratio, flow volumes) and computes
    population-deviation z-scores that flag accounts behaving anomalously
    relative to the full observed network.

    A supervised RandomForestClassifier is fitted on these graph features with
    node-level suspicion labels derived from transaction-level ``is_suspicious``
    annotations. At prediction time the same feature pipeline is applied and
    risk scores are returned per account node.

Feature set
    Raw graph metrics (from TransactionGraphBuilder.node_features()):
        in_degree, out_degree, total_inflow_gbp, total_outflow_gbp,
        net_position_gbp, pass_through_ratio, betweenness_centrality,
        pagerank, in_txn_count, out_txn_count

    Population deviation z-scores (computed relative to training population):
        betweenness_zscore, pagerank_zscore, pass_through_zscore,
        inflow_zscore, outflow_zscore

Regulatory alignment
    FATF Recommendation R.10 requires firms to understand the nature and
    purpose of business relationships. Betweenness centrality and pass-through
    ratio directly operationalise this by identifying bridge accounts that
    channel funds for others — a structural signal for layering.

    MLR 2017 Reg 28 requires Enhanced Due Diligence (EDD) for higher-risk
    customers. The risk tier output (LOW / MEDIUM / HIGH / CRITICAL) maps
    directly to EDD trigger thresholds for MLRO review.

    FCA SYSC 10A requires documented records of automated decisions. The
    ``explain()`` method produces SHAP-based reason codes for each scored
    node, providing an auditable rationale suitable for SAR documentation
    under POCA 2002 s.330.

Architecture note
    Imports only from fincrime_ml.aml and fincrime_ml.core. No imports from
    fincrime_ml.fraud permitted (ADR 001).

Author: Temidayo Akindahunsi
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from fincrime_ml.aml.graph import TransactionGraphBuilder
from fincrime_ml.core.base import BasePipeline, PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column specification
# ---------------------------------------------------------------------------

# Raw graph metrics produced by TransactionGraphBuilder.node_features()
RAW_GRAPH_FEATURES: list[str] = [
    "in_degree",
    "out_degree",
    "total_inflow_gbp",
    "total_outflow_gbp",
    "net_position_gbp",
    "pass_through_ratio",
    "betweenness_centrality",
    "pagerank",
    "in_txn_count",
    "out_txn_count",
]

# Z-score deviation features derived from raw graph metrics
DEVIATION_FEATURES: list[str] = [
    "betweenness_zscore",
    "pagerank_zscore",
    "pass_through_zscore",
    "inflow_zscore",
    "outflow_zscore",
]

# Mapping from z-score name to its source raw feature
_ZSCORE_SOURCE_MAP: dict[str, str] = {
    "betweenness_zscore": "betweenness_centrality",
    "pagerank_zscore": "pagerank",
    "pass_through_zscore": "pass_through_ratio",
    "inflow_zscore": "total_inflow_gbp",
    "outflow_zscore": "total_outflow_gbp",
}

ALL_FEATURE_COLS: list[str] = RAW_GRAPH_FEATURES + DEVIATION_FEATURES

# Risk tier thresholds — aligned to UK bank alert escalation conventions
_RISK_TIERS: list[tuple[float, str]] = [
    (0.85, "CRITICAL"),
    (0.65, "HIGH"),
    (0.30, "MEDIUM"),
    (0.0, "LOW"),
]


def _assign_risk_tier(score: float) -> str:
    """Map a continuous [0, 1] risk score to a categorical risk tier.

    Tier boundaries are calibrated to UK bank SAR escalation thresholds:
        CRITICAL (>=0.85): Immediate MLRO referral warranted.
        HIGH     (>=0.65): Enhanced monitoring, EDD trigger.
        MEDIUM   (>=0.30): Standard monitoring uplift.
        LOW      (<0.30):  Routine activity.

    Args:
        score: Continuous risk score in [0, 1].

    Returns:
        Risk tier label string.
    """
    for threshold, tier in _RISK_TIERS:
        if score >= threshold:
            return tier
    return "LOW"


# ---------------------------------------------------------------------------
# GraphScorer
# ---------------------------------------------------------------------------


class GraphScorer(BasePipeline):
    """AML graph-based anomaly scorer using centrality and flow deviation features.

    Combines node-level graph metrics from a transaction network with
    population-deviation z-scores to score each account's AML risk.
    A supervised RandomForestClassifier is trained on labelled transaction
    data; at inference time the same graph feature extraction is applied to
    new transaction batches.

    Label convention: AML pipelines use ``is_suspicious`` (not ``is_fraud``,
    which is reserved for the fraud domain per ADR 001).

    Example::

        from fincrime_ml.aml.models.graph_scorer import GraphScorer
        from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

        gen = SyntheticAMLGenerator(n_accounts=1000, seed=42)
        df = gen.generate(n_transactions=10_000, suspicious_rate=0.05)

        scorer = GraphScorer()
        scorer.train(df, label_col="is_suspicious")

        scores = scorer.predict(df)
        explanations = scorer.explain(df)

    Attributes:
        config: PipelineConfig instance.
        model: Fitted RandomForestClassifier (None before train()).
        feature_names: Ordered list of feature columns used for training.
        n_estimators: Number of trees in the RandomForest.
        max_depth: Maximum tree depth (None = unlimited).
        sender_col: Transaction DataFrame column for the sending account.
        receiver_col: Transaction DataFrame column for the receiving account.
    """

    LABEL_COL: str = "is_suspicious"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        n_estimators: int = 200,
        max_depth: int | None = 10,
        sender_col: str = "sender_account_id",
        receiver_col: str = "receiver_account_id",
    ) -> None:
        super().__init__(config)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sender_col = sender_col
        self.receiver_col = receiver_col

        self._graph_builder = TransactionGraphBuilder(
            sender_col=sender_col,
            receiver_col=receiver_col,
        )
        # Population statistics fitted during train() for consistent z-scores
        self._population_stats: dict[str, tuple[float, float]] = {}
        self._shap_explainer: Any = None

    # ------------------------------------------------------------------
    # Public API — BasePipeline implementation
    # ------------------------------------------------------------------

    def prepare_features(
        self,
        df: pd.DataFrame,
        use_fitted_stats: bool = False,
    ) -> pd.DataFrame:
        """Build transaction graph and extract node-level feature matrix.

        Constructs the transaction network via TransactionGraphBuilder, extracts
        raw centrality and flow metrics per node, then appends population-relative
        z-score deviation features.

        When ``use_fitted_stats=True`` (inference mode), z-scores are computed
        using mean/std stored from the training population, making scores
        comparable across batches. When False (training mode), statistics are
        derived from the current data and stored for later inference calls.

        Args:
            df: Transaction DataFrame with sender_col and receiver_col columns.
            use_fitted_stats: If True, apply stored training population statistics
                for z-score computation. Set True at inference time.

        Returns:
            Node-level DataFrame with one row per unique account and columns
            covering all features in ALL_FEATURE_COLS, plus ``node_id``
            and ``is_mule``.

        Raises:
            KeyError: If required columns are absent from df.
        """
        self._graph_builder.build(df)
        node_df = self._graph_builder.node_features()

        for zscore_col, source_col in _ZSCORE_SOURCE_MAP.items():
            col_vals = node_df[source_col].values.astype(float)

            if use_fitted_stats and source_col in self._population_stats:
                mean_val, std_val = self._population_stats[source_col]
            else:
                mean_val = float(np.mean(col_vals))
                std_val = float(np.std(col_vals))
                self._population_stats[source_col] = (mean_val, std_val)

            node_df[zscore_col] = (col_vals - mean_val) / std_val if std_val > 0 else 0.0

        return node_df

    def train(
        self,
        df: pd.DataFrame,
        label_col: str = "is_suspicious",
    ) -> "GraphScorer":
        """Train the graph-based AML scorer on labelled transaction data.

        Workflow:
            1. Build the transaction network and extract node features.
            2. Aggregate transaction-level labels to node level: a node is
               labelled suspicious if it appears as sender or receiver in any
               transaction with ``label_col == 1``.
            3. Fit a RandomForestClassifier with inverse-frequency class weighting
               to handle the typical 2–8% AML suspicious rate.
            4. Pre-compute a SHAP TreeExplainer for the fitted model.

        Primary metric is AUC-PR (average_precision_score), consistent with the
        FinCrime-ML convention for imbalanced financial crime datasets.

        Args:
            df: Labelled transaction DataFrame.
            label_col: Binary target column (1 = suspicious). Defaults to
                ``is_suspicious``.

        Returns:
            Self (for method chaining).

        Raises:
            KeyError: If label_col is absent from df.
            ValueError: If no positive labels are found after node aggregation.
        """
        if label_col not in df.columns:
            raise KeyError(f"GraphScorer.train: label column '{label_col}' not found in DataFrame.")

        node_df = self.prepare_features(df, use_fitted_stats=False)

        # Aggregate is_suspicious labels to node level: suspicious if ever
        # involved (as sender or receiver) in a flagged transaction
        sender_labels = df.groupby(self.sender_col)[label_col].max()
        sender_labels.index.name = "node_id"
        receiver_labels = df.groupby(self.receiver_col)[label_col].max()
        receiver_labels.index.name = "node_id"

        node_labels = (
            pd.concat([sender_labels, receiver_labels])
            .groupby(level=0)
            .max()
            .rename("is_suspicious")
            .reset_index()
        )

        node_df = node_df.merge(node_labels, on="node_id", how="left")
        node_df["is_suspicious"] = node_df["is_suspicious"].fillna(0).astype(int)

        n_pos = int(node_df["is_suspicious"].sum())
        if n_pos == 0:
            raise ValueError(
                "GraphScorer.train: no suspicious nodes found after label aggregation. "
                "Verify that the input DataFrame contains positive is_suspicious labels."
            )

        feature_cols = [c for c in ALL_FEATURE_COLS if c in node_df.columns]
        self.feature_names = feature_cols

        x_train = node_df[feature_cols].values
        y = node_df["is_suspicious"].values

        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight={0: 1.0, 1: scale_pos_weight},
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(x_train, y)
        self._is_fitted = True

        self._shap_explainer = shap.TreeExplainer(self.model)

        self._log_audit(
            "train",
            {
                "n_nodes": len(node_df),
                "n_suspicious_nodes": n_pos,
                "suspicious_rate_pct": round(100 * n_pos / len(node_df), 2),
                "n_features": len(feature_cols),
                "label_col": label_col,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
            },
        )
        logger.info(
            "GraphScorer.train: %d nodes, %d suspicious (%.1f%%), %d features",
            len(node_df),
            n_pos,
            100 * n_pos / len(node_df),
            len(feature_cols),
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all account nodes in the transaction graph.

        Builds the transaction network from df, extracts graph features using
        training-population z-score statistics, and returns a risk score for
        each unique account node.

        Args:
            df: Transaction DataFrame. May be labelled or unlabelled; any
                label column is ignored at scoring time.

        Returns:
            DataFrame with one row per account node, sorted descending by
            risk_score. Columns: node_id, risk_score, risk_tier,
            model_version, scored_at.

        Raises:
            RuntimeError: If train() has not been called.
        """
        self._check_fitted()

        node_df = self.prepare_features(df, use_fitted_stats=True)
        x_score = node_df[self.feature_names].values

        risk_scores = self.model.predict_proba(x_score)[:, 1]

        n_high = int(np.sum(risk_scores >= 0.65))

        results = (
            pd.DataFrame(
                {
                    "node_id": node_df["node_id"].values,
                    "risk_score": np.round(risk_scores, 4),
                    "risk_tier": [_assign_risk_tier(float(s)) for s in risk_scores],
                    "model_version": self.config.version,
                    "scored_at": datetime.utcnow().isoformat(),
                }
            )
            .sort_values("risk_score", ascending=False)
            .reset_index(drop=True)
        )

        self._log_audit(
            "predict",
            {
                "n_nodes_scored": len(results),
                "n_high_risk": n_high,
                "n_critical": int((results["risk_tier"] == "CRITICAL").sum()),
            },
        )
        logger.info(
            "GraphScorer.predict: %d nodes scored, %d flagged HIGH or above",
            len(results),
            n_high,
        )
        return results

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate SHAP-based feature explanations for each scored node.

        Computes SHAP values for the suspicious-class probability using the
        pre-fitted TreeExplainer. Returns per-node SHAP values alongside the
        top-3 reason codes ranked by absolute SHAP magnitude.

        This output supports MLRO review obligations under MLR 2017 Reg 28
        and FCA SYSC 10A audit record-keeping for automated decision systems.

        Args:
            df: Transaction DataFrame (same data passed to predict()).

        Returns:
            DataFrame with columns: node_id, shap_{feature} for each feature,
            top_reason_1..3 (feature name), top_shap_1..3 (signed SHAP value).

        Raises:
            RuntimeError: If train() has not been called.
        """
        self._check_fitted()

        node_df = self.prepare_features(df, use_fitted_stats=True)
        x_explain = node_df[self.feature_names].values

        raw_shap = self._shap_explainer.shap_values(x_explain)
        # SHAP API varies by version:
        #   >=0.46: returns ndarray of shape (n_samples, n_features, n_classes)
        #   <0.46:  returns list [class_0_arr, class_1_arr]
        shap_raw = np.array(raw_shap)
        if shap_raw.ndim == 3:
            shap_arr = shap_raw[:, :, 1]  # class 1 = suspicious
        elif isinstance(raw_shap, list):
            shap_arr = np.array(raw_shap[1])
        else:
            shap_arr = shap_raw

        shap_cols = [f"shap_{f}" for f in self.feature_names]
        shap_df = pd.DataFrame(shap_arr, columns=shap_cols)
        shap_df.insert(0, "node_id", node_df["node_id"].values)

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

        self._log_audit("explain", {"n_nodes_explained": len(shap_df)})
        logger.info("GraphScorer.explain: %d nodes explained", len(shap_df))
        return shap_df

    # ------------------------------------------------------------------
    # Feature importance (convenience — not required by BasePipeline)
    # ------------------------------------------------------------------

    def feature_importances(self) -> pd.DataFrame:
        """Return mean decrease in impurity feature importances from the fitted model.

        Args: None (uses fitted model).

        Returns:
            DataFrame with columns feature and importance, sorted descending.

        Raises:
            RuntimeError: If train() has not been called.
        """
        self._check_fitted()
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
