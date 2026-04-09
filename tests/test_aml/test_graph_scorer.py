"""
tests/test_aml/test_graph_scorer.py
=====================================
Unit tests for the graph-based AML anomaly scorer.

Covers: prepare_features, train, predict, explain, feature_importances,
and the module-level _assign_risk_tier helper.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fincrime_ml.aml.models.graph_scorer import GraphScorer, _assign_risk_tier
from fincrime_ml.core.base import BasePipeline, PipelineConfig
from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def aml_df() -> pd.DataFrame:
    gen = SyntheticAMLGenerator(n_accounts=300, seed=99)
    return gen.generate(n_transactions=3_000, suspicious_rate=0.08)


@pytest.fixture(scope="module")
def fitted_scorer(aml_df) -> GraphScorer:
    scorer = GraphScorer(n_estimators=50, max_depth=5)
    scorer.train(aml_df, label_col="is_suspicious")
    return scorer


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Four-node graph with two suspicious and two legitimate transactions."""
    return pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "sender_account_id": "MULE1",
                "receiver_account_id": "MULE2",
                "amount_gbp": 9_500.0,
                "timestamp": "2024-03-01 10:00:00",
                "is_mule_sender": 1,
                "is_mule_receiver": 1,
                "is_suspicious": 1,
            },
            {
                "transaction_id": "T2",
                "sender_account_id": "MULE2",
                "receiver_account_id": "CLEAN1",
                "amount_gbp": 8_900.0,
                "timestamp": "2024-03-01 14:00:00",
                "is_mule_sender": 1,
                "is_mule_receiver": 0,
                "is_suspicious": 1,
            },
            {
                "transaction_id": "T3",
                "sender_account_id": "CLEAN1",
                "receiver_account_id": "CLEAN2",
                "amount_gbp": 120.0,
                "timestamp": "2024-03-02 09:00:00",
                "is_mule_sender": 0,
                "is_mule_receiver": 0,
                "is_suspicious": 0,
            },
            {
                "transaction_id": "T4",
                "sender_account_id": "CLEAN2",
                "receiver_account_id": "CLEAN1",
                "amount_gbp": 50.0,
                "timestamp": "2024-03-02 11:00:00",
                "is_mule_sender": 0,
                "is_mule_receiver": 0,
                "is_suspicious": 0,
            },
        ]
    )


# ---------------------------------------------------------------------------
# Inheritance and configuration
# ---------------------------------------------------------------------------


def test_graph_scorer_inherits_base_pipeline():
    scorer = GraphScorer()
    assert isinstance(scorer, BasePipeline)


def test_graph_scorer_accepts_custom_config():
    cfg = PipelineConfig(random_state=7, version="0.2.0")
    scorer = GraphScorer(config=cfg)
    assert scorer.config.version == "0.2.0"
    assert scorer.config.random_state == 7


def test_graph_scorer_default_label_col():
    assert GraphScorer.LABEL_COL == "is_suspicious"


def test_graph_scorer_model_none_before_train():
    scorer = GraphScorer()
    assert scorer.model is None
    assert not scorer._is_fitted


# ---------------------------------------------------------------------------
# _assign_risk_tier
# ---------------------------------------------------------------------------


def test_risk_tier_critical():
    assert _assign_risk_tier(0.90) == "CRITICAL"


def test_risk_tier_critical_boundary():
    assert _assign_risk_tier(0.85) == "CRITICAL"


def test_risk_tier_high():
    assert _assign_risk_tier(0.70) == "HIGH"


def test_risk_tier_high_boundary():
    assert _assign_risk_tier(0.65) == "HIGH"


def test_risk_tier_medium():
    assert _assign_risk_tier(0.50) == "MEDIUM"


def test_risk_tier_medium_boundary():
    assert _assign_risk_tier(0.30) == "MEDIUM"


def test_risk_tier_low():
    assert _assign_risk_tier(0.10) == "LOW"


def test_risk_tier_zero():
    assert _assign_risk_tier(0.0) == "LOW"


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------


def test_prepare_features_returns_dataframe(aml_df):
    scorer = GraphScorer()
    node_df = scorer.prepare_features(aml_df)
    assert isinstance(node_df, pd.DataFrame)


def test_prepare_features_has_node_id(aml_df):
    scorer = GraphScorer()
    node_df = scorer.prepare_features(aml_df)
    assert "node_id" in node_df.columns


def test_prepare_features_has_centrality_cols(aml_df):
    scorer = GraphScorer()
    node_df = scorer.prepare_features(aml_df)
    for col in ("betweenness_centrality", "pagerank", "pass_through_ratio"):
        assert col in node_df.columns


def test_prepare_features_has_deviation_cols(aml_df):
    scorer = GraphScorer()
    node_df = scorer.prepare_features(aml_df)
    for col in ("betweenness_zscore", "pagerank_zscore", "pass_through_zscore"):
        assert col in node_df.columns


def test_prepare_features_one_row_per_node(aml_df):
    scorer = GraphScorer()
    node_df = scorer.prepare_features(aml_df)
    assert node_df["node_id"].nunique() == len(node_df)


def test_prepare_features_population_stats_stored(aml_df):
    scorer = GraphScorer()
    scorer.prepare_features(aml_df, use_fitted_stats=False)
    assert "betweenness_centrality" in scorer._population_stats
    assert "pagerank" in scorer._population_stats


def test_prepare_features_fitted_stats_applied(aml_df):
    """Z-scores computed with fitted stats differ from batch-derived stats."""
    scorer = GraphScorer()
    node_df_train = scorer.prepare_features(aml_df, use_fitted_stats=False)
    # Apply stored stats — should produce identical result on the same data
    node_df_infer = scorer.prepare_features(aml_df, use_fitted_stats=True)
    # betweenness_zscore should be numerically equal since same data
    assert node_df_train["betweenness_zscore"].values == pytest.approx(
        node_df_infer["betweenness_zscore"].values, abs=1e-6
    )


def test_prepare_features_minimal_df(minimal_df):
    scorer = GraphScorer()
    node_df = scorer.prepare_features(minimal_df)
    # 4 distinct accounts: MULE1, MULE2, CLEAN1, CLEAN2
    assert len(node_df) == 4


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_returns_self(aml_df):
    scorer = GraphScorer(n_estimators=20, max_depth=3)
    result = scorer.train(aml_df)
    assert result is scorer


def test_train_sets_is_fitted(aml_df):
    scorer = GraphScorer(n_estimators=20, max_depth=3)
    scorer.train(aml_df)
    assert scorer._is_fitted is True


def test_train_sets_model(aml_df):
    scorer = GraphScorer(n_estimators=20, max_depth=3)
    scorer.train(aml_df)
    assert scorer.model is not None


def test_train_sets_feature_names(aml_df):
    scorer = GraphScorer(n_estimators=20, max_depth=3)
    scorer.train(aml_df)
    assert len(scorer.feature_names) > 0


def test_train_sets_shap_explainer(aml_df):
    scorer = GraphScorer(n_estimators=20, max_depth=3)
    scorer.train(aml_df)
    assert scorer._shap_explainer is not None


def test_train_missing_label_col_raises(aml_df):
    scorer = GraphScorer(n_estimators=20, max_depth=3)
    with pytest.raises(KeyError, match="label column"):
        scorer.train(aml_df, label_col="nonexistent_col")


def test_train_no_positives_raises(minimal_df):
    """If all transactions are legitimate, training should raise ValueError."""
    df_clean = minimal_df.copy()
    df_clean["is_suspicious"] = 0
    scorer = GraphScorer(n_estimators=10, max_depth=3)
    with pytest.raises(ValueError, match="no suspicious nodes"):
        scorer.train(df_clean)


def test_train_audit_log_populated(aml_df):
    scorer = GraphScorer(n_estimators=20, max_depth=3)
    scorer.train(aml_df)
    assert len(scorer.audit_log) >= 1
    assert scorer.audit_log[0]["event"] == "train"


def test_train_all_feature_cols_present(aml_df):
    """All expected feature columns should be in feature_names after training."""
    from fincrime_ml.aml.models.graph_scorer import ALL_FEATURE_COLS

    scorer = GraphScorer(n_estimators=20, max_depth=3)
    scorer.train(aml_df)
    for col in ALL_FEATURE_COLS:
        assert col in scorer.feature_names


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_returns_dataframe(fitted_scorer, aml_df):
    result = fitted_scorer.predict(aml_df)
    assert isinstance(result, pd.DataFrame)


def test_predict_columns(fitted_scorer, aml_df):
    result = fitted_scorer.predict(aml_df)
    for col in ("node_id", "risk_score", "risk_tier", "model_version", "scored_at"):
        assert col in result.columns


def test_predict_risk_score_in_unit_interval(fitted_scorer, aml_df):
    result = fitted_scorer.predict(aml_df)
    assert (result["risk_score"] >= 0.0).all()
    assert (result["risk_score"] <= 1.0).all()


def test_predict_risk_tier_valid_values(fitted_scorer, aml_df):
    result = fitted_scorer.predict(aml_df)
    valid_tiers = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert set(result["risk_tier"].unique()).issubset(valid_tiers)


def test_predict_sorted_descending(fitted_scorer, aml_df):
    result = fitted_scorer.predict(aml_df)
    assert result["risk_score"].is_monotonic_decreasing


def test_predict_one_row_per_node(fitted_scorer, aml_df):
    result = fitted_scorer.predict(aml_df)
    assert result["node_id"].nunique() == len(result)


def test_predict_before_train_raises():
    scorer = GraphScorer()
    with pytest.raises(RuntimeError, match="fitted"):
        scorer.predict(
            pd.DataFrame(
                {
                    "transaction_id": ["T1"],
                    "sender_account_id": ["A"],
                    "receiver_account_id": ["B"],
                    "amount_gbp": [100.0],
                    "timestamp": ["2024-01-01"],
                    "is_mule_sender": [0],
                    "is_mule_receiver": [0],
                }
            )
        )


def test_predict_audit_log_updated(aml_df):
    scorer = GraphScorer(n_estimators=20, max_depth=3)
    scorer.train(aml_df)
    scorer.predict(aml_df)
    events = [e["event"] for e in scorer.audit_log]
    assert "predict" in events


def test_predict_model_version_matches_config(aml_df):
    cfg = PipelineConfig(version="0.9.9")
    scorer = GraphScorer(config=cfg, n_estimators=20, max_depth=3)
    scorer.train(aml_df)
    result = scorer.predict(aml_df)
    assert (result["model_version"] == "0.9.9").all()


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_returns_dataframe(fitted_scorer, aml_df):
    result = fitted_scorer.explain(aml_df)
    assert isinstance(result, pd.DataFrame)


def test_explain_has_node_id(fitted_scorer, aml_df):
    result = fitted_scorer.explain(aml_df)
    assert "node_id" in result.columns


def test_explain_has_shap_cols(fitted_scorer, aml_df):
    result = fitted_scorer.explain(aml_df)
    shap_cols = [c for c in result.columns if c.startswith("shap_")]
    assert len(shap_cols) == len(fitted_scorer.feature_names)


def test_explain_has_reason_code_cols(fitted_scorer, aml_df):
    result = fitted_scorer.explain(aml_df)
    for rank in range(1, 4):
        assert f"top_reason_{rank}" in result.columns
        assert f"top_shap_{rank}" in result.columns


def test_explain_reason_codes_are_feature_names(fitted_scorer, aml_df):
    result = fitted_scorer.explain(aml_df)
    valid_features = set(fitted_scorer.feature_names)
    for col in ("top_reason_1", "top_reason_2", "top_reason_3"):
        assert set(result[col].unique()).issubset(valid_features)


def test_explain_row_count_matches_predict(fitted_scorer, aml_df):
    scores = fitted_scorer.predict(aml_df)
    explanations = fitted_scorer.explain(aml_df)
    assert len(scores) == len(explanations)


def test_explain_before_train_raises():
    scorer = GraphScorer()
    with pytest.raises(RuntimeError, match="fitted"):
        scorer.explain(
            pd.DataFrame(
                {
                    "transaction_id": ["T1"],
                    "sender_account_id": ["A"],
                    "receiver_account_id": ["B"],
                    "amount_gbp": [100.0],
                    "timestamp": ["2024-01-01"],
                    "is_mule_sender": [0],
                    "is_mule_receiver": [0],
                }
            )
        )


def test_explain_audit_log_updated(aml_df):
    scorer = GraphScorer(n_estimators=20, max_depth=3)
    scorer.train(aml_df)
    scorer.explain(aml_df)
    events = [e["event"] for e in scorer.audit_log]
    assert "explain" in events


# ---------------------------------------------------------------------------
# feature_importances
# ---------------------------------------------------------------------------


def test_feature_importances_returns_dataframe(fitted_scorer):
    df = fitted_scorer.feature_importances()
    assert isinstance(df, pd.DataFrame)


def test_feature_importances_columns(fitted_scorer):
    df = fitted_scorer.feature_importances()
    assert "feature" in df.columns
    assert "importance" in df.columns


def test_feature_importances_sum_to_one(fitted_scorer):
    df = fitted_scorer.feature_importances()
    assert df["importance"].sum() == pytest.approx(1.0, abs=1e-4)


def test_feature_importances_sorted_descending(fitted_scorer):
    df = fitted_scorer.feature_importances()
    assert df["importance"].is_monotonic_decreasing


def test_feature_importances_before_train_raises():
    scorer = GraphScorer()
    with pytest.raises(RuntimeError, match="fitted"):
        scorer.feature_importances()


# ---------------------------------------------------------------------------
# End-to-end: mule accounts score higher than clean accounts
# ---------------------------------------------------------------------------


def test_mule_accounts_score_higher_than_clean(aml_df):
    """On average, nodes linked to suspicious transactions should score higher."""
    scorer = GraphScorer(n_estimators=100, max_depth=6)
    scorer.train(aml_df)
    scores = scorer.predict(aml_df)

    # Identify mule nodes from the generator data
    mule_senders = set(aml_df.loc[aml_df["is_mule_sender"] == 1, "sender_account_id"])
    mule_receivers = set(aml_df.loc[aml_df["is_mule_receiver"] == 1, "receiver_account_id"])
    mule_nodes = mule_senders | mule_receivers

    mule_scores = scores.loc[scores["node_id"].isin(mule_nodes), "risk_score"]
    clean_scores = scores.loc[~scores["node_id"].isin(mule_nodes), "risk_score"]

    assert mule_scores.mean() > clean_scores.mean(), (
        f"Expected mule mean score ({mule_scores.mean():.3f}) > "
        f"clean mean score ({clean_scores.mean():.3f})"
    )
