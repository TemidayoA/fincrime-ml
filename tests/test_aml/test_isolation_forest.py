"""
tests/test_aml/test_isolation_forest.py
=========================================
Unit tests for the unsupervised AML Isolation Forest baseline.

Covers: prepare_features, train (no-label), predict, explain, evaluate,
and the module-level _assign_risk_tier helper.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fincrime_ml.aml.models.isolation_forest import (
    AMLIsolationForest,
    _assign_risk_tier,
)
from fincrime_ml.core.base import BasePipeline, PipelineConfig
from fincrime_ml.core.data.synth_aml import SyntheticAMLGenerator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def aml_df() -> pd.DataFrame:
    gen = SyntheticAMLGenerator(n_accounts=300, seed=11)
    return gen.generate(n_transactions=3_000, suspicious_rate=0.06)


@pytest.fixture(scope="module")
def fitted_model(aml_df) -> AMLIsolationForest:
    model = AMLIsolationForest(n_estimators=50)
    model.train(aml_df)
    return model


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Small DataFrame with both suspicious and legitimate transactions."""
    rows = []
    for i in range(20):
        is_sus = int(i < 3)
        rows.append(
            {
                "transaction_id": f"T{i:04d}",
                "sender_account_id": f"C{i:03d}",
                "receiver_account_id": f"C{(i+1):03d}",
                "amount_gbp": 9_200.0 if is_sus else 45.0,
                "hour_of_day": 2 if is_sus else 14,
                "day_of_week": 1,
                "layering_depth": 1 if is_sus else 0,
                "structuring_flag": 1 if is_sus else 0,
                "rapid_movement_flag": 0,
                "is_suspicious": is_sus,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inheritance and configuration
# ---------------------------------------------------------------------------


def test_inherits_base_pipeline():
    assert isinstance(AMLIsolationForest(), BasePipeline)


def test_accepts_custom_config():
    cfg = PipelineConfig(random_state=7, version="1.0.0")
    model = AMLIsolationForest(config=cfg)
    assert model.config.version == "1.0.0"


def test_default_label_col():
    assert AMLIsolationForest.LABEL_COL == "is_suspicious"


def test_model_none_before_train():
    model = AMLIsolationForest()
    assert model.model is None
    assert not model._is_fitted


# ---------------------------------------------------------------------------
# _assign_risk_tier
# ---------------------------------------------------------------------------


def test_risk_tier_critical():
    assert _assign_risk_tier(0.90) == "CRITICAL"


def test_risk_tier_critical_boundary():
    assert _assign_risk_tier(0.85) == "CRITICAL"


def test_risk_tier_high():
    assert _assign_risk_tier(0.70) == "HIGH"


def test_risk_tier_medium():
    assert _assign_risk_tier(0.50) == "MEDIUM"


def test_risk_tier_low():
    assert _assign_risk_tier(0.20) == "LOW"


def test_risk_tier_zero():
    assert _assign_risk_tier(0.0) == "LOW"


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------


def test_prepare_features_returns_dataframe(aml_df):
    model = AMLIsolationForest()
    feat = model.prepare_features(aml_df)
    assert isinstance(feat, pd.DataFrame)


def test_prepare_features_has_log_amount(aml_df):
    model = AMLIsolationForest()
    feat = model.prepare_features(aml_df)
    assert "log_amount_gbp" in feat.columns


def test_prepare_features_includes_core_cols(aml_df):
    model = AMLIsolationForest()
    feat = model.prepare_features(aml_df)
    for col in ("amount_gbp", "hour_of_day", "day_of_week", "structuring_flag"):
        assert col in feat.columns


def test_prepare_features_optional_cols_included_when_present(aml_df):
    model = AMLIsolationForest()
    feat = model.prepare_features(aml_df)
    # SyntheticAMLGenerator output has is_mule_sender / is_mule_receiver
    assert "is_mule_sender" in feat.columns
    assert "is_mule_receiver" in feat.columns


def test_prepare_features_optional_cols_absent_when_missing(aml_df):
    df_no_mule = aml_df.drop(columns=["is_mule_sender", "is_mule_receiver"], errors="ignore")
    model = AMLIsolationForest()
    feat = model.prepare_features(df_no_mule)
    assert "is_mule_sender" not in feat.columns
    assert "is_mule_receiver" not in feat.columns


def test_prepare_features_row_count_preserved(aml_df):
    model = AMLIsolationForest()
    feat = model.prepare_features(aml_df)
    assert len(feat) == len(aml_df)


def test_prepare_features_log_amount_non_negative(aml_df):
    model = AMLIsolationForest()
    feat = model.prepare_features(aml_df)
    assert (feat["log_amount_gbp"] >= 0).all()


def test_prepare_features_missing_required_col_raises():
    bad_df = pd.DataFrame({"amount_gbp": [100.0], "hour_of_day": [12]})
    model = AMLIsolationForest()
    with pytest.raises(KeyError, match="required columns missing"):
        model.prepare_features(bad_df)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_returns_self(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    result = model.train(aml_df)
    assert result is model


def test_train_sets_is_fitted(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    assert model._is_fitted is True


def test_train_sets_model(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    assert model.model is not None


def test_train_sets_feature_names(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    assert len(model.feature_names) > 0


def test_train_sets_shap_explainer(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    assert model._shap_explainer is not None


def test_train_sets_score_anchors(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    assert model._score_max >= model._score_min


def test_train_ignores_label_col(aml_df):
    """train() must succeed whether or not label_col exists in the DataFrame."""
    model_labelled = AMLIsolationForest(n_estimators=20)
    model_labelled.train(aml_df, label_col="is_suspicious")

    model_unlabelled = AMLIsolationForest(n_estimators=20)
    df_no_label = aml_df.drop(columns=["is_suspicious"])
    model_unlabelled.train(df_no_label, label_col="is_suspicious")  # col absent, still fits

    assert model_labelled._is_fitted
    assert model_unlabelled._is_fitted


def test_train_too_few_rows_raises():
    model = AMLIsolationForest(n_estimators=10)
    tiny_df = pd.DataFrame(
        {
            "amount_gbp": [100.0],
            "hour_of_day": [10],
            "day_of_week": [1],
            "layering_depth": [0],
            "structuring_flag": [0],
            "rapid_movement_flag": [0],
        }
    )
    with pytest.raises(ValueError, match="at least 2 rows"):
        model.train(tiny_df)


def test_train_audit_log_populated(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    assert len(model.audit_log) >= 1
    assert model.audit_log[0]["event"] == "train"


def test_train_audit_log_marks_unsupervised(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    assert model.audit_log[0]["supervised"] is False


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_returns_dataframe(fitted_model, aml_df):
    result = fitted_model.predict(aml_df)
    assert isinstance(result, pd.DataFrame)


def test_predict_has_required_columns(fitted_model, aml_df):
    result = fitted_model.predict(aml_df)
    for col in ("risk_score", "risk_tier", "model_version", "scored_at"):
        assert col in result.columns


def test_predict_includes_transaction_id_when_present(fitted_model, aml_df):
    result = fitted_model.predict(aml_df)
    assert "transaction_id" in result.columns


def test_predict_omits_transaction_id_when_absent(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    df_no_id = aml_df.drop(columns=["transaction_id"])
    model.train(df_no_id)
    result = model.predict(df_no_id)
    assert "transaction_id" not in result.columns


def test_predict_risk_score_in_unit_interval(fitted_model, aml_df):
    result = fitted_model.predict(aml_df)
    assert (result["risk_score"] >= 0.0).all()
    assert (result["risk_score"] <= 1.0).all()


def test_predict_risk_tier_valid_values(fitted_model, aml_df):
    result = fitted_model.predict(aml_df)
    assert set(result["risk_tier"].unique()).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def test_predict_sorted_descending(fitted_model, aml_df):
    result = fitted_model.predict(aml_df)
    assert result["risk_score"].is_monotonic_decreasing


def test_predict_row_count(fitted_model, aml_df):
    result = fitted_model.predict(aml_df)
    assert len(result) == len(aml_df)


def test_predict_before_train_raises():
    model = AMLIsolationForest()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(
            pd.DataFrame(
                {
                    "amount_gbp": [100.0],
                    "hour_of_day": [12],
                    "day_of_week": [1],
                    "layering_depth": [0],
                    "structuring_flag": [0],
                    "rapid_movement_flag": [0],
                }
            )
        )


def test_predict_audit_log_updated(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    model.predict(aml_df)
    events = [e["event"] for e in model.audit_log]
    assert "predict" in events


def test_predict_version_from_config(aml_df):
    cfg = PipelineConfig(version="2.0.0")
    model = AMLIsolationForest(config=cfg, n_estimators=20)
    model.train(aml_df)
    result = model.predict(aml_df)
    assert (result["model_version"] == "2.0.0").all()


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_returns_dataframe(fitted_model, aml_df):
    result = fitted_model.explain(aml_df)
    assert isinstance(result, pd.DataFrame)


def test_explain_has_shap_cols(fitted_model, aml_df):
    result = fitted_model.explain(aml_df)
    shap_cols = [c for c in result.columns if c.startswith("shap_")]
    assert len(shap_cols) == len(fitted_model.feature_names)


def test_explain_has_reason_code_cols(fitted_model, aml_df):
    result = fitted_model.explain(aml_df)
    for rank in range(1, 4):
        assert f"top_reason_{rank}" in result.columns
        assert f"top_shap_{rank}" in result.columns


def test_explain_reason_codes_are_feature_names(fitted_model, aml_df):
    result = fitted_model.explain(aml_df)
    valid = set(fitted_model.feature_names)
    for col in ("top_reason_1", "top_reason_2", "top_reason_3"):
        assert set(result[col].unique()).issubset(valid)


def test_explain_row_count_matches_predict(fitted_model, aml_df):
    scores = fitted_model.predict(aml_df)
    explanations = fitted_model.explain(aml_df)
    assert len(scores) == len(explanations)


def test_explain_includes_transaction_id_when_present(fitted_model, aml_df):
    result = fitted_model.explain(aml_df)
    assert "transaction_id" in result.columns


def test_explain_before_train_raises():
    model = AMLIsolationForest()
    with pytest.raises(RuntimeError, match="fitted"):
        model.explain(
            pd.DataFrame(
                {
                    "amount_gbp": [100.0],
                    "hour_of_day": [12],
                    "day_of_week": [1],
                    "layering_depth": [0],
                    "structuring_flag": [0],
                    "rapid_movement_flag": [0],
                }
            )
        )


def test_explain_audit_log_updated(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    model.explain(aml_df)
    events = [e["event"] for e in model.audit_log]
    assert "explain" in events


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_dict(fitted_model, aml_df):
    metrics = fitted_model.evaluate(aml_df)
    assert isinstance(metrics, dict)


def test_evaluate_has_auc_pr(fitted_model, aml_df):
    metrics = fitted_model.evaluate(aml_df)
    assert "auc_pr" in metrics


def test_evaluate_has_roc_auc(fitted_model, aml_df):
    metrics = fitted_model.evaluate(aml_df)
    assert "roc_auc" in metrics


def test_evaluate_metrics_in_unit_interval(fitted_model, aml_df):
    metrics = fitted_model.evaluate(aml_df)
    assert 0.0 <= metrics["auc_pr"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_missing_label_col_raises(fitted_model, aml_df):
    with pytest.raises(KeyError, match="label column"):
        fitted_model.evaluate(aml_df, label_col="nonexistent")


def test_evaluate_before_train_raises():
    model = AMLIsolationForest()
    with pytest.raises(RuntimeError, match="fitted"):
        model.evaluate(
            pd.DataFrame(
                {
                    "amount_gbp": [100.0, 200.0],
                    "hour_of_day": [12, 3],
                    "day_of_week": [1, 5],
                    "layering_depth": [0, 1],
                    "structuring_flag": [0, 1],
                    "rapid_movement_flag": [0, 0],
                    "is_suspicious": [0, 1],
                }
            )
        )


def test_evaluate_audit_log_updated(aml_df):
    model = AMLIsolationForest(n_estimators=20)
    model.train(aml_df)
    model.evaluate(aml_df)
    events = [e["event"] for e in model.audit_log]
    assert "evaluate" in events


# ---------------------------------------------------------------------------
# End-to-end: suspicious transactions score higher on average
# ---------------------------------------------------------------------------


def test_suspicious_transactions_score_higher_on_average(minimal_df):
    """Structuring + high-amount + unusual-hour transactions should outscore legitimate."""
    model = AMLIsolationForest(n_estimators=100)
    model.train(minimal_df)
    scores = model.predict(minimal_df)

    # Merge labels back via transaction_id
    labelled = scores.merge(
        minimal_df[["transaction_id", "is_suspicious"]], on="transaction_id", how="left"
    )

    sus_mean = labelled.loc[labelled["is_suspicious"] == 1, "risk_score"].mean()
    legit_mean = labelled.loc[labelled["is_suspicious"] == 0, "risk_score"].mean()

    assert (
        sus_mean > legit_mean
    ), f"Expected suspicious mean ({sus_mean:.3f}) > legitimate mean ({legit_mean:.3f})"


# ---------------------------------------------------------------------------
# No-label scenario: train without is_suspicious column at all
# ---------------------------------------------------------------------------


def test_train_and_predict_without_any_labels(aml_df):
    """Core use case: fully unsupervised — no label column present anywhere."""
    df_unlabelled = aml_df.drop(columns=["is_suspicious"])
    model = AMLIsolationForest(n_estimators=30)
    model.train(df_unlabelled)
    scores = model.predict(df_unlabelled)
    assert len(scores) == len(df_unlabelled)
    assert (scores["risk_score"].between(0.0, 1.0)).all()
