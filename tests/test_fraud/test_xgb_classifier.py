"""
tests/test_fraud/test_xgb_classifier.py
=========================================
Unit and integration tests for the XGBoost fraud classifier.
"""

import numpy as np
import pandas as pd
import pytest

from fincrime_ml.core.data.synth_cards import SyntheticTransactionGenerator
from fincrime_ml.fraud.models.xgb_classifier import (
    FEATURE_COLS,
    XGBFraudClassifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    """Small synthetic training set with enough fraud for stable CV."""
    gen = SyntheticTransactionGenerator(n_accounts=200, seed=42)
    return gen.generate(n_transactions=1_000, fraud_rate=0.10)


@pytest.fixture(scope="module")
def fitted_clf(train_df) -> XGBFraudClassifier:
    """Pre-fitted classifier shared across tests in this module."""
    clf = XGBFraudClassifier(n_cv_folds=3, seed=42)
    clf.train(train_df)
    return clf


# ---------------------------------------------------------------------------
# FEATURE_COLS constant tests
# ---------------------------------------------------------------------------


def test_feature_cols_non_empty():
    assert len(FEATURE_COLS) > 0


def test_feature_cols_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_feature_cols_all_strings():
    assert all(isinstance(c, str) for c in FEATURE_COLS)


# ---------------------------------------------------------------------------
# prepare_features tests
# ---------------------------------------------------------------------------


def test_prepare_features_returns_correct_columns(train_df):
    clf = XGBFraudClassifier()
    X = clf.prepare_features(train_df)
    assert list(X.columns) == FEATURE_COLS


def test_prepare_features_row_count_preserved(train_df):
    clf = XGBFraudClassifier()
    X = clf.prepare_features(train_df)
    assert len(X) == len(train_df)


def test_prepare_features_all_numeric(train_df):
    clf = XGBFraudClassifier()
    X = clf.prepare_features(train_df)
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def test_prepare_features_no_inf_values(train_df):
    clf = XGBFraudClassifier()
    X = clf.prepare_features(train_df)
    assert not np.isinf(X.to_numpy()).any()


# ---------------------------------------------------------------------------
# train() tests
# ---------------------------------------------------------------------------


def test_train_returns_self(train_df):
    clf = XGBFraudClassifier(n_cv_folds=2, seed=0)
    result = clf.train(train_df)
    assert result is clf


def test_train_sets_is_fitted(train_df):
    clf = XGBFraudClassifier(n_cv_folds=2, seed=0)
    clf.train(train_df)
    assert clf._is_fitted is True


def test_train_model_is_not_none(fitted_clf):
    assert fitted_clf.model is not None


def test_train_cv_results_populated(fitted_clf):
    assert len(fitted_clf.cv_results) == fitted_clf.n_cv_folds


def test_train_cv_results_have_correct_keys(fitted_clf):
    for result in fitted_clf.cv_results:
        assert "fold" in result
        assert "auc_pr" in result
        assert "roc_auc" in result
        assert "n_train" in result
        assert "n_val" in result


def test_train_auc_pr_in_valid_range(fitted_clf):
    for result in fitted_clf.cv_results:
        assert 0.0 <= result["auc_pr"] <= 1.0


def test_train_roc_auc_in_valid_range(fitted_clf):
    for result in fitted_clf.cv_results:
        assert 0.0 <= result["roc_auc"] <= 1.0


def test_train_audit_log_populated(fitted_clf):
    assert len(fitted_clf.audit_log) > 0
    events = [e["event"] for e in fitted_clf.audit_log]
    assert "train" in events


def test_train_raises_on_missing_label_col(train_df):
    clf = XGBFraudClassifier(n_cv_folds=2)
    with pytest.raises(ValueError, match="label column"):
        clf.train(train_df, label_col="nonexistent_col")


def test_train_raises_on_non_binary_label(train_df):
    clf = XGBFraudClassifier(n_cv_folds=2)
    bad_df = train_df.copy()
    bad_df["is_fraud"] = 2  # non-binary
    with pytest.raises(ValueError, match="binary"):
        clf.train(bad_df)


# ---------------------------------------------------------------------------
# predict() tests
# ---------------------------------------------------------------------------


def test_predict_returns_dataframe(fitted_clf, train_df):
    result = fitted_clf.predict(train_df)
    assert isinstance(result, pd.DataFrame)


def test_predict_output_columns(fitted_clf, train_df):
    result = fitted_clf.predict(train_df)
    for col in ["transaction_id", "risk_score", "risk_tier", "model_version", "scored_at"]:
        assert col in result.columns


def test_predict_row_count_matches_input(fitted_clf, train_df):
    result = fitted_clf.predict(train_df)
    assert len(result) == len(train_df)


def test_predict_risk_score_in_range(fitted_clf, train_df):
    result = fitted_clf.predict(train_df)
    assert result["risk_score"].between(0.0, 1.0).all()


def test_predict_risk_tier_valid_values(fitted_clf, train_df):
    result = fitted_clf.predict(train_df)
    assert result["risk_tier"].isin(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).all()


def test_predict_fraud_rows_higher_scores(fitted_clf, train_df):
    """Fraud rows should have higher average risk scores than legitimate rows."""
    result = fitted_clf.predict(train_df)
    result["is_fraud"] = train_df["is_fraud"].values
    fraud_avg = result[result["is_fraud"] == 1]["risk_score"].mean()
    legit_avg = result[result["is_fraud"] == 0]["risk_score"].mean()
    assert fraud_avg > legit_avg


def test_predict_adds_audit_log_entry(fitted_clf, train_df):
    n_before = len(fitted_clf.audit_log)
    fitted_clf.predict(train_df.head(10))
    assert len(fitted_clf.audit_log) > n_before


def test_predict_raises_if_not_fitted(train_df):
    clf = XGBFraudClassifier()
    with pytest.raises(RuntimeError, match="not been fitted"):
        clf.predict(train_df)


def test_predict_without_transaction_id_col(fitted_clf, train_df):
    """predict() must not fail when transaction_id column is absent."""
    df_no_id = train_df.drop(columns=["transaction_id"])
    result = fitted_clf.predict(df_no_id)
    assert "transaction_id" in result.columns


# ---------------------------------------------------------------------------
# explain() tests
# ---------------------------------------------------------------------------


def test_explain_returns_dataframe(fitted_clf, train_df):
    result = fitted_clf.explain(train_df.head(20))
    assert isinstance(result, pd.DataFrame)


def test_explain_output_has_transaction_id(fitted_clf, train_df):
    result = fitted_clf.explain(train_df.head(20))
    assert "transaction_id" in result.columns


def test_explain_top_reason_columns_present(fitted_clf, train_df):
    result = fitted_clf.explain(train_df.head(20), top_n=3)
    for i in range(1, 4):
        assert f"top_reason_{i}" in result.columns


def test_explain_top_reasons_are_valid_feature_names(fitted_clf, train_df):
    result = fitted_clf.explain(train_df.head(20), top_n=2)
    for col in ["top_reason_1", "top_reason_2"]:
        assert result[col].isin(FEATURE_COLS).all()


def test_explain_row_count_matches_input(fitted_clf, train_df):
    result = fitted_clf.explain(train_df.head(30))
    assert len(result) == 30


def test_explain_raises_if_not_fitted(train_df):
    clf = XGBFraudClassifier()
    with pytest.raises(RuntimeError, match="not been fitted"):
        clf.explain(train_df.head(5))


# ---------------------------------------------------------------------------
# cv_summary() and mean_cv_auc_pr() tests
# ---------------------------------------------------------------------------


def test_cv_summary_returns_dataframe(fitted_clf):
    summary = fitted_clf.cv_summary()
    assert isinstance(summary, pd.DataFrame)


def test_cv_summary_row_count(fitted_clf):
    summary = fitted_clf.cv_summary()
    assert len(summary) == fitted_clf.n_cv_folds


def test_mean_cv_auc_pr_in_range(fitted_clf):
    score = fitted_clf.mean_cv_auc_pr()
    assert 0.0 <= score <= 1.0


def test_cv_summary_raises_before_training():
    clf = XGBFraudClassifier()
    with pytest.raises(RuntimeError, match="no CV results"):
        clf.cv_summary()


def test_mean_cv_auc_pr_raises_before_training():
    clf = XGBFraudClassifier()
    with pytest.raises(RuntimeError, match="no CV results"):
        clf.mean_cv_auc_pr()


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


def test_train_reproducible_with_same_seed(train_df):
    clf1 = XGBFraudClassifier(n_cv_folds=2, seed=99)
    clf2 = XGBFraudClassifier(n_cv_folds=2, seed=99)
    clf1.train(train_df)
    clf2.train(train_df)
    scores1 = clf1.predict(train_df)["risk_score"].values
    scores2 = clf2.predict(train_df)["risk_score"].values
    np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)
