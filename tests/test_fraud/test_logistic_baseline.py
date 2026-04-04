"""
tests/test_fraud/test_logistic_baseline.py
============================================
Unit tests for the logistic regression fraud baseline.
"""

import numpy as np
import pandas as pd
import pytest

from fincrime_ml.core.data.synth_cards import SyntheticTransactionGenerator
from fincrime_ml.fraud.models.logistic_baseline import LogisticFraudBaseline
from fincrime_ml.fraud.models.xgb_classifier import FEATURE_COLS, XGBFraudClassifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    gen = SyntheticTransactionGenerator(n_accounts=200, seed=42)
    return gen.generate(n_transactions=1_000, fraud_rate=0.10)


@pytest.fixture(scope="module")
def fitted_baseline(train_df) -> LogisticFraudBaseline:
    baseline = LogisticFraudBaseline(n_cv_folds=3, seed=42)
    baseline.train(train_df)
    return baseline


# ---------------------------------------------------------------------------
# prepare_features tests
# ---------------------------------------------------------------------------


def test_prepare_features_returns_feature_cols(train_df):
    baseline = LogisticFraudBaseline()
    X = baseline.prepare_features(train_df)
    assert list(X.columns) == FEATURE_COLS


def test_prepare_features_all_numeric(train_df):
    baseline = LogisticFraudBaseline()
    X = baseline.prepare_features(train_df)
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def test_prepare_features_row_count(train_df):
    baseline = LogisticFraudBaseline()
    X = baseline.prepare_features(train_df)
    assert len(X) == len(train_df)


# ---------------------------------------------------------------------------
# train() tests
# ---------------------------------------------------------------------------


def test_train_returns_self(train_df):
    baseline = LogisticFraudBaseline(n_cv_folds=2, seed=0)
    result = baseline.train(train_df)
    assert result is baseline


def test_train_sets_is_fitted(train_df):
    baseline = LogisticFraudBaseline(n_cv_folds=2, seed=0)
    baseline.train(train_df)
    assert baseline._is_fitted is True


def test_train_model_is_not_none(fitted_baseline):
    assert fitted_baseline.model is not None


def test_train_scaler_is_fitted(fitted_baseline):
    assert hasattr(fitted_baseline.scaler, "mean_")


def test_train_cv_results_populated(fitted_baseline):
    assert len(fitted_baseline.cv_results) == fitted_baseline.n_cv_folds


def test_train_cv_results_keys(fitted_baseline):
    for r in fitted_baseline.cv_results:
        for key in ("fold", "auc_pr", "roc_auc", "n_train", "n_val"):
            assert key in r


def test_train_auc_pr_in_range(fitted_baseline):
    for r in fitted_baseline.cv_results:
        assert 0.0 <= r["auc_pr"] <= 1.0


def test_train_audit_log_has_train_event(fitted_baseline):
    events = [e["event"] for e in fitted_baseline.audit_log]
    assert "train" in events


def test_train_raises_on_missing_label(train_df):
    baseline = LogisticFraudBaseline(n_cv_folds=2)
    with pytest.raises(ValueError, match="label column"):
        baseline.train(train_df, label_col="missing_col")


def test_train_raises_on_non_binary_label(train_df):
    baseline = LogisticFraudBaseline(n_cv_folds=2)
    bad = train_df.copy()
    bad["is_fraud"] = 2
    with pytest.raises(ValueError, match="binary"):
        baseline.train(bad)


# ---------------------------------------------------------------------------
# predict() tests
# ---------------------------------------------------------------------------


def test_predict_returns_dataframe(fitted_baseline, train_df):
    result = fitted_baseline.predict(train_df)
    assert isinstance(result, pd.DataFrame)


def test_predict_output_columns(fitted_baseline, train_df):
    result = fitted_baseline.predict(train_df)
    for col in ["transaction_id", "risk_score", "risk_tier", "model_version", "scored_at"]:
        assert col in result.columns


def test_predict_row_count(fitted_baseline, train_df):
    result = fitted_baseline.predict(train_df)
    assert len(result) == len(train_df)


def test_predict_risk_score_range(fitted_baseline, train_df):
    result = fitted_baseline.predict(train_df)
    assert result["risk_score"].between(0.0, 1.0).all()


def test_predict_risk_tier_valid(fitted_baseline, train_df):
    result = fitted_baseline.predict(train_df)
    assert result["risk_tier"].isin(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).all()


def test_predict_fraud_rows_higher_scores(fitted_baseline, train_df):
    result = fitted_baseline.predict(train_df)
    result["is_fraud"] = train_df["is_fraud"].values
    fraud_avg = result[result["is_fraud"] == 1]["risk_score"].mean()
    legit_avg = result[result["is_fraud"] == 0]["risk_score"].mean()
    assert fraud_avg > legit_avg


def test_predict_raises_if_not_fitted(train_df):
    baseline = LogisticFraudBaseline()
    with pytest.raises(RuntimeError, match="not been fitted"):
        baseline.predict(train_df)


def test_predict_without_transaction_id(fitted_baseline, train_df):
    df_no_id = train_df.drop(columns=["transaction_id"])
    result = fitted_baseline.predict(df_no_id)
    assert "transaction_id" in result.columns


# ---------------------------------------------------------------------------
# explain() tests
# ---------------------------------------------------------------------------


def test_explain_returns_dataframe(fitted_baseline, train_df):
    result = fitted_baseline.explain(train_df.head(20))
    assert isinstance(result, pd.DataFrame)


def test_explain_has_transaction_id(fitted_baseline, train_df):
    result = fitted_baseline.explain(train_df.head(20))
    assert "transaction_id" in result.columns


def test_explain_top_reason_columns(fitted_baseline, train_df):
    result = fitted_baseline.explain(train_df.head(20), top_n=3)
    for i in range(1, 4):
        assert f"top_reason_{i}" in result.columns


def test_explain_top_reasons_are_feature_cols(fitted_baseline, train_df):
    result = fitted_baseline.explain(train_df.head(20), top_n=2)
    for col in ["top_reason_1", "top_reason_2"]:
        assert result[col].isin(FEATURE_COLS).all()


def test_explain_consistent_across_rows(fitted_baseline, train_df):
    """LR reason codes are coefficient-based so must be the same for all rows."""
    result = fitted_baseline.explain(train_df.head(50), top_n=2)
    assert result["top_reason_1"].nunique() == 1
    assert result["top_reason_2"].nunique() == 1


def test_explain_raises_if_not_fitted(train_df):
    baseline = LogisticFraudBaseline()
    with pytest.raises(RuntimeError, match="not been fitted"):
        baseline.explain(train_df.head(5))


# ---------------------------------------------------------------------------
# feature_importance() tests
# ---------------------------------------------------------------------------


def test_feature_importance_returns_dataframe(fitted_baseline):
    fi = fitted_baseline.feature_importance()
    assert isinstance(fi, pd.DataFrame)


def test_feature_importance_columns(fitted_baseline):
    fi = fitted_baseline.feature_importance()
    for col in ("feature", "coefficient", "abs_coefficient", "rank"):
        assert col in fi.columns


def test_feature_importance_row_count(fitted_baseline):
    fi = fitted_baseline.feature_importance()
    assert len(fi) == len(FEATURE_COLS)


def test_feature_importance_sorted_descending(fitted_baseline):
    fi = fitted_baseline.feature_importance()
    assert fi["abs_coefficient"].is_monotonic_decreasing


def test_feature_importance_rank_starts_at_1(fitted_baseline):
    fi = fitted_baseline.feature_importance()
    assert fi["rank"].iloc[0] == 1


def test_feature_importance_raises_if_not_fitted():
    baseline = LogisticFraudBaseline()
    with pytest.raises(RuntimeError, match="not been fitted"):
        baseline.feature_importance()


# ---------------------------------------------------------------------------
# cv_summary() and mean_cv_auc_pr() tests
# ---------------------------------------------------------------------------


def test_cv_summary_returns_dataframe(fitted_baseline):
    assert isinstance(fitted_baseline.cv_summary(), pd.DataFrame)


def test_cv_summary_row_count(fitted_baseline):
    assert len(fitted_baseline.cv_summary()) == fitted_baseline.n_cv_folds


def test_mean_cv_auc_pr_in_range(fitted_baseline):
    score = fitted_baseline.mean_cv_auc_pr()
    assert 0.0 <= score <= 1.0


def test_cv_summary_raises_before_train():
    with pytest.raises(RuntimeError, match="no CV results"):
        LogisticFraudBaseline().cv_summary()


def test_mean_cv_auc_pr_raises_before_train():
    with pytest.raises(RuntimeError, match="no CV results"):
        LogisticFraudBaseline().mean_cv_auc_pr()


# ---------------------------------------------------------------------------
# compare_with_xgb() tests
# ---------------------------------------------------------------------------


def test_compare_with_xgb_returns_dataframe(fitted_baseline, train_df):
    xgb_clf = XGBFraudClassifier(n_cv_folds=2, seed=42)
    xgb_clf.train(train_df)
    comparison = fitted_baseline.compare_with_xgb(xgb_clf)
    assert isinstance(comparison, pd.DataFrame)


def test_compare_with_xgb_columns(fitted_baseline, train_df):
    xgb_clf = XGBFraudClassifier(n_cv_folds=2, seed=42)
    xgb_clf.train(train_df)
    comparison = fitted_baseline.compare_with_xgb(xgb_clf)
    for col in ("feature", "lr_rank", "xgb_rank", "rank_delta"):
        assert col in comparison.columns


def test_compare_with_xgb_row_count(fitted_baseline, train_df):
    xgb_clf = XGBFraudClassifier(n_cv_folds=2, seed=42)
    xgb_clf.train(train_df)
    comparison = fitted_baseline.compare_with_xgb(xgb_clf)
    assert len(comparison) == len(FEATURE_COLS)


def test_compare_with_xgb_rank_delta_non_negative(fitted_baseline, train_df):
    xgb_clf = XGBFraudClassifier(n_cv_folds=2, seed=42)
    xgb_clf.train(train_df)
    comparison = fitted_baseline.compare_with_xgb(xgb_clf)
    assert (comparison["rank_delta"] >= 0).all()


def test_compare_with_xgb_raises_on_unfitted_xgb(fitted_baseline):
    unfitted_xgb = XGBFraudClassifier()
    with pytest.raises(ValueError, match="no fitted model"):
        fitted_baseline.compare_with_xgb(unfitted_xgb)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_train_reproducible(train_df):
    b1 = LogisticFraudBaseline(n_cv_folds=2, seed=7)
    b2 = LogisticFraudBaseline(n_cv_folds=2, seed=7)
    b1.train(train_df)
    b2.train(train_df)
    s1 = b1.predict(train_df)["risk_score"].values
    s2 = b2.predict(train_df)["risk_score"].values
    np.testing.assert_array_almost_equal(s1, s2, decimal=6)
